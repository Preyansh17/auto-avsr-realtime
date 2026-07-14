#!/usr/bin/env python3
"""Streaming eval for CarelessWhisper/WhisperRT (arXiv 2508.12301) causal Whisper.

Third streaming-Whisper route, next to whisper_streaming_eval.py
(SimulStreaming/AlignAtt over an UNMODIFIED checkpoint). CarelessWhisper is a
genuinely causal retrain: LoRA-finetuned encoder with causal attention masks,
O(1) per-chunk decode with KV caches, chunk sizes down to 40ms -- the route
flagged in results/whisper_streaming_investigation_2026-07-09.md as the only
real lever below the ~1.4s AlignAtt latency floor.

Released checkpoints download anonymously from the public HF repo
MLSpeech/CarelessWhisper-Streaming (no login needed -- this script resolves
them itself because upstream hardcodes token=True).

Mirrors whisper_streaming_eval.py's contract: one utterance == one stream,
--manifest-tsv with path/ref columns, shared light normalizer from
asr_baselines.metrics, latency in SIMULATED REAL TIME (chunk k's text can
exist no earlier than (k+1)*chunk + compute(chunk k)).

Metrics: WER/CER, RTF, ms/chunk, TTFT, and (greedy decode only, utterances
that fit one 30s context window) word commit lag from the model's own
streaming timestamps. Beam decode has no timestamps in this codebase, so beam
runs report WER/TTFT only -- same "lag unavailable" caveat class as Nemotron.

Requires the CarelessWhisper-streaming checkout on PYTHONPATH
(--careless-dir, github.com/tomer9080/CarelessWhisper-streaming; CC BY-NC 4.0
for its original code -- research use only).

  python -m asr_baselines.carelesswhisper_streaming_eval \
    --model small --chunk-ms 300 \
    --manifest-tsv $EXP/whisper_largev3_specaug_splitv1_legal_seed1/test_hyps.tsv \
    --careless-dir /scratch/$USER/third_party/CarelessWhisper-streaming \
    --out cw_streaming_hyps.tsv

--model is a released size (base/small/large-v2, with --chunk-ms one of their
released granularities) OR a path to a finetuned .pt from their train.py
(chunk size then comes from the checkpoint's cfg).
"""

import argparse
import os
import re
import sys
import time
import types

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_baselines.metrics import compute_wer_cer  # noqa: E402

# whisper special-token markup (<|notimestamps|> etc.) leaks into decoded text
# on some paths; never count it as words.
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def _stub_pyaudio():
    """whisper_rt.audio imports pyaudio at module top for the microphone path,
    which we never use; a stub avoids needing portaudio on the cluster."""
    try:
        import pyaudio  # noqa: F401
    except Exception:
        stub = types.ModuleType("pyaudio")
        stub.paInt16 = 8  # only read as a default-arg value, never used
        sys.modules["pyaudio"] = stub


def read_manifest_tsv(path):
    items = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        pi, ri = header.index("path"), header.index("ref")
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) > max(pi, ri):
                items.append({"path": cols[pi], "ref": cols[ri]})
    return items


def stable_commit_times(snapshots, final_words):
    """snapshots: [(emit_sim, text)] per chunk. For each final word i, the
    emit time of the earliest chunk from which the prefix final_words[:i+1]
    was already in place and never revised afterwards."""
    commits = [None] * len(final_words)
    for emit_sim, text in snapshots:
        words = text.split()
        for i in range(len(final_words)):
            prefix_ok = len(words) > i and words[: i + 1] == final_words[: i + 1]
            if prefix_ok:
                if commits[i] is None:
                    commits[i] = emit_sim
            else:
                commits[i] = None  # revised later -> not committed yet
    return commits


def word_end_times(result, tokenizer):
    """(word, end_seconds) pairs from a greedy DecodingResult's timed_tokens
    (text tokens interleaved with whisper timestamp tokens). Returns None if
    timestamps are absent (beam decode) or unparseable."""
    timed = getattr(result, "timed_tokens", None)
    if not timed or len(timed) <= 1:
        return None
    ends, words, cur = [], [], []
    last_ts = None
    for t in timed:
        if t >= tokenizer.timestamp_begin:
            last_ts = (t - tokenizer.timestamp_begin) * 0.02
            if cur:
                for w in tokenizer.decode(cur).split():
                    words.append(w)
                    ends.append(last_ts)
                cur = []
        elif t < tokenizer.eot:
            cur.append(t)
    if cur and last_ts is not None:
        for w in tokenizer.decode(cur).split():
            words.append(w)
            ends.append(last_ts)
    return list(zip(words, ends)) if words else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="released size (base/small/large-v2) or path to a finetuned .pt")
    p.add_argument("--chunk-ms", type=int, default=300,
                   help="released granularity to fetch for a named --model (40/100/200/300/1000); "
                        "with --rcs, the actual streaming chunk size (multiple of 20)")
    p.add_argument("--rcs", action="store_true", help="use the random-chunk-size checkpoint")
    p.add_argument("--extra-initial-blocks", type=int, default=None,
                   help="RCS models only: extra lookahead blocks (their README uses 2)")
    p.add_argument("--multilingual", action="store_true")
    p.add_argument("--manifest-tsv", required=True, help="TSV with path/ref columns")
    p.add_argument("--careless-dir", required=True, help="CarelessWhisper-streaming checkout")
    p.add_argument("--out", default=None)
    p.add_argument("--beams", type=int, default=0,
                   help="0 = greedy (enables word-lag via streaming timestamps); "
                        ">0 = beam search (WER/TTFT only)")
    p.add_argument("--no-ca-kv-cache", action="store_true",
                   help="disable the cross-attention KV cache (their recommended default is on)")
    p.add_argument("--sa-kv-cache", action="store_true", help="also cache decoder self-attention")
    p.add_argument("--no-word-lag", action="store_true",
                   help="skip streaming timestamps / word-lag even in greedy mode")
    p.add_argument("--language", default="en")
    p.add_argument("--max-sec-context", type=int, default=30)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    _stub_pyaudio()
    sys.path.insert(0, args.careless_dir)
    import whisper_rt
    from whisper_rt.audio import SAMPLE_RATE, MyStream, SpectrogramStream, load_audio
    from whisper_rt.streaming_decoding import DecodingOptions
    from whisper_rt.tokenizer import get_tokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve named checkpoints ourselves: the HF repo is public and ungated,
    # but upstream's load_streaming_model hardcodes token=True, which fails
    # without a stored login. Anonymous download, then pass the local path
    # (their path branch reads gran/rank/extra_gran_blocks from the ckpt cfg).
    model_ref = args.model
    if not os.path.isfile(model_ref):
        from huggingface_hub import hf_hub_download
        subname = (f"{args.chunk_ms}-multi" if args.multilingual
                   else "rcs" if args.rcs else str(args.chunk_ms))
        filename = whisper_rt._STREAMING_MODELS_HF[args.model][subname]
        model_ref = hf_hub_download(repo_id="MLSpeech/CarelessWhisper-Streaming",
                                    filename=filename, repo_type="model")
    model = whisper_rt.load_streaming_model(
        name=model_ref,
        gran=args.chunk_ms,
        multilingual=args.multilingual,
        varying_chunk_size=args.rcs,
        device=device,
    )
    model.eval()

    ms_gran = args.chunk_ms if args.rcs else model.encoder.gran * 20
    assert ms_gran % 20 == 0, "chunk must be a multiple of 20ms"
    chunk_s = ms_gran / 1000.0
    chunk_samples = int(ms_gran / 1000 * SAMPLE_RATE)
    extra_blocks = (args.extra_initial_blocks if args.extra_initial_blocks is not None
                    else model.encoder.extra_gran_blocks)
    # 0 (not None) is their greedy path: DecodingTask does `beam_size > 0`
    # unguarded, and the timestamp branch checks `beam_size == 0`.
    beam_size = args.beams if args.beams and args.beams > 0 else 0
    greedy = beam_size == 0

    opts = DecodingOptions(
        language=args.language,
        gran=ms_gran // 20,
        single_frame_mel=True,
        without_timestamps=True,
        beam_size=beam_size,
        temperature=0,
        length_penalty=None,
        look_ahead_blocks=extra_blocks,
        patience=None,
        stream_decode=True,
        use_kv_cache=args.sa_kv_cache,
        use_ca_kv_cache=not args.no_ca_kv_cache,
        # word-lag needs the greedy decoder's timestamps_map; their dataclass
        # defaults this True but their own CLI passes False -- be explicit.
        streaming_timestamps=greedy and not args.no_word_lag,
    )

    try:
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=getattr(model, "num_languages", 99),
            language=args.language,
            task="transcribe",
        )
    except Exception:
        tokenizer = None

    items = read_manifest_tsv(args.manifest_tsv)
    reset_len = args.max_sec_context * SAMPLE_RATE + 360  # mirrors their transcribe()
    is_cuda = device.startswith("cuda")

    refs, hyps, rows = [], [], []
    total_audio = total_compute = 0.0
    total_chunks = 0
    ttft_sims, word_lags = [], []
    n_reset_utts = 0

    for it in items:
        # Label CSV paths point at video-only lip-crop mp4s; audio is the
        # sibling 16kHz .wav (same convention as patient_audio.load_waveform).
        wav_path = os.path.splitext(it["path"])[0] + ".wav"
        src = wav_path if os.path.isfile(wav_path) else it["path"]
        # Patient wavs are already 16kHz mono; read directly (whisper's
        # load_audio shells out to ffmpeg, absent on the cluster nodes).
        try:
            import soundfile as sf
            audio, sr = sf.read(src, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                raise ValueError(f"{src}: {sr}Hz, need ffmpeg resample")
        except Exception:
            audio = load_audio(src)
        dur = len(audio) / SAMPLE_RATE
        total_audio += dur

        # MyStream floor-divides the buffer into whole chunks, silently
        # dropping a partial tail (up to chunk_ms of real speech) -- pad to a
        # whole multiple so the tail is decoded.
        pad = (-len(audio)) % chunk_samples
        buf = torch.from_numpy(np.pad(audio, (0, pad)) if pad else audio)
        stream = MyStream(ms_gran, simulate_stream=True, wav_file=buf,
                          pad_trim=False, use_latency=False)
        stream.open_stream()

        model.reset(use_stream=True)
        spec = SpectrogramStream(n_mels=model.dims.n_mels)
        frames = []
        committed = ""       # text finalized by 30s context resets
        snapshots = []       # (emit_sim, full text so far) per chunk
        last_result = None
        did_reset = False

        for k, frame in enumerate(stream.read()):
            frames.extend(frame)
            if len(frames) >= reset_len:  # positional-embedding limit: fresh window
                frame = np.concatenate((frames[-360:], frame))
                frames = list(frame)
                model.reset(use_stream=True)
                spec.reset()
                if last_result is not None:
                    committed = (committed + " " + SPECIAL_TOKEN_RE.sub("", last_result.text)).strip()
                did_reset = True

            audio_avail = min((k + 1) * chunk_s, dur)
            if is_cuda:
                torch.cuda.synchronize()
            tc = time.time()
            frame_tensor = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
            mel = spec.calc_mel_with_new_frame(frame_tensor.to(model.device))
            result = model.decode(mel.squeeze(0), opts)
            if is_cuda:
                torch.cuda.synchronize()
            compute = time.time() - tc

            total_compute += compute
            total_chunks += 1
            last_result = result
            text_now = (committed + " " + SPECIAL_TOKEN_RE.sub("", result.text)).strip()
            emit_sim = audio_avail + compute
            snapshots.append((emit_sim, text_now))

        # TTFT: earliest chunk whose text was non-empty
        utt_ttft = next((e for e, t in snapshots if t), None)
        if utt_ttft is not None:
            ttft_sims.append(utt_ttft)

        hyp = snapshots[-1][1] if snapshots else ""
        refs.append(it["ref"])
        hyps.append(hyp)
        rows.append((it["path"], it["ref"], hyp))

        # Word commit lag: greedy only (beam decode carries no timestamps),
        # single-window utterances only (timestamps are window-relative).
        if did_reset:
            n_reset_utts += 1
        elif greedy and tokenizer is not None and last_result is not None and hyp:
            try:
                pairs = word_end_times(last_result, tokenizer)
                if pairs:
                    final_words = hyp.split()
                    commits = stable_commit_times(snapshots, final_words)
                    tw = [w for w, _ in pairs]
                    if tw == final_words:  # only trust exact alignment
                        for (_, end), c in zip(pairs, commits):
                            if c is not None:
                                word_lags.append(c - end)
            except Exception as e:  # timestamps are best-effort, never fatal
                print(f"[warn] word-lag skipped for {it['path']}: {e}", file=sys.stderr)

        print(f"[{len(rows)}/{len(items)}] ref={it['ref']!r} hyp={hyp!r}", flush=True)

    m = compute_wer_cer(refs, hyps)
    rtf = total_compute / total_audio if total_audio else float("nan")
    print(f"N={len(refs)}  streaming WER={m['wer']:.4f}  CER={m['cer']:.4f}  "
          f"RTF={rtf:.3f}  chunks={total_chunks}  "
          f"{1000 * total_compute / max(total_chunks, 1):.1f} ms/chunk  "
          f"({total_compute:.1f}s compute / {total_audio:.1f}s audio)  "
          f"chunk={ms_gran}ms beams={args.beams} ca_kv_cache={not args.no_ca_kv_cache}")

    def pctl(xs, q):
        return sorted(xs)[min(len(xs) - 1, int(q * (len(xs) - 1)))]

    import statistics
    if ttft_sims:
        print(f"TTFT (stream start -> first text, simulated real-time): "
              f"mean {statistics.mean(ttft_sims):.2f}s  p50 {pctl(ttft_sims, .5):.2f}s  "
              f"p95 {pctl(ttft_sims, .95):.2f}s  (n={len(ttft_sims)}; includes any "
              f"leading silence before the first word)")
    if word_lags:
        print(f"word commit lag (word spoken -> text stable, simulated real-time): "
              f"mean {statistics.mean(word_lags):.2f}s  p50 {pctl(word_lags, .5):.2f}s  "
              f"p95 {pctl(word_lags, .95):.2f}s  (n={len(word_lags)} words)")
    elif not greedy:
        print("word commit lag: unavailable with beam decode (no streaming timestamps)")
    if n_reset_utts:
        print(f"[note] {n_reset_utts} utterances exceeded the {args.max_sec_context}s "
              f"context window (word-lag skipped for those)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("path\tref\thyp\n")
            for path, ref, hyp in rows:
                f.write(f"{path}\t{ref}\t{hyp}\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
