#!/usr/bin/env python3
"""Streaming eval for finetuned Whisper via SimulStreaming (AlignAtt policy).

Whisper counterpart to nemotron_streaming_eval.py. Whisper's encoder is
non-causal over a fixed 30s window, so unlike Nemotron/Emformer there is no
native carried-state streaming mode; SimulStreaming (github.com/ufal/SimulStreaming,
UFAL 2025) wraps an unmodified Whisper checkpoint with the AlignAtt policy:
audio is fed in fixed segments (default 1.2s), and the decoder emits only the
tokens whose encoder-decoder attention stays far enough from the end of the
audio buffer (--frame-threshold) to be safe against future revisions. This is
pseudo-streaming (recompute over a growing buffer, not O(1) per chunk), so
ms/chunk grows with buffer length -- report it alongside RTF, and compare
latency to Nemotron/Emformer with that caveat.

Mirrors nemotron_streaming_eval.py's contract: one utterance == one stream
(fresh decoder/KV state per file), per-chunk wall-clock latency, and the shared
light normalizer from asr_baselines.metrics so WER is comparable across models.

Requires the SimulStreaming checkout on PYTHONPATH (--simulstreaming-dir) and
an OpenAI-format checkpoint (convert HF with convert_hf_to_openai.py; verify
the conversion is BIT-EXACT against the source state dict and decodes a known
clip offline first -- a silently fp16-degraded conversion produced pure
hallucinations, see results/whisper_streaming_investigation_2026-07-09.md).

  python -m asr_baselines.whisper_streaming_eval \
    --model-path $EXP/openai_format/large-v3-patient-ft-v2.pt \
    --manifest-tsv $EXP/whisper_largev3_specaug_3way_full_seed1/val_hyps.tsv \
    --simulstreaming-dir /scratch/$USER/third_party/SimulStreaming \
    --out streaming_hyps.tsv

--manifest-tsv is any TSV with header path\tref\t... (e.g. an offline eval's
val_hyps.tsv, which pins the exact same test-set membership as the offline
number being compared against).
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_baselines.metrics import compute_wer_cer, normalize  # noqa: E402
from asr_baselines.mfa_timings import (  # noqa: E402
    build_textgrid_index, ref_word_ends, match_hyp_to_ref,
)


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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="OpenAI-format .pt checkpoint")
    p.add_argument("--manifest-tsv", required=True, help="TSV with path/ref columns")
    p.add_argument("--simulstreaming-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--segment-length", type=float, default=1.2,
                   help="audio fed to the model in segments of this many seconds; "
                        "the latency knob, analogous to Nemotron's chunk size")
    p.add_argument("--frame-threshold", type=int, default=25,
                   help="AlignAtt: emit a token only if its most-attended encoder frame "
                        "is at least this many frames (0.02s each) before buffer end")
    p.add_argument("--beams", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--mfa-aligned-dir", default=None)
    p.add_argument("--mfa-split-root", default="/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1")
    args = p.parse_args()

    sys.path.insert(0, args.simulstreaming_dir)
    from simulstreaming.whisper.simul_whisper.whisper.audio import load_audio, SAMPLE_RATE
    import simulstreaming_whisper as sw

    factory_args = SimpleNamespace(
        model_path=args.model_path,
        cif_ckpt_path=None,
        frame_threshold=args.frame_threshold,
        audio_min_len=0.0,
        audio_max_len=30.0,
        beams=args.beams,
        decoder=None,
        task=args.task,
        # No CIF model exists for large-v3; without never_fire the last decoded
        # word of every chunk is unconditionally trimmed, which on short
        # utterances eats real words.
        never_fire=True,
        init_prompt=None,
        static_init_prompt=None,
        max_context_tokens=None,
        logdir=None,
        lan=args.language,
        min_chunk_size=args.segment_length,
        log_level="WARNING",
    )
    asr, online = sw.simul_asr_factory(factory_args)

    items = read_manifest_tsv(args.manifest_tsv)
    seg_samples = int(args.segment_length * SAMPLE_RATE)
    refs, hyps, rows = [], [], []
    total_audio = 0.0
    total_elapsed = 0.0
    total_chunks = 0
    # Latency in SIMULATED REAL TIME: the eval loop feeds audio as fast as the
    # GPU eats it, but a live user records chunk k over [k*seg, (k+1)*seg], so
    # its text can appear no earlier than (k+1)*seg + compute(chunk k). Valid
    # while per-chunk compute < segment (RTF<1: previous chunk's compute
    # overlaps the next chunk's recording).
    ttft_sims = []           # stream start -> first text visible (includes leading silence)
    ttft_from_speech_sims = []  # first word's own end-of-speech -> its text visible
    word_lags = []           # word finished being SPOKEN -> its text visible (all words)
    mfa_word_lags = []       # same, anchored to MFA ground truth instead
    mfa_ttft_sims = []
    mfa_index = build_textgrid_index(args.mfa_aligned_dir) if getattr(args, "mfa_aligned_dir", None) else None
    mfa_covered = mfa_missing = 0
    if mfa_index is not None:
        print(f"MFA reference: {len(mfa_index)} TextGrids indexed from {args.mfa_aligned_dir}")

    for it in items:
        # Label CSV paths point at the lip-crop mp4s, which are VIDEO-ONLY;
        # the audio lives in a sibling 16kHz .wav (same convention as
        # patient_audio.load_waveform).
        wav_path = os.path.splitext(it["path"])[0] + ".wav"
        audio = load_audio(wav_path if os.path.isfile(wav_path) else it["path"])
        total_audio += len(audio) / SAMPLE_RATE

        online.init()  # fresh stream: resets audio buffer, decoder context, KV state
        pieces = []
        utt_ttft = None
        emitted = []   # (word_text, emit_sim) for MFA matching
        t0 = time.time()
        for k, off in enumerate(range(0, len(audio), seg_samples)):
            chunk = audio[off:off + seg_samples]
            online.insert_audio_chunk(chunk)
            audio_avail = min(off + seg_samples, len(audio)) / SAMPLE_RATE
            tc = time.time()
            if off + seg_samples >= len(audio):
                out = online.finish()  # last chunk: flush with is_last set
            else:
                out = online.process_iter()
            compute = time.time() - tc
            total_chunks += 1
            if out and out.get("text"):
                pieces.append(out["text"])
                emit_sim = audio_avail + compute  # earliest wall time (from stream
                # start) this text could exist in a real-time deployment
                words = out.get("words", [])
                if utt_ttft is None:
                    utt_ttft = emit_sim
                    ttft_sims.append(emit_sim)
                    if words:  # anchor to the first word's own spoken-end time,
                        # not stream start -- excludes leading silence
                        ttft_from_speech_sims.append(emit_sim - words[0]["end"])
                for w in words:
                    word_lags.append(emit_sim - w["end"])
                    emitted.append((str(w.get("word", w.get("text", ""))), emit_sim))
        total_elapsed += time.time() - t0

        if mfa_index is not None and emitted:
            ref_pairs = ref_word_ends(wav_path, mfa_index, args.mfa_split_root)
            if ref_pairs is None:
                mfa_missing += 1
            else:
                mfa_covered += 1
                hyp_norm = [normalize(w).split()[0] if normalize(w) else ""
                            for w, _ in emitted]
                ref_norm = [w for w, _ in ref_pairs]
                for k, (hi, ri) in enumerate(match_hyp_to_ref(hyp_norm, ref_norm)):
                    lag = emitted[hi][1] - ref_pairs[ri][1]
                    mfa_word_lags.append(lag)
                    if k == 0:
                        mfa_ttft_sims.append(lag)

        hyp = " ".join(pieces).strip()
        refs.append(it["ref"])
        hyps.append(hyp)
        rows.append((it["path"], it["ref"], hyp))
        print(f"[{len(rows)}/{len(items)}] ref={it['ref']!r} hyp={hyp!r}", flush=True)

    m = compute_wer_cer(refs, hyps)
    rtf = total_elapsed / total_audio if total_audio else float("nan")
    print(f"N={len(refs)}  streaming WER={m['wer']:.4f}  CER={m['cer']:.4f}  "
          f"RTF={rtf:.3f}  chunks={total_chunks}  "
          f"{1000*total_elapsed/max(total_chunks,1):.1f} ms/chunk  "
          f"({total_elapsed:.1f}s / {total_audio:.1f}s audio)  "
          f"segment={args.segment_length}s frame_threshold={args.frame_threshold}")

    def pctl(xs, q):
        return sorted(xs)[min(len(xs) - 1, int(q * (len(xs) - 1)))]
    if ttft_sims:
        import statistics
        print(f"TTFT (stream start -> first text, simulated real-time): "
              f"mean {statistics.mean(ttft_sims):.2f}s  p50 {pctl(ttft_sims, .5):.2f}s  "
              f"p95 {pctl(ttft_sims, .95):.2f}s  (n={len(ttft_sims)}; includes any "
              f"leading silence before the first word)")
    if ttft_from_speech_sims:
        import statistics
        print(f"TTFT from first word spoken (excludes leading silence): "
              f"mean {statistics.mean(ttft_from_speech_sims):.2f}s  "
              f"p50 {pctl(ttft_from_speech_sims, .5):.2f}s  "
              f"p95 {pctl(ttft_from_speech_sims, .95):.2f}s  (n={len(ttft_from_speech_sims)})")
    if mfa_word_lags:
        print(f"[MFA ground truth] TTFT from first word spoken: "
              f"mean {statistics.mean(mfa_ttft_sims):.2f}s  "
              f"p50 {pctl(mfa_ttft_sims, .5):.2f}s  "
              f"p95 {pctl(mfa_ttft_sims, .95):.2f}s  (n={len(mfa_ttft_sims)})")
        print(f"[MFA ground truth] word commit lag: "
              f"mean {statistics.mean(mfa_word_lags):.2f}s  "
              f"p50 {pctl(mfa_word_lags, .5):.2f}s  "
              f"p95 {pctl(mfa_word_lags, .95):.2f}s  "
              f"(n={len(mfa_word_lags)} matched; {mfa_covered} aligned, {mfa_missing} unaligned)")
    if word_lags:
        import statistics
        print(f"word commit lag (word spoken -> text visible): "
              f"mean {statistics.mean(word_lags):.2f}s  p50 {pctl(word_lags, .5):.2f}s  "
              f"p95 {pctl(word_lags, .95):.2f}s  (n={len(word_lags)} words)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("path\tref\thyp\n")
            for path, ref, hyp in rows:
                f.write(f"{path}\t{ref}\t{hyp}\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
