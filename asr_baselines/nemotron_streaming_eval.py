#!/usr/bin/env python3
"""Cache-aware streaming eval for Nemotron (chunked decode, carried state).

Follow-up to nemotron_eval.py's offline transcribe(). Nemotron
(nvidia/nemotron-speech-streaming-en-0.6b) is a cache-aware FastConformer-RNNT
-- built to run in fixed-size audio chunks with encoder state
(cache_last_channel/cache_last_time) carried across chunks, the same
streaming contract as the AV Emformer. Offline transcribe() decodes the whole
utterance at once and reports a WER ceiling that chunked/causal inference may
not actually reach -- this script reports the real streaming-mode WER, plus
per-chunk latency (ms/chunk), the number actually comparable to the AV
Emformer's own streaming latency.

Uses NeMo's own cache-aware streaming utilities (same API as NeMo's
examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
and the "Cache Aware Streaming" tutorial notebook) rather than reimplementing
chunk framing / cache bookkeeping: CacheAwareStreamingAudioBuffer for chunk
framing + model.conformer_stream_step() for the carried-state decode loop.
This API is version sensitive (like everything else in the nemotron phase --
see requirements-nemotron.txt); it has NOT been smoke-tested against the
installed nemo_toolkit==2.7.3 yet (no local NeMo install to verify against --
must be smoke-tested on a real compute-node job before trusting the numbers).

  python -m asr_baselines.nemotron_streaming_eval \
    --model $EXP/nemotron_patient.nemo --manifest val.jsonl \
    --out streaming_hyps.tsv
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_baselines.metrics import compute_wer_cer  # noqa: E402


def read_manifest(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--chunk-size", type=int, default=None,
                   help="encoder chunk size in frames; default keeps the model's "
                        "pretrained streaming_cfg (nemotron ships tuned for low-latency "
                        "streaming already)")
    p.add_argument("--left-chunks", type=int, default=None,
                   help="left-context chunks kept in cache; default keeps pretrained config")
    p.add_argument("--online-normalization", action="store_true",
                   help="use running mean/var feature normalization instead of "
                        "per-utterance stats -- offline transcribe() normalizes over the "
                        "FULL utterance, which peeks at future audio a true streaming "
                        "deployment can't see; this flag removes that unfairness")
    args = p.parse_args()

    import torch
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    if args.model.endswith(".nemo") and os.path.isfile(args.model):
        model = nemo_asr.models.ASRModel.restore_from(args.model)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    model.eval()

    if args.chunk_size is not None or args.left_chunks is not None:
        model.encoder.setup_streaming_params(chunk_size=args.chunk_size, left_chunks=args.left_chunks)
    cfg = model.encoder.streaming_cfg
    print(f"streaming_cfg: chunk_size={cfg.chunk_size} shift_size={cfg.shift_size} "
          f"pre_encode_cache_size={cfg.pre_encode_cache_size}")

    items = read_manifest(args.manifest)
    refs, hyps = [], []
    total_audio = 0.0
    total_elapsed = 0.0
    total_chunks = 0
    device = next(model.parameters()).device

    for it in items:
        path = it["audio_filepath"]
        total_audio += it.get("duration", 0.0)

        # One utterance == one stream: fresh buffer + cache per file. Matches
        # a real single-stream deployment (one live audio feed), and keeps
        # per-utterance timing directly comparable to the AV Emformer's own
        # per-utterance streaming latency.
        streaming_buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=args.online_normalization)
        streaming_buffer.append_audio_file(path, stream_id=-1)

        cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(batch_size=1)
        previous_hypotheses = None
        pred_out_stream = None
        transcribed_texts = None

        t0 = time.time()
        for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
            with torch.inference_mode():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio.to(device),
                    processed_signal_length=chunk_lengths.to(device),
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=cfg.drop_extra_pre_encoded if step_num != 0 else 0,
                    return_transcription=True,
                )
            total_chunks += 1
        total_elapsed += time.time() - t0

        hyp = transcribed_texts[0]
        hyp = hyp.text if hasattr(hyp, "text") else hyp
        refs.append(it["text"])
        hyps.append(hyp)

    m = compute_wer_cer(refs, hyps)
    rtf = total_elapsed / total_audio if total_audio else float("nan")
    print(f"N={len(refs)}  streaming WER={m['wer']:.4f}  CER={m['cer']:.4f}  "
          f"RTF={rtf:.3f}  chunks={total_chunks}  "
          f"{1000*total_elapsed/max(total_chunks,1):.1f} ms/chunk  "
          f"({total_elapsed:.1f}s / {total_audio:.1f}s audio)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("path\tref\thyp\n")
            for it, ref, hyp in zip(items, refs, hyps):
                f.write(f"{it['audio_filepath']}\t{ref}\t{hyp}\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
