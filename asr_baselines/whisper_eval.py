#!/usr/bin/env python3
"""Evaluate a finetuned (or zero-shot) Whisper on patient data.

  python -m asr_baselines.whisper_eval --model $EXP/whisper_small_green/best \
    --root-dir $ROOT --label-file val.csv --out hyps.tsv

Pass a HF hub id (e.g. openai/whisper-small) for the zero-shot baseline.
"""

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_baselines.metrics import compute_wer_cer  # noqa: E402
from asr_baselines.patient_audio import load_audio_examples, load_waveform  # noqa: E402

DEFAULT_SPM = os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--root-dir", required=True)
    p.add_argument("--label-file", required=True)
    p.add_argument("--sp-model-path", default=DEFAULT_SPM)
    p.add_argument("--language", default="english")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--prefix", choices=["lang_task", "sot_sot"], default="lang_task",
                   help="decoder prompt: lang_task = <sot><lang><task><notimestamps> (HF default); sot_sot = "
                        "<sot><sot><notimestamps>, what whisper_finetune.py trained on before 2026-10-20 "
                        "(its collator never stripped the tokenizer's <sot>)")
    p.add_argument("--out", default=None, help="TSV of path\\tref\\thyp")
    args = p.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    forced = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    gen_kw = {"forced_decoder_ids": forced}
    if args.prefix == "sot_sot":
        sot = model.generation_config.decoder_start_token_id
        prompt = [sot, sot, model.generation_config.no_timestamps_token_id]
        gen_kw = {"decoder_input_ids": torch.tensor([prompt], device=device)}

    examples = load_audio_examples(args.root_dir, args.label_file, args.sp_model_path,
                                   max_frames=args.max_frames)
    refs, hyps, rows = [], [], []
    for ex in examples:
        wav = load_waveform(ex.path).numpy()
        feats = processor.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            ids = model.generate(feats, num_beams=args.num_beams, **gen_kw)
        hyp = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
        refs.append(ex.text)
        hyps.append(hyp)
        rows.append((ex.path, ex.text, hyp))

    m = compute_wer_cer(refs, hyps)
    print(f"N={len(refs)}  WER={m['wer']:.4f}  CER={m['cer']:.4f}  "
          f"words={m['n_words']} chars={m['n_chars']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("path\tref\thyp\n")
            for path, ref, hyp in rows:
                f.write(f"{path}\t{ref}\t{hyp}\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
