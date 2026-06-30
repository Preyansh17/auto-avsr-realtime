#!/usr/bin/env python3
"""Finetune Whisper (small/medium) on patient dysarthric speech.

Green et al. (Interspeech 2021) recipe: update ONLY the first ~5 encoder layers,
freeze the rest of the encoder + the whole decoder, and regularize with
SpecAugment (cut frequency masking, blown-up time masking). The patient set is
tiny + repetitive, so this freeze + heavy time masking is the overfitting brake.

Whisper has SpecAugment built in (WhisperConfig.apply_spec_augment +
mask_time_*/mask_feature_*); we drive it from CLI flags whose Green-leaning
defaults mirror asr_baselines/configs/specaug_green.yaml. The standalone
specaugment.py is for the framework-independent ablation / visualization; the
WER ablation here just toggles --no-specaug.

Checkpoint selection is on WER, NOT val_loss -- the AV repo already established
val_loss anti-correlates with WER on this data.

Example:
  python -m asr_baselines.whisper_finetune \
    --model openai/whisper-small \
    --root-dir $ROOT --train-file train.csv --val-file val.csv \
    --output-dir $EXP/whisper_small_green --epochs 30 --unfreeze-encoder-layers 5
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--root-dir", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("--sp-model-path", default=DEFAULT_SPM)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--language", default="english")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-frames", type=int, default=None, help="drop clips longer than N video frames")
    p.add_argument("--max-label-length", type=int, default=448)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gen-num-beams", type=int, default=1)
    # Green freeze: train first N encoder layers, freeze the rest + decoder.
    p.add_argument("--unfreeze-encoder-layers", type=int, default=5,
                   help="train encoder layers [0, N); 0 = freeze whole encoder; -1 = train all")
    p.add_argument("--unfreeze-conv", action="store_true",
                   help="also train the conv stem (conv1/conv2); Green trains only the 5 layers")
    p.add_argument("--train-decoder", action="store_true", help="also unfreeze the decoder (off = Green)")
    # SpecAugment (Whisper built-in). Green-leaning defaults: heavy time, light freq.
    p.add_argument("--no-specaug", action="store_true", help="ablation: disable SpecAugment")
    p.add_argument("--mask-time-prob", type=float, default=0.5)
    p.add_argument("--mask-time-length", type=int, default=40)
    p.add_argument("--mask-time-min-masks", type=int, default=2)
    p.add_argument("--mask-feature-prob", type=float, default=0.1)
    p.add_argument("--mask-feature-length", type=int, default=13)
    p.add_argument("--mask-feature-min-masks", type=int, default=1)
    return p.parse_args()


def apply_freeze(model, args):
    enc = model.model.encoder
    n = args.unfreeze_encoder_layers
    if n == -1 and args.train_decoder:
        return  # full finetune
    for prm in model.parameters():
        prm.requires_grad = False
    layers = enc.layers
    train_idx = range(len(layers)) if n == -1 else range(min(n, len(layers)))
    for i in train_idx:
        for prm in layers[i].parameters():
            prm.requires_grad = True
    if args.unfreeze_conv:
        for mod in (enc.conv1, enc.conv2):
            for prm in mod.parameters():
                prm.requires_grad = True
    if args.train_decoder:
        for prm in model.model.decoder.parameters():
            prm.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Freeze: training encoder layers {list(train_idx)} "
          f"(conv={args.unfreeze_conv}, decoder={args.train_decoder}); "
          f"trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total:.1%})")


def configure_specaug(model, args):
    cfg = model.config
    if args.no_specaug:
        cfg.apply_spec_augment = False
        print("SpecAugment: DISABLED (ablation)")
        return
    cfg.apply_spec_augment = True
    cfg.mask_time_prob = args.mask_time_prob
    cfg.mask_time_length = args.mask_time_length
    cfg.mask_time_min_masks = args.mask_time_min_masks
    cfg.mask_feature_prob = args.mask_feature_prob
    cfg.mask_feature_length = args.mask_feature_length
    cfg.mask_feature_min_masks = args.mask_feature_min_masks
    print(f"SpecAugment: time(prob={cfg.mask_time_prob}, len={cfg.mask_time_length}, "
          f"min={cfg.mask_time_min_masks})  feature(prob={cfg.mask_feature_prob}, "
          f"len={cfg.mask_feature_length}, min={cfg.mask_feature_min_masks})")


class WhisperPatientDataset(torch.utils.data.Dataset):
    """Maps (waveform, text) -> {input_features, labels} via the processor."""

    def __init__(self, examples, processor, max_label_length):
        self.examples = examples
        self.processor = processor
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        wav = load_waveform(ex.path).numpy()
        feats = self.processor.feature_extractor(wav, sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(text=ex.text).input_ids[: self.max_label_length]
        return {"input_features": feats, "labels": labels}


class DataCollatorSpeechSeq2Seq:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # strip the leading BOS the model prepends itself
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    args = parse_args()
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    # English transcription, no forced language prefix to overfit against.
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model.config.use_cache = False  # required with gradient_checkpointing

    configure_specaug(model, args)
    apply_freeze(model, args)

    train_ex = load_audio_examples(args.root_dir, args.train_file, args.sp_model_path,
                                   max_frames=args.max_frames)
    val_ex = load_audio_examples(args.root_dir, args.val_file, args.sp_model_path,
                                 max_frames=args.max_frames)
    print(f"train={len(train_ex)} val={len(val_ex)} examples")
    train_ds = WhisperPatientDataset(train_ex, processor, args.max_label_length)
    val_ds = WhisperPatientDataset(val_ex, processor, args.max_label_length)
    collator = DataCollatorSpeechSeq2Seq(processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        m = compute_wer_cer(refs, hyps)
        return {"wer": m["wer"], "cer": m["cer"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=(args.precision == "fp16"),
        bf16=(args.precision == "bf16"),
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.max_label_length,
        generation_num_beams=args.gen_num_beams,
        logging_steps=10,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "best"))
    processor.save_pretrained(os.path.join(args.output_dir, "best"))
    metrics = trainer.evaluate()
    print("Final val:", {k: metrics[k] for k in metrics if "wer" in k or "cer" in k or "loss" in k})


if __name__ == "__main__":
    main()
