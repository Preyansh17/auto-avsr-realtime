#!/usr/bin/env python3
"""Train/fine-tune the streaming Emformer RNN-T AVSR model on patient data.

Two starting points:
  --model-source bootstrap  fine-tune the pretrained device_avsr weights
                            (architecture=device, 44x44 face/ROI frames;
                            requires the tutorial spm_unigram_1023 model)
  --model-source scratch    train from random init (either architecture)

Label CSVs must carry 1023-piece token ids
(scripts/regenerate_patient_labels.py).
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import torch

# argparse.Namespace is stored in Lightning checkpoints; allow it under PyTorch 2.6+
torch.serialization.add_safe_globals([argparse.Namespace])

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from online_avsr.average_checkpoints import ensemble  # noqa: E402
from online_avsr.checkpoint import (  # noqa: E402
    load_validated_state_dict,
    preflight_environment,
    sha256_file,
)
from online_avsr.data_module import PatientAVDataModule  # noqa: E402
from online_avsr.lora import SCOPE_PREFIXES, inject_lora, merge_lora  # noqa: E402
from online_avsr.module import OnlineAVSRModule  # noqa: E402
from online_avsr.patient_dataset import discover_patient_records  # noqa: E402
from online_avsr.text import (  # noqa: E402
    load_sentencepiece_model,
    train_sentencepiece_from_texts,
)

DEFAULT_BOOTSTRAP = os.path.join(PROJECT_ROOT, "cpts", "online_avsr_bootstrap.ckpt")


def resolve_label_file(root_dir: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if os.path.isabs(value):
        return value
    if os.path.isfile(value):
        return os.path.abspath(value)
    direct = os.path.join(root_dir, value)
    if os.path.isfile(direct):
        return direct
    return os.path.join(root_dir, "labels", value)


def ensure_sp_model(args, train_file, val_file) -> str:
    if os.path.isfile(args.sp_model_path):
        return args.sp_model_path
    if not args.generate_sp_model:
        raise FileNotFoundError(
            f"SentencePiece model not found: {args.sp_model_path}. "
            "Run scripts/download_assets.py (bootstrap) or pass --generate-sp-model (scratch only)."
        )
    if args.model_source == "bootstrap":
        raise SystemExit(
            "--generate-sp-model is incompatible with --model-source bootstrap: "
            "the pretrained weights are tied to the tutorial spm_unigram_1023 vocabulary."
        )
    texts = []
    for label_file in (train_file, val_file):
        if label_file:
            records = discover_patient_records(args.root_dir, label_file=label_file)
            texts.extend(r.text for r in records if r.text)
    prefix = os.path.splitext(args.sp_model_path)[0]
    return train_sentencepiece_from_texts(texts, prefix)


def build_trainer(args, run_dir):
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=10,
        filename="{epoch}-{val_loss:.4f}",
    )
    from pytorch_lightning.strategies import DDPStrategy

    strategy = DDPStrategy(find_unused_parameters=False) if args.gpus * args.num_nodes > 1 else "auto"
    return Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        strategy=strategy,
        num_nodes=args.num_nodes,
        default_root_dir=run_dir,
        max_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=10.0,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        sync_batchnorm=args.gpus * args.num_nodes > 1,
    )


def write_manifest(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-source", choices=["bootstrap", "scratch"], default="bootstrap")
    parser.add_argument("--architecture", choices=["device", "recipe"], default=None,
                        help="Default: device for bootstrap, recipe for scratch")
    parser.add_argument("--modality", choices=["audiovisual", "audio", "video"], default="audiovisual",
                        help="audio/video-only drop the other frontend + fusion (reuse pretrained frontend + RNN-T)")
    parser.add_argument("--segment-length", type=int, default=None)
    parser.add_argument("--right-context-length", type=int, default=None)
    parser.add_argument("--frame-size", type=int, default=None,
                        help="Video input resolution (default: 44 for device, 88 for recipe)")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR", os.path.join(PROJECT_ROOT, "exp")))
    parser.add_argument("--exp-name", default=os.environ.get("EXP_NAME", "patient_online_avsr"))
    parser.add_argument("--sp-model-path", default=os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model"))
    parser.add_argument("--generate-sp-model", action="store_true")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "55")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "0")),
                        help="Stop after this many optimizer steps (mirrors the offline runs' max_steps; 0 = use epochs)")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", "1")))
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("GPUS", "1")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    parser.add_argument("--accumulate-grad-batches", type=int, default=int(os.environ.get("ACCUMULATE_GRAD_BATCHES", "4")))
    parser.add_argument("--precision", default=os.environ.get("PRECISION", "32-true"),
                        help="Lightning precision, e.g. bf16-mixed on A100")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    parser.add_argument("--max-videos", type=int, default=int(os.environ.get("MAX_VIDEOS", "0")) or None)
    parser.add_argument("--max-frames", type=int, default=int(os.environ.get("MAX_FRAMES", "600")),
                        help="Drop clips longer than this many frames (RNNT loss memory)")
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "0.0008")))
    parser.add_argument("--resume-from-checkpoint", default=os.environ.get("RESUME_FROM_CHECKPOINT") or None)
    parser.add_argument("--pretrained-model-path", default=os.environ.get("PRETRAINED_MODEL_PATH") or None)
    parser.add_argument("--ensemble-last", type=int, default=10,
                        help="Average the last N epoch checkpoints after training (0 to disable)")
    parser.add_argument("--lora", action="store_true",
                        help="Freeze the pretrained weights and train low-rank adapters only")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-scopes", nargs="+", default=["encoder", "predictor", "joiner", "fusion"],
                        choices=sorted(SCOPE_PREFIXES) + ["all"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.architecture is None:
        args.architecture = "device" if args.model_source == "bootstrap" else "recipe"
    if args.frame_size is None:
        args.frame_size = 44 if args.architecture == "device" else 88
    if args.model_source == "bootstrap" and not args.pretrained_model_path:
        args.pretrained_model_path = DEFAULT_BOOTSTRAP
        if not os.path.isfile(args.pretrained_model_path):
            raise FileNotFoundError(
                f"{args.pretrained_model_path} not found; run scripts/download_assets.py "
                "then scripts/bootstrap_from_jit.py"
            )
    if args.lora and not args.pretrained_model_path:
        raise SystemExit("--lora requires pretrained weights (--model-source bootstrap or --pretrained-model-path)")

    env = preflight_environment()
    train_file = resolve_label_file(args.root_dir, args.train_file)
    val_file = resolve_label_file(args.root_dir, args.val_file) or train_file
    test_file = resolve_label_file(args.root_dir, args.test_file) or val_file
    for label in (train_file, val_file):
        if label and not os.path.isfile(label):
            raise FileNotFoundError(f"Label file not found: {label}")

    sp_model_path = ensure_sp_model(args, train_file, val_file)
    sp_model = load_sentencepiece_model(sp_model_path)
    run_dir = os.path.join(args.exp_dir, f"{args.exp_name}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        "command": sys.argv,
        "preflight": env,
        "model_source": args.model_source,
        "architecture": args.architecture,
        "segment_length": args.segment_length,
        "right_context_length": args.right_context_length,
        "frame_size": args.frame_size,
        "root_dir": args.root_dir,
        "train_file": train_file,
        "val_file": val_file,
        "test_file": test_file,
        "sp_model_path": sp_model_path,
        "sp_model_sha256": sha256_file(sp_model_path),
        "pretrained_model_path": args.pretrained_model_path,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "precision": args.precision,
        "learning_rate": args.learning_rate,
        "max_frames": args.max_frames,
        "lora": {
            "enabled": args.lora,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "scopes": args.lora_scopes,
        } if args.lora else {"enabled": False},
    }
    write_manifest(os.path.join(run_dir, "run_manifest.json"), manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    data_module = PatientAVDataModule(
        root_dir=args.root_dir,
        sp_model=sp_model,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.max_videos,
        max_frames=args.max_frames,
        frame_size=args.frame_size,
    )
    model = OnlineAVSRModule(args=args, sp_model=sp_model)
    if args.pretrained_model_path:
        state_dict = load_validated_state_dict(args.pretrained_model_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained initialization. missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 10:
            print(f"WARNING: many missing keys; check --architecture. First few: {list(missing)[:5]}")
    if args.lora:
        replaced, trainable, total = inject_lora(
            model, args.lora_scopes, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
        )
        print(f"LoRA: wrapped {replaced} Linear layers; trainable {trainable / 1e6:.2f}M "
              f"of {total / 1e6:.2f}M params ({trainable / total:.1%})")

    trainer = build_trainer(args, run_dir)
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)

    if trainer.is_global_zero:
        if args.lora:
            # Fold adapters into plain Linear weights -> standard checkpoint
            # for eval.py / demo_realtime.py (last.ckpt keeps the LoRA form
            # for resuming). Merge from the BEST checkpoint (lowest val_loss),
            # NOT the in-memory last-epoch weights: on these small patient sets
            # the schedule overfits hard in late epochs (val_loss climbs ~15 ->
            # 60), so the final weights are far worse than the best epoch.
            best_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
            if best_path and os.path.isfile(best_path):
                best_sd = torch.load(best_path, map_location="cpu")["state_dict"]
                best_sd = {k: v for k, v in best_sd.items() if not k.startswith("loss.")}
                missing, unexpected = model.load_state_dict(best_sd, strict=False)
                if len(missing) > 10:
                    print(f"WARNING: best-ckpt load missing={len(missing)} unexpected={len(unexpected)}")
                print(f"Restored best checkpoint for merge: {best_path}")
            else:
                print("WARNING: best checkpoint unavailable; merging last-epoch weights")
            merged = merge_lora(model)
            merged_path = os.path.join(run_dir, "model_lora_merged.pth")
            torch.save({"state_dict": model.state_dict(), "lora": manifest["lora"]}, merged_path)
            print(f"Merged {merged} LoRA layers (from best) -> {merged_path}")
        elif args.ensemble_last:
            avg_path = ensemble(run_dir, last_n=args.ensemble_last)
            print(f"Averaged checkpoint: {avg_path}" if avg_path else "Too few checkpoints to average.")
    print(f"Training complete. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
