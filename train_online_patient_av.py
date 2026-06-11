#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from online_avsr.checkpoint import (  # noqa: E402
    load_validated_state_dict,
    preflight_environment,
)
from online_avsr.data import PatientAVDataModule, discover_patient_records  # noqa: E402
from online_avsr.module import OnlineAVSRModule  # noqa: E402
from online_avsr.text import (  # noqa: E402
    load_sentencepiece_model,
    train_sentencepiece_from_texts,
)


def resolve_label_file(root_dir: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if os.path.isabs(value):
        return value
    direct = os.path.join(root_dir, value)
    if os.path.isfile(direct):
        return direct
    return os.path.join(root_dir, "labels", value)


def collect_texts(root_dir: str, label_files: List[Optional[str]]) -> List[str]:
    texts = []
    for label_file in label_files:
        if not label_file:
            continue
        records = discover_patient_records(root_dir, label_file=label_file)
        texts.extend(record.text for record in records if record.text)
    if not texts:
        records = discover_patient_records(root_dir)
        texts.extend(record.text for record in records if record.text)
    return texts


def ensure_sp_model(args, train_file, val_file) -> str:
    if os.path.isfile(args.sp_model_path):
        return args.sp_model_path
    if not args.generate_sp_model:
        raise FileNotFoundError(
            f"SentencePiece model not found: {args.sp_model_path}. "
            "Pass --generate-sp-model to build one from patient labels."
        )
    prefix = os.path.splitext(args.sp_model_path)[0]
    texts = collect_texts(args.root_dir, [train_file, val_file])
    return train_sentencepiece_from_texts(texts, prefix)


def build_trainer(args, run_dir):
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
        filename="{epoch}-{val_loss:.4f}",
    )
    callbacks = [checkpoint, LearningRateMonitor(logging_interval="step")]

    common = {
        "default_root_dir": run_dir,
        "max_epochs": args.epochs,
        "num_nodes": args.num_nodes,
        "callbacks": callbacks,
        "gradient_clip_val": 10.0,
    }

    try:
        from pytorch_lightning.strategies import DDPStrategy

        strategy = DDPStrategy(find_unused_parameters=False) if args.gpus * args.num_nodes > 1 else "auto"
        return Trainer(
            accelerator="gpu",
            devices=args.gpus,
            strategy=strategy,
            **common,
        )
    except (ImportError, TypeError):
        kwargs = dict(common)
        kwargs["gpus"] = args.gpus
        if args.resume_from_checkpoint:
            kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint
        return Trainer(**kwargs)


def write_manifest(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train online patient AVSR Emformer RNN-T.")
    parser.add_argument("--model-source", choices=["scratch"], default="scratch")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--exp-dir", default=os.environ.get("EXP_DIR", "/scratch/th3482/experiments/online_avsr"))
    parser.add_argument("--exp-name", default=os.environ.get("EXP_NAME", "patient_online_avsr"))
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--generate-sp-model", action="store_true")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "55")))
    parser.add_argument("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", "1")))
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("GPUS", "1")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    parser.add_argument("--max-videos", type=int, default=int(os.environ.get("MAX_VIDEOS", "0")) or None)
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "0.0008")))
    parser.add_argument("--resume-from-checkpoint", default=os.environ.get("RESUME_FROM_CHECKPOINT"))
    parser.add_argument("--pretrained-model-path", default=os.environ.get("PRETRAINED_MODEL_PATH"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    env = preflight_environment()
    train_file = resolve_label_file(args.root_dir, args.train_file)
    val_file = resolve_label_file(args.root_dir, args.val_file) or train_file
    test_file = resolve_label_file(args.root_dir, args.test_file) or val_file

    for label in [train_file, val_file]:
        if label and not os.path.isfile(label):
            raise FileNotFoundError(f"Label file not found: {label}")

    sp_model_path = ensure_sp_model(args, train_file, val_file)
    sp_model = load_sentencepiece_model(sp_model_path)
    run_dir = os.path.join(args.exp_dir, f"{args.exp_name}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        "command": sys.argv,
        "preflight": env,
        "root_dir": args.root_dir,
        "train_file": train_file,
        "val_file": val_file,
        "test_file": test_file,
        "sp_model_path": sp_model_path,
        "epochs": args.epochs,
        "num_nodes": args.num_nodes,
        "gpus": args.gpus,
        "batch_size": args.batch_size,
        "pretrained_model_path": args.pretrained_model_path,
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
    )
    model = OnlineAVSRModule(args=args, sp_model=sp_model)
    if args.pretrained_model_path:
        state_dict = load_validated_state_dict(args.pretrained_model_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained initialization. missing={len(missing)} unexpected={len(unexpected)}")

    trainer = build_trainer(args, run_dir)
    try:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)
    except TypeError:
        trainer.fit(model, datamodule=data_module)

    print(f"Training complete. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
