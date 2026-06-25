#!/usr/bin/env python3
"""Re-merge a LoRA epoch checkpoint into a plain merged .pth for eval.

train.py used to fold adapters from the *last* epoch, which on small patient
sets is badly overfit (val_loss can climb ~15 -> 60 in late epochs) instead of
the best. This converts any saved epoch ckpt -- which keeps the LoRA-wrapped
form -- into model_lora_merged-style weights so you can eval the BEST
checkpoint without retraining.

Usage:
  python scripts/merge_lora_ckpt.py \
    --ckpt /scratch/.../lora_all_legal_only_av_j11714048_*/epoch=15-val_loss=14.7900.ckpt \
    --out  /scratch/.../lora_all_legal_only_av_j11714048_*/model_lora_merged_best.pth
"""

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from online_avsr.lora import inject_lora, merge_lora  # noqa: E402
from online_avsr.module import OnlineAVSRModule  # noqa: E402
from online_avsr.text import load_sentencepiece_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="A LoRA-form epoch checkpoint (epoch=..-val_loss=..ckpt or last.ckpt)")
    p.add_argument("--out", required=True, help="Where to write the merged .pth")
    p.add_argument("--sp-model", default=os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model"),
                   help="Fallback SentencePiece model if the checkpoint hparams omit one")
    return p.parse_args()


def main():
    a = parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck.get("hyper_parameters", {}).get("args")
    if args is None:
        raise SystemExit("Checkpoint has no hyper_parameters.args; cannot rebuild the model")
    if not getattr(args, "lora", False):
        raise SystemExit("Checkpoint was not a LoRA run (args.lora is False); nothing to merge")

    sp_path = getattr(args, "sp_model_path", None) or a.sp_model
    sp_model = load_sentencepiece_model(sp_path)
    model = OnlineAVSRModule(args=args, sp_model=sp_model)
    inject_lora(model, args.lora_scopes, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    state_dict = {k: v for k, v in ck["state_dict"].items() if not k.startswith("loss.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 10:
        print(f"WARNING: missing={len(missing)} unexpected={len(unexpected)} "
              f"(first missing: {list(missing)[:5]}) -- check architecture/scopes", file=sys.stderr)

    merged = merge_lora(model)
    lora_manifest = {
        "scopes": list(args.lora_scopes),
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "lora": lora_manifest}, a.out)
    print(f"Merged {merged} LoRA layers from {os.path.basename(a.ckpt)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
