#!/usr/bin/env python3
"""Extract the pretrained device_avsr weights into an eager, fine-tunable
OnlineAVSRModule checkpoint.

The published streaming AVSR model exists only as a TorchScript archive.
Its state_dict key layout matches OnlineAVSRModule(architecture="device")
(audio_frontend.*, video_frontend.linear.*, fusion.*, model.*), so the
transfer is a direct load, gated by a coverage report and a numerical
parity test (features + first-chunk hypothesis, JIT vs eager).

Usage:
  python scripts/bootstrap_from_jit.py --dump-keys
  python scripts/bootstrap_from_jit.py --out cpts/online_avsr_bootstrap.ckpt
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from online_avsr.module import OnlineAVSRModule  # noqa: E402
from online_avsr.streaming import SentencePieceTokenProcessor  # noqa: E402
from online_avsr.text import load_sentencepiece_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jit-path", default=os.path.join(PROJECT_ROOT, "cpts", "device_avsr_model.pt"))
    parser.add_argument("--sp-model-path", default=os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "cpts", "online_avsr_bootstrap.ckpt"))
    parser.add_argument("--report", default=os.path.join(PROJECT_ROOT, "cpts", "bootstrap_report.json"))
    parser.add_argument("--segment-length", type=int, default=32)
    parser.add_argument("--right-context-length", type=int, default=4)
    parser.add_argument("--dump-keys", action="store_true", help="Print the JIT state_dict tree and exit")
    parser.add_argument(
        "--prefix-map", nargs="*", default=[],
        help="Extra key rewrites as old=new prefixes, applied before loading",
    )
    parser.add_argument("--skip-parity", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.jit_path):
        raise FileNotFoundError(f"{args.jit_path} not found; run scripts/download_assets.py first")

    jit = torch.jit.load(args.jit_path, map_location="cpu")
    jit.eval()
    jit_sd = dict(jit.state_dict())

    if args.dump_keys:
        for k, v in jit_sd.items():
            print(k, tuple(v.shape))
        return 0

    for rewrite in args.prefix_map:
        old, new = rewrite.split("=", 1)
        jit_sd = {(new + k[len(old):] if k.startswith(old) else k): v for k, v in jit_sd.items()}

    sp_model = load_sentencepiece_model(args.sp_model_path)
    module = OnlineAVSRModule(
        args=SimpleNamespace(
            architecture="device",
            segment_length=args.segment_length,
            right_context_length=args.right_context_length,
        ),
        sp_model=sp_model,
    )
    module.eval()

    missing, unexpected = module.load_state_dict(jit_sd, strict=False)
    eager_sd = module.state_dict()
    mapped = [k for k in eager_sd if k in jit_sd and eager_sd[k].shape == jit_sd[k].shape]
    n_mapped_params = sum(eager_sd[k].numel() for k in mapped)
    n_total_params = sum(v.numel() for v in eager_sd.values())
    coverage = n_mapped_params / n_total_params

    report = {
        "jit_path": args.jit_path,
        "architecture": "device",
        "segment_length": args.segment_length,
        "right_context_length": args.right_context_length,
        "eager_keys": len(eager_sd),
        "jit_keys": len(jit_sd),
        "mapped_keys": len(mapped),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "param_coverage": coverage,
    }
    print(f"mapped {len(mapped)}/{len(eager_sd)} keys, {coverage:.1%} of eager parameters")
    if missing:
        print(f"missing ({len(missing)}): {list(missing)[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"unexpected ({len(unexpected)}): {list(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}")
    if coverage < 0.95:
        print("WARNING: <95% parameter coverage; the bootstrap is incomplete. "
              "Inspect --dump-keys output and use --prefix-map.", file=sys.stderr)

    if not args.skip_parity:
        torch.manual_seed(0)
        frames = args.segment_length + args.right_context_length  # one streaming window
        video = torch.randn(1, frames, 1, 44, 44)
        audio = torch.randn(1, frames * 640, 1)
        with torch.inference_mode():
            jit_feats = jit(audio, video)
            eager_feats = module.encode_av(audio, video)
            feat_diff = (jit_feats - eager_feats).abs().max().item()
            report["feature_max_abs_diff"] = feat_diff
            print(f"feature parity: max|diff| = {feat_diff:.2e}")

            from torchaudio.models import RNNTBeamSearch

            tp = SentencePieceTokenProcessor(sp_model)
            length = torch.tensor([jit_feats.size(1)])
            jit_hyp, _ = RNNTBeamSearch(jit.model, module.blank_idx).infer(jit_feats, length, 10)
            eager_hyp, _ = module.decoder.infer(eager_feats, length, 10)
            jit_tokens, eager_tokens = jit_hyp[0][0], eager_hyp[0][0]
            report["jit_hypothesis"] = tp(jit_tokens)
            report["eager_hypothesis"] = tp(eager_tokens)
            report["hypothesis_match"] = jit_tokens == eager_tokens
            print(f"first-chunk hypothesis match: {report['hypothesis_match']}")
        ok = feat_diff < 1e-4 and report["hypothesis_match"]
        report["parity_ok"] = ok
        if not ok:
            print("WARNING: parity test failed; keep using the JIT model for inference "
                  "(demo_realtime.py --jit-model) and treat this checkpoint as suspect.", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": module.state_dict(), "bootstrap": report}, args.out)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
