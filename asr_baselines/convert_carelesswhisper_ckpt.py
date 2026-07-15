#!/usr/bin/env python3
"""Convert a CarelessWhisper LoRA finetune's Lightning .ckpt to the
dims/cfg/state_dict .pt format whisper_rt.load_streaming_model expects.

Lightning saves state_dict keys prefixed "model." (LoRAStreamedWhisper wraps
self.model = StreamingWhisper(...)), and doesn't write the flat
{dims, cfg: {gran, rank, extra_gran_blocks}, state_dict} shape their loader
wants -- both mismatches would silently produce garbage (or KeyErrors) if
handed to carelesswhisper_streaming_eval.py directly.

Hard rule from this project's earlier Whisper-checkpoint corruption incident
(see results/whisper_streaming_investigation_2026-07-09.md): every checkpoint
conversion must assert bit-exact against its source before being trusted.
Here that means loading the converted .pt back and diffing every tensor
against the original Lightning state_dict -- max diff must be exactly 0.0.

Requires the CarelessWhisper-streaming checkout on PYTHONPATH (its own
Config/LoRAStreamedWhisper classes are pickled into the .ckpt).

DO NOT run this on the login node for large-v2 OR full-finetune checkpoints
(FULL_FINETUNE=1) -- the Lightning .ckpt bundles full optimizer state
alongside the weights, and it's the optimizer state size, not raw model
size, that OOMs the login node: large-v2's LoRA run (7.1M trainable) and
small's full-finetune run (240M trainable) both got SIGKILL'd (exit 137, no
error message -- looks like a silent hang unless you check the real exit
code) despite `small`-full-FT being the smaller BASE model. Submit as a
small CPU sbatch job instead (--mem=32GB has been enough for both cases so
far).

  python -m asr_baselines.convert_carelesswhisper_ckpt \
    --ckpt .../checkpoint/checkpoint-epoch=0004.ckpt \
    --careless-dir /scratch/$USER/third_party/CarelessWhisper-streaming \
    --out /scratch/$USER/carelesswhisper/ckpts/legal_train_epoch4.pt
"""

import argparse
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Lightning .ckpt from training_code/train.py")
    p.add_argument("--careless-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sys.path.insert(0, args.careless_dir)
    import torch

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    prefix = "model."
    sd = ckpt["state_dict"]
    bad = [k for k in sd if not k.startswith(prefix)]
    if bad:
        raise RuntimeError(f"unexpected state_dict keys without '{prefix}' prefix: {bad[:5]}")
    stripped = {k[len(prefix):]: v for k, v in sd.items()}

    out_ckpt = {
        "dims": ckpt["dims"],
        "cfg": {
            "gran": hp["enc_emb_gran"],
            "rank": hp["rank"],
            "extra_gran_blocks": hp["enc_context"],
        },
        "state_dict": stripped,
    }
    torch.save(out_ckpt, args.out)

    # Bit-exact verify: reload from disk, diff every tensor against the
    # in-memory source. Not "close enough" -- exactly 0.0, per project rule.
    reloaded = torch.load(args.out, map_location="cpu", weights_only=False)
    assert reloaded["state_dict"].keys() == stripped.keys(), "key set changed across save/load"
    max_diff = 0.0
    for k in stripped:
        a, b = stripped[k], reloaded["state_dict"][k]
        assert a.shape == b.shape and a.dtype == b.dtype, f"{k}: shape/dtype changed"
        d = (a.float() - b.float()).abs().max().item()
        max_diff = max(max_diff, d)
    if max_diff != 0.0:
        import os
        os.remove(args.out)
        raise RuntimeError(f"conversion NOT bit-exact (max diff {max_diff}) -- deleted {args.out}")

    print(f"OK: bit-exact conversion, {len(stripped)} tensors, max diff {max_diff}")
    print(f"epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')}")
    print(f"gran={out_ckpt['cfg']['gran']} rank={out_ckpt['cfg']['rank']} "
          f"extra_gran_blocks={out_ckpt['cfg']['extra_gran_blocks']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
