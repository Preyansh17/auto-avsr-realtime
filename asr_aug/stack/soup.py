"""Uniform weight average ("soup") of the val-selected best/ checkpoints of one arm's seeds.

    python soup.py --arm E2 --seeds 1 2 3 4 5     # -> /scratch/pa2753/stack/runs/<arm>_soup/best

All seeds start from the same pretrained large-v3 and share the frozen decoder, so only encoder weights differ.
Tensors are averaged in fp32 one shard at a time; non-weight files (config, tokenizer) are copied from the first seed.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

R = "/scratch/pa2753/stack/runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    a = ap.parse_args()
    srcs = [f"{R}/{a.arm}_seed{s}/best" for s in a.seeds]
    out = f"{R}/{a.arm}_soup/best"
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(f"{srcs[0]}/model.safetensors.index.json"))
    for f in os.listdir(srcs[0]):
        if not f.endswith(".safetensors") and f != "training_args.bin":
            shutil.copy(f"{srcs[0]}/{f}", out)
    shards = sorted(set(idx["weight_map"].values()))
    for sh in shards:
        acc, dtypes = {}, {}
        for src in srcs:
            with safe_open(f"{src}/{sh}", framework="pt") as fh:
                for k in fh.keys():
                    t = fh.get_tensor(k)
                    dtypes[k] = t.dtype
                    acc[k] = acc.get(k, 0) + t.float()
        save_file({k: (v / len(srcs)).to(dtypes[k]).contiguous() for k, v in acc.items()}, f"{out}/{sh}",
                  metadata={"format": "pt"})
        print(f"{sh}: {len(acc)} tensors averaged over {len(srcs)} seeds")
    json.dump({"arm": a.arm, "seeds": a.seeds, "sources": srcs}, open(f"{R}/{a.arm}_soup/soup.json", "w"), indent=1)


if __name__ == "__main__":
    main()
