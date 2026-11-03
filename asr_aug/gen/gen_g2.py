"""Generate G2 clips (asr_aug/gen/README.md, amendment 3): IndexTTS-1.5 + the trained ProtoDisent GPT, the same plan rows
and references as Higgs/G1 (gen/g1/<v>/plan.json), pathology class 6 (semg1: old, s1) or 7 (semg2: s2).

    python gen_g2.py --voice s1 --run /scratch/pa2753/gen/g2/run [--shard k --nshard N] [--limit N]

Resumable: a row whose clip exists is skipped; each clip is written to a temp file and renamed. Each row is seeded
from md5(rel). Repo default sampling (top_p 0.8, top_k 30, temperature 1.0), in fp32 as the repo runs it (fp16 gives
NaN sampling probabilities with the trained GPT); no duration post-processing.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch

G = "/scratch/pa2753/gen"
REPO = f"{G}/code/Controllable-Dysarthric-Speech-Synthesis"
CLASS = {"old": 6, "s1": 6, "s2": 7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, choices=list(CLASS))
    ap.add_argument("--run", required=True, help="run dir with model/ (symlinks + checkpoints) and g2.yaml")
    ap.add_argument("--out-root", default=f"{G}/root/g2")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="N evenly spaced plan rows only (smoke)")
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    from indextts.inference import IndexTTS

    ckpt = f"{a.run}/model/checkpoints_dys_spk_grl_exp1/gpt_best.pth"
    tts = IndexTTS(cfg_path=f"{a.run}/g2.yaml", model_dir=f"{a.run}/model", is_fp16=False, use_cuda_kernel=False,
                   gpt_checkpoint=ckpt)
    rows = json.load(open(f"{G}/g1/{a.voice}/plan.json"))
    if a.limit:
        rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
    todo = [r for r in rows if not os.path.exists(f"{a.out_root}/{r['rel']}")][a.shard::a.nshard]
    print(f"{len(rows)} rows, {len(todo)} to generate in shard {a.shard}/{a.nshard}", flush=True)
    t0 = time.time()
    for i, r in enumerate(todo):
        out = f"{a.out_root}/{r['rel']}"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        s = int(hashlib.md5(r["rel"].encode()).hexdigest()[:8], 16)
        random.seed(s), np.random.seed(s), torch.manual_seed(s)
        tmp = out[:-4] + ".tmp.wav"
        tts.infer(audio_prompt=r["ref_wav"], text=r["text"], output_path=tmp, pathology_ids=CLASS[a.voice])
        os.replace(tmp, out)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(todo)} {(time.time() - t0) / (i + 1):.2f}s/clip", flush=True)
    print("generation done", flush=True)


if __name__ == "__main__":
    main()
