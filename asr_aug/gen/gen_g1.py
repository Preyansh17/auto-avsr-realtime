"""Generate a G1 voice's plan rows with the merged IndexTTS-2 + LoRA model (g1.sbatch runs this in the idx2 env).

    python gen_g1.py --voice-dir /scratch/pa2753/gen/g1/s1 --out-root /scratch/pa2753/gen/root/g1
    python gen_g1.py ... --out-root /scratch/pa2753/gen/root/g1d --dur-scale 0.528   # G1d (amendment 2)

Resumable: a row whose clip exists is skipped; each clip is written to a temp file and renamed. Each row is seeded from
its rel path, so a requeued job regenerates the same clips. IndexTTS-2 default sampling settings.
"""
import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--dur-scale", type=float, default=1.0,
                    help="multiplies the s2mel length (needs g1d.sbatch's infer_v2.py patch when != 1)")
    ap.add_argument("--limit", type=int, default=0, help="N evenly spaced plan rows only (smoke)")
    ap.add_argument("--shard", type=int, default=0, help="this worker takes todo rows shard::nshard")
    ap.add_argument("--nshard", type=int, default=1,
                    help="workers sharing one GPU (batch-1 TTS leaves a single process mostly GPU-idle)")
    a = ap.parse_args()
    from indextts.infer_v2 import IndexTTS2

    md = f"{a.voice_dir}/model"
    tts = IndexTTS2(cfg_path=f"{md}/config.yaml", model_dir=md, use_fp16=True, use_cuda_kernel=False)
    tts.dur_scale = a.dur_scale
    rows = json.load(open(f"{a.voice_dir}/plan.json"))
    if a.limit:
        rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
    todo = [r for r in rows if not os.path.exists(f"{a.out_root}/{r['rel']}")]
    todo = todo[a.shard::a.nshard]  # per-row seeds, so output does not depend on the sharding
    print(f"{len(rows)} rows, {len(rows) - len(todo)} done, {len(todo)} to generate", flush=True)
    t0 = time.time()
    for i, r in enumerate(todo):
        out = f"{a.out_root}/{r['rel']}"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        s = int(hashlib.md5(r["rel"].encode()).hexdigest()[:8], 16)
        random.seed(s), np.random.seed(s), torch.manual_seed(s)
        tmp = out[:-4] + ".tmp.wav"
        tts.infer(spk_audio_prompt=r["ref_wav"], text=r["text"], output_path=tmp, verbose=False)
        os.replace(tmp, out)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(todo)} {(time.time() - t0) / (i + 1):.2f}s/clip", flush=True)
    print("generation done", flush=True)


if __name__ == "__main__":
    main()
