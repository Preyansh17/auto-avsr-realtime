"""Generate a G3 voice's plan rows with VoxCPM2 + that voice's LoRA (asr_aug/gen/README.md, amendment 4).

    python gen_g3.py --voice s1 --lora /scratch/pa2753/gen/g3/s1/lora/latest [--shard k --nshard N] [--limit N]

Text only (the LoRA carries the voice), repo defaults: cfg 2.0, 10 timesteps, retry_badcase on (3 tries, ratio 6.0),
no normalisation, no denoiser; torch.compile off. Each row is seeded from md5(rel). Rows come from g1/<v>/plan.json,
clips go to root/g3/<rel> at the model's 48 kHz. Resumable: a row whose clip exists is skipped; temp file, then rename.
"""
import argparse
import hashlib
import json
import os
import time

import soundfile as sf

G = "/scratch/pa2753/gen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, choices=("old", "s1", "s2"))
    ap.add_argument("--lora", required=True, help="LoRA checkpoint dir (lora_weights.safetensors + lora_config.json)")
    ap.add_argument("--out-root", default=f"{G}/root/g3")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="N evenly spaced plan rows only (smoke)")
    a = ap.parse_args()
    from voxcpm.core import VoxCPM
    from voxcpm.model.voxcpm import LoRAConfig

    info = json.load(open(f"{a.lora}/lora_config.json"))
    model = VoxCPM.from_pretrained(hf_model_id=f"{G}/models/VoxCPM2", load_denoiser=False, optimize=False,
                                   lora_config=LoRAConfig(**info["lora_config"]), lora_weights_path=a.lora)
    sr = model.tts_model.sample_rate
    rows = json.load(open(f"{G}/g1/{a.voice}/plan.json"))
    if a.limit:
        rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
    # shard first, then drop existing clips: workers start at different times, so filtering first shifts the shards
    todo = [r for r in rows[a.shard::a.nshard] if not os.path.exists(f"{a.out_root}/{r['rel']}")]
    print(f"{len(rows)} rows, {len(todo)} to generate in shard {a.shard}/{a.nshard}, sr {sr}", flush=True)
    t0 = time.time()
    for i, r in enumerate(todo):
        out = f"{a.out_root}/{r['rel']}"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        seed = int(hashlib.md5(r["rel"].encode()).hexdigest()[:8], 16)
        text = r["text"].strip()
        wav = model.generate(text=text[:1].upper() + text[1:], cfg_value=2.0, inference_timesteps=10, normalize=False,
                             denoise=False, retry_badcase=True, seed=seed)
        tmp = out[:-4] + ".tmp.wav"
        sf.write(tmp, wav, sr)
        os.replace(tmp, out)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(todo)} {(time.time() - t0) / (i + 1):.2f}s/clip", flush=True)
    print("generation done", flush=True)


if __name__ == "__main__":
    main()
