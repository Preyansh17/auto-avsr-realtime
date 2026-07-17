import os
import torch
from transformers import WhisperForConditionalGeneration

EXP_ROOT = "/scratch/pa2753/experiments/whisper_asr"
DOMAINS = ["legal", "legacy", "merged"]
SEEDS = [1, 2, 3]

for domain in DOMAINS:
    print(f"\n===== souping {domain} (seeds {SEEDS}) =====")
    ckpt_dirs = [f"{EXP_ROOT}/whisper_largev3_specaug_splitv1_{domain}_seed{s}/best" for s in SEEDS]
    for d in ckpt_dirs:
        assert os.path.isdir(d), f"missing {d}"

    models = [WhisperForConditionalGeneration.from_pretrained(d) for d in ckpt_dirs]
    sds = [m.state_dict() for m in models]

    keys = set(sds[0].keys())
    for sd in sds[1:]:
        assert set(sd.keys()) == keys, "state dict key mismatch across seeds"

    avg_sd = {}
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in sds], dim=0)
        avg_sd[k] = stacked.mean(dim=0)
        # preserve original dtype of the base model's param
        avg_sd[k] = avg_sd[k].to(sds[0][k].dtype)

    n_nan = sum(int(torch.isnan(v).any()) for v in avg_sd.values())
    n_inf = sum(int(torch.isinf(v).any()) for v in avg_sd.values())
    print(f"{domain}: averaged {len(avg_sd)} tensors, nan_tensors={n_nan} inf_tensors={n_inf}")
    assert n_nan == 0 and n_inf == 0, f"soup for {domain} produced NaN/Inf"

    soup_model = models[0]
    soup_model.load_state_dict(avg_sd)

    out_dir = f"{EXP_ROOT}/whisper_largev3_specaug_splitv1_{domain}_soup/best"
    os.makedirs(out_dir, exist_ok=True)
    soup_model.save_pretrained(out_dir)
    # processor/tokenizer files are identical across seeds -- copy from seed1
    from transformers import WhisperProcessor
    proc = WhisperProcessor.from_pretrained(ckpt_dirs[0])
    proc.save_pretrained(out_dir)
    print(f"{domain}: soup saved to {out_dir}")

    del models, sds, avg_sd, soup_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\nDone souping all domains.")
