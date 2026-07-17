import sys, os, torch
sys.path.insert(0, "/scratch/pa2753/third_party")
from convert_hf_to_openai import convert_tfms_to_openai_whisper, reverse_rename_keys
from transformers import WhisperForConditionalGeneration

EXP_ROOT = "/scratch/pa2753/experiments/whisper_asr"
DOMAINS = ["legal", "legacy", "merged"]

results = []
for domain in DOMAINS:
    tag = f"splitv1_{domain}_soup"
    exp_dir = f"{EXP_ROOT}/whisper_largev3_specaug_{tag}"
    hf_ckpt = f"{exp_dir}/best"
    out_dir = f"{exp_dir}/openai_format"
    out_pt = f"{out_dir}/large-v3-{tag}.pt"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n===== converting {tag} =====")
    if not os.path.isdir(hf_ckpt):
        print(f"SKIP: {hf_ckpt} does not exist")
        results.append((tag, "MISSING_CKPT"))
        continue

    convert_tfms_to_openai_whisper(hf_ckpt, out_pt)

    # Structural verification (not bit-exact -- soup isn't from a single source):
    # confirm the converted state dict's keys/shapes/values exactly match the HF
    # source we just converted (round-trip correctness of the converter itself,
    # which IS still bit-exact-checkable even though the soup weights themselves
    # are an average).
    hf = WhisperForConditionalGeneration.from_pretrained(hf_ckpt)
    sd_hf = {k: v.cpu() for k, v in hf.state_dict().items()}
    sd_hf = reverse_rename_keys(sd_hf)
    del sd_hf["proj_out.weight"]
    sd_hf = {k.replace("model.", "", 1): v for k, v in sd_hf.items()}
    sd_new = torch.load(out_pt, map_location="cpu")["model_state_dict"]

    key_mismatch = set(sd_hf) != set(sd_new)
    worst = max((sd_hf[k] - sd_new[k]).abs().max().item() for k in sd_new if k in sd_hf) if not key_mismatch else float("inf")
    dtypes = {str(v.dtype) for v in sd_new.values()}
    bit_exact = (not key_mismatch) and worst == 0.0 and dtypes == {"torch.float32"}

    print(f"{tag}: key_mismatch={key_mismatch} worst_diff={worst} dtypes={dtypes} converter_round_trip_exact={bit_exact}")
    if not bit_exact:
        print(f"ABORT: {tag} conversion round-trip NOT exact -- refusing to trust this checkpoint")
        results.append((tag, f"NOT_EXACT diff={worst}"))
        if os.path.exists(out_pt):
            os.remove(out_pt)
        continue

    results.append((tag, f"OK -> {out_pt}"))
    del hf, sd_hf, sd_new

print("\n===== SUMMARY =====")
for tag, status in results:
    print(f"{tag}: {status}")
n_ok = sum(1 for _, s in results if s.startswith("OK"))
print(f"\n{n_ok}/{len(results)} converted and verified.")
