"""Step 0 (CPU): percent / % / digit forms in the 193 in-home references and F1 predictions; rescore consistently."""
import json, re, sys
from collections import Counter
sys.path.insert(0, "/scratch/pa2753/synvid/endclip")
from endlib import normalize, normalize_pct, clip_stats, summarize, paired_bootstrap

S = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/semg3_rcv_20261109_v1/roles/dc_home_all.jsonl"
P = "/scratch/th3482/Dysarthric_ASR/runs/s3v34/home_eval_v1/bp/decgrid/F1/c0.1_b10/evaluation/dc_home_all.predictions.json"
rows = {r["rel_path"]: r for r in map(json.loads, open(S))}
preds = json.load(open(P))
assert len(preds) == 193 and all(p["rel_path"] in rows for p in preds)

def forms(t):
    return dict(percent_word=len(re.findall(r"\bpercent\b", t, re.I)), pct_sign=t.count("%"), digit_tokens=len(re.findall(r"\d+", t)))

c = {k: Counter() for k in ("ref_raw", "ref_transcript", "pred")}
for p in preds:
    r = rows[p["rel_path"]]
    c["ref_raw"].update(forms(r["raw_sentence"])); c["ref_transcript"].update(forms(r["transcript"])); c["pred"].update(forms(p["prediction"]))
print("form counts (tokens over 193 clips):")
for k, v in c.items(): print(f"  {k:15s}", dict(v))
print("clips with % in raw ref:", sum("%" in rows[p["rel_path"]]["raw_sentence"] for p in preds),
      "| with 'percent' word in raw ref:", sum(bool(re.search(r"\bpercent\b", rows[p["rel_path"]]["raw_sentence"], re.I)) for p in preds),
      "| with digits in raw ref:", sum(bool(re.search(r"\d", rows[p["rel_path"]]["raw_sentence"])) for p in preds))
print("normaliser checks:")
for t in ("NINETY PERCENT", "90%", "90 %", "ninety %", "85 percent", "eighty five%"):
    print(f"  {t!r:18s} delivered -> {normalize(t)!r:22s} consistent -> {normalize_pct(t)!r}")

A = [clip_stats(rows[p["rel_path"]]["raw_sentence"], p["prediction"]) for p in preds]
B = [clip_stats(rows[p["rel_path"]]["raw_sentence"], p["prediction"], normalize_pct) for p in preds]
# check our scoring matches the stored edit distances
mism = sum(a["err"] != p["edit_distance"] for a, p in zip(A, preds))
print("delivered normaliser:", summarize(A), "| stored-edit mismatches:", mism)
print("consistent normaliser:", summarize(B))
print("bootstrap consistent - delivered:", paired_bootstrap(A, B, keys=("last_wrong", "first_wrong", "err")))

print("\nlast-word-wrong clips (delivered normaliser): last_ref -> hyp_last | raw ref | prediction")
lw = Counter()
for a, p in zip(A, preds):
    if a["last_wrong"]:
        lw[a["last_ref"]] += 1
        print(f"  {a['last_ref']:>10s} -> {a['hyp_last']:<10s} | {rows[p['rel_path']]['raw_sentence']} | {p['prediction']}")
print("last ref word among last-wrong clips:", lw.most_common(10))
print("clips whose ref ends in 'percent' (delivered norm):", sum(a["last_ref"] == "percent" for a in A),
      "| of which last wrong:", sum(a["last_ref"] == "percent" and a["last_wrong"] for a in A))
print("clips whose ref ends in 'percent' (consistent norm):", sum(b["last_ref"] == "percent" for b in B),
      "| of which last wrong:", sum(b["last_ref"] == "percent" and b["last_wrong"] for b in B))
json.dump(dict(delivered=A, consistent=B, rel=[p["rel_path"] for p in preds]), open("/scratch/pa2753/synvid/endclip/step0_stats.json", "w"))
