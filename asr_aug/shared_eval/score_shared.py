"""Score any system's hypotheses on one shared patient role (Tianyu's split tree, LATEST_combined/roles/<role>.jsonl).
Hyps TSV: header row, must have a `hyp` column and a `take_id` or `path` column (take_id = file stem). Refs always come
from the role file, never from the TSV, so every system is scored on identical rows and text. Missing or extra rows abort.
Normalization: lowercase, hyphen -> space, drop chars outside [a-z' ], collapse spaces (same as asr_aug/report_*.py).
Usage: score_shared.py --role s1_test --tsv runA/test_hyps.tsv runB/test_hyps.tsv ... [--label X] [--rows out.tsv]
Several TSVs = several seeds of one system: prints each, then mean±sd, and corpus WER over all seeds."""
import argparse, csv, hashlib, json, os, re, statistics as st
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()
def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]
def stem(p): return os.path.splitext(os.path.basename(p))[0]
ap = argparse.ArgumentParser()
ap.add_argument("--role", required=True); ap.add_argument("--tsv", nargs="+", required=True)
ap.add_argument("--label", default=""); ap.add_argument("--rows")
a = ap.parse_args()
rf = os.path.join(ROLES, f"{a.role}.jsonl"); raw = open(rf, "rb").read()
ref = {x["take_id"]: norm(x["transcript"]) for x in map(json.loads, raw.decode().splitlines())}
print(f"role {a.role}: {len(ref)} rows, {os.path.realpath(rf)} sha256 {hashlib.sha256(raw).hexdigest()[:12]}")
wers, E, N, out = [], 0, 0, []
for t in a.tsv:
    rows = list(csv.DictReader(open(t), delimiter="\t"))
    hyp = {r["take_id"] if "take_id" in r else stem(r["path"]): r["hyp"] or "" for r in rows}
    miss, extra = ref.keys() - hyp.keys(), hyp.keys() - ref.keys()
    if miss or extra: raise SystemExit(f"{t}: row mismatch, missing {len(miss)} extra {len(extra)} e.g. {sorted(miss | extra)[:3]}")
    e = n = 0
    for k, r in ref.items():
        h = norm(hyp[k]); d = ed(r.split(), h.split()); e += d; n += len(r.split()); out.append((a.label, t, k, r, h, d))
    wers.append(100 * e / n); E += e; N += n
    print(f"{a.label} {t}: WER {wers[-1]:.2f}% ({e}/{n})")
if len(wers) > 1:
    print(f"{a.label} {a.role} seeds={len(wers)} mean {st.mean(wers):.2f} sd {st.stdev(wers):.2f}  corpus {100 * E / N:.2f}%")
if a.rows:
    with open(a.rows, "w") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["label", "tsv", "take_id", "ref", "hyp", "errors"]); w.writerows(out)
