"""A18: choose extra generated-clip sentences so every W type appears in >= 5 generated clips (S clips + DailyDialog,
then S re-samples with new generator seeds). Leak-checked against every patient split role file, V0 T/val and the pool.
Run with /scratch/pa2753/envs/idx2/bin/python (g2p_en) from /scratch/pa2753/synvid/t2m2/code."""
import glob, json, os, random, re, sys
from collections import Counter
sys.path.insert(0, "/scratch/pa2753/synvid/t2m2/code")
from phonemize import item, norm
SV = "/scratch/pa2753/synvid"; OUT = f"{SV}/a18"; K = 5
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
W = set(open(f"{SV}/v0/split/W.txt").read().split())
S = jl(f"{SV}/v0/split/roles/v0_S.jsonl")
txt = lambda r: [r.get(k) for k in ("raw_sentence", "transcript", "norm", "text") if isinstance(r.get(k), str)]
# leak set: exact lowercased + normalised forms of every transcript we must not train on
block, nroles = set(), 0
for f in glob.glob("/scratch/th3482/Dysarthric_ASR/data/patients/splits/*/roles/*.jsonl"):
    try:
        rows = jl(f)
    except Exception as e:
        print("skip", f, e); continue
    nroles += 1
    for r in rows:
        for t in txt(r):
            block |= {t.strip().lower(), " ".join(norm(t))}
for role in ("v0_T", "v0_val"):
    for r in jl(f"{SV}/v0/split/roles/{role}.jsonl"):
        for t in txt(r):
            block |= {t.strip().lower(), " ".join(norm(t))}
for r in jl(f"{SV}/t2m2/pool/pool.jsonl"):
    for t in txt(r):
        block |= {t.strip().lower(), " ".join(norm(t))}
# coverage from the 169 S clips (V3 set d, one clip each)
cov = Counter()
for r in S:
    for w in set(r["norm"].split()) & W:
        cov[w] += 1
dd, seen_dd = [], set()
for r in jl(f"{SV}/t2m2/text/dd_pool.jsonl"):
    ws = norm(r["norm"]); t = " ".join(ws)
    if not (3 <= len(ws) <= 10) or t in seen_dd:
        continue
    seen_dd.add(t)
    if t in block or r["text"].strip().lower() in block or not (set(ws) & W):
        continue
    dd.append(t)
leaked = sum(1 for t in seen_dd if t in block)
random.Random(20261118).shuffle(dd)
by_w = {w: [t for t in dd if w in t.split()] for w in W}
extra, used = [], set()
while True:
    need = sorted((w for w in W if cov[w] < K), key=lambda w: (len([t for t in by_w[w] if t not in used]), w))
    need = [w for w in need if any(t not in used for t in by_w[w])]
    if not need:
        break
    t = next(t for t in by_w[need[0]] if t not in used)
    used.add(t); extra.append(t)
    for w in set(t.split()) & W:
        cov[w] += 1
# fill the rest by re-sampling S sentences with new generator seeds
res = {}  # seed -> list of S indices
for w in sorted(W):
    idx = [i for i, r in enumerate(S) if w in r["norm"].split()]
    k = 0
    while cov[w] < K:
        i = idx[k % len(idx)]; seed = 2 + k // len(idx) if len(idx) else None
        res.setdefault(seed, []).append(i)
        for v in set(S[i]["norm"].split()) & W:
            cov[v] += 1
        k += 1
os.makedirs(f"{OUT}/x", exist_ok=True)
with open(f"{OUT}/x/items.jsonl", "w") as f:
    for j, t in enumerate(extra):
        f.write(json.dumps(item(f"x_{j:04d}", t)) + "\n")
for seed, ids in res.items():
    os.makedirs(f"{OUT}/r{seed}", exist_ok=True)
    with open(f"{OUT}/r{seed}/items.jsonl", "w") as f:
        for j, i in enumerate(ids):
            f.write(json.dumps(item(f"r{seed}_{j:04d}", S[i]["norm"]) | dict(s_index=i, s_id=S[i]["id"])) + "\n")
rep = dict(W=len(W), K=K, role_files_checked=nroles, block_texts=len(block), dd_candidates=len(dd), dd_blocked=leaked,
           s_clips=len(S), dd_extra=len(extra), resample={str(k): len(v) for k, v in res.items()},
           min_cov=min(cov[w] for w in W), W_cov_hist=dict(sorted(Counter(min(cov[w], 20) for w in W).items())),
           W_with_resample=sorted({w for k, v in res.items() for i in v for w in set(S[i]["norm"].split()) & W}))
json.dump(rep, open(f"{OUT}/select_report.json", "w"), indent=1); print(json.dumps(rep))
