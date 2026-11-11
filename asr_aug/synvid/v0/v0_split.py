"""V0 real-video oracle (PLAN.md 4.3): frozen held-out-word split inside the 912-clip training pool.

W = word types that occur in 2..6 pool clips, are outside the s3 50-word set and have no clip among the generator's
heldout30. Types are taken in a seeded random order until the test set T holds N_T clips; every pool clip containing
a W word goes to T (test) or S (support), and every W type has at least one clip in each. VAL = seeded clips from the
rest, stratified by source, used only for checkpoint selection (it contains no W word).
Arms: A0 = pool - T - S - VAL, A1 = pool - T - VAL (= A0 + S). No eval role is read.
"""
import collections, hashlib, json, random, sys
from pathlib import Path

SEED, N_T, N_VAL = 20261114, 100, 60
POOL = Path("/scratch/pa2753/synvid/t2m2/pool/pool.jsonl")
CROPS = Path("/scratch/pa2753/synvid/t2m2/crops/npz")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/pa2753/synvid/v0/split")

P = [r for r in map(json.loads, POOL.read_text().splitlines()) if r.get("role") in ("gen_train", "heldout")]
assert len(P) == 912 and len({r["id"] for r in P}) == 912
by_id = {r["id"]: r for r in P}
words = {r["id"]: r["norm"].split() for r in P}
s3 = {w for r in P if r["src"] == "s3" for w in words[r["id"]]}
clips_of = collections.defaultdict(list)
for r in P:
    for w in sorted(set(words[r["id"]])):
        clips_of[w].append(r["id"])
held = {r["id"] for r in P if r["role"] == "heldout"}
cand = sorted(w for w, c in clips_of.items() if 2 <= len(c) <= 6 and w not in s3 and not held & set(c))

rng = random.Random(SEED)
order = cand[:]; rng.shuffle(order)
T, S, W = set(), set(), []
for w in order:
    if len(T) >= N_T:
        break
    C = clips_of[w]
    un = [c for c in C if c not in T and c not in S]; rng.shuffle(un)
    need = (not any(c in T for c in C)) + (not any(c in S for c in C))
    if len(un) < need:
        continue  # cannot give this type both a test and a support clip
    if not any(c in T for c in C):
        T.add(un.pop())
    if not any(c in S for c in C):
        S.add(un.pop())
    S.update(un)
    W.append(w)
Wset = set(W)
assert not T & S
rest = [r["id"] for r in P if r["id"] not in T and r["id"] not in S]
assert not any(Wset & set(words[i]) for i in rest), "a W word is left in the base pool"
val = []
for src in ("s3", "s3ov", "v34", "real"):
    ids = sorted(i for i in rest if by_id[i]["src"] == src and i not in held)
    val += rng.sample(ids, round(N_VAL * len(ids) / sum(1 for i in rest if i not in held)))
val = set(val)
A0 = [i for i in rest if i not in val]
A1 = A0 + sorted(S)


def row(i):
    r = by_id[i]; p = CROPS / f"{i}.npz"
    assert p.exists(), p
    return dict(id=i, take_id=i, rel_path=f"{i}.npz", domain=r["src"], raw_sentence=r["transcript"], transcript=r["transcript"],
                norm=r["norm"], video_path=str(p))


(OUT / "roles").mkdir(parents=True, exist_ok=True)
roles = {"v0_A0_train": A0, "v0_A1_train": A1, "v0_S": sorted(S), "v0_T": sorted(T), "v0_val": sorted(val)}
sha = {}
for name, ids in roles.items():
    txt = "".join(json.dumps(row(i)) + "\n" for i in ids)
    (OUT / "roles" / f"{name}.jsonl").write_text(txt); sha[name] = hashlib.sha256(txt.encode()).hexdigest()
(OUT / "W.txt").write_text("\n".join(sorted(W)) + "\n")
src_n = lambda ids: dict(collections.Counter(by_id[i]["src"] for i in ids))
tokT = [w for i in T for w in words[i]]
rep = dict(seed=SEED, pool=len(P), candidate_types=len(cand), W_types=len(W),
           clips={k: len(v) for k, v in roles.items()}, by_source={k: src_n(v) for k, v in roles.items()},
           T_tokens=len(tokT), T_W_tokens=sum(w in Wset for w in tokT),
           S_W_tokens=sum(w in Wset for i in S for w in words[i]),
           support_clips_per_W_type=dict(sorted(collections.Counter(sum(c in S for c in clips_of[w]) for w in W).items())),
           T_tokens_seen_in_A0=sum(w in {x for i in A0 for x in words[i]} for w in tokT),
           T_other_unseen_tokens=sum(w not in Wset and w not in {x for i in A0 for x in words[i]} for w in tokT),
           W_in_val=sum(w in Wset for i in val for w in words[i]), sha256=sha)
(OUT / "split_report.json").write_text(json.dumps(rep, indent=1) + "\n")
print(json.dumps(rep, indent=1))
