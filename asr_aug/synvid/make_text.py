"""Phase 2.1: generation text pool (plan 3.3 with A4: 3 to 8 words).

Sources, fixed before generating:
  pool B  /scratch/pa2753/asr_aug/data/pool_B.json, sentences with 3-8 words (others dropped, A4)
  LS-LM   2,000 sentences of 3-8 words from librispeech-lm-norm.txt.gz (openslr 11), uniform reservoir sample with
          random.Random(20261106) over every qualifying line (only [A-Z'] words, no single-letter-only lines)
Leak rules (hard): a sentence is dropped if, after score_shared normalisation, it equals or is within one word edit
(substitution, insertion or deletion) of any sentence of any non-train role in LATEST_combined/roles; it is also dropped
if it equals an s3_train or s3ov_train sentence. Duplicates inside the pool are removed. The run fails (exit 1) if any kept
sentence still matches a held-out sentence (re-check), so a leak can never be written silently.
Output: --out text.json, a list of {id, text, norm, source, words}, plus text_report.json (counts per drop reason).

Usage (CPU node): python make_text.py --out /scratch/pa2753/synvid/data/text.json
"""
import argparse, glob, gzip, json, os, random, re, sys

ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
POOL_B = "/scratch/pa2753/asr_aug/data/pool_B.json"
LSLM = "/scratch/pa2753/synvid/data/librispeech-lm-norm.txt.gz"
SEED, N_LS, MINW, MAXW = 20261106, 2000, 3, 8


def norm(t):  # asr_aug/shared_eval/score_shared.py
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def within_one(a, b):
    """word-level edit distance <= 1"""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) <= 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def load_roles():
    held, train = {}, set()
    for f in sorted(glob.glob(f"{ROLES}/*.jsonl")):
        role = os.path.basename(f)[:-6]
        for l in open(f):
            if not l.strip():
                continue
            n = norm(json.loads(l)["transcript"])
            if role.endswith("_train"):
                if role in ("s3_train", "s3ov_train"):
                    train.add(n)
            else:
                held.setdefault(n, set()).add(role)
    return held, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    held, train = load_roles()
    held_by_len = {}
    for h in held:
        held_by_len.setdefault(len(h.split()), []).append(h.split())

    def leak(n):
        w = n.split()
        if n in held:
            return "exact_heldout"
        for L in (len(w) - 1, len(w), len(w) + 1):
            for h in held_by_len.get(L, []):
                if within_one(w, h):
                    return "one_edit_heldout"
        if n in train:
            return "exact_s3_train"
        return None

    rep = {"pool_B_in": 0, "pool_B_len": 0, "ls_qualifying_lines": 0}
    cands = []
    for p in json.load(open(POOL_B)):
        rep["pool_B_in"] += 1
        n = norm(p["text"])
        if MINW <= len(n.split()) <= MAXW:
            cands.append(("B", p["text"], n))
        else:
            rep["pool_B_len"] += 1
    rng = random.Random(SEED)
    res, k = [], 0
    ok = re.compile(r"^[A-Z']+( [A-Z']+){%d,%d}$" % (MINW - 1, MAXW - 1))
    with gzip.open(LSLM, "rt") as f:
        for line in f:
            line = line.strip()
            if not ok.match(line) or all(len(t) == 1 for t in line.split()):
                continue
            k += 1
            if len(res) < 4 * N_LS:  # oversample 4x so leak / duplicate drops still leave N_LS
                res.append(line)
            else:
                j = rng.randrange(k)
                if j < 4 * N_LS:
                    res[j] = line
    rep["ls_qualifying_lines"] = k
    rng.shuffle(res)
    seen, out, drops = set(), [], {}
    for src, t, n in cands + [("LS", r, norm(r)) for r in res]:
        if src == "LS" and sum(1 for o in out if o["source"] == "LS") >= N_LS:
            continue
        why = "duplicate" if n in seen else leak(n)
        if why:
            drops[f"{src}:{why}"] = drops.get(f"{src}:{why}", 0) + 1
            continue
        seen.add(n)
        text = t if src == "B" else t.capitalize()
        out.append(dict(id=f"t{len(out):05d}", text=text, norm=n, source=src, words=len(n.split())))
    for o in out:  # re-check, fail hard on any leak
        if leak(o["norm"]) is not None:
            print("LEAK", o)
            sys.exit(1)
    rep.update(drops=drops, kept=len(out), kept_B=sum(o["source"] == "B" for o in out),
               kept_LS=sum(o["source"] == "LS" for o in out), heldout_sentences=len(held), seed=SEED)
    json.dump(out, open(a.out, "w"), indent=1)
    json.dump(rep, open(os.path.join(os.path.dirname(a.out), "text_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
