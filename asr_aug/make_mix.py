"""Training CSV for one augmentation arm and seed.

    python make_mix.py --arm M1 --seed 1 --out train.csv

Arms: R (real only, train CSV order unchanged); S (all kept synthetic, A+B); M<r> (all real + round(r * n_real) kept synthetic, half pool A / half pool B);
A<r> / B<r> (real + r * n_real from one pool); D<r> (as M<r> but from the duration+silence-only filter);
V<r> (re-voicing, revoice/README.md: real + round(r * n_real) kept re-voiced clips; r <= 1 draws that many distinct
source clips and one kept copy of each, r > 1 takes one copy per source first and fills from the second copies;
Vall = real + every kept re-voiced clip).
The synthetic subset is drawn with the training seed. Real rows = split_v1 merged train.
"""
import argparse
import collections
import json
import random

DATA = "/scratch/pa2753/asr_aug/data"
RV = "/scratch/pa2753/asr_aug/revoice/data"
REAL = "/scratch/pa2753/avsr_realtime/labels/splitv1/merged/train_spm1023.csv"


def rows(p):
    return [l for l in open(p).read().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(f"mix-{a.arm}-{a.seed}")
    real = rows(REAL)
    kind, r = a.arm[0], (float(a.arm[1:]) if a.arm[1:] not in ("", "all") else None)
    if kind == "R":
        open(a.out, "w").write("\n".join(real) + "\n")
        print(f"R seed {a.seed}: {len(real)} real -> {a.out}")
        return
    if kind == "V":
        src_of = {i["rel_path"]: i["src"] for i in json.load(open(f"{RV}/rv_info.json"))}
        by_src = collections.defaultdict(list)
        for row in rows(f"{RV}/rv_kept.csv"):
            by_src[src_of[row.split(",")[1]]].append(row)
        srcs = sorted(by_src)
        n = sum(map(len, by_src.values())) if r is None else round(r * len(real))
        if n <= len(srcs):
            syn = [rng.choice(sorted(by_src[s])) for s in rng.sample(srcs, n)]
        else:
            first = {s: rng.choice(sorted(by_src[s])) for s in srcs}
            rest = sorted(x for s in srcs for x in by_src[s] if x != first[s])
            assert n - len(srcs) <= len(rest), f"need {n}, have {len(srcs) + len(rest)} kept re-voiced clips"
            syn = list(first.values()) + rng.sample(rest, n - len(srcs))
        out = real + syn
        rng.shuffle(out)
        open(a.out, "w").write("\n".join(out) + "\n")
        print(f"{a.arm} seed {a.seed}: {len(real)} real + {len(syn)} re-voiced ({len({x for x in syn})} distinct) -> {a.out}")
        return
    stem = "synth_durkept" if kind == "D" else "synth_kept"
    pa, pb = rows(f"{DATA}/{stem}_A.csv"), rows(f"{DATA}/{stem}_B.csv")
    if kind == "S":
        syn, out_real = pa + pb, []
    else:
        n = round(r * len(real))
        if kind in "MD":
            na = n // 2
            nb = n - na
            assert na <= len(pa) and nb <= len(pb), f"need {na}/{nb}, have {len(pa)}/{len(pb)}"
            syn = rng.sample(pa, na) + rng.sample(pb, nb)
        else:
            pool = pa if kind == "A" else pb
            assert n <= len(pool), f"need {n}, pool {kind} has {len(pool)}"
            syn = rng.sample(pool, n)
        out_real = real
    out = out_real + syn
    rng.shuffle(out)
    open(a.out, "w").write("\n".join(out) + "\n")
    print(f"{a.arm} seed {a.seed}: {len(out_real)} real + {len(syn)} synthetic -> {a.out}")


if __name__ == "__main__":
    main()
