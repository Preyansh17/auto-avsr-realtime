"""Train CSV for one TORGO arm x seed.

    python make_mix.py --arm R|M0.5|M0.5sc --seed 4 --out run/train.csv

R = real train (1693 clips, unchanged). M<r> = real + for each speaker round(r * that speaker's real train count) clips
drawn from that speaker's judge-kept synthetic clips (seeded by arm and seed), shuffled together. Suffix "sc" draws from
synth_kept_silcal.csv (calibrated silence floor, filter_synth.py --silence calibrated) instead of synth_kept.csv.
"""
import argparse
import collections
import csv
import json
import os
import random
import re

D = "/scratch/pa2753/torgo_aug/data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    real = [l for l in open(f"{D}/train_spm1023.csv").read().splitlines() if l]
    if a.arm == "R":
        open(a.out, "w").write("\n".join(real) + "\n")
        print(f"R: {len(real)} real")
        return
    m = re.fullmatch(r"M([\d.]+)(sc)?", a.arm)
    r, kept_csv = float(m.group(1)), "synth_kept_silcal.csv" if m.group(2) else "synth_kept.csv"
    spk_of = lambda rel: rel.split("/")[0].split("_")[2]  # FT_TG_<spk>_... # noqa: E731
    n_real = collections.Counter(l.split(",")[1].split("/")[0] for l in real)
    kept = collections.defaultdict(list)
    for l in open(f"{D}/{kept_csv}").read().splitlines():
        if l:
            kept[spk_of(l.split(",")[1])].append(l)
    rng = random.Random(f"torgo-mix-{a.arm.removesuffix('sc')}-{a.seed}")
    syn = []
    for spk, n in sorted(n_real.items()):
        k = round(r * n)
        assert len(kept[spk]) >= k, f"{spk}: need {k}, kept {len(kept[spk])}"
        syn += rng.sample(sorted(kept[spk]), k)
    rows = real + syn
    rng.shuffle(rows)
    open(a.out, "w").write("\n".join(rows) + "\n")
    print(f"{a.arm} seed {a.seed}: {len(real)} real + {len(syn)} synthetic {dict(collections.Counter(spk_of(s.split(',')[1]) for s in syn))}")


if __name__ == "__main__":
    main()
