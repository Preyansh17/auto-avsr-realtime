"""Training CSV for one old->Oct-1 semg1 arm and seed (README "Oct 1 session" section).

    python make_mix_oct1.py --arm ONS --seed 1 --out train.csv

Real sources: O = split_v1 merged train (legal+legacy, 314 clips); N = Oct 1 s1_train (240 clips).
Arms: F suffix (NF, ONF) = real + oov-text synthetic of --fold (2-fold oov experiment).
Arms: O, N, ON (O+N); append S for synthetic: OS, NS, ONS = real + round(0.5 * n_real) kept synthetic clips,
half pool A / half pool B, drawn with the training seed. Synthetic sentences equal to any s1_val/test/oov sentence are
dropped before drawing. Eval is always s1_val/test/oov (clean for ASR and for the Higgs adapter, trained on old data only).
"""
import argparse
import json
import random
import re

DATA = "/scratch/pa2753/asr_aug/data"
OLD = "/scratch/pa2753/avsr_realtime/labels/splitv1/merged/train_spm1023.csv"
L1 = "/scratch/pa2753/multi_patient/labels"
RATIO = 0.5


def rows(p):
    return [l for l in open(p).read().splitlines() if l.strip()]


def norm(t):
    return re.sub(r"\s+", " ", t.lower().replace("-", " ")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", choices=["A", "B"], help="arm ends in F: synthetic = this fold's oov-text clips (filter_fold.py)")
    a = ap.parse_args()
    rng = random.Random(f"oct1-{a.arm}-{a.seed}")
    fold_flag = a.arm.endswith("F")
    syn_flag = a.arm.endswith("S") or fold_flag
    base = a.arm[:-1] if syn_flag else a.arm
    assert base in ("O", "N", "ON"), a.arm
    real = (rows(OLD) if "O" in base else []) + (rows(f"{L1}/s1_train_spm1023.csv") if "N" in base else [])
    syn = []
    if fold_flag:
        # oov-text synthetic from one fold only; the evaluation sentences of the other fold stay unseen. Pool is small,
        # so cap at what the filtered fold holds (reported).
        assert a.fold
        pool = rows(f"{DATA}/synth_fold{a.fold}.csv")
        n = min(round(RATIO * len(real)), len(pool))
        syn = rng.sample(pool, n)
    elif syn_flag:
        W = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/w50s350_v1/roles"
        held = {norm(json.loads(l)["transcript"]) for r in ("val", "test", "oov") for l in open(f"{W}/s1_{r}.jsonl")}
        info = {i["rel_path"]: i["norm"] for i in json.load(open(f"{DATA}/synth_info.json"))}
        pools = []
        for p in "AB":
            ok = [r for r in rows(f"{DATA}/synth_kept_{p}.csv") if info[r.split(",")[1]] not in held]
            pools.append(ok)
        n = round(RATIO * len(real))
        na = n // 2
        assert na <= len(pools[0]) and n - na <= len(pools[1]), (n, len(pools[0]), len(pools[1]))
        syn = rng.sample(pools[0], na) + rng.sample(pools[1], n - na)
    out = real + syn
    rng.shuffle(out)
    open(a.out, "w").write("\n".join(out) + "\n")
    print(f"{a.arm} seed {a.seed}: {len(real)} real + {len(syn)} synthetic -> {a.out}")


if __name__ == "__main__":
    main()
