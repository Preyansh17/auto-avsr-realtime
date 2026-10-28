"""Training CSV for one item-3 stack arm and seed (PLAN.md 3.3 item 3, pre-registered in asr_aug/stack/README.md).

    python make_mix.py --arm E2 --seed 1 --out train.csv

E1 = O (split_v1 merged train, 314) + s1 (semg1 Oct 1 train, 240): Tianyu's documented semg1 recipe data.
E2 = E1 + s2 (semg2 train, 240) + synthetic, 0.5 x real per voice, each voice's share half pool A / half pool B:
     semg1 voice: round(0.5 * (314 + 240)) clips, half from the ONS pool (old-data adapter B_r64,
     asr_aug/data/synth_kept_{A,B}.csv) and half from the Oct 1 adapter pool (stack/data/synth_kept_s1_{A,B}.csv);
     semg2 voice: round(0.5 * 240) clips from stack/data/synth_kept_s2_{A,B}.csv.
Synthetic sentences equal to any s1 or s2 val/test/oov sentence are dropped before drawing (the stack pools already
exclude them; the ONS pool is re-checked here). Draws use the training seed.
"""
import argparse
import json
import random
import re

OLD = "/scratch/pa2753/avsr_realtime/labels/splitv1/merged/train_spm1023.csv"
L = "/scratch/pa2753/multi_patient/labels"
H = "/scratch/pa2753/asr_aug/data"
S = "/scratch/pa2753/stack/data"
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
RATIO = 0.5


def rows(p):
    return [l for l in open(p).read().splitlines() if l.strip()]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def draw(rng, pools, n, held, text_of):
    """n clips, half from pools[0] (A) and the rest from pools[1] (B), excluding held text."""
    ok = [[r for r in rows(p) if norm(text_of[r.split(",")[1]]) not in held] for p in pools]
    na = n // 2
    assert na <= len(ok[0]) and n - na <= len(ok[1]), (pools, n, len(ok[0]), len(ok[1]))
    return rng.sample(ok[0], na) + rng.sample(ok[1], n - na)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["E1", "E2"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(f"stack-{a.arm}-{a.seed}")
    old, s1, s2 = rows(OLD), rows(f"{L}/s1_train_spm1023.csv"), rows(f"{L}/s2_train_spm1023.csv")
    out = old + s1
    if a.arm == "E2":
        held = {norm(json.loads(l)["transcript"]) for p in ("s1", "s2") for r in ("val", "test", "oov")
                for l in open(f"{ROLES}/{p}_{r}.jsonl")}
        text_of = {i["rel_path"]: i["norm"] for i in json.load(open(f"{H}/synth_info.json"))}
        text_of.update({i["rel_path"]: i["norm"] for i in json.load(open(f"{S}/synth_info.json"))})
        n1 = round(RATIO * (len(old) + len(s1)))
        n1_old = n1 // 2
        syn = draw(rng, [f"{H}/synth_kept_A.csv", f"{H}/synth_kept_B.csv"], n1_old, held, text_of)
        syn += draw(rng, [f"{S}/synth_kept_s1_A.csv", f"{S}/synth_kept_s1_B.csv"], n1 - n1_old, held, text_of)
        syn += draw(rng, [f"{S}/synth_kept_s2_A.csv", f"{S}/synth_kept_s2_B.csv"], round(RATIO * len(s2)), held, text_of)
        out = out + s2 + syn
        print(f"E2 synthetic: semg1 old-adapter {n1_old}, semg1 Oct 1 adapter {n1 - n1_old}, semg2 {round(RATIO * len(s2))}")
    rng.shuffle(out)
    open(a.out, "w").write("\n".join(out) + "\n")
    print(f"{a.arm} seed {a.seed}: {len(out)} rows -> {a.out}")


if __name__ == "__main__":
    main()
