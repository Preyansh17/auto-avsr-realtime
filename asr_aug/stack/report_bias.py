"""Item-1 biasing report (pre-registered in asr_aug/stack/README.md, with amendment 1). Prints tables and contrasts.

    python report_bias.py [--seeds 1 2 3 4 5]

Per run (runs/<arm>_seed<k>, runs/<arm>_soup), from <run>/bias:
  beam    beam_none_0_<set>_hyps.tsv
  prompt  prompt_<bias>_0_<set>_hyps.tsv
  ngram   ngram_<bias>_<lam>_<set>_hyps.tsv, lam = that run's val choice (selection.json)
  llm     llm_<bias>_<mu>_<set>_hyps.tsv,    mu  = that run's val choice (llm_selection.json)
Primary: each clean method minus beam on s1_oov, 5 single seeds, per arm; paired two-level bootstrap (report_stack.boot);
Holm over the 3 methods within each arm. "Helps" = Holm p < 0.05, CI upper < 0, and s1_test point diff <= +2 pp.
Loops: hyp >= 2 x ref words + 5, or a word 4-gram occurring 3+ times (amendment 1).
"""
import argparse
import collections
import csv
import json
import os
import random
import statistics as st

from report_stack import R, boot, errs, norm, refs, wer

METHODS = ("beam", "prompt", "ngram", "llm")
TEST = ("s1_test", "s1_oov", "s2_test", "s2_oov")
LAMS = (0.1, 0.2, 0.3, 0.5)
MUS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
BAR = {"s1_oov": 10.0, "s2_oov": 2.1}  # Tianyu's best OOV (Notion)


def hyps(tsv):
    out = {}
    for r in csv.DictReader(open(tsv), delimiter="\t"):
        k = r["take_id"] if "take_id" in r else os.path.splitext(os.path.basename(r["path"]))[0]
        out[k] = norm(r["hyp"] or "")
    return out


def tag(run, method, bias):
    if method == "beam":
        return "beam_none_0"
    if method == "prompt":
        return f"prompt_{bias}_0"
    if method == "ngram":
        return f"ngram_{bias}_{json.load(open(f'{run}/bias/selection.json'))['lam'][bias]:g}"
    return f"llm_{bias}_{json.load(open(f'{run}/bias/llm_selection.json'))[bias]['mu']:g}"


def loop(ref, h):
    w = h.split()
    g = collections.Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    return len(w) >= 2 * len(ref.split()) + 5 or bool(g and max(g.values()) >= 3)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    rng = random.Random(0)
    REF = {s: refs(s) for s in TEST}
    systems = {arm: {"single": [f"{R}/{arm}_seed{k}" for k in a.seeds], "soup": [f"{R}/{arm}_soup"]} for arm in ("E1", "E2")}
    E, loops, sel = {}, collections.Counter(), collections.defaultdict(list)
    for arm, kinds in systems.items():
        for kind, runs in kinds.items():
            runs = [r for r in runs if os.path.exists(f"{r}/bias/llm_selection.json")]
            if not runs:
                continue
            for r in runs:
                sel[arm, kind].append((os.path.basename(r), json.load(open(f"{r}/bias/selection.json"))["lam"],
                                       {b: v["mu"] for b, v in json.load(open(f"{r}/bias/llm_selection.json")).items()}))
            for m in METHODS:
                for b in (("none",) if m == "beam" else ("clean", "words")):
                    for s in TEST:
                        es = []
                        for r in runs:
                            h = hyps(f"{r}/bias/{tag(r, m, b)}_{s}_hyps.tsv")
                            es.append(errs(REF[s], h))
                            loops[arm, kind, m, b, s] += sum(loop(REF[s][k], h[k]) for k in REF[s])
                        E[arm, kind, m, b, s] = es

    print("Selected lam / mu per run (val):")
    for (arm, kind), v in sel.items():
        for name, lam, mu in v:
            print(f"  {name}: lam {lam}  mu {mu}")

    print("\nWER % (single: mean ± sd over seeds; soup: one model). Bar: Tianyu s1_oov 10.0, s2_oov 2.1")
    print("| arm | system | method | bias | " + " | ".join(TEST) + " | loops (" + "/".join(TEST) + ") |")
    print("|---|---|---|---|" + "---|" * (len(TEST) + 1))
    for (arm, kind, m, b, s0) in [k for k in E if k[4] == TEST[0]]:
        cells = []
        for s in TEST:
            v = [wer(e) for e in E[arm, kind, m, b, s]]
            cells.append(f"{st.mean(v):.2f} ± {st.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}")
        lp = "/".join(str(loops[arm, kind, m, b, s]) for s in TEST)
        print(f"| {arm} | {kind} ({len(E[arm, kind, m, b, s0])}) | {m} | {b} | " + " | ".join(cells) + f" | {lp} |")

    print("\nPRIMARY: clean method - beam, s1_oov, single seeds; Holm over 3 methods per arm")
    for arm in ("E1", "E2"):
        if len(E.get((arm, "single", "beam", "none", "s1_oov"), [])) != len(a.seeds):
            print(f"  {arm}: incomplete ({len(E.get((arm, 'single', 'beam', 'none', 's1_oov'), []))} of {len(a.seeds)} seeds), skipped")
            continue
        res = [boot(E[arm, "single", m, "clean", "s1_oov"], E[arm, "single", "beam", "none", "s1_oov"], rng)
               for m in ("prompt", "ngram", "llm")]
        adj = holm([r[3] for r in res])
        for m, r, pa in zip(("prompt", "ngram", "llm"), res, adj):
            t = st.mean(map(wer, E[arm, "single", m, "clean", "s1_test"])) - st.mean(map(wer, E[arm, "single", "beam", "none", "s1_test"]))
            ok = pa < 0.05 and r[2] < 0 and t <= 2
            print(f"  {arm} {m}: diff {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] p={r[3]:.4f} Holm p={pa:.4f}; "
                  f"s1_test diff {t:+.2f} -> {'HELPS' if ok else 'no claim'}")

    print("\nSecondary (no claims): method - beam")
    for arm in ("E1", "E2"):
        for kind in ("single", "soup"):
            for m in ("prompt", "ngram", "llm"):
                for b in ("clean", "words"):
                    for s in ("s1_oov", "s2_oov"):
                        k = (arm, kind, m, b, s)
                        if k in E and not (kind == "single" and b == "clean" and s == "s1_oov"):
                            print(f"  {arm} {kind} {m} {b} {s}: diff %+.2f [%+.2f, %+.2f] p=%.4f"
                                  % boot(E[k], E[arm, kind, "beam", "none", s], rng))

    print("\nDescriptive: full grids on OOV, single seeds, mean WER (val picked one value per run)")
    for arm in ("E1", "E2"):
        runs = [r for r in systems[arm]["single"] if os.path.exists(f"{r}/bias/llm_selection.json")]
        for s in ("s1_oov", "s2_oov"):
            for b in ("clean", "words"):
                ng = [st.mean(wer(errs(REF[s], hyps(f"{r}/bias/ngram_{b}_{l:g}_{s}_hyps.tsv"))) for r in runs) for l in LAMS]
                lm = [st.mean(wer(errs(REF[s], hyps(f"{r}/bias/llm_{b}_{u:g}_{s}_hyps.tsv"))) for r in runs) for u in MUS]
                nl = [sum(loop(REF[s][k], hyps(f"{r}/bias/ngram_{b}_{l:g}_{s}_hyps.tsv")[k]) for r in runs for k in REF[s]) for l in LAMS]
                print(f"  {arm} {s} {b}: ngram " + " ".join(f"{l:g}:{v:.1f}" for l, v in zip(LAMS, ng))
                      + f" (loops {nl}) | llm " + " ".join(f"{u:g}:{v:.1f}" for u, v in zip(MUS, lm)))


if __name__ == "__main__":
    main()
