"""POST-HOC (not pre-registered) paired per-sentence analysis of the Phase 4 final table.

The pre-registered C1b compares the seed range of the 40-sentence mean PER with a gap, so it is dominated by seed noise
on n=40. Here the unit is the SENTENCE: per-sentence PER (mean over the 3 seeds for FT and stitch5) minus the real
clips' per-sentence PER (mean over repeats), bootstrap over sentences. Also splits near-duplicate sentences (word-level
edit distance <= 1 to a tts_train sentence) from far ones, as the README promised.
    python post_hoc_paired.py <dir with final/pb_out/chunk_*/items_scored.csv and splits.json>
"""
import csv, glob, json, re, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "."
sp = json.load(open(f"{D}/splits.json"))
norm = lambda t: re.sub(r"[^a-z' ]", "", t.lower()).split()
train = [norm(sp["clips"][u]["transcript"]) for u in sp["tts_train"]]
test_sents = {" ".join(norm(sp["clips"][u]["transcript"])) for u in sp["tts_test"]}

def wed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[len(b)]

near = {s: min(wed(s.split(), t) for t in train) <= 1 for s in test_sents}
rows = []
for f in sorted(glob.glob(f"{D}/final/pb_out/chunk_*/items_scored.csv")):
    rows += list(csv.DictReader(open(f)))
lab = lambda r: "real" if r["source"] == "real" else r["severity"]  # run label of synthetic rows is in `severity`


def vals(src, col):
    out = {}
    for r in rows:
        if lab(r) == src and r[col] not in ("", None):
            out.setdefault(" ".join(norm(r["transcript"])), []).append(float(r[col]))
    return {k: float(np.mean(v)) for k, v in out.items()}
srcs = sorted({lab(r) for r in rows})
def group(prefix):
    return [s for s in srcs if s.startswith(prefix)]
GROUPS = {"A_r16 e12": group("FT_A_r16_best"), "B_r64 e8": group("FT_B_r64_best"), "C_r16_lr5e-5 e12": group("FT_C_r16_lr5e-5_best"),
          "stitch5 (3 seeds)": group("R1_stitch5"), "stitch10": ["R1b_stitch10_s1"], "lowpitch5": ["LP_low5_s1"], "median1": ["R2_median1_s1"]}
rng = np.random.default_rng(0)

def per_sentence(col, g):
    runs = [vals(s, col) for s in GROUPS[g]]
    real = vals("real", col)
    keys = sorted(set(real) & set.intersection(*[set(r) for r in runs]))
    return keys, np.array([np.mean([r[k] for r in runs]) - real[k] for k in keys])

def boot(x, f, n=4000):
    idx = rng.integers(0, len(x), (n, len(x)))
    v = np.array([f(x[i]) for i in idx])
    return np.percentile(v, [2.5, 97.5])

out = []
for col in ("per", "artp"):
    out.append(f"\n## {col.upper()}: per-sentence (synth - real), n sentences, bootstrap 95% CI over sentences\n")
    out.append("| condition | n | bias (mean d) | |bias| | MAE (mean abs d) | MAE 95% CI | near (n) bias / MAE | far (n) bias / MAE |")
    out.append("|---|---|---|---|---|---|---|---|")
    base_keys, base_d = per_sentence(col, "stitch5 (3 seeds)")
    for g in GROUPS:
        keys, d = per_sentence(col, g)
        nm = np.array([near.get(k, False) for k in keys])
        lo, hi = boot(np.abs(d), np.mean)
        cell = lambda m: f"({int(m.sum())}) {d[m].mean():+.3f} / {np.abs(d[m]).mean():.3f}" if m.sum() else "-"
        out.append(f"| {g} | {len(d)} | {d.mean():+.3f} | {abs(d.mean()):.3f} | {np.abs(d).mean():.3f} | [{lo:.3f}, {hi:.3f}] | {cell(nm)} | {cell(~nm)} |")
    out.append(f"\nPaired vs zero-shot stitch5 on the same sentences ({col.upper()}), FT minus stitch5; negative = FT closer to real:\n")
    out.append("| FT run | d|bias| (|mean d_FT| - |mean d_s5|) [95% CI] | d MAE (mean|d_FT| - mean|d_s5|) [95% CI] | frac sentences closer to real |")
    out.append("|---|---|---|---|")
    for g in ("A_r16 e12", "B_r64 e8", "C_r16_lr5e-5 e12"):
        keys, d = per_sentence(col, g)
        kk = sorted(set(keys) & set(base_keys))
        a = np.array([d[keys.index(k)] for k in kk]); b = np.array([base_d[base_keys.index(k)] for k in kk])
        pair = np.stack([a, b], 1)
        fb = lambda p: abs(p[:, 0].mean()) - abs(p[:, 1].mean())
        fm = lambda p: np.abs(p[:, 0]).mean() - np.abs(p[:, 1]).mean()
        bl = boot(pair, fb); ml = boot(pair, fm)
        out.append(f"| {g} | {fb(pair):+.3f} [{bl[0]:+.3f}, {bl[1]:+.3f}] | {fm(pair):+.3f} [{ml[0]:+.3f}, {ml[1]:+.3f}] | {(np.abs(a) < np.abs(b)).mean():.0%} |")
out.insert(0, f"Test sentences: {len(test_sents)}; near-duplicate of a tts_train sentence (word edit distance <= 1): {sum(near.values())}; far: {len(near) - sum(near.values())}.")
print("\n".join(out))
