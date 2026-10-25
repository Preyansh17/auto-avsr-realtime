"""T3: strong conditioning controls, paired per-clip CE deltas vs matched on the 40 gate2_diag clips.

    python t3_conditioning.py <out_dir>

(a) different-speaker target audio (TORGO, closest duration), patient text + patient reference
(b) patient target, different-speaker reference (k = 1/3/5 TORGO clips stitched, their prompts as reference text)
(c) target text from a different tts_train sentence of the closest character length
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FT_DIR, L, TORGO_SPK, cached_codes, load_splits, torgo_clips  # noqa: E402

sys.path.insert(0, FT_DIR)
from gate2_diag import ce_matrix, stats  # noqa: E402

DEV = "cuda:0"
KEYS = ("all", "cb0", "early", "cb0_early", "eos")


def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / max(1, len(a | b))


@torch.no_grad()
def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sp = load_splits()
    clips, train = sp["clips"], sp["tts_train"]
    proc, model = L.load_models(DEV)
    pool = L.RefPool(clips, train)
    rng = random.Random(0)
    sel = sorted(train)[:: max(1, len(train) // 40)][:40]  # gate2_diag.py selection
    codes = {u: cached_codes(u) for u in sel}

    tg = torgo_clips(min_words=1)
    tg_sent = [r for r in tg if len(r[1].split()) >= 3 and r[2] <= 6.0]
    print(f"TORGO {TORGO_SPK}: {len(tg)} clips usable, {len(tg_sent)} with >= 3 words", flush=True)
    tcode_cache = {}

    def tcodes(path):
        if path not in tcode_cache:
            tcode_cache[path] = L.encode_wave(proc, L.load_wave(path))
        return tcode_cache[path]

    used_a = set()
    trng = random.Random(7)
    rows = []
    for i, u in enumerate(sel):
        k = [1, 3, 5][i % 3]
        rw, rt, _ = pool.sample(rng, u, k=k)  # same draw sequence as gate2_diag -> same matched references
        rc = L.encode_wave(proc, rw)
        text = clips[u]["transcript"]
        dur = codes[u].shape[1] / 25.0
        # (a) TORGO clip closest in duration, each used at most once
        cand = sorted((abs(d - dur), p) for p, _, d in tg if p not in used_a)
        pa = cand[0][1]
        used_a.add(pa)
        # (b) k TORGO sentence clips stitched (<= 45 s)
        picks = trng.sample(tg_sent, k)
        b_wave = L.stitch_waves([L.load_wave(p) for p, _, _ in picks])
        b_text = " ".join(t for _, t, _ in picks)
        b_codes = L.encode_wave(proc, b_wave)
        # (c) closest character length, different sentence, low word overlap
        n_u = clips[u]["norm"]
        cc = [(abs(len(clips[v]["norm"]) - len(n_u)), v) for v in train
              if clips[v]["norm"] != n_u and jaccard(clips[v]["norm"], n_u) < 0.5]
        pc = min(cc)[1]

        def run(rt_, rc_, text_, codes_):
            return stats(*ce_matrix(model, L.build_example(proc, rt_, rc_, text_, codes_, True), DEV))

        r = {"utt": u, "k": k, "dur": dur,
             "matched": run(rt, rc, text, codes[u]),
             "a_other_speaker_audio": run(rt, rc, text, tcodes(pa)),
             "b_other_speaker_ref": run(b_text, b_codes, text, codes[u]),
             "c_lenmatched_text": run(rt, rc, clips[pc]["transcript"], codes[u]),
             "a_path": pa, "a_dur": tcodes(pa).shape[1] / 25.0, "b_paths": [p for p, _, _ in picks], "b_text": b_text,
             "b_ref_sec": len(b_wave) / L.SR, "c_partner": pc, "c_text": clips[pc]["transcript"],
             "c_len_ratio": len(clips[pc]["norm"]) / max(1, len(n_u)), "c_jaccard": jaccard(clips[pc]["norm"], n_u)}
        rows.append(r)
        print(f"{i + 1}/{len(sel)} {u[:8]} matched={r['matched']['all']:.3f} a={r['a_other_speaker_audio']['all']:.3f} "
              f"b={r['b_other_speaker_ref']['all']:.3f} c={r['c_lenmatched_text']['all']:.3f} "
              f"(c len ratio {r['c_len_ratio']:.2f})", flush=True)

    summary, verdict = {}, {}
    for var in ("a_other_speaker_audio", "b_other_speaker_ref", "c_lenmatched_text"):
        for kk in KEYS:
            d = np.array([r[var][kk] - r["matched"][kk] for r in rows])
            d = d[~np.isnan(d)]
            m, sem = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
            summary[f"{var}:{kk}"] = {"mean_diff": m, "sem": sem, "z": m / sem if sem > 0 else float("inf"),
                                      "frac_worse": float((d > 0).mean()), "n": int(len(d))}
        s = summary[f"{var}:all"]
        verdict[var] = bool(s["mean_diff"] > 0 and s["mean_diff"] >= 5 * s["sem"] and s["frac_worse"] >= 0.90)
    verdict["T3"] = "PASS" if verdict["a_other_speaker_audio"] and verdict["c_lenmatched_text"] else "FAIL"
    verdict["reference_conditioning_used"] = verdict["b_other_speaker_ref"]
    base = {kk: float(np.nanmean([r["matched"][kk] for r in rows])) for kk in KEYS}
    cl = np.array([r["c_len_ratio"] for r in rows])
    print("\nmatched mean CE:", {kk: round(v, 3) for kk, v in base.items()})
    for kk, v in summary.items():
        print(f"  {kk:30s} {v['mean_diff']:+.3f} +- {v['sem']:.3f} (z {v['z']:.1f})  worse in {100 * v['frac_worse']:.0f}% (n={v['n']})")
    print(f"(c) char-length ratio: median {np.median(cl):.3f}, range {cl.min():.2f}-{cl.max():.2f}")
    print("verdict:", verdict, flush=True)
    json.dump({"torgo_speaker": TORGO_SPK, "matched_mean": base, "summary": summary, "verdict": verdict,
               "c_len_ratio": {"median": float(np.median(cl)), "min": float(cl.min()), "max": float(cl.max())},
               "rows": rows}, open(f"{out}/t3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
