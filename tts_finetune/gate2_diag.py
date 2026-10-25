"""Sharper version of gate 2's matched-vs-shuffled control.

Teacher-forced CE is dominated by the audio prefix (same speaker, so shuffled audio from another clip is nearly
as predictable as the real one), so the all-token mean is a weak detector of text/reference conditioning. This
measures where conditioning must show up, over 40 clips with paired differences (variant - matched):
  all non-EOS tokens | codebook 0 only | first 20 target frames | codebook 0, first 20 frames | EOS tokens
Variants: matched; text from the clip of most different length (shortest<->longest pairing); audio from that clip.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import higgs_ft_lib as L  # noqa: E402

DATA = os.environ.get("HIGGS_FT_DATA", "/scratch/pa2753/higgs_ft/data")
EARLY = 20


def ce_matrix(model, ex, dev):
    logits = L.forward_logits(model, ex, dev)
    tgt = ex["labels_pos"].to(dev)[1:]
    pred = logits[:-1].float()
    ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), ignore_index=L.IGNORE,
                         reduction="none").reshape(tgt.shape)
    return ce, tgt


def stats(ce, tgt):
    valid = tgt != L.IGNORE
    eos = tgt == L.EOS_ID
    non = valid & ~eos
    rows = valid.any(1).nonzero().squeeze(1)[:EARLY]
    early = torch.zeros_like(valid)
    early[rows] = True
    cb0 = torch.zeros_like(valid)
    cb0[:, 0] = True

    def m(mask):
        return float(ce[mask].mean()) if int(mask.sum()) else float("nan")

    return {"all": m(non), "cb0": m(non & cb0), "early": m(non & early), "cb0_early": m(non & cb0 & early),
            "eos": m(eos)}


@torch.no_grad()
def main():
    dev = "cuda:0"
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sp = json.load(open(f"{DATA}/splits.json"))
    clips, train = sp["clips"], sp["tts_train"]
    proc, model = L.load_models(dev)
    pool = L.RefPool(clips, train)
    rng = random.Random(0)
    sel = sorted(train)[:: max(1, len(train) // 40)][:40]
    by_dur = sorted(sel, key=lambda u: clips[u]["duration_sec"])
    partner = {u: by_dur[-1 - i] for i, u in enumerate(by_dur)}
    codes = {u: torch.load(f"{DATA}/codes/{u}.pt").long() for u in sel}
    rows = []
    for i, u in enumerate(sel):
        p = partner[u]
        rw, rt, _ = pool.sample(rng, u, k=[1, 3, 5][i % 3])
        rc = L.encode_wave(proc, rw)

        def ex(text_u, audio_u):
            return L.build_example(proc, rt, rc, clips[text_u]["transcript"], codes[audio_u], True)

        r = {"utt": u, "partner": p,
             "matched": stats(*ce_matrix(model, ex(u, u), dev)),
             "text_partner": stats(*ce_matrix(model, ex(p, u), dev)),
             "audio_partner": stats(*ce_matrix(model, ex(u, p), dev))}
        rows.append(r)
        print(f"{i + 1}/{len(sel)} {u[:8]} matched all={r['matched']['all']:.3f} cb0_early={r['matched']['cb0_early']:.3f}", flush=True)

    summary = {}
    for var in ("text_partner", "audio_partner"):
        for k in ("all", "cb0", "early", "cb0_early", "eos"):
            d = np.array([r[var][k] - r["matched"][k] for r in rows if not math.isnan(r[var][k] - r["matched"][k])])
            summary[f"{var}:{k}"] = {"mean_diff": float(d.mean()), "sem": float(d.std(ddof=1) / math.sqrt(len(d))),
                                     "frac_worse": float((d > 0).mean()), "n": int(len(d))}
    base = {k: float(np.nanmean([r["matched"][k] for r in rows])) for k in ("all", "cb0", "early", "cb0_early", "eos")}
    print("\nmatched mean CE:", {k: round(v, 3) for k, v in base.items()})
    print("paired diff (variant - matched); positive = conditioning matters:")
    for k, v in summary.items():
        print(f"  {k:22s} {v['mean_diff']:+.3f} +- {v['sem']:.3f}  worse in {100 * v['frac_worse']:.0f}% of clips (n={v['n']})")
    json.dump({"matched_mean": base, "summary": summary, "rows": rows}, open(f"{out}/gate2_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
