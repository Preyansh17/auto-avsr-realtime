"""Collect and filter the re-voiced clips (design B).

    python filter_rv.py collect   # metadata -> data/rv_all.csv (label CSV), data/rv_info.json, data/real_train_speech_frac.json
    (judge: real-only Whisper large-v3 R seed 1 decodes real val + data/rv_all.csv; see judge_rv.sbatch)
    python filter_rv.py score     # data/rv_kept.csv, data/rv_filter_clips.csv, data/rv_filter_report.md

Rules (pre-registered in README.md before generation; a clip is kept only if it passes all):
  dur     duration / duration of its own source clip outside [0.5, 2.0], or > 15 s
  silent  energy-VAD speech fraction below the 5th percentile of the real ASR-train clips (calibrated floor,
          the TORGO lesson: a fixed 0.3 floor removed real speech)
  judge   word errors or insertions of the R seed-1 judge above the 95th percentile of the same judge on real ASR val
          (val is independent of this adapter: its TTS split drops every ASR val/test sentence). Every text is an
          ASR-train sentence, so the word-level rule measures audio, not vocabulary.
Label rows use dataset synth_higgs_rv (ASR root symlink -> /scratch/pa2753/gen_runs) and the SOURCE clip's token ids.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

R = "/scratch/pa2753/asr_aug/revoice"
DATA = f"{R}/data"
ROOT = "/scratch/pa2753/asr_aug/root"
GEN = "/scratch/pa2753/gen_runs"
DS = "synth_higgs_rv"
LABELS = "/scratch/pa2753/avsr_realtime/labels/splitv1/merged"
REPO = "/home/pa2753/auto-avsr-realtime"
DUR_LO, DUR_HI, MAX_SEC, SIL_PCTL, PCTL = 0.5, 2.0, 15.0, 5, 95
sys.path.insert(0, REPO)
sys.path.insert(0, "/scratch/pa2753/asr_aug/code")
from filter_synth import judge_scores, speech_fraction  # noqa: E402  (patient filter: same VAD + judge scoring)


def mono(path):
    w, sr = sf.read(path, dtype="float32")
    return (w.mean(axis=1) if w.ndim > 1 else w), sr


def cmd_collect(_a) -> None:
    link = f"{ROOT}/{DS}"
    if not os.path.islink(link):
        os.symlink(GEN, link)
    plan = json.load(open(f"{DATA}/plan.json"))
    src_of = {(u["variant"], u["prompts"][0]["prompt_id"]): u["src_label"] for u in plan["units"]}
    real = {f"{r[0]},{r[1]}": r for r in (l.split(",") for l in open(f"{LABELS}/train_spm1023.csv").read().splitlines() if l)}
    rows, info = [], []
    for m in sorted(glob.glob(f"{GEN}/FT_RV_clean_c*/synthetic/metadata/*.csv")):
        for r in csv.DictReader(open(m)):
            src = src_of[(r["variant"], r["prompt_id"])]
            ds, rel_src, _, tok = real[src]
            rel = os.path.relpath(r["audio_path"], GEN)
            assert "," not in rel, rel
            dur = float(r["duration_sec"])
            src_wav = os.path.splitext(f"{ROOT}/{ds}/{rel_src}")[0] + ".wav"
            rows.append(f"{DS},{rel},{max(1, round(dur * 25))},{tok}")
            info.append({"rel_path": rel, "src": src, "copy": int(r["variant"].rsplit("_c", 1)[1]), "duration": dur,
                         "src_duration": sf.info(src_wav).duration, "norm": r["norm_transcript"]})
    open(f"{DATA}/rv_all.csv", "w").write("\n".join(rows) + "\n")
    json.dump(info, open(f"{DATA}/rv_info.json", "w"), indent=0)
    fr = []
    for ds, rel, _, _ in real.values():
        w, sr = mono(os.path.splitext(f"{ROOT}/{ds}/{rel}")[0] + ".wav")
        fr.append(speech_fraction(w, sr))
    json.dump(fr, open(f"{DATA}/real_train_speech_frac.json", "w"))
    print(f"{len(rows)} re-voiced clips; copies {collections.Counter(i['copy'] for i in info)}; "
          f"real-train speech fraction p5 {np.percentile(fr, SIL_PCTL):.3f}")


def cmd_score(a) -> None:
    info = json.load(open(f"{DATA}/rv_info.json"))
    sil = float(np.percentile(json.load(open(f"{DATA}/real_train_speech_frac.json")), SIL_PCTL))
    real = judge_scores(a.real_val_tsv)
    syn = judge_scores(a.synth_tsv)
    thr = {k: float(np.percentile([v[k] for v in real.values()], PCTL)) for k in ("wer", "ins")}
    clips = []
    for i in info:
        path = f"{ROOT}/{DS}/{i['rel_path']}"
        w, sr = mono(path)
        ratio = i["duration"] / i["src_duration"]
        sfr = speech_fraction(w, sr)
        j = syn[path]
        why = []
        if not (DUR_LO <= ratio <= DUR_HI) or i["duration"] > MAX_SEC:
            why.append("dur")
        if sfr < sil:
            why.append("silent")
        if j["wer"] > thr["wer"] or j["ins"] > thr["ins"]:
            why.append("judge")
        clips.append({**i, "dur_ratio": round(ratio, 3), "speech_frac": round(sfr, 3), "judge_wer": round(j["wer"], 3),
                      "judge_ins": j["ins"], "judge_hyp": j["hyp"], "reject": "+".join(why), "kept": not why})
    with open(f"{DATA}/rv_filter_clips.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(clips[0]))
        wr.writeheader()
        wr.writerows(clips)
    rows = {l.split(",")[1]: l for l in open(f"{DATA}/rv_all.csv").read().splitlines() if l}
    open(f"{DATA}/rv_kept.csv", "w").write("\n".join(rows[c["rel_path"]] for c in clips if c["kept"]) + "\n")
    cnt = collections.Counter(r for c in clips for r in c["reject"].split("+") if r)
    kept_src = collections.Counter(c["src"] for c in clips if c["kept"])
    ratios = np.array([c["dur_ratio"] for c in clips])
    L = ["# Re-voicing filter report", "",
         f"Duration ratio to own source clip in [{DUR_LO}, {DUR_HI}], max {MAX_SEC} s; speech fraction >= {sil:.3f} "
         f"({SIL_PCTL}th pct of real train); judge thresholds ({PCTL}th pct of real val, n={len(real)}): "
         f"WER {thr['wer']:.3f}, insertions {thr['ins']:.1f}.", "",
         f"Generated {len(clips)}; kept {sum(c['kept'] for c in clips)}; rejected dur {cnt['dur']}, silent {cnt['silent']}, "
         f"judge {cnt['judge']}.",
         f"Source clips with >=1 kept copy: {len(kept_src)} / {len({c['src'] for c in clips})}; with 2: "
         f"{sum(v >= 2 for v in kept_src.values())}.",
         f"Duration ratio median {np.median(ratios):.3f} (IQR {np.percentile(ratios, 25):.3f}-{np.percentile(ratios, 75):.3f}).",
         f"Judge WER on kept clips: mean {np.mean([c['judge_wer'] for c in clips if c['kept']]):.3f}; real val mean "
         f"{np.mean([v['wer'] for v in real.values()]):.3f}."]
    open(f"{DATA}/rv_filter_report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    s = sub.add_parser("score")
    s.add_argument("--real-val-tsv", default=f"{DATA}/judge_real_val.tsv")
    s.add_argument("--synth-tsv", default=f"{DATA}/judge_rv.tsv")
    a = ap.parse_args()
    {"collect": cmd_collect, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
