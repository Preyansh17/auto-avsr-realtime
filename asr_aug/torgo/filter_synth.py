"""Collect and filter the TORGO synthetic clips (per-speaker Higgs adapters).

    python filter_synth.py collect   # metadata -> data/synth_all.csv, synth_info.json, real_rate.json, judge input CSVs
    (judge: zero-shot whisper-large-v3 decodes data/real_train_spm1023.csv and data/synth_all.csv; see judge.sbatch)
    python filter_synth.py score     # thresholds, per-clip table, data/synth_kept.csv, synth_durkept.csv, filter_report.md
    python filter_synth.py score --silence calibrated --tag _silcal   # recalibrated silence rule (2026-10-16)

Rules (a clip is kept only if it passes all):
  dur     duration / (speaker x type median real-train duration per word, times n_words) outside [0.5, 2.0], or > 15 s
  silent  energy-VAD speech fraction below a floor. "fixed" (the M0.5 runs of 2026-10-16): 0.3. "calibrated": the 5th
          percentile of that speaker's REAL TRAIN clips of the same type. The fixed floor rejected 31-37% of real
          single-word clips of M02/M03/M04 (short words in long recordings), so it was mostly a speaker/word filter.
  judge   zero-shot Whisper large-v3 CER above the 95th percentile of that same speaker's REAL TRAIN clips of the same
          type (word / sentence). The judge is not trained on any TORGO or synthetic data, so (unlike the patient study)
          it does not filter by a narrow vocabulary, and calibration never touches ASR val/test.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys

import numpy as np
import soundfile as sf

T = "/scratch/pa2753/torgo_aug"
DATA = f"{T}/data"
ROOT = f"{T}/root"
GEN = "/scratch/pa2753/gen_runs"
SYNTH_DS = "synth_higgs_tg"
REPO = "/home/pa2753/auto-avsr-realtime"
SPM = f"{REPO}/spm/spm_unigram_1023.model"
DUR_LO, DUR_HI, MAX_SEC, MIN_SPEECH_FRAC, PCTL, SIL_PCTL = 0.5, 2.0, 15.0, 0.3, 95, 5
sys.path.insert(0, REPO)


def speech_fraction(wav: np.ndarray, sr: int) -> float:
    n = int(0.025 * sr)
    if len(wav) < n:
        return 0.0
    frames = wav[: len(wav) // n * n].reshape(-1, n)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    return float(np.mean((db > db.max() - 35) & (db > -60)))


def cmd_collect(_a) -> None:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=SPM)
    rows, info = [], []
    for spk_file in sorted(glob.glob(f"{DATA}/text_[FM]0[0-9].json")):
        spk = spk_file.split("_")[-1][:-5]
        want = {e["norm"]: e["type"] for e in json.load(open(spk_file))}
        for m in sorted(glob.glob(f"{GEN}/FT_TG_{spk}_*/synthetic/metadata/*.csv")):
            for r in csv.DictReader(open(m)):
                if r["norm_transcript"] not in want:
                    continue
                rel = os.path.relpath(r["audio_path"], GEN)
                ids = sp.encode(r["norm_transcript"])
                assert sp.decode(ids) == r["norm_transcript"]
                dur = float(r["duration_sec"])
                rows.append(f"{SYNTH_DS},{rel},{max(1, round(dur * 25))},{' '.join(map(str, ids))}")
                info.append({"rel_path": rel, "speaker": spk, "norm": r["norm_transcript"],
                             "type": want[r["norm_transcript"]], "duration": dur})
    open(f"{DATA}/synth_all.csv", "w").write("\n".join(rows) + "\n")
    json.dump(info, open(f"{DATA}/synth_info.json", "w"), indent=0)
    items = [i for i in csv.DictReader(open(f"{DATA}/items.csv")) if i["split"] == "train"]
    rate = collections.defaultdict(list)
    for i in items:
        rate[f"{i['speaker']}|{i['type']}"].append(float(i["duration"]) / len(i["text"].split()))
    json.dump({k: statistics.median(v) for k, v in rate.items()}, open(f"{DATA}/real_rate.json", "w"), indent=1)
    print(f"{len(rows)} synthetic clips; per speaker {collections.Counter(i['speaker'] for i in info)}")


def scores(tsv: str) -> dict[str, dict]:
    from asr_baselines.metrics import _levenshtein, normalize
    out = {}
    for r in csv.DictReader(open(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        ref, hyp = normalize(r["ref"]), normalize(r["hyp"])
        out[r["path"]] = {"cer": _levenshtein(list(ref), list(hyp)) / max(1, len(ref)), "hyp": hyp}
    return out


def real_speech_frac() -> dict[str, list[float]]:
    """Speech fraction of every real train clip, by speaker|type (cached in data/real_speech_frac.json)."""
    path = f"{DATA}/real_speech_frac.json"
    if not os.path.exists(path):
        by = collections.defaultdict(list)
        for i in csv.DictReader(open(f"{DATA}/items.csv")):
            if i["split"] == "train":
                w, sr = sf.read(i["wav"], dtype="float32")
                by[f"{i['speaker']}|{i['type']}"].append(speech_fraction(w.mean(axis=1) if w.ndim > 1 else w, sr))
        json.dump(by, open(path, "w"))
    return json.load(open(path))


def cmd_score(a) -> None:
    rate = json.load(open(f"{DATA}/real_rate.json"))
    sil = {k: (float(np.percentile(v, SIL_PCTL)) if a.silence == "calibrated" else MIN_SPEECH_FRAC)
           for k, v in real_speech_frac().items()}
    info = json.load(open(f"{DATA}/synth_info.json"))
    items = {os.path.join(ROOT, "torgo_raw", os.path.relpath(i["wav"], "/scratch/th3482/TORGO/raw")): i
             for i in csv.DictReader(open(f"{DATA}/items.csv")) if i["split"] == "train"}
    real = scores(a.real_tsv)
    real_by = collections.defaultdict(list)
    for p, s in real.items():
        i = items[p]
        real_by[f"{i['speaker']}|{i['type']}"].append(s["cer"])
    thr = {k: float(np.percentile(v, PCTL)) for k, v in real_by.items()}
    syn = scores(a.synth_tsv)
    clips = []
    for i in info:
        path = f"{ROOT}/{SYNTH_DS}/{i['rel_path']}"
        wav, sr = sf.read(path, dtype="float32")
        wav = wav.mean(axis=1) if wav.ndim > 1 else wav
        key = f"{i['speaker']}|{i['type']}"
        ratio = i["duration"] / (len(i["norm"].split()) * rate[key])
        sfr = speech_fraction(wav, sr)
        j = syn[path]
        why = []
        if not (DUR_LO <= ratio <= DUR_HI) or i["duration"] > MAX_SEC:
            why.append("dur")
        if sfr < sil[key]:
            why.append("silent")
        if j["cer"] > thr[key]:
            why.append("judge")
        clips.append({**i, "dur_ratio": round(ratio, 3), "speech_frac": round(sfr, 3), "judge_cer": round(j["cer"], 3),
                      "judge_hyp": j["hyp"], "reject": "+".join(why), "kept": not why,
                      "dur_kept": not ({"dur", "silent"} & set(why))})
    with open(f"{DATA}/filter_clips{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clips[0]))
        w.writeheader()
        w.writerows(clips)
    rows = {l.split(",")[1]: l for l in open(f"{DATA}/synth_all.csv").read().splitlines() if l}
    for flag, stem in (("kept", "synth_kept"), ("dur_kept", "synth_durkept")):
        open(f"{DATA}/{stem}{a.tag}.csv", "w").write("\n".join(rows[c["rel_path"]] for c in clips if c[flag]) + "\n")
    L = ["# TORGO synthetic filter report", "",
         f"Judge = zero-shot whisper-large-v3; threshold = {PCTL}th pct CER of the speaker's real train clips per type.",
         f"Silence floor: {a.silence}" + (f" ({SIL_PCTL}th pct of real train speech fraction)" if a.silence == "calibrated"
                                          else f" ({MIN_SPEECH_FRAC})"), "",
         "| speaker | type | real-train CER thr | silence floor | generated | kept | dur | silent | judge |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(thr):
        spk, typ = key.split("|")
        cs = [c for c in clips if c["speaker"] == spk and c["type"] == typ]
        cnt = collections.Counter(r for c in cs for r in c["reject"].split("+") if r)
        L.append(f"| {spk} | {typ} | {thr[key]:.3f} | {sil[key]:.3f} | {len(cs)} | {sum(c['kept'] for c in cs)} | {cnt['dur']} | "
                 f"{cnt['silent']} | {cnt['judge']} |")
    open(f"{DATA}/filter_report{a.tag}.md", "w").write("\n".join(L) + "\n")
    json.dump({"judge": thr, "silence": sil}, open(f"{DATA}/thresholds{a.tag}.json", "w"), indent=1)
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    s = sub.add_parser("score")
    s.add_argument("--real-tsv", default=f"{DATA}/judge_real_train.tsv")
    s.add_argument("--synth-tsv", default=f"{DATA}/judge_synth.tsv")
    s.add_argument("--silence", choices=["fixed", "calibrated"], default="fixed")
    s.add_argument("--tag", default="", help="suffix for every output file, so a variant never overwrites the M0.5 inputs")
    a = ap.parse_args()
    {"collect": cmd_collect, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
