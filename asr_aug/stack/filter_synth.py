"""Collect and filter the item-3 stack synthetic clips (per-patient Higgs adapters, s1 = semg1 Oct 1, s2 = semg2).

    python filter_synth.py collect   # metadata -> data/synth_all_<p>.csv, synth_info.json, real_rate.json, root links
    (judge: judge.sbatch decodes data/synth_all_<p>.csv with that patient's real-only finetuned large-v3)
    python filter_synth.py score     # per-clip table, data/synth_kept_<p>_{A,B}.csv, filter_report.md

Rules are the H filter (asr_aug/filter_synth.py) with per-patient pieces; a clip is kept only if it passes all:
  dur     duration / (n_words * that patient's real-train median sec/word) outside [0.5, 2.0], or > 15 s
  silent  energy-VAD speech fraction < 0.3
  judge   pool A: WER > 0.576 or insertions > 1; pool B: CER > 0.521. These are the H thresholds, frozen, NOT re-derived:
          the judges here (s1: oct1_N seed 1, s2: multi_patient s2 lang_task seed 1) reach 1-3% val WER, so a
          95th-percentile-of-real-val threshold would be ~0 and reject on vocabulary rather than audio quality.
Label rows use dataset synth_higgs_b8 (= gen_runs), like the ONS pool, so old and new synthetic rows share one root.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics
import sys

import numpy as np
import soundfile as sf

S = "/scratch/pa2753/stack"
DATA = f"{S}/data"
ROOT = f"{S}/root"
GEN = "/scratch/pa2753/gen_runs"
SYNTH_DS = "synth_higgs_b8"
REPO = "/home/pa2753/auto-avsr-realtime"
SPM = f"{REPO}/spm/spm_unigram_1023.model"
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
LINKS = {"patient_retinaface_legacy": "/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1/patient_retinaface_legacy",
         "patient_retinaface_legal298": "/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1/patient_retinaface_legal298",
         "semg1_oct1_retinaface": "/scratch/th3482/Dysarthric_ASR/data/patients/splits/w50s350_v1/semg1_oct1_retinaface",
         "semg2_oct1_retinaface": "/scratch/th3482/Dysarthric_ASR/data/patients/splits/w50s350_v1/semg2_oct1_retinaface",
         SYNTH_DS: GEN}
PATIENTS = ["s1", "s2"]
DUR_LO, DUR_HI, MAX_SEC, MIN_SPEECH_FRAC = 0.5, 2.0, 15.0, 0.3
THR = {"wer": 0.576, "ins": 1.0, "cer": 0.521}  # H filter_report.md, frozen
sys.path.insert(0, REPO)


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


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
    os.makedirs(ROOT, exist_ok=True)
    for name, target in LINKS.items():
        if not os.path.islink(f"{ROOT}/{name}"):
            os.symlink(target, f"{ROOT}/{name}")
    pool_of = {e["norm"]: e["pool"] for e in json.load(open(f"{DATA}/text.json"))}
    info, rate = [], {}
    for p in PATIENTS:
        rows, seen = [], set()
        for m in sorted(glob.glob(f"{GEN}/FT_ST_{p}_*/synthetic/metadata/*.csv")):
            for r in csv.DictReader(open(m)):
                if r["norm_transcript"] not in pool_of or r["prompt_id"] in seen:
                    continue
                seen.add(r["prompt_id"])
                rel = os.path.relpath(r["audio_path"], GEN)
                assert "," not in rel, rel
                ids = sp.encode(r["norm_transcript"])
                assert sp.decode(ids) == r["norm_transcript"], r["norm_transcript"]
                dur = float(r["duration_sec"])
                rows.append([SYNTH_DS, rel, str(max(1, round(dur * 25))), " ".join(map(str, ids))])
                info.append({"patient": p, "rel_path": rel, "prompt_id": r["prompt_id"], "norm": r["norm_transcript"],
                             "pool": pool_of[r["norm_transcript"]], "duration": dur})
        with open(f"{DATA}/synth_all_{p}.csv", "w") as f:
            f.writelines(",".join(r) + "\n" for r in rows)
        rates = [sf.info(x["audio_path"]).duration / len(norm(x["transcript"]).split())
                 for x in map(json.loads, open(f"{ROLES}/{p}_train.jsonl"))]
        rate[p] = statistics.median(rates)
        print(f"{p}: {len(rows)} clips, pools {collections.Counter(i['pool'] for i in info if i['patient'] == p)}, "
              f"real sec/word median {rate[p]:.3f}")
    json.dump(info, open(f"{DATA}/synth_info.json", "w"), indent=0)
    json.dump(rate, open(f"{DATA}/real_rate.json", "w"))


def wer_parts(ref: str, hyp: str) -> tuple[int, int]:
    """(edits, insertions) of a word alignment."""
    r, h = ref.split(), hyp.split()
    d = [[(j, j) for j in range(len(h) + 1)]]
    for i in range(1, len(r) + 1):
        row = [(i, 0)]
        for j in range(1, len(h) + 1):
            sub = d[i - 1][j - 1][0] + (r[i - 1] != h[j - 1])
            row.append(min([(sub, d[i - 1][j - 1][1]), (d[i - 1][j][0] + 1, d[i - 1][j][1]),
                            (row[j - 1][0] + 1, row[j - 1][1] + 1)]))
        d.append(row)
    return d[-1][-1]


def cmd_score(_a) -> None:
    from asr_baselines.metrics import _levenshtein, normalize
    rate = json.load(open(f"{DATA}/real_rate.json"))
    info = json.load(open(f"{DATA}/synth_info.json"))
    judge = {}
    for p in PATIENTS:
        for r in csv.DictReader(open(f"{DATA}/judge_synth_{p}.tsv"), delimiter="\t"):
            judge[r["path"]] = (normalize(r["ref"]), normalize(r["hyp"]))
    clips, reasons = [], collections.Counter()
    for i in info:
        path = f"{ROOT}/{SYNTH_DS}/{i['rel_path']}"
        ref, hyp = judge[path]
        e, ins = wer_parts(ref, hyp)
        wer, cer = e / max(1, len(ref.split())), _levenshtein(list(ref), list(hyp)) / max(1, len(ref))
        wav, sr = sf.read(path, dtype="float32")
        wav = wav.mean(axis=1) if wav.ndim > 1 else wav
        ratio = i["duration"] / (len(i["norm"].split()) * rate[i["patient"]])
        sfr = speech_fraction(wav, sr)
        why = []
        if not (DUR_LO <= ratio <= DUR_HI) or i["duration"] > MAX_SEC:
            why.append("dur")
        if sfr < MIN_SPEECH_FRAC:
            why.append("silent")
        if (wer > THR["wer"] or ins > THR["ins"]) if i["pool"] == "A" else cer > THR["cer"]:
            why.append("judge")
        reasons.update(f"{i['patient']}_{i['pool']}_{w}" for w in (why or ["kept"]))
        clips.append({**i, "dur_ratio": round(ratio, 3), "speech_frac": round(sfr, 3), "judge_wer": round(wer, 3),
                      "judge_ins": ins, "judge_cer": round(cer, 3), "judge_hyp": hyp, "reject": "+".join(why), "kept": not why})
    with open(f"{DATA}/filter_clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clips[0]))
        w.writeheader()
        w.writerows(clips)
    rep = ["# Stack synthetic filter report", "",
           f"Thresholds (H, frozen): {THR}; real sec/word median {rate}.", "",
           "| patient | pool | clips | kept | kept % | dur | silent | judge |", "|---|---|---|---|---|---|---|---|"]
    for p in PATIENTS:
        rows = {r.split(",")[1]: r for r in open(f"{DATA}/synth_all_{p}.csv").read().splitlines() if r.strip()}
        for pool in "AB":
            cs = [c for c in clips if c["patient"] == p and c["pool"] == pool]
            kept = [c for c in cs if c["kept"]]
            with open(f"{DATA}/synth_kept_{p}_{pool}.csv", "w") as f:
                f.writelines(rows[c["rel_path"]] + "\n" for c in kept)
            cnt = lambda k: sum(k in c["reject"].split("+") for c in cs)
            rep.append(f"| {p} | {pool} | {len(cs)} | {len(kept)} | {100 * len(kept) / max(1, len(cs)):.1f}% | "
                       f"{cnt('dur')} | {cnt('silent')} | {cnt('judge')} |")
    open(f"{DATA}/filter_report.md", "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "score"])
    a = ap.parse_args()
    {"collect": cmd_collect, "score": cmd_score}[a.cmd](a)
