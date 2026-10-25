"""Collect the Higgs synthetic clips into repo label CSVs and filter out bad generations.

    python filter_synth.py collect   # metadata -> data/synth_all.csv (+ combined root, real durations)
    (judge decodes: whisper_eval on real val and on synth_all.csv; see judge.sbatch)
    python filter_synth.py score     # per-clip table, thresholds, kept/rejected CSVs, report

Label CSV rows are the repo's "dataset_name,rel_path,frames,token_ids" (online_avsr/patient_dataset.read_label_csv),
all relative to one combined root (ROOT below) that symlinks the two split_v1 patient datasets plus the generation
folder, so real and synthetic rows can share one --root-dir. frames = round(duration * 25) like the real video rows.

Rejection rules (a clip is kept only if it passes all):
  dur     duration / (n_words * real-train median sec/word) outside [DUR_LO, DUR_HI], or > MAX_SEC (ramble / short)
  silent  energy-VAD speech fraction < MIN_SPEECH_FRAC
  judge   ASR judge (real-only Whisper large-v3, R seed 1) disagrees with the target text more than it does on
          95% of real val clips: pool A by WER or insertion count, pool B by CER (its new words are unseen by
          the judge, so word-level scores would mostly measure vocabulary, not audio quality).
Writes data/synth_kept_{A,B}.csv, data/synth_durkept_{A,B}.csv (duration+silence rules only, ablation),
data/filter_clips.csv, data/filter_report.md, and samples/<reason>/ symlinks for listening.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import statistics
import sys

import numpy as np
import soundfile as sf

BASE = "/scratch/pa2753/asr_aug"
DATA = f"{BASE}/data"
ROOT = f"{BASE}/root"
SPLIT_V1 = "/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1"
LABELS = "/scratch/pa2753/avsr_realtime/labels/splitv1/merged"
GEN = "/scratch/pa2753/gen_runs"
SYNTH_DS = "synth_higgs_b8"
REPO = "/home/pa2753/auto-avsr-realtime"
SPM = f"{REPO}/spm/spm_unigram_1023.model"
DUR_LO, DUR_HI, MAX_SEC = 0.5, 2.0, 15.0
MIN_SPEECH_FRAC = 0.3
PCTL = 95

sys.path.insert(0, REPO)


def sp_model():
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=SPM)


def read_rows(path: str) -> list[list[str]]:
    return [l.split(",") for l in open(path).read().splitlines() if l.strip()]


def speech_fraction(wav: np.ndarray, sr: int) -> float:
    """Fraction of 25 ms frames whose energy is within 35 dB of the clip's loudest frame and above an absolute floor."""
    n = int(0.025 * sr)
    if len(wav) < n:
        return 0.0
    frames = wav[: len(wav) // n * n].reshape(-1, n)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    return float(np.mean((db > db.max() - 35) & (db > -60)))


def cmd_collect(_args) -> None:
    os.makedirs(ROOT, exist_ok=True)
    for name, target in (("patient_retinaface_legal298", f"{SPLIT_V1}/patient_retinaface_legal298"),
                         ("patient_retinaface_legacy", f"{SPLIT_V1}/patient_retinaface_legacy"), (SYNTH_DS, GEN)):
        link = f"{ROOT}/{name}"
        if not os.path.islink(link):
            os.symlink(target, link)
    sp = sp_model()
    pool_of = {e["norm"]: e["pool"] for p in "AB" for e in json.load(open(f"{DATA}/pool_{p}.json"))}
    meta = sorted(glob.glob(f"{GEN}/FT_AUG_B8_*/synthetic/metadata/*.csv"))
    rows, info, seen = [], [], set()
    for m in meta:
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
            info.append({"rel_path": rel, "prompt_id": r["prompt_id"], "norm": r["norm_transcript"],
                         "pool": pool_of[r["norm_transcript"]], "duration": dur, "ref": r["variant"].split("_sents_")[0]})
    with open(f"{DATA}/synth_all.csv", "w") as f:
        f.writelines(",".join(r) + "\n" for r in rows)
    json.dump(info, open(f"{DATA}/synth_info.json", "w"), indent=0)
    # real-train speaking rate (sec per word) from the actual wavs
    rates = []
    for ds, rel, _, tok in read_rows(f"{LABELS}/train_spm1023.csv"):
        wav = os.path.splitext(f"{ROOT}/{ds}/{rel}")[0] + ".wav"
        text = sp.decode([int(t) for t in tok.split()])
        rates.append(sf.info(wav).duration / len(text.split()))
    json.dump({"sec_per_word_median": statistics.median(rates), "n_train": len(rates)},
              open(f"{DATA}/real_rate.json", "w"))
    print(f"{len(rows)} synthetic clips from {len(meta)} metadata files; pools {collections.Counter(i['pool'] for i in info)}; "
          f"real sec/word median {statistics.median(rates):.3f}")


def wer_parts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(edits, insertions, ref_len, hyp_len) of a word alignment."""
    r, h = ref.split(), hyp.split()
    d = [[(j, j) for j in range(len(h) + 1)]]  # (cost, insertions)
    for i in range(1, len(r) + 1):
        row = [(i, 0)]
        for j in range(1, len(h) + 1):
            sub = d[i - 1][j - 1][0] + (r[i - 1] != h[j - 1])
            cands = [(sub, d[i - 1][j - 1][1]), (d[i - 1][j][0] + 1, d[i - 1][j][1]), (row[j - 1][0] + 1, row[j - 1][1] + 1)]
            row.append(min(cands))
        d.append(row)
    return d[-1][-1][0], d[-1][-1][1], len(r), len(h)


def cer(ref: str, hyp: str) -> float:
    from asr_baselines.metrics import _levenshtein
    return _levenshtein(list(ref), list(hyp)) / max(1, len(ref))


def judge_scores(tsv: str) -> dict[str, dict]:
    from asr_baselines.metrics import normalize
    out = {}
    for r in csv.DictReader(open(tsv), delimiter="\t"):
        ref, hyp = normalize(r["ref"]), normalize(r["hyp"])
        e, ins, n, _ = wer_parts(ref, hyp)
        out[r["path"]] = {"wer": e / max(1, n), "ins": ins, "cer": cer(ref, hyp), "hyp": hyp}
    return out


def cmd_score(args) -> None:
    rate = json.load(open(f"{DATA}/real_rate.json"))["sec_per_word_median"]
    info = json.load(open(f"{DATA}/synth_info.json"))
    real = judge_scores(args.real_val_tsv)
    syn = judge_scores(args.synth_tsv)
    thr = {k: float(np.percentile([v[k] for v in real.values()], PCTL)) for k in ("wer", "ins", "cer")}
    real_pass = np.mean([v["wer"] <= thr["wer"] and v["ins"] <= thr["ins"] for v in real.values()])
    clips, reasons = [], collections.Counter()
    for i in info:
        path = f"{ROOT}/{SYNTH_DS}/{i['rel_path']}"
        j = syn[path]
        wav, sr = sf.read(path, dtype="float32")
        wav = wav.mean(axis=1) if wav.ndim > 1 else wav
        n_words = len(i["norm"].split())
        ratio = i["duration"] / (n_words * rate)
        sf_ = speech_fraction(wav, sr)
        why = []
        if not (DUR_LO <= ratio <= DUR_HI) or i["duration"] > MAX_SEC:
            why.append("dur")
        if sf_ < MIN_SPEECH_FRAC:
            why.append("silent")
        bad_judge = (j["wer"] > thr["wer"] or j["ins"] > thr["ins"]) if i["pool"] == "A" else j["cer"] > thr["cer"]
        if bad_judge:
            why.append("judge")
        reasons.update(why or ["kept"])
        clips.append({**i, "dur_ratio": round(ratio, 3), "speech_frac": round(sf_, 3), "judge_wer": round(j["wer"], 3),
                      "judge_ins": j["ins"], "judge_cer": round(j["cer"], 3), "judge_hyp": j["hyp"],
                      "reject": "+".join(why), "kept": not why, "dur_kept": not ({"dur", "silent"} & set(why))})
    with open(f"{DATA}/filter_clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clips[0]))
        w.writeheader()
        w.writerows(clips)
    rows = {r[1]: r for r in read_rows(f"{DATA}/synth_all.csv")}
    for flag, stem in (("kept", "synth_kept"), ("dur_kept", "synth_durkept")):
        for p in "AB":
            with open(f"{DATA}/{stem}_{p}.csv", "w") as f:
                f.writelines(",".join(rows[c["rel_path"]]) + "\n" for c in clips if c["pool"] == p and c[flag])
    # listening samples
    rng = random.Random(0)
    for reason in ("kept", "dur", "silent", "judge"):
        d = f"{BASE}/samples/{reason}"
        os.makedirs(d, exist_ok=True)
        pick = [c for c in clips if (c["kept"] if reason == "kept" else reason in c["reject"].split("+"))]
        for c in rng.sample(pick, min(10, len(pick))):
            dst = f"{d}/{c['pool']}_{c['prompt_id']}.wav"
            if not os.path.exists(dst):
                os.symlink(f"{ROOT}/{SYNTH_DS}/{c['rel_path']}", dst)
    L = ["# Synthetic filter report", "",
         f"Real-train median {rate:.3f} s/word; duration window [{DUR_LO}, {DUR_HI}] x expected, max {MAX_SEC} s; "
         f"speech fraction >= {MIN_SPEECH_FRAC}.",
         f"Judge thresholds ({PCTL}th pct of real val, n={len(real)}): WER {thr['wer']:.3f}, insertions {thr['ins']:.1f}, "
         f"CER {thr['cer']:.3f}; real val pass rate under the pool-A rule {real_pass:.1%}.", "",
         "| pool | clips | kept | kept % | dur | silent | judge | dur+silent-only kept |", "|---|---|---|---|---|---|---|---|"]
    for p in "AB":
        cs = [c for c in clips if c["pool"] == p]
        cnt = collections.Counter(r for c in cs for r in c["reject"].split("+") if r)
        k = sum(c["kept"] for c in cs)
        L.append(f"| {p} | {len(cs)} | {k} | {k / len(cs):.1%} | {cnt['dur']} | {cnt['silent']} | {cnt['judge']} | "
                 f"{sum(c['dur_kept'] for c in cs)} |")
    L += ["", "Per reference prompt (kept %):", ""]
    for ref in sorted({c["ref"] for c in clips}):
        cs = [c for c in clips if c["ref"] == ref]
        L.append(f"- {ref}: {sum(c['kept'] for c in cs) / len(cs):.1%} of {len(cs)}")
    open(f"{DATA}/filter_report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    s = sub.add_parser("score")
    s.add_argument("--real-val-tsv", default=f"{DATA}/judge_real_val.tsv")
    s.add_argument("--synth-tsv", default=f"{DATA}/judge_synth.tsv")
    args = ap.parse_args()
    {"collect": cmd_collect, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
