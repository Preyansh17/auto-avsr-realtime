"""Collect and filter the fold-text synthetic clips (2-fold oov experiment, PLAN.md section H2).

    python filter_fold.py   # -> data/synth_fold{A,B}.csv (duration+silence rules only), data/fold_report.txt
Same label-row format and duration/silence rules as filter_synth.py; no judge (it knows only 76 words).
"""
import collections, csv, glob, json, os, sys
import soundfile as sf
sys.path.insert(0, "/scratch/pa2753/asr_aug/code")
import filter_synth as fs

rate = json.load(open(f"{fs.DATA}/real_rate.json"))["sec_per_word_median"]
sp = fs.sp_model()
rep = []
for F in "AB":
    sents = {e["norm"] for e in json.load(open(f"{fs.DATA}/sents_fold{F}.json"))}
    rows, cnt, seen = [], collections.Counter(), set()
    for m in sorted(glob.glob(f"{fs.GEN}/FT_AUG_B8fold{F}_*/synthetic/metadata/*.csv")):
        for r in csv.DictReader(open(m)):
            if r["norm_transcript"] not in sents or r["audio_path"] in seen:
                continue
            seen.add(r["audio_path"])
            dur = float(r["duration_sec"]); n = len(r["norm_transcript"].split())
            wav, sr = sf.read(r["audio_path"], dtype="float32")
            wav = wav.mean(axis=1) if wav.ndim > 1 else wav
            why = []
            if not (fs.DUR_LO <= dur / (n * rate) <= fs.DUR_HI) or dur > fs.MAX_SEC: why.append("dur")
            if fs.speech_fraction(wav, sr) < fs.MIN_SPEECH_FRAC: why.append("silent")
            cnt.update(why or ["kept"])
            if not why:
                ids = sp.encode(r["norm_transcript"]); assert sp.decode(ids) == r["norm_transcript"]
                rows.append([fs.SYNTH_DS, os.path.relpath(r["audio_path"], fs.GEN), str(max(1, round(dur * 25))), " ".join(map(str, ids))])
    open(f"{fs.DATA}/synth_fold{F}.csv", "w").writelines(",".join(r) + "\n" for r in rows)
    rep.append(f"fold {F}: {len(sents)} sentences, {len(seen)} clips, {dict(cnt)}")
open(f"{fs.DATA}/fold_report.txt", "w").write("\n".join(rep) + "\n"); print("\n".join(rep))
