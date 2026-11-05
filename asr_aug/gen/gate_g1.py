"""G1 generator gate (asr_aug/gen/README.md, Step A): filter the G1 clips exactly like their Higgs counterparts, then
compare kept G1 vs kept Higgs on the step-0 metrics.

    python gate_g1.py collect       # CPU: label CSVs per voice (data/g1/synth_all_<v>.csv) + per-clip info
    (gate_g1.sbatch: judge decodes per voice, the same judge and prefix the Higgs filter used)
    python gate_g1.py score         # CPU: H filter rules + frozen thresholds -> data/g1/filter_clips{,_old}.csv, kept CSVs
    python gate_g1.py diag --batch 16   # GPU: asr_aug/stack/diag_synth.py run on real s1/s2 + G1 s1/s2 clips
    python gate_g1.py report        # CPU: data/g1/gate_report.md (rule below)
    GEN=g1d python gate_g1.py ...   # G1d (README amendment 2): clips in gen/root/g1d, outputs in data/g1d, same rule

Filter per voice (as Higgs): dur ratio in [0.5, 2.0] of the real-train median sec/word and <= 15 s; speech fraction
>= 0.3; judge pool A WER <= 0.576 and insertions <= 1, pool B CER <= 0.521.
  old: judge = split_v1 real-only large-v3 R seed 1 (no prefix), rate = asr_aug/data/real_rate.json, pools = asr_aug pools
  s1:  judge = asr_aug oct1_N seed 1 (lang_task), rate/pools = stack/data
  s2:  judge = multi_patient s2 lang_task seed 1 (lang_task), rate/pools = stack/data
Gate (pre-registered in README Step A), per patient s1 / s2 on kept clips (pools A+B):
  keep rate >= 0.5 x Higgs keep rate, AND G1 at least as close to real train as Higgs on >= 2 of
  |mean capped zs_wer - real|, |mean sec/word - real|, mean spk_sim (higher = closer).
  G1 goes to Step B only if s1 passes. The old voice is reported (keep rate) but has no real-clip diag set.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/scratch/pa2753/stack/code")
from filter_synth import norm, speech_fraction, wer_parts  # noqa: E402  (stack copy: same rules as the Higgs filter)

G = "/scratch/pa2753/gen"
GEN = os.environ.get("GEN", "g1")
DESC = {"g1": "IndexTTS-2 + LoRA", "g1d": "IndexTTS-2 + LoRA + duration control",
        "g2": "ProtoDisent (IndexTTS-1.5 + pathology prefix + GRL)", "g3": "VoxCPM2 + LoRA"}
DATA = f"{G}/data/{GEN}"
FROOT = f"{G}/fl" if GEN == "g1" else f"{G}/fl_{GEN}"  # FROOT/root/synth_higgs_b8 -> gen/root/<GEN>, so rel paths and
# dataset name match the Higgs rows
DS = "synth_higgs_b8"
SPM = "/home/pa2753/auto-avsr-realtime/spm/spm_unigram_1023.model"
VOICES = ["old", "s1", "s2"]
DUR_LO, DUR_HI, MAX_SEC, MIN_SPEECH_FRAC = 0.5, 2.0, 15.0, 0.3
THR = {"wer": 0.576, "ins": 1.0, "cer": 0.521}  # H filter, frozen
HIGGS_CLIPS = {"old": "/scratch/pa2753/asr_aug/data/filter_clips.csv", "s": "/scratch/pa2753/stack/data/filter_clips.csv"}


def rate_pools(v):
    if v == "old":
        A = "/scratch/pa2753/asr_aug/data"
        return (json.load(open(f"{A}/real_rate.json"))["sec_per_word_median"],
                {e["norm"]: p for p in "AB" for e in json.load(open(f"{A}/pool_{p}.json"))})
    S = "/scratch/pa2753/stack/data"
    return json.load(open(f"{S}/real_rate.json"))[v], {e["norm"]: e["pool"] for e in json.load(open(f"{S}/text.json"))}


def cmd_collect(_a):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=SPM)
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(f"{FROOT}/root", exist_ok=True)
    if not os.path.islink(f"{FROOT}/root/{DS}"):
        os.symlink(f"{G}/root/{GEN}", f"{FROOT}/root/{DS}")
    info = []
    for v in VOICES:
        _, pool_of = rate_pools(v)
        rows = []
        for r in json.load(open(f"{G}/g1/{v}/plan.json")):
            n = norm(r["text"])
            ids = sp.encode(n)
            assert sp.decode(ids) == n, n
            dur = sf.info(f"{G}/root/{GEN}/{r['rel']}").duration
            rows.append(f"{DS},{r['rel']},{max(1, round(dur * 25))},{' '.join(map(str, ids))}\n")
            info.append({"patient": v, "rel_path": r["rel"], "prompt_id": os.path.basename(r["rel"])[:-4], "norm": n,
                         "pool": pool_of[n], "duration": dur})
        open(f"{DATA}/synth_all_{v}.csv", "w").writelines(rows)
        print(v, len(rows), collections.Counter(i["pool"] for i in info if i["patient"] == v))
    json.dump(info, open(f"{DATA}/synth_info.json", "w"), indent=0)


def cmd_score(_a):
    sys.path.insert(0, "/home/pa2753/auto-avsr-realtime")
    from asr_baselines.metrics import _levenshtein, normalize
    info = json.load(open(f"{DATA}/synth_info.json"))
    judge = {}
    for v in VOICES:
        # raw tab-separated; a hyp may open with a quote mark ('"I want...'), which default csv quoting reads across lines
        for r in csv.DictReader(open(f"{DATA}/judge_{v}.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE):
            judge[r["path"]] = (normalize(r["ref"]), normalize(r["hyp"]))
    rates = {v: rate_pools(v)[0] for v in VOICES}
    clips = []
    for i in info:
        path = f"{FROOT}/root/{DS}/{i['rel_path']}"
        ref, hyp = judge[path]
        e, ins = wer_parts(ref, hyp)
        wer, cer = e / max(1, len(ref.split())), _levenshtein(list(ref), list(hyp)) / max(1, len(ref))
        wav, sr = sf.read(path, dtype="float32")
        wav = wav.mean(axis=1) if wav.ndim > 1 else wav
        ratio = i["duration"] / (len(i["norm"].split()) * rates[i["patient"]])
        sfr = speech_fraction(wav, sr)
        why = []
        if not (DUR_LO <= ratio <= DUR_HI) or i["duration"] > MAX_SEC:
            why.append("dur")
        if sfr < MIN_SPEECH_FRAC:
            why.append("silent")
        if (wer > THR["wer"] or ins > THR["ins"]) if i["pool"] == "A" else cer > THR["cer"]:
            why.append("judge")
        clips.append({**i, "dur_ratio": round(ratio, 3), "speech_frac": round(sfr, 3), "judge_wer": round(wer, 3),
                      "judge_ins": ins, "judge_cer": round(cer, 3), "judge_hyp": hyp, "reject": "+".join(why),
                      "kept": not why})
    for name, keep in (("filter_clips.csv", lambda c: c["patient"] != "old"), ("filter_clips_old.csv", lambda c: c["patient"] == "old")):
        with open(f"{DATA}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(clips[0]))
            w.writeheader()
            w.writerows(c for c in clips if keep(c))
    for v in VOICES:
        rows = {r.split(",")[1]: r for r in open(f"{DATA}/synth_all_{v}.csv").read().splitlines() if r.strip()}
        for pool in "AB":
            kept = [c for c in clips if c["patient"] == v and c["pool"] == pool and c["kept"]]
            open(f"{DATA}/synth_kept_{v}_{pool}.csv", "w").writelines(rows[c["rel_path"]] + "\n" for c in kept)
    print(collections.Counter((c["patient"], c["pool"], c["reject"] or "kept") for c in clips))


def cmd_diag(a):
    import diag_synth as d  # stack copy; its globals are pointed at the G1 clips, the code is unchanged
    d.DATA, d.S, d.OUT = DATA, FROOT, f"{DATA}/diag_clips.csv"
    d.cmd_run(a)


def keep_rates(path, voices):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["patient"] in voices:
            k = out.setdefault(r["patient"], [0, 0])
            k[0] += r["kept"] == "True"
            k[1] += 1
    return {v: k[0] / k[1] for v, k in out.items()}


def summary(path):
    rows = list(csv.DictReader(open(path)))
    m = {}
    for p in ("s1", "s2"):
        real = [r for r in rows if r["patient"] == p and r["set"] == "real_train"]
        syn = [r for r in rows if r["patient"] == p and r["set"] == "synth" and r["kept"] == "True"]
        f = lambda g, k, cap=False: statistics.mean(min(1.0, float(r[k])) if cap else float(r[k]) for r in g)
        m[p] = {"n": len(syn), "real_wer": f(real, "zs_wer", True), "wer": f(syn, "zs_wer", True),
                "real_spw": f(real, "sec_per_word"), "spw": f(syn, "sec_per_word"), "sim": f(syn, "spk_sim"),
                "real_pause": f(real, "pause_frac"), "pause": f(syn, "pause_frac")}
    return m


def cmd_report(_a):
    kr_g = {**keep_rates(f"{DATA}/filter_clips.csv", {"s1", "s2"}), **keep_rates(f"{DATA}/filter_clips_old.csv", {"old"})}
    kr_h = keep_rates(HIGGS_CLIPS["s"], {"s1", "s2"})
    old_h = [r for r in csv.DictReader(open(HIGGS_CLIPS["old"]))]
    kr_h["old"] = sum(r["kept"] == "True" for r in old_h) / len(old_h)
    H, Gm = summary("/scratch/pa2753/stack/data/diag_clips.csv"), summary(f"{DATA}/diag_clips.csv")
    out = [f"# {GEN.upper()} gate: {DESC.get(GEN, GEN)} vs Higgs "
           f"(`asr_aug/gen/gate_g1.py`, GEN={GEN})", "",
           "Kept clips, pools A+B. zs_wer capped at 1 (zero-shot whisper-large-v3). Real = that patient's real train role.", "",
           f"| voice | keep rate Higgs | keep rate {GEN.upper()} | {GEN.upper()} >= 0.5 x Higgs |", "|---|---|---|---|"]
    for v in VOICES:
        out.append(f"| {v} | {kr_h[v]:.3f} | {kr_g[v]:.3f} | {kr_g[v] >= 0.5 * kr_h[v]} |")
    out += ["", "| patient | gen | n kept | zs_wer (real) | sec/word (real) | spk_sim | pause frac (real) |", "|---|---|---|---|---|---|---|"]
    verdict = {}
    for p in ("s1", "s2"):
        for name, m in (("Higgs", H[p]), (GEN.upper(), Gm[p])):
            out.append(f"| {p} | {name} | {m['n']} | {m['wer']:.3f} ({m['real_wer']:.3f}) | {m['spw']:.3f} ({m['real_spw']:.3f}) | "
                       f"{m['sim']:.3f} | {m['pause']:.3f} ({m['real_pause']:.3f}) |")
        h, g = H[p], Gm[p]
        closer = [abs(g["wer"] - g["real_wer"]) <= abs(h["wer"] - h["real_wer"]),
                  abs(g["spw"] - g["real_spw"]) <= abs(h["spw"] - h["real_spw"]), g["sim"] >= h["sim"]]
        ok_keep = kr_g[p] >= 0.5 * kr_h[p]
        verdict[p] = ok_keep and sum(closer) >= 2
        out.append(f"|  | closer than Higgs (wer, rate, sim) | {closer} | keep ok {ok_keep} | **{'PASS' if verdict[p] else 'FAIL'}** | | |")
    out += ["", f"**Gate: s1 {'PASS' if verdict['s1'] else 'FAIL'}, s2 {'PASS' if verdict['s2'] else 'FAIL'}. "
            f"{GEN.upper()} {'goes' if verdict['s1'] else 'does not go'} to Step B (decided by s1).**"]
    open(f"{DATA}/gate_report.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("collect", "score", "report"):
        sub.add_parser(c)
    sub.add_parser("diag").add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    {"collect": cmd_collect, "score": cmd_score, "diag": cmd_diag, "report": cmd_report}[a.cmd](a)
