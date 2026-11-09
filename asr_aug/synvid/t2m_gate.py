"""Gate G3 for text-to-mouth step 1 (README A11). Applies to set b; set a and the real held-out crops are reported with
the same columns.
  validity     clip built, every word realised, every frame 96x96 decodable; share >= 0.98
  readability  mean capped WER of the Huge judge (judge_<set>.jsonl; an unbuilt clip counts as 1) <= 0.372
  sharpness    median Laplacian variance / median over real s3_train crops >= 0.7 (gate_vid.lapvar)
  likeness     aperture range (p95 - p5), speed (p95 of the smoothed aperture velocity x 25), active sec/word
               (lip_activity span rule, frozen thr): synthetic median inside the real s3_train per-clip p10..p90
  v1.8         descriptive only, read from t2m/v18_scores.json when present
Usage: python t2m_gate.py  (writes t2m/gate_g3.md and .json)
"""
import glob, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_vid import lapvar  # noqa: E402
from lip_activity import aperture, velocity, span  # noqa: E402

SV = "/scratch/pa2753/synvid"; T = f"{SV}/t2m"


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def stats(ap, words, thr):
    v = velocity(ap)
    s = span(v, thr)
    return dict(range=float(np.percentile(ap, 95) - np.percentile(ap, 5)), speed=float(np.percentile(v, 95) * 25),
                spw=None if s is None else (s[1] - s[0] + 1) / 25 / words)


def decodable(p):
    cap, n = cv2.VideoCapture(p), 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (96, 96):
            return False
        n += 1
    return n > 0


def main():
    thr = json.load(open(f"{SV}/p1/thr.json"))["thr"]
    lip = {c["take_id"]: c for c in json.load(open(f"{SV}/p1/lip_activity_s3_train.json"))["per_clip"]}
    real = []
    for p in sorted(glob.glob(f"{SV}/p1/crop240/landmarks/*.npy")):
        t = os.path.basename(p)[:-4]
        a = aperture(np.load(p))
        if a is not None and t in lip and lip[t].get("words"):
            real.append(stats(a, lip[t]["words"], thr))
    band = {k: (float(np.percentile([r[k] for r in real if r[k] is not None], 10)),
                float(np.percentile([r[k] for r in real if r[k] is not None], 90))) for k in ("range", "speed", "spw")}
    real_sharp = float(np.median([x for x in (lapvar(p) for p in sorted(glob.glob(f"{SV}/p1/crop240/crops/*.mp4"))) if x]))
    v18 = json.load(open(f"{T}/v18_scores.json")) if os.path.exists(f"{T}/v18_scores.json") else {}
    held = open(f"{T}/heldout.txt").read().split()
    res = {}
    for name in ("a", "b", "a_real"):
        if name == "a_real":
            rows = [dict(sid="a_" + t, path=f"{SV}/p1/crop240/crops/{t}.mp4", valid=True, src=t) for t in held]
        else:
            rows = jl(f"{T}/{name}/rows.jsonl")
        jd = {r["take_id"]: r for r in jl(f"{T}/judge_{name}.jsonl")}
        ok = [r for r in rows if r.get("valid") and r.get("path") and decodable(r["path"])]
        key = lambda r: os.path.basename(r["path"])[:-4] if r.get("path") else None  # judge take_id = file stem
        caps = [jd[key(r)]["wer_cap1"] if key(r) in jd else 1.0 for r in rows]
        sharp = [x for x in (lapvar(r["path"]) for r in ok) if x]
        st = []
        for r in ok:
            if name == "a_real":
                a = aperture(np.load(f"{SV}/p1/crop240/landmarks/{r['src']}.npy")); w = lip.get(r["src"], {}).get("words")
            else:
                a = np.load(f"{T}/{name}/ap/{r['sid']}.npy"); w = r["words"]
            if a is not None and w:
                st.append(stats(a, w, thr))
        med = {k: (float(np.median([s[k] for s in st if s[k] is not None])) if any(s[k] is not None for s in st) else None)
               for k in ("range", "speed", "spw")}
        r_ = dict(n=len(rows), validity=len(ok) / max(1, len(rows)), readability=float(np.mean(caps)) if caps else None,
                  wer0=float(np.mean([jd[k]["wer"] == 0 for k in jd])) if jd else None,
                  sharpness_ratio=float(np.median(sharp)) / real_sharp if sharp else 0.0, likeness=med,
                  in_vocab_word_rate=(sum(r.get("in_vocab", 0) for r in rows) / max(1, sum(r.get("words", 0) for r in rows)))
                  if name != "a_real" else None, v18=v18.get(name))
        r_["pass"] = dict(validity=r_["validity"] >= 0.98, readability=r_["readability"] is not None and r_["readability"] <= 0.372,
                          sharpness=r_["sharpness_ratio"] >= 0.7,
                          likeness=all(med[k] is not None and band[k][0] <= med[k] <= band[k][1] for k in band))
        r_["passes_all"] = all(r_["pass"].values())
        res[name] = r_
    out = dict(real_band_p10_p90=band, real_sharpness=real_sharp, sets=res,
               g3=("PASS" if res["b"]["passes_all"] else "FAIL"), stop_rule_a_gt_0_6=res["a"]["readability"] > 0.6)
    json.dump(out, open(f"{T}/gate_g3.json", "w"), indent=1)
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    L = ["# Gate G3, text-to-mouth step 1 (README A11)", "",
         "Real s3_train per-clip p10..p90: aperture range %s..%s, speed %s..%s, active sec/word %s..%s" % (
             f(band['range'][0]), f(band['range'][1]), f(band['speed'][0], 2), f(band['speed'][1], 2), f(band['spw'][0]), f(band['spw'][1])), "",
         "| set | n | in-vocab words | validity (>=0.98) | Huge readability (<=0.372) | Huge WER-0 | sharpness (>=0.7) | "
         "range | speed | sec/word | likeness | v1.8 pooled / capped | G3 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        v = r["v18"] or {}
        L.append(f"| {name} | {r['n']} | {f(r['in_vocab_word_rate'], 2)} | {f(r['validity'], 2)} | {f(r['readability'])} | "
                 f"{f(r['wer0'], 2)} | {f(r['sharpness_ratio'], 2)} | {f(r['likeness']['range'])} | {f(r['likeness']['speed'], 2)} | "
                 f"{f(r['likeness']['spw'])} | {'ok' if r['pass']['likeness'] else 'out'} | "
                 f"{f(v.get('pooled_wer'), 1)} / {f(v.get('capmean'))} | {'PASS' if r['passes_all'] else 'FAIL'} |")
    L += ["", f"G3 (set b): {out['g3']}. Stop rule (set a readability > 0.6): {'fires' if out['stop_rule_a_gt_0_6'] else 'does not fire'}.",
          "D2 G-LSft on the same 100 sentences: Huge 0.839, v1.8 93.59% / 0.863."]
    open(f"{T}/gate_g3.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
