"""Phase 3.2 generator gate, part 1 (plan 3.4). One row per candidate, every rule as pre-registered in README.md.

Per candidate dir pilot/<GEN>/ (from pilot_gen.sbatch): gen/*.mp4, crop/crop_log_*.jsonl, crop/crops, crop/landmarks,
judge.jsonl, gen/gen_log.jsonl or gen_wall.txt.
  crop success   clip has a generated video, a crop, frames_out == frames_in and no frame without a face; share of 100
                 >= 0.98
  readability    mean judge WER capped at 1 over all 100 pilot clips (a clip without a crop counts as 1)
                 <= mean capped judge WER of real s3ov_train + 0.15
  template leak  share of judged clips whose judge output has lower WER against the template's own sentence than
                 against the target sentence; <= 0.05
  rate           median active sec/word of the output (lip_activity.py, frozen threshold) within 15% of real s3_train
  sharpness      per clip, median over frames of the variance of the Laplacian of the grayscale 96x96 mouth crop;
                 median over synthetic clips / median over real s3_train crops (same crop pipeline) >= 0.7
  too easy       share of judge WER 0, synthetic vs real s3ov_train (descriptive)
A candidate passes if it meets every rule; if more than one passes, the lowest readability WER is chosen.
Usage: python gate_vid.py --gens ls keysync glsft --out results/synvid_gate/gate_report.md
"""
import argparse, glob, json, os, re, subprocess, sys

import cv2
import numpy as np

SV = "/scratch/pa2753/synvid"
HERE = os.path.dirname(os.path.abspath(__file__))


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def wer(ref, hyp):
    R = norm(ref).split()
    return ed(R, norm(hyp).split()) / max(1, len(R))


def lapvar(path):
    cap, v = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        v.append(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    cap.release()
    return float(np.median(v)) if v else None


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", required=True)
    ap.add_argument("--pilot", default=f"{SV}/pilot/pilot.jsonl")
    ap.add_argument("--real-judge", default=f"{SV}/p1/judge_huge_s3ov_train.jsonl")
    ap.add_argument("--real-crops", default=f"{SV}/p1/crop240/crops")
    ap.add_argument("--real-lip", default=f"{SV}/p1/lip_activity_s3_train.json")
    ap.add_argument("--thr", default=f"{SV}/p1/thr.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pilot = jl(a.pilot)
    rj = jl(a.real_judge)
    real_wer = float(np.mean([r["wer_cap1"] for r in rj]))
    real_zero = float(np.mean([r["wer"] == 0 for r in rj]))
    real_spw = json.load(open(a.real_lip))["active_sec_per_word"]["median"]
    real_sharp = float(np.median([x for x in (lapvar(p) for p in sorted(glob.glob(a.real_crops + "/*.mp4"))) if x]))
    res = {}
    for g in a.gens:
        d = f"{SV}/pilot/{g}"
        logs = {r["take_id"]: r for f in glob.glob(f"{d}/crop/crop_log_*.jsonl") for r in jl(f)}
        jd = {r["take_id"]: r for r in jl(f"{d}/judge.jsonl")}
        ok = [p["pid"] for p in pilot if p["pid"] in logs and logs[p["pid"]]["ok"]
              and logs[p["pid"]]["no_face_frames"] == 0 and logs[p["pid"]]["frames_out"] == logs[p["pid"]]["frames_in"]]
        caps = [jd[p["pid"]]["wer_cap1"] if p["pid"] in jd else 1.0 for p in pilot]
        leak = [wer(p["template_text"], jd[p["pid"]]["hyp"]) < wer(p["text"], jd[p["pid"]]["hyp"])
                for p in pilot if p["pid"] in jd]
        lip_out = f"{d}/lip_activity.json"
        subprocess.run([sys.executable, os.path.join(HERE, "lip_activity.py"), "--lm-dir", f"{d}/crop/landmarks",
                        "--rows", f"{d}/crop_rows.jsonl", "--out", lip_out, "--thr-file", a.thr], check=True,
                       stdout=subprocess.DEVNULL)
        spw = json.load(open(lip_out))["active_sec_per_word"]["median"]
        sharp = [x for x in (lapvar(f"{d}/crop/crops/{p}.mp4") for p in ok) if x]
        gl = jl(f"{d}/gen/gen_log.jsonl")
        gen_sec = float(np.median([r["sec"] for r in gl if "error" not in r])) if gl else None
        if gen_sec is None and os.path.exists(f"{d}/gen_wall.txt"):
            w = open(f"{d}/gen_wall.txt").read().split("GEN_WALL_SECONDS=")[1].split()[0]
            gen_sec = float(w) / max(1, len(glob.glob(f"{d}/gen/*.mp4")))
        r = dict(crop_success=len(ok) / len(pilot), readability=float(np.mean(caps)),
                 readability_judged_only=float(np.mean([jd[k]["wer_cap1"] for k in jd])) if jd else None,
                 template_leak=float(np.mean(leak)) if leak else 1.0, active_spw=spw, rate_ratio=spw / real_spw,
                 sharpness_ratio=(float(np.median(sharp)) / real_sharp) if sharp else 0.0,
                 too_easy=float(np.mean([jd[k]["wer"] == 0 for k in jd])) if jd else None, n_judged=len(jd),
                 gpu_sec_per_clip=gen_sec, peak_mib=max((x.get("peak_mib", 0) for x in gl), default=None))
        r["pass"] = dict(crop=r["crop_success"] >= 0.98, readability=r["readability"] <= real_wer + 0.15,
                         leak=r["template_leak"] <= 0.05, rate=abs(r["rate_ratio"] - 1) <= 0.15,
                         sharpness=r["sharpness_ratio"] >= 0.7)
        r["passes_all"] = all(r["pass"].values())
        res[g] = r
    passing = sorted([g for g in res if res[g]["passes_all"]], key=lambda g: res[g]["readability"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(real=dict(s3ov_train_wer_cap1=real_wer, s3ov_train_wer0=real_zero, s3_train_active_spw=real_spw,
                             s3_train_sharpness=real_sharp), gens=res, chosen=passing[0] if passing else None),
              open(a.out.replace(".md", ".json"), "w"), indent=1)
    L = ["# Generator gate, part 1 (pilot: 100 sentences x 100 s3_train templates)", "",
         f"Real references: s3ov_train judge WER (capped) {real_wer:.3f}, WER-0 share {real_zero:.2f}; "
         f"s3_train active sec/word {real_spw:.3f}; s3_train crop sharpness {real_sharp:.1f}.", "",
         "| candidate | crop success (>=0.98) | readability WER (<= %.3f) | template leak (<=0.05) | rate ratio (0.85-1.15) "
         "| sharpness ratio (>=0.7) | WER-0 share | GPU s/clip | pass |" % (real_wer + 0.15),
         "|---|---|---|---|---|---|---|---|---|"]
    for g, r in res.items():
        m = lambda k: "" if r["pass"][k] else " (fail)"
        L.append(f"| {g} | {r['crop_success']:.2f}{m('crop')} | {r['readability']:.3f}{m('readability')} | "
                 f"{r['template_leak']:.3f}{m('leak')} | {r['rate_ratio']:.2f}{m('rate')} | "
                 f"{r['sharpness_ratio']:.2f}{m('sharpness')} | {r['too_easy'] if r['too_easy'] is None else round(r['too_easy'], 2)} | "
                 f"{r['gpu_sec_per_clip'] if r['gpu_sec_per_clip'] is None else round(r['gpu_sec_per_clip'], 1)} | "
                 f"{'PASS' if r['passes_all'] else 'FAIL'} |")
    L += ["", f"Chosen: {passing[0] if passing else 'none passes'}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
