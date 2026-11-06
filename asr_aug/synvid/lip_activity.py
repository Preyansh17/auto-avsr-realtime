"""Active speaking span from FAN 68-point inner-lip landmarks (plan A3).

Per frame: aperture a_t = mean(|p61-p67|, |p62-p66|, |p63-p65|) / |p36-p45| (inner-lip vertical opening over
outer-eye-corner distance, so it is scale-free). NaN frames (no face) are linearly interpolated.
Activity: v_t = |a_{t+1} - a_{t-1}| / 2, smoothed with a 5-frame moving average.
A frame is active if v_t > THR. THR is fixed once from real s3_train clips (0.25 x the pooled 95th percentile
of v over all real frames) and frozen in thr.json, so synthetic clips are measured with the same rule.
Span = first to last active frame, after dropping active runs shorter than 3 frames.
lead-in = frames before the span / 25; tail = frames after it / 25; active_s = span length / 25.

Usage: python lip_activity.py --lm-dir DIR --rows s3_train.jsonl --out OUT.json [--thr-file thr.json | --fit-thr]
"""
import argparse, glob, json, os, re

import numpy as np

FPS = 25.0


def norm_words(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip().split()


def aperture(lm):
    lm = lm.astype(np.float64)
    eye = np.linalg.norm(lm[:, 36] - lm[:, 45], axis=1)
    op = np.mean([np.linalg.norm(lm[:, i] - lm[:, j], axis=1) for i, j in ((61, 67), (62, 66), (63, 65))], axis=0)
    a = op / eye
    t = np.arange(len(a))
    ok = np.isfinite(a)
    if ok.sum() < 2:
        return None
    return np.interp(t, t[ok], a[ok])


def velocity(a):
    v = np.zeros_like(a)
    v[1:-1] = np.abs(a[2:] - a[:-2]) / 2
    v[0], v[-1] = v[1], v[-2]
    return np.convolve(v, np.ones(5) / 5, mode="same")


def span(v, thr, min_run=3):
    act = v > thr
    idx = np.where(act)[0]
    if len(idx) == 0:
        return None
    runs, s = [], idx[0]
    for p, q in zip(idx[:-1], idx[1:]):
        if q != p + 1:
            runs.append((s, p)); s = q
    runs.append((s, idx[-1]))
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_run]
    if not runs:
        return None
    return runs[0][0], runs[-1][1]


def q(xs):
    xs = np.asarray(xs, float)
    return dict(n=int(len(xs)), p10=float(np.percentile(xs, 10)), median=float(np.median(xs)),
                p90=float(np.percentile(xs, 90)), mean=float(xs.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-dir", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr-file", default=None)
    ap.add_argument("--fit-thr", action="store_true")
    a = ap.parse_args()
    words = {}
    for l in open(a.rows):
        if l.strip():
            r = json.loads(l)
            words[r["take_id"]] = len(norm_words(r["transcript"]))
    vs = {}
    for f in sorted(glob.glob(os.path.join(a.lm_dir, "*.npy"))):
        tid = os.path.basename(f)[:-4]
        ap_ = aperture(np.load(f))
        if ap_ is not None and tid in words:
            vs[tid] = (ap_, velocity(ap_))
    if a.fit_thr:
        thr = 0.25 * float(np.percentile(np.concatenate([v for _, v in vs.values()]), 95))
        json.dump(dict(thr=thr, rule="0.25 x pooled p95 of smoothed |d aperture/dt| over real s3_train frames",
                       n_clips=len(vs)), open(a.thr_file or os.path.join(os.path.dirname(a.out), "thr.json"), "w"),
                  indent=1)
    else:
        thr = json.load(open(a.thr_file))["thr"]
    per, lead, tail, act, spw = [], [], [], [], []
    for tid, (ap_, v) in vs.items():
        n = len(v)
        s = span(v, thr)
        if s is None:
            per.append(dict(take_id=tid, frames=n, span=None)); continue
        L, T, A = s[0] / FPS, (n - 1 - s[1]) / FPS, (s[1] - s[0] + 1) / FPS
        lead.append(L); tail.append(T); act.append(A); spw.append(A / words[tid])
        per.append(dict(take_id=tid, frames=n, start=int(s[0]), end=int(s[1]), lead_s=L, tail_s=T, active_s=A,
                        active_spw=A / words[tid], words=words[tid]))
    out = dict(thr=thr, n_clips=len(vs), n_with_span=len(act), lead_in_s=q(lead), tail_s=q(tail),
               active_s=q(act), active_sec_per_word=q(spw), per_clip=per)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_clip"}, indent=1))


if __name__ == "__main__":
    main()
