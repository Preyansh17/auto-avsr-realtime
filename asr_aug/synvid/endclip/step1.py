"""Step 1 (CPU): lead-in / tail after the last lip movement on the in-home crops the model uses (devcrop npz).
Signal: the npz 'aperture' (phone front end: inner-lip opening / mouth width, per 25 fps frame). Velocity, smoothing,
min-run and the 0.25 x pooled-p95 threshold rule are the same as asr_aug/synvid/lip_activity.py; the threshold is
fitted once on all 193 in-home clips (the FAN s3_train threshold is on a different aperture scale)."""
import json, sys
import numpy as np
sys.path.insert(0, "/scratch/pa2753/synvid/code"); sys.path.insert(0, "/scratch/pa2753/synvid/endclip")
from lip_activity import velocity, span, q

S = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/semg3_rcv_20261109_v1/roles/dc_home_all.jsonl"
K = "/scratch/th3482/Dysarthric_ASR/data/patients/semg3/packages/semg3-talk-labeled-20261107/manifest.jsonl"
rows = [json.loads(l) for l in open(S)]
man = {m["phrase_id"]: m for m in map(json.loads, open(K))}
st0 = json.load(open("/scratch/pa2753/synvid/endclip/step0_stats.json"))
lw = dict(zip(st0["rel"], st0["delivered"]))

sig = {}
for r in rows:
    z = np.load(r["video_path"])
    a = z["aperture"].astype(np.float64); miss = z["missing"]; T = len(a)
    t = np.arange(T); ok = np.isfinite(a) & ~miss
    a = np.interp(t, t[ok], a[ok]) if ok.sum() >= 2 else None
    sig[r["rel_path"]] = (a, T, int(miss.sum()), int(miss[-10:].sum()))
thr = 0.25 * float(np.percentile(np.concatenate([velocity(a) for a, *_ in sig.values() if a is not None]), 95))

per = []
for r in rows:
    a, T, nmiss, miss_end = sig[r["rel_path"]]
    m = man[r["phrase_id"]]; tm = m.get("timing", {})
    s = span(velocity(a), thr) if a is not None else None
    d = dict(rel=r["rel_path"], T=T, dur_s=T / 25, missing=nmiss, missing_last10=miss_end,
             video_frames=tm.get("video_frames"), video_dur_s=tm.get("video_duration_s"),
             press_s=tm.get("press_s"), release_s=tm.get("release_s"), rec_end_s=tm.get("recording_end_s"),
             last_wrong=lw[r["rel_path"]]["last_wrong"], first_wrong=lw[r["rel_path"]]["first_wrong"],
             last_ref=lw[r["rel_path"]]["last_ref"])
    if s:
        d.update(lead_s=s[0] / 25, tail_s=(T - 1 - s[1]) / 25, end_active_s=(s[1] + 1) / 25)
        if d["release_s"] is not None:
            d["release_minus_lastmove_s"] = d["release_s"] - d["end_active_s"]
    per.append(d)

def grp(key, sel):
    xs = [p[key] for p in per if sel(p) and p.get(key) is not None]
    return q(xs) if xs else None

out = dict(thr=thr, n=len(per), n_span=sum("tail_s" in p for p in per))
for k in ("lead_s", "tail_s", "release_minus_lastmove_s"):
    out[k] = dict(all=grp(k, lambda p: True), last_wrong=grp(k, lambda p: p["last_wrong"]), last_ok=grp(k, lambda p: not p["last_wrong"]),
                  last_wrong_nonpercent=grp(k, lambda p: p["last_wrong"] and p["last_ref"] != "percent"))
# rank-sum (Mann-Whitney U, normal approx) on tail, last-wrong vs last-ok
def mwu(x, y):
    allv = np.concatenate([x, y]); ranks = allv.argsort().argsort() + 1.0
    # average ties
    for v in np.unique(allv):
        idx = allv == v; ranks[idx] = ranks[idx].mean()
    U = ranks[:len(x)].sum() - len(x) * (len(x) + 1) / 2; n1, n2 = len(x), len(y)
    z = (U - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    from math import erf, sqrt
    return dict(U=float(U), z=float(z), p_two_sided=float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))), auc=float(U / (n1 * n2)))
x = np.array([p["tail_s"] for p in per if p["last_wrong"] and "tail_s" in p]); y = np.array([p["tail_s"] for p in per if not p["last_wrong"] and "tail_s" in p])
out["tail_mwu_lastwrong_vs_ok"] = mwu(x, y)
x2 = np.array([p["tail_s"] for p in per if p["last_wrong"] and p["last_ref"] != "percent" and "tail_s" in p])
out["tail_mwu_lastwrong_nonpercent_vs_ok"] = mwu(x2, y)
out["short_tail_counts"] = {f"tail<{c}s": dict(last_wrong=int((x < c).sum()), n_wrong=len(x), last_ok=int((y < c).sum()), n_ok=len(y)) for c in (0.1, 0.2, 0.3, 0.4)}
# processed clip vs raw recording
dd = [p["dur_s"] - p["video_dur_s"] for p in per if p["video_dur_s"]]
out["crop_dur_minus_video_dur_s"] = q(dd)
out["post_release_s"] = q([p["rec_end_s"] - p["release_s"] for p in per if p["release_s"] is not None])
out["crop_end_minus_release_s"] = q([p["dur_s"] - p["release_s"] for p in per if p["release_s"] is not None])
out["n_missing_timing"] = sum(p["release_s"] is None for p in per)
out["clips_missing_in_last10"] = sum(p["missing_last10"] > 0 for p in per)
json.dump(dict(summary=out, per_clip=per), open("/scratch/pa2753/synvid/endclip/step1.json", "w"), indent=1)
print(json.dumps(out, indent=1))
