"""Gate G4 (A13 + A16) for the V2 text-to-mouth generator.

  realref   write real heldout30 crops as mp4 + rows (judge reference, sharpness/likeness reference)
  aperture  train a small CNN aperture regressor on real gen-train devcrop frames (MediaPipe aperture target);
            FAN cannot run on 96x96 mouth crops, so likeness uses this one instrument on real and generated clips
  score     combine judge outputs + validity + sharpness + likeness -> g4_report.json / g4_report.md
Frozen thresholds (A13): readability mean capped WER <= 0.372 on a, b, d; validity >= 0.98 on b; sharpness
median Laplacian var gen/real >= 0.7; likeness medians inside real p10..p90; stop rule a > 0.6.
"""
import argparse, json, os, random, subprocess
from statistics import median

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SV = "/scratch/pa2753/synvid"
G4 = os.environ.get("T2M2_G4", f"{SV}/t2m2/g4")
GEN = os.environ.get("T2M2_G", f"{SV}/t2m2/gen")
FFMPEG = f"{SV}/envs/latentsync/bin/ffmpeg"
THR = dict(read=0.372, valid=0.98, sharp=0.7, stop=0.6)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def u8(npz):
    with np.load(npz) as z:
        return np.clip(np.rint(z["gray96"].astype(np.float32)), 0, 255).astype(np.uint8)


def write_mp4(fr, path):
    bgr = np.repeat(fr[..., None], 3, -1)
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "96x96", "-r", "25",
                    "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", path], input=bgr.tobytes(), check=True)


def lapvar(fr):  # T,96,96 uint8 -> median per-frame Laplacian variance
    x = torch.from_numpy(fr).float()[:, None]
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)[None, None]
    return float(F.conv2d(x, k).var((1, 2, 3)).median())


def cmd_realref(a):
    os.makedirs(f"{G4}/real_a/clips", exist_ok=True)
    rows = []
    for it in jl(f"{G4}/a/items.jsonl"):
        p = f"{G4}/real_a/clips/{it['sid']}.mp4"
        write_mp4(u8(f"{SV}/t2m2/crops/npz/{it['real_id']}.npz"), p)
        rows.append(dict(sid=it["sid"], take_id=it["sid"], video_path=p, npz_path=f"{SV}/t2m2/crops/npz/{it['real_id']}.npz",
                         transcript=it["text"], words=len(it["text"].split()), valid=True))
    with open(f"{G4}/real_a/rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows))


class ApNet(nn.Module):
    def __init__(s):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        s.f = nn.Sequential(c(1, 32), c(32, 64), c(64, 128), c(128, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))

    def forward(s, x):
        return s.f(x).squeeze(1)


def cmd_aperture(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    clips = [c for c in torch.load(f"{GEN}/pack.pt") if not c["val"]]
    X, Y = [], []
    for c in clips:
        with np.load(f"{SV}/t2m2/crops/npz/{c['id']}.npz") as z:
            ap, miss = z["aperture"].astype(np.float32), z["missing"]
        ok = ~miss & np.isfinite(ap)
        X.append(c["frames"][torch.from_numpy(ok)]); Y.append(torch.from_numpy(ap[ok]))
    X, Y = torch.cat(X), torch.cat(Y)
    mu, sd = Y.mean().item(), Y.std().item()
    idx = torch.randperm(len(X)); va, tr = idx[:3000], idx[3000:]
    m = ApNet().to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3)
    for step in range(1, a.steps + 1):
        b = tr[torch.randint(len(tr), (256,))]
        x = X[b].to(dev).float()[:, None] / 255
        loss = F.smooth_l1_loss(m(x), ((Y[b] - mu) / sd).to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.cat([m(X[va[i:i + 512]].to(dev).float()[:, None] / 255).cpu() for i in range(0, len(va), 512)]) * sd + mu
    r = float(np.corrcoef(p.numpy(), Y[va].numpy())[0, 1])
    torch.save(dict(m=m.state_dict(), mu=mu, sd=sd), f"{G4}/apnet.pth")
    json.dump(dict(frames=len(X), val_r=r), open(f"{G4}/apnet_report.json", "w"))
    print(dict(frames=len(X), val_r=r))


def aperture_series(m, mu, sd, fr, dev):
    with torch.no_grad():
        return (m(torch.from_numpy(fr).to(dev).float()[:, None] / 255).cpu().numpy() * sd + mu)


def likeness(ap, words, thr):
    ap = np.convolve(ap, np.ones(3) / 3, mode="same")
    sp = np.abs(np.diff(ap)) * 25
    act = np.nonzero(sp > thr)[0]
    span = (act[-1] - act[0] + 1) / 25 if len(act) else 0.0
    return dict(range=float(np.percentile(ap, 95) - np.percentile(ap, 5)), speed=float(np.percentile(sp, 95)),
                spw=span / max(words, 1))


def cmd_score(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(f"{G4}/apnet.pth"); m = ApNet().to(dev); m.load_state_dict(st["m"]); m.eval()
    mu, sd = st["mu"], st["sd"]
    # real reference: gen-train pool clips (instrument = ApNet on both sides)
    clips = [c for c in torch.load(f"{GEN}/pack.pt") if not c["val"]]
    random.Random(7).shuffle(clips)
    ref = clips[:300]
    real_ap = [(aperture_series(m, mu, sd, c["frames"].numpy(), dev), len(c["text"].split())) for c in ref]
    speeds = np.concatenate([np.abs(np.diff(np.convolve(x, np.ones(3) / 3, mode="same"))) * 25 for x, _ in real_ap])
    thr = 0.25 * float(np.percentile(speeds, 95))
    real_l = [likeness(x, w, thr) for x, w in real_ap]
    band = {k: (float(np.percentile([d[k] for d in real_l], 10)), float(np.percentile([d[k] for d in real_l], 90)))
            for k in real_l[0]}
    real_sharp = median(lapvar(c["frames"].numpy()) for c in ref)
    rep = dict(thr_speed=thr, band=band, real_sharp=real_sharp, sets={})
    for s in ["a", "b", "d", "real_a"]:
        rp = f"{G4}/{s}/rows.jsonl"
        if not os.path.exists(rp):
            continue
        rows = jl(rp)
        jd = {r["take_id"]: r for r in jl(f"{G4}/{s}/judge.jsonl")} if os.path.exists(f"{G4}/{s}/judge.jsonl") else {}
        caps = [jd[r["sid"]]["wer_cap1"] if r["sid"] in jd else 1.0 for r in rows]
        errs = sum(jd[r["sid"]]["errors"] for r in rows if r["sid"] in jd)
        nw = sum(jd[r["sid"]]["ref_words"] for r in rows if r["sid"] in jd)
        lk, sh = [], []
        for r in rows:
            fr = u8(r["npz_path"])
            lk.append(likeness(aperture_series(m, mu, sd, fr, dev), r["words"], thr)); sh.append(lapvar(fr))
        med = {k: median(d[k] for d in lk) for k in lk[0]}
        rep["sets"][s] = dict(n=len(rows), judged=len(jd), validity=sum(r["valid"] for r in rows) / len(rows),
                              readability=sum(caps) / len(caps), pooled_wer=errs / max(nw, 1), wer0=sum(c == 0 for c in caps) / len(caps),
                              sharp=median(sh) / real_sharp, likeness=med,
                              like_ok=all(band[k][0] <= med[k] <= band[k][1] for k in med))
    S = rep["sets"]
    g = lambda s, k: S.get(s, {}).get(k)
    checks = {"a_read": g("a", "readability") is not None and g("a", "readability") <= THR["read"],
              "b_read": g("b", "readability") is not None and g("b", "readability") <= THR["read"],
              "d_read": g("d", "readability") is not None and g("d", "readability") <= THR["read"],
              "b_valid": g("b", "validity") is not None and g("b", "validity") >= THR["valid"],
              "sharp": all(S[s]["sharp"] >= THR["sharp"] for s in "abd" if s in S),
              "likeness": all(S[s]["like_ok"] for s in "abd" if s in S)}
    rep["checks"] = checks
    rep["G4"] = "PASS" if all(checks.values()) else "FAIL"
    rep["stop_rule_fires"] = g("a", "readability") is not None and g("a", "readability") > THR["stop"]
    json.dump(rep, open(f"{G4}/g4_report.json", "w"), indent=1)
    lines = ["| set | n | validity | readability (<=0.372) | pooled WER | WER-0 | sharp (>=0.7) | likeness range/speed/spw | in band |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s, d in S.items():
        L = d["likeness"]
        lines.append(f"| {s} | {d['n']} | {d['validity']:.2f} | {d['readability']:.3f} | {100*d['pooled_wer']:.1f}% | {d['wer0']:.2f} | "
                     f"{d['sharp']:.2f} | {L['range']:.3f}/{L['speed']:.3f}/{L['spw']:.3f} | {d['like_ok']} |")
    lines.append(f"\nReal band p10..p90: " + ", ".join(f"{k} {v[0]:.3f}..{v[1]:.3f}" for k, v in band.items()))
    lines.append(f"\nChecks: {checks}\n\nG4: **{rep['G4']}**; stop rule (a > 0.6) fires: {rep['stop_rule_fires']}")
    open(f"{G4}/g4_report.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["realref", "aperture", "score"])
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    dict(realref=cmd_realref, aperture=cmd_aperture, score=cmd_score)[a.cmd](a)


if __name__ == "__main__":
    main()
