"""Room-simulated copies of the stack's kept Higgs clips for arm E2R (PLAN.md 3.3 item 5; pre-registered in README.md).

    python rir_synth.py build    # CPU: noise bank + SNR prior from real train clips, then every kept clip -> RIR_DIR

Higgs output is clean studio audio; the patient clips are room recordings (lit review 1.4, Labrak et al. 2026). Each
synthetic clip gets, deterministically from its rel_path:
  1. a simulated RIR: unit direct path + exponentially decaying Gaussian tail starting 1-5 ms later,
     RT60 ~ U(0.15, 0.50) s, direct-to-reverberant ratio ~ U(0, 10) dB (small clinic / office rooms);
  2. additive noise cut from the real recordings' own noise floor (the quietest 20 ms frames of that patient's real
     train clips; s1 voice -> s1_train, s2 voice -> s2_train), at an SNR drawn from the same patient's real per-clip
     SNR estimates (speech level = mean power of frames within 35 dB of the clip max, noise = mean of the quietest 10%).
Output keeps the original clip length (reverb tail truncated) and RMS, so label lengths and loudness are unchanged; only
room and channel change. Written as 16 kHz wav at RIR_DIR/<same rel_path>, served through root/synth_rir.
Only train-role real clips are read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_synth import load_16k  # noqa: E402

SR = 16000
S = "/scratch/pa2753/stack"
H = "/scratch/pa2753/asr_aug/data"
GEN = "/scratch/pa2753/gen_runs"  # = root/synth_higgs_b8
RIR_DIR = f"{S}/rir_audio"        # = root/synth_rir
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
# kept pools and the patient whose room they get (the ONS pool is the semg1 old-data adapter)
POOLS = {f"{H}/synth_kept_A.csv": "s1", f"{H}/synth_kept_B.csv": "s1",
         f"{S}/data/synth_kept_s1_A.csv": "s1", f"{S}/data/synth_kept_s1_B.csv": "s1",
         f"{S}/data/synth_kept_s2_A.csv": "s2", f"{S}/data/synth_kept_s2_B.csv": "s2"}
FR = int(0.02 * SR)


def frame_pow(w):
    f = w[: len(w) // FR * FR].reshape(-1, FR)
    return f, np.mean(f ** 2, axis=1) + 1e-12


def real_stats(p):
    """Noise bank (quietest 20% of frames, concatenated) and per-clip SNR estimates (dB) for one patient."""
    bank, snrs = [], []
    for l in open(f"{ROLES}/{p}_train.jsonl"):
        f, pw = frame_pow(load_16k(json.loads(l)["audio_path"]))
        db = 10 * np.log10(pw)
        order = np.argsort(pw)
        bank.append(f[order[: max(1, len(pw) // 5)]].reshape(-1))
        noise = pw[order[: max(1, len(pw) // 10)]].mean()
        speech = pw[db > db.max() - 35].mean()
        snrs.append(10 * np.log10(speech / noise))
    return np.concatenate(bank), np.array(snrs)


def rir(rng):
    rt60, drr = rng.uniform(0.15, 0.50), rng.uniform(0, 10)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.908 * t / rt60)  # -60 dB at rt60
    tail[: int(rng.uniform(0.001, 0.005) * SR)] = 0
    tail *= np.sqrt(10 ** (-drr / 10) / np.sum(tail ** 2))  # direct energy 1
    tail[0] = 1.0
    return tail, rt60, drr


def room(w, bank, snrs, key):
    rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    h, rt60, drr = rir(rng)
    y = np.convolve(w, h)[: len(w)]
    snr = float(rng.choice(snrs))
    st = rng.integers(0, len(bank) - len(y)) if len(bank) > len(y) else 0
    nz = np.resize(bank[st:], len(y))
    _, pw = frame_pow(y) if len(y) >= FR else (None, np.array([np.mean(y ** 2) + 1e-12]))
    db = 10 * np.log10(pw)
    speech = pw[db > db.max() - 35].mean()
    y = y + nz * np.sqrt(speech / 10 ** (snr / 10) / (np.mean(nz ** 2) + 1e-12))
    y *= np.sqrt(np.mean(w ** 2) / (np.mean(y ** 2) + 1e-12))  # keep the original loudness
    peak = np.abs(y).max()
    if peak > 0.99:
        y *= 0.99 / peak
    return y.astype(np.float32), rt60, drr, snr


def cmd_build():
    stats = {p: real_stats(p) for p in ("s1", "s2")}
    for p, (b, s) in stats.items():
        print(f"{p}: noise bank {len(b) / SR:.1f} s, real SNR dB mean {s.mean():.1f} median {np.median(s):.1f} "
              f"[p10 {np.percentile(s, 10):.1f}, p90 {np.percentile(s, 90):.1f}]", flush=True)
    log, n = [], 0
    for pool, p in POOLS.items():
        for r in csv.reader(open(pool)):
            rel = r[1]
            out = f"{RIR_DIR}/{rel}"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            y, rt60, drr, snr = room(load_16k(f"{GEN}/{rel}"), *stats[p], rel)
            sf.write(out, y, SR)
            log.append({"rel_path": rel, "patient": p, "rt60": round(rt60, 3), "drr": round(drr, 2), "snr": round(snr, 2)})
            n += 1
            if n % 500 == 0:
                print(n, flush=True)
    with open(f"{S}/data/rir_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"wrote {n} clips -> {RIR_DIR}; log {S}/data/rir_log.csv")


if __name__ == "__main__":
    assert sys.argv[1:] == ["build"], __doc__
    cmd_build()
