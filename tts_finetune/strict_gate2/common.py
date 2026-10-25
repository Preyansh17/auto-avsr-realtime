"""Shared helpers for the strict gate-2 tests (T1-T4). See THRESHOLDS.md for what each test must show."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FT_DIR = os.path.dirname(HERE)
sys.path.insert(0, FT_DIR)
import higgs_ft_lib as L  # noqa: E402

ROOT = os.environ.get("HIGGS_FT_ROOT", "/scratch/pa2753/higgs_ft")
DATA = f"{ROOT}/data"
OUT_ROOT = f"{ROOT}/strict_gate2/out"
TORGO_SPK = os.environ.get("TORGO_SPK", "/scratch/th3482/TORGO/raw/M03")  # read only
SEED_BASE = 260619825
GEN_PARAMS = {"temperature": 1.0, "top_k": 50, "top_p": 0.95, "max_new_tokens": 2048,
              "ras_win_len": 7, "ras_win_max_num_repeat": 2}  # build_plan_ft.DEFAULT


def load_splits():
    return json.load(open(f"{DATA}/splits.json"))


def load_refs():
    return json.load(open(f"{ROOT}/refs/refs_ft.json"))["refs"]


def cached_codes(u):
    return torch.load(f"{DATA}/codes/{u}.pt").long()


class GenCapture:
    """Records what the real inference path does inside Client.generate: the raw audio_lm_head output at the last
    position of every forward call (= the logits each generated frame was sampled from, before any logits processor)
    and the full audio_input_ids handed to processor.decode (reference frames + generated frames)."""

    def __init__(self, model, processor):
        self.model, self.proc = model, processor
        self.logits, self.frames, self._h = [], None, None
        self._orig_decode = processor.decode

    def __enter__(self):
        self.logits, self.frames = [], None
        self._h = self.model.audio_lm_head.register_forward_hook(
            lambda m, i, o: self.logits.append(o[0, -1].detach().float().cpu()))

        def dec(ids):
            self.frames = ids.detach().cpu().clone()
            return self._orig_decode(ids)

        self.proc.decode = dec
        return self

    def __exit__(self, *a):
        self._h.remove()
        self.proc.decode = self._orig_decode


def base_hf_model(model):
    """The HiggsAudioV2ForConditionalGeneration under a peft wrapper (or the model itself)."""
    m = model
    while not hasattr(m, "audio_lm_head"):
        m = m.base_model if hasattr(m, "base_model") and m.base_model is not m else m.model
    return m


def first_eos(col):
    idx = (col == L.EOS_ID).nonzero()
    return int(idx[0]) if len(idx) else -1


# ----------------------------------------------------------------------------- logit comparison
def cell_stats(I, T, x):
    """I, T: (n, 8, V) logits; x: (n, 8) sampled tokens. Per-cell numpy arrays."""
    lsI, lsT = F.log_softmax(I.float(), -1), F.log_softmax(T.float(), -1)
    d = (lsI - lsT).abs()
    lpI, lpT = lsI.gather(-1, x[..., None]).squeeze(-1), lsT.gather(-1, x[..., None]).squeeze(-1)
    tv = 0.5 * (lsI.exp() - lsT.exp()).abs().sum(-1)
    return {"meanD": d.mean(-1).numpy(), "maxD": d.amax(-1).numpy(), "dLP": (lpI - lpT).abs().numpy(),
            "agree": (lsI.argmax(-1) == lsT.argmax(-1)).numpy(), "tv": tv.numpy(),
            "lpI": lpI.numpy(), "lpT": lpT.numpy()}


def pool(stats_list):
    cat = {k: np.concatenate([s[k].reshape(-1) for s in stats_list]) for k in stats_list[0]}
    return {"n_cells": int(cat["meanD"].size),
            "median_meanD": float(np.median(cat["meanD"])), "p99_meanD": float(np.percentile(cat["meanD"], 99)),
            "max_meanD": float(cat["meanD"].max()),
            "median_maxD": float(np.median(cat["maxD"])), "p99_maxD": float(np.percentile(cat["maxD"], 99)),
            "max_maxD": float(cat["maxD"].max()),
            "mean_dLP": float(cat["dLP"].mean()), "p99_dLP": float(np.percentile(cat["dLP"], 99)),
            "agree": float(cat["agree"].mean()), "median_tv": float(np.median(cat["tv"])),
            "mean_lp_sampled_inf": float(cat["lpI"].mean()), "mean_lp_sampled_train": float(cat["lpT"].mean())}


# ----------------------------------------------------------------------------- text / ASR
def wer(ref: str, hyp: str) -> float:
    r, h = L.normalize_text(ref).split(), L.normalize_text(hyp).split()
    if not r:
        return float(len(h) > 0)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def asr_whisper(paths, dev="cuda:0"):
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    p = WhisperProcessor.from_pretrained("openai/whisper-small")
    m = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", torch_dtype=torch.float16).to(dev).eval()
    out = []
    for path in paths:
        w = librosa.load(path, sr=16000, mono=True)[0]
        feats = p(w, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
        with torch.inference_mode():
            ids = m.generate(feats, language="en", task="transcribe", max_new_tokens=128)
        out.append(p.batch_decode(ids, skip_special_tokens=True)[0].strip())
    del m
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- TORGO (read only)
def torgo_clips(min_words=1):
    """[(wav_path, prompt_text, duration_sec)] for head-mic clips of TORGO_SPK with a spoken-text prompt."""
    import soundfile as sf

    rows = []
    for pf in sorted(glob.glob(f"{TORGO_SPK}/Session*/prompts/*.txt")):
        sess = os.path.dirname(os.path.dirname(pf))
        wav = f"{sess}/wav_headMic/{os.path.basename(pf)[:-4]}.wav"
        if not os.path.exists(wav):
            continue
        t = " ".join(open(pf, errors="ignore").read().split())
        if not t or "/" in t or "[" in t or "xxx" in t.lower() or len(t.split()) < min_words:
            continue
        info = sf.info(wav)
        dur = info.frames / info.samplerate
        if 0.8 <= dur <= 12.0:
            rows.append((wav, t, dur))
    return rows
