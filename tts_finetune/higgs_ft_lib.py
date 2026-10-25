"""Shared pieces for the Higgs Audio v2 LoRA finetune: example construction, labels, loss, model loading.

Everything here mirrors what inference does (higgs_gen/gen_higgs.py `Client.generate`) so that a
training example is exactly "the inference prompt + the target clip's audio tokens":

    system   "Generate audio following instruction"
    user     <reference text>                      (normalize_for_higgs)
    assistant<reference audio>                     (one <|AUDIO_OUT|> block, expanded like the processor does)
    user     <target text>
    assistant<target audio>                        <- the only span that carries loss

Audio tokens follow the processor's layout: codes (8, T) -> [BOS, codes, EOS] -> delay pattern
-> (T+9, 8) frames. Labels sit on the same positions as the frames; the loss shifts by one
(logits at frame k predict frame k+1). Stream-BOS labels are masked (the logits processor forces
them at generation time); stream-EOS labels are kept so the model still learns when to stop.
"""
from __future__ import annotations

import io
import math
import os
import random
import re
import sys

import numpy as np

HIGGS_GEN = os.environ.get("HIGGS_GEN", "/scratch/pa2753/higgs_gen")
if HIGGS_GEN not in sys.path:
    sys.path.insert(0, HIGGS_GEN)
from gen_higgs import normalize_for_higgs, normalize_text, display_text, stable_id  # noqa: E402,F401

MODEL = "bosonai/higgs-audio-v2-generation-3B-base"
SYSTEM = "Generate audio following instruction"
SR = 24000
NUM_CB = 8
BOS_ID, EOS_ID = 1024, 1025
IGNORE = -100
# make_refs.py recipe
SILENCE_SEC, FINAL_PEAK, MAX_STITCH_SOURCE_SEC = 0.3, 0.98, 45.0
MAX_SEQ_LEN = 1900  # model max_position_embeddings is 2048; keep headroom

LORA_TARGET_REGEX = (
    r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj)"
    r"|audio_mlp\.(gate_proj|up_proj|down_proj))"
)


# ----------------------------------------------------------------------------- models
def load_models(device: str, model_path: str = MODEL):
    """Same loading as gen_higgs.Client.__init__ (bf16 model, audio tokenizer on `device`, eval)."""
    import torch
    from transformers import AutoProcessor
    from transformers.models.higgs_audio_v2 import HiggsAudioV2ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
        model_path, device_map=device,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    model.eval()
    processor.audio_tokenizer.to(device)
    processor.audio_tokenizer.eval()
    return processor, model


def make_client(processor, model, device: str, model_path: str = MODEL):
    """A gen_higgs.Client around an already-loaded model (so `Client.generate` is reused unchanged)."""
    import torch
    from gen_higgs import Client

    c = Client.__new__(Client)
    c.torch, c.device, c.model_path = torch, device, model_path
    c.processor, c.model, c.sr, c._ref_cache = processor, model, SR, {}
    return c


# ----------------------------------------------------------------------------- audio
def load_wave(path: str) -> np.ndarray:
    import librosa

    return librosa.load(path, sr=SR, mono=True)[0].astype(np.float32)


def stitch_waves(waves: list[np.ndarray]) -> np.ndarray:
    """higgs_gen/make_refs.py `stitch`, on in-memory 24 kHz arrays, then the PCM_16 file round trip
    the real reference wav goes through (written by soundfile, re-read by librosa at inference)."""
    import soundfile as sf

    silence = np.zeros(int(round(SR * SILENCE_SEC)), dtype=np.float32)
    chunks = []
    for i, w in enumerate(waves):
        chunks.append(w.astype(np.float32))
        if i != len(waves) - 1:
            chunks.append(silence)
    y = np.concatenate(chunks)
    peak = float(np.max(np.abs(y)))
    if peak > FINAL_PEAK:
        y = y * (FINAL_PEAK / peak)
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV")  # default subtype for WAV is PCM_16, as in make_refs
    buf.seek(0)
    return sf.read(buf, dtype="float32")[0]


def encode_wave(processor, wave: np.ndarray):
    """Audio -> codes (8, T) long, via the processor's own feature extractor + audio tokenizer."""
    import torch

    ai = processor.feature_extractor(wave, padding=False, sampling_rate=SR, return_tensors="pt")
    ai.pop("padding_mask", None)
    ai = ai.to(processor.audio_tokenizer.device)
    with torch.inference_mode():
        codes = processor.audio_tokenizer.encode(**ai).audio_codes
    return codes[0].detach().cpu().long()


def build_delay_pattern(codes):
    """(8, T) codes -> (T+9, 8) frames: [BOS, codes, EOS] with the per-codebook delay of the processor."""
    import torch

    cb, T = codes.shape
    seq = torch.cat([torch.full((cb, 1), BOS_ID, dtype=torch.long), codes.long(),
                     torch.full((cb, 1), EOS_ID, dtype=torch.long)], dim=1)
    L = seq.shape[1]
    out = torch.ones((cb, L + cb - 1), dtype=torch.long)
    bos_mask = torch.tril(out, -1) > 0
    eos_mask = torch.triu(out, L) > 0
    data_mask = ~(bos_mask | eos_mask)
    out[bos_mask] = BOS_ID
    out[data_mask] = seq.reshape(-1)
    out[eos_mask] = EOS_ID
    return out.transpose(0, 1).contiguous()


def revert_delay_pattern(frames):
    """(T+9, 8) frames -> (8, T) codes (inverse of build_delay_pattern, BOS/EOS rows stripped)."""
    import torch

    n, cb = frames.shape
    L = n - cb + 1
    seq = torch.stack([frames[i : i + L, i] for i in range(cb)], dim=0)  # (8, T+2)
    return seq[:, 1:-1]


# ----------------------------------------------------------------------------- examples
def build_example(processor, ref_text: str, ref_codes, tgt_text: str, tgt_codes, eos_labels: bool = True):
    """One batch-of-1 training example. `*_codes` are (8, T) long CPU tensors.

    Returns model inputs (input_ids, attention_mask, audio_input_ids, audio_input_ids_mask), plus
    `audio_labels` (1, N, 8) in the HF forward() convention and `labels_pos` (L, 8): the same labels
    laid out per text position (-100 off audio positions), which is what the loss below consumes.
    """
    import torch

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": normalize_for_higgs(ref_text)},
        {"role": "assistant", "content": [{"type": "audio"}]},
        {"role": "user", "content": normalize_for_higgs(tgt_text)},
        {"role": "assistant", "content": [{"type": "audio"}]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ref_f, tgt_f = build_delay_pattern(ref_codes), build_delay_pattern(tgt_codes)
    sizes = iter([ref_f.shape[0], tgt_f.shape[0]])
    text = re.sub(re.escape(processor.audio_token), lambda _: processor.get_audio_tokens(next(sizes)), prompt)
    enc = processor.tokenizer(text, padding=True, padding_side="left", return_tensors="pt")

    frames = torch.cat([ref_f, tgt_f], dim=0)  # (N, 8)
    N = frames.shape[0]
    labels = torch.full_like(frames, IGNORE)
    tgt_lab = tgt_f.clone()
    tgt_lab[tgt_lab == BOS_ID] = IGNORE
    if not eos_labels:
        tgt_lab[tgt_lab == EOS_ID] = IGNORE
    labels[ref_f.shape[0]:] = tgt_lab

    cfg_ids = enc["input_ids"][0]
    apos = (cfg_ids == processor.audio_token_id) | (cfg_ids == processor.audio_delay_token_id)
    if int(apos.sum()) != N:
        raise RuntimeError(f"audio positions {int(apos.sum())} != audio frames {N}")
    labels_pos = torch.full((cfg_ids.shape[0], NUM_CB), IGNORE, dtype=torch.long)
    labels_pos[apos] = labels
    return {
        "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
        "audio_input_ids": frames[None], "audio_input_ids_mask": torch.ones(1, N, dtype=torch.bool),
        "audio_labels": labels[None], "labels_pos": labels_pos, "n_ref_frames": ref_f.shape[0],
    }


MODEL_KEYS = ("input_ids", "attention_mask", "audio_input_ids", "audio_input_ids_mask")


def forward_logits(model, ex, device):
    inputs = {k: ex[k].to(device) for k in MODEL_KEYS}
    out = model(**inputs, use_cache=False)
    L = ex["input_ids"].shape[1]
    cb_size = out.logits.shape[-1] // NUM_CB
    return out.logits.reshape(L, NUM_CB, cb_size)


def example_ce(model, ex, device, labels_pos=None):
    """Per-codebook summed CE and valid-token counts over the labelled span. Differentiable."""
    import torch
    import torch.nn.functional as F

    logits = forward_logits(model, ex, device)
    lab = (ex["labels_pos"] if labels_pos is None else labels_pos).to(device)
    pred, tgt = logits[:-1].float(), lab[1:]
    sums, counts = [], []
    for c in range(NUM_CB):
        valid = tgt[:, c] != IGNORE
        counts.append(valid.sum())
        sums.append(F.cross_entropy(pred[:, c, :], tgt[:, c], ignore_index=IGNORE, reduction="sum"))
    return torch.stack(sums), torch.stack(counts)


# ----------------------------------------------------------------------------- reference pool
class RefPool:
    """Draws training references: k (1..5) other-sentence clips from the pool, stitched like make_refs."""

    def __init__(self, clips: dict, pool_ids: list[str]):
        self.clips = clips
        self.pool_ids = [u for u in pool_ids if clips[u]["duration_sec"] >= 1.0]
        self._waves: dict[str, np.ndarray] = {}

    def wave(self, utt_id: str) -> np.ndarray:
        if utt_id not in self._waves:
            self._waves[utt_id] = load_wave(self.clips[utt_id]["audio_path"])
        return self._waves[utt_id]

    def sample(self, rng: random.Random, target_id: str, k: int | None = None, max_k: int = 5):
        """-> (ref_wave, ref_text, source_ids). Distinct sentences, none equal to the target's."""
        k = k or rng.randint(1, max_k)
        tgt_norm = self.clips[target_id]["norm"]
        order = [u for u in self.pool_ids if self.clips[u]["norm"] != tgt_norm]
        rng.shuffle(order)
        picked, norms, total = [], set(), 0.0
        for u in order:
            c = self.clips[u]
            if c["norm"] in norms or total + c["duration_sec"] > MAX_STITCH_SOURCE_SEC:
                continue
            picked.append(u)
            norms.add(c["norm"])
            total += c["duration_sec"]
            if len(picked) == k:
                break
        # make_refs writes even a 1-clip reference through `stitch`, so it always takes this path
        wave = stitch_waves([self.wave(u) for u in picked])
        text = " ".join(self.clips[u]["transcript"].strip() for u in picked)
        return wave, text, picked


def cosine_lr(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
