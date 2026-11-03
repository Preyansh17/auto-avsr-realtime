"""Step 0 of PLAN.md 3.3 item 5: are our filtered Higgs clips easier / cleaner than the patients' real clips?

    python diag_synth.py run      # GPU: zero-shot large-v3 + WavLM-SV over real and synthetic clips -> data/diag_clips.csv
    python diag_synth.py report   # CPU: data/diag_report.md

Sets per patient (s1 = semg1 Oct 1, s2 = semg2): real = <p>_train role (240) and <p>_val (30); synthetic = every
stack clip (data/filter_clips.csv), split into kept / rejected and pool A (known-word) / B (open-vocab).
Metrics per clip:
  zs_wer      zero-shot openai/whisper-large-v3 WER (greedy, English): how impaired the speech sounds to a model that
              never saw this patient. The fine-tuned judge is not used for this: it trained on the real train clips.
  sec_per_word, pause_frac (1 - energy-VAD speech fraction, same VAD as the filter)
  spk_sim     cosine of WavLM-base-plus-sv x-vector to the centroid of that patient's real train clips
              (real train clips: leave-one-out centroid)
Decision rule, fixed before any number is read (2026-10-29): "synthetic is easier" for a patient if kept pool-A
zs_wer is below real-train zs_wer by 10 pp or more (mean over clips). Everything else is descriptive.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_synth import norm, speech_fraction, wer_parts  # noqa: E402  same normaliser, VAD and WER as the filter

S = "/scratch/pa2753/stack"
DATA = f"{S}/data"
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
OUT = f"{DATA}/diag_clips.csv"
# microsoft/wavlm-base-plus-sv ships only a .bin; the whisper_asr env torch is < 2.6, which transformers refuses for .bin,
# so it was re-saved as safetensors with the nemo31 env.
SV = "/scratch/pa2753/hf_models/wavlm-base-plus-sv-st"


def load_16k(path):
    import librosa
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav


def clip_list():
    rows = []
    for p in ("s1", "s2"):
        for split in ("train", "val"):
            for l in open(f"{ROLES}/{p}_{split}.jsonl"):
                r = json.loads(l)
                rows.append({"patient": p, "set": f"real_{split}", "pool": "", "kept": "",
                             "path": r["audio_path"], "ref": norm(r["transcript"])})
    for r in csv.DictReader(open(f"{DATA}/filter_clips.csv")):
        rows.append({"patient": r["patient"], "set": "synth", "pool": r["pool"], "kept": r["kept"],
                     "path": f"{S}/root/synth_higgs_b8/{r['rel_path']}", "ref": r["norm"]})
    return rows


def cmd_run(a):
    import torch
    from transformers import (AutoFeatureExtractor, WavLMForXVector, WhisperForConditionalGeneration,
                              WhisperProcessor)
    rows = clip_list()
    dev = "cuda"
    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", dtype=torch.float16).to(dev).eval()
    fe = AutoFeatureExtractor.from_pretrained(SV)
    sv = WavLMForXVector.from_pretrained(SV).to(dev).eval()
    emb = []
    for i in range(0, len(rows), a.batch):
        b = rows[i:i + a.batch]
        wavs = [load_16k(r["path"]) for r in b]
        feats = proc(wavs, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
        with torch.no_grad():
            ids = asr.generate(feats, language="english", task="transcribe", num_beams=1, max_new_tokens=128)
            # one clip at a time so batch padding never enters the x-vector; clips under 1 s are zero-padded to 1 s
            # (the TDNN kernel needs more frames than a near-empty clip gives)
            e = torch.cat([sv(**{k: v.to(dev) for k, v in fe(np.pad(w, (0, max(0, 16000 - len(w)))), sampling_rate=16000,
                                                               return_tensors="pt").items()}).embeddings for w in wavs])
        hyps = proc.batch_decode(ids, skip_special_tokens=True)
        emb.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        for r, w, h in zip(b, wavs, hyps):
            n = max(1, len(r["ref"].split()))
            err, _ins = wer_parts(r["ref"], norm(h))
            r.update(hyp=norm(h), zs_wer=round(err / n, 4), dur=round(len(w) / 16000, 3),
                     sec_per_word=round(len(w) / 16000 / n, 4), pause_frac=round(1 - speech_fraction(w, 16000), 4))
        print(f"{i + len(b)}/{len(rows)}", flush=True)
    E = np.concatenate(emb)
    for p in ("s1", "s2"):
        tr = [k for k, r in enumerate(rows) if r["patient"] == p and r["set"] == "real_train"]
        tot = E[tr].sum(0)
        for k, r in enumerate(rows):
            if r["patient"] != p:
                continue
            c = tot - E[k] if k in tr else tot  # leave-one-out for the real train clips themselves
            r["spk_sim"] = round(float(E[k] @ (c / np.linalg.norm(c))), 4)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


def cmd_report(_a):
    rows = list(csv.DictReader(open(OUT)))
    groups = [("real train", lambda r: r["set"] == "real_train"), ("real val", lambda r: r["set"] == "real_val"),
              ("synth A kept", lambda r: r["set"] == "synth" and r["pool"] == "A" and r["kept"] == "True"),
              ("synth A rejected", lambda r: r["set"] == "synth" and r["pool"] == "A" and r["kept"] != "True"),
              ("synth B kept", lambda r: r["set"] == "synth" and r["pool"] == "B" and r["kept"] == "True"),
              ("synth B rejected", lambda r: r["set"] == "synth" and r["pool"] == "B" and r["kept"] != "True")]
    keys = ["zs_wer", "sec_per_word", "pause_frac", "spk_sim"]
    out = ["# Step 0: Higgs synthetic vs real patient clips (`asr_aug/stack/diag_synth.py`)", "",
           "Means (median in brackets). zs_wer = zero-shot whisper-large-v3 WER, higher = sounds more impaired.", ""]
    verdict = []
    for p in ("s1", "s2"):
        out += [f"## {p}", "", "| set | n | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
        m = {}
        for name, f in groups:
            g = [r for r in rows if r["patient"] == p and f(r)]
            if not g:
                continue
            cells = []
            for k in keys:
                v = [float(r[k]) for r in g]
                cells.append(f"{statistics.mean(v):.3f} [{statistics.median(v):.3f}]")
            m[name] = statistics.mean(float(r["zs_wer"]) for r in g)
            out.append(f"| {name} | {len(g)} | " + " | ".join(cells) + " |")
        gap = m["real train"] - m["synth A kept"]
        verdict.append(f"- {p}: real-train zs_wer {m['real train']:.3f} vs kept pool-A {m['synth A kept']:.3f} "
                       f"(gap {gap * 100:+.1f} pp) -> {'EASIER (rule met)' if gap >= 0.10 else 'not easier by the rule'}")
        out.append("")
    out += ["## Decision rule (fixed before reading): synthetic easier if kept pool-A zs_wer <= real-train - 10 pp", ""] + verdict
    open(f"{DATA}/diag_report.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--batch", type=int, default=16)
    sub.add_parser("report")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)
