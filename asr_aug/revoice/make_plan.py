"""Re-voicing plan (design B, 2026-10-20): every ASR-train clip says its own sentence again, with ITSELF as the Higgs
reference (audio + transcript), through the clean-split adapter (TTS never saw ASR val/test).

    python make_plan.py [--copies 2]   # -> data/plan.json, data/plan_audit.json

One unit per (clip, copy): out_root gen_runs/FT_RV_clean_c<copy>, prompt_id rv_<sha of the clip's rel_path>, seed base
SEED1 + copy - 1 (gen_higgs then offsets it per prompt id). Text = the TTS manifest transcript of the same recording
(matched by file stem; casing kept), whose normalised form must equal the ASR label text. Reference audio = the
clip's 16 kHz split_v1 wav under the ASR root. Asserted: every reference and every text is ASR-train. Train clips whose
sentence is also an ASR val/test sentence are skipped (README amendment 2), so no text is in the held-out sets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

HIGGS_FT = "/scratch/pa2753/higgs_ft"
sys.path.insert(0, HIGGS_FT)
from higgs_ft_lib import MODEL, SYSTEM, normalize_text  # noqa: E402

R = "/scratch/pa2753/asr_aug/revoice"
ROOT = "/scratch/pa2753/asr_aug/root"
LABELS = "/scratch/pa2753/avsr_realtime/labels/splitv1/merged"
MANIFEST = "/scratch/th3482/TORGO/runs/patient_higgs_whisper_20260702_072656/manifests/patient_real_manifest.csv"
ADAPTER = "/scratch/pa2753/higgs_ft_clean/runs/B_r64/epoch_08"
CLEAN_SPLITS = "/scratch/pa2753/higgs_ft_clean/data/splits.json"
GEN = "/scratch/pa2753/gen_runs"
SPM = "/home/pa2753/auto-avsr-realtime/spm/spm_unigram_1023.model"
SEED1 = 260619825
DEFAULT = {"temperature": 1.0, "top_k": 50, "top_p": 0.95, "max_new_tokens": 2048,
           "ras_win_len": 7, "ras_win_max_num_repeat": 2}


def rows(p):
    return [l.split(",") for l in open(p).read().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=2)
    a = ap.parse_args()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=SPM)
    text_of = lambda tok: sp.decode([int(t) for t in tok.split()])  # noqa: E731
    held = {text_of(r[3]) for s in ("val", "test") for r in rows(f"{LABELS}/{s}_spm1023.csv")}
    man = {os.path.splitext(os.path.basename(m["audio_path"]))[0]: m for m in csv.DictReader(open(MANIFEST))}
    # the adapter must be the clean one: its split excludes every ASR val/test sentence
    clean = json.load(open(CLEAN_SPLITS))
    assert set(clean["log"]["asr_holdout"]) == {"val", "test"}, clean["log"]["asr_holdout"]
    cfg = json.load(open(os.path.join(os.path.dirname(ADAPTER), "config.json")))
    assert cfg["split_hash"] == clean["log"]["split_hash"], (cfg["split_hash"], clean["log"]["split_hash"])
    tts_norms = {clean["clips"][u]["norm"] for u in clean["tts_train"]}
    assert not (tts_norms & {normalize_text(h) for h in held}), "adapter trained on an ASR val/test sentence"

    units, audit = [], {"clips": 0, "copies": a.copies, "adapter": ADAPTER, "split_hash": cfg["split_hash"],
                        "skipped_heldout_text": []}
    for ds, rel, _, tok in rows(f"{LABELS}/train_spm1023.csv"):
        norm = text_of(tok)
        stem = os.path.splitext(os.path.basename(rel))[0]
        m = man[stem]
        assert normalize_text(m["transcript"]) == norm, (m["transcript"], norm)
        if norm in held:  # split_v1 is split by recording: 11 train clips say one of 10 val sentences (0 test)
            audit["skipped_heldout_text"].append(f"{ds}/{rel}")
            continue
        wav = os.path.splitext(f"{ROOT}/{ds}/{rel}")[0] + ".wav"
        assert os.path.isfile(wav), wav
        pid = "rv_" + hashlib.sha1(f"{ds}/{rel}".encode()).hexdigest()[:12]
        audit["clips"] += 1
        for c in range(1, a.copies + 1):
            variant = f"FT_RV_clean_c{c}"
            units.append({"name": f"{variant}__{pid}", "variant": variant, "out_root": f"{GEN}/{variant}",
                          "speaker": "bingge", "ref_wav": wav, "ref_text": m["transcript"], "ref_count": 1,
                          "params": dict(DEFAULT), "system_prompt": SYSTEM, "seed_base": SEED1 + c - 1,
                          "adapter": ADAPTER, "src_label": f"{ds},{rel}",
                          "prompts": [{"prompt_id": pid, "text": m["transcript"]}]})
    os.makedirs(f"{R}/data", exist_ok=True)
    json.dump({"model_path": MODEL, "units": units}, open(f"{R}/data/plan.json", "w"), indent=0)
    audit.update({"units": len(units), "heldout_sentences": len(held), "tts_train_clips": len(clean["tts_train"])})
    json.dump(audit, open(f"{R}/data/plan_audit.json", "w"), indent=1)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
