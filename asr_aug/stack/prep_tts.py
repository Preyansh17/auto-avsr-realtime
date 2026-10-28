"""Per-patient Higgs LoRA data for the item-3 stack (PLAN.md 3.3): semg1 Oct 1 (s1) and semg2 (s2), in tts_finetune's
splits.json / refs_ft.json format. Same as asr_aug/torgo/prep_tts.py, with the shared role files as the source.

    python prep_tts.py            # -> /scratch/pa2753/stack/higgs/{s1,s2}/{data/splits.json, refs/}

Only the patient's ASR-train role clips are used, so the TTS never hears an ASR val / test / oov clip or sentence.
tts_val = 10% of the train texts, loss logging only; the adapter epoch is fixed in advance (B recipe, epoch 8).
Generation reference: ft_stitch5 = 5 distinct-text train clips, 1.5-12 s, <= 45 s total.
Then, per patient: HIGGS_FT_DATA=<p>/data python tts_finetune/prep_data.py tokenize (tts_train.sbatch does this).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

import soundfile as sf

ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
OUT = "/scratch/pa2753/stack/higgs"
SEED = 20261026
PATIENTS = ["s1", "s2"]
sys.path.insert(0, "/scratch/pa2753/higgs_gen")
sys.path.insert(0, "/scratch/pa2753/higgs_ft")


def role(p: str, r: str) -> list[dict]:
    return [json.loads(l) for l in open(f"{ROLES}/{p}_{r}.jsonl")]


def main() -> None:
    from make_refs import stitch
    from prep_data import normalize_text

    summary = {}
    for p in PATIENTS:
        clips = {}
        for x in role(p, "train"):
            clips[f"{p}_{x['take_id']}"] = {"audio_path": x["audio_path"], "transcript": x["transcript"].strip(),
                                           "norm": normalize_text(x["transcript"]),
                                           "duration_sec": sf.info(x["audio_path"]).duration, "type": "sentence"}
        held = {normalize_text(x["transcript"]) for r in ("val", "test", "oov") for x in role(p, r)}
        held_audio = {x["audio_path"] for r in ("val", "test", "oov") for x in role(p, r)}
        assert not ({c["norm"] for c in clips.values()} & held), f"{p}: train text overlaps val/test/oov"
        rng = random.Random(f"{SEED}-{p}")
        texts = sorted({c["norm"] for c in clips.values()})
        rng.shuffle(texts)
        val_texts = set(texts[: max(2, round(0.1 * len(texts)))])
        split = {"tts_train": sorted(u for u, c in clips.items() if c["norm"] not in val_texts),
                 "tts_val": sorted(u for u, c in clips.items() if c["norm"] in val_texts), "tts_test": []}
        h = hashlib.sha1(json.dumps(split).encode()).hexdigest()[:12]
        tr = split["tts_train"]
        log = {"patient": p, "clips": {k: len(v) for k, v in split.items()}, "split_seed": SEED, "split_hash": h,
               "train_minutes": round(sum(clips[u]["duration_sec"] for u in tr) / 60, 2)}
        for d in ("data", "refs", "runs"):
            os.makedirs(f"{OUT}/{p}/{d}", exist_ok=True)
        json.dump({**split, "clips": clips, "log": log, "asr_dropped_utt_ids": []},
                  open(f"{OUT}/{p}/data/splits.json", "w"), indent=1)
        cands = [u for u in tr if 1.5 <= clips[u]["duration_sec"] <= 12.0]
        rng.shuffle(cands)
        picked, norms, total = [], set(), 0.0
        for u in cands:
            if clips[u]["norm"] in norms or total + clips[u]["duration_sec"] > 45.0:
                continue
            picked.append(u)
            norms.add(clips[u]["norm"])
            total += clips[u]["duration_sec"]
            if len(picked) == 5:
                break
        assert len(picked) == 5, (p, len(picked))
        out = f"{OUT}/{p}/refs/ft_stitch5.wav"
        st = stitch([clips[u]["audio_path"] for u in picked], out)
        refs = {"ft_stitch5": {"audio_path": out, "transcript": " ".join(clips[u]["transcript"] for u in picked),
                               "source_utt_ids": picked, "source_audio_paths": [clips[u]["audio_path"] for u in picked],
                               "duration_sec": st["duration_sec"], "origin": f"{p} ASR-train role only"}}
        json.dump({"refs": refs, "skipped": {}, "split_hash": h}, open(f"{OUT}/{p}/refs/refs_ft.json", "w"), indent=1)
        assert not ({c["audio_path"] for c in clips.values()} & held_audio)
        assert not (set(refs["ft_stitch5"]["source_audio_paths"]) & held_audio)
        log["ref_sec"] = round(st["duration_sec"], 1)
        summary[p] = log
        print(json.dumps(log))
    json.dump(summary, open(f"{OUT}/prep_summary.json", "w"), indent=1)
    print("OK: all TTS clips and references are ASR-train only")


if __name__ == "__main__":
    main()
