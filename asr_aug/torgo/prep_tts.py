"""Per-speaker Higgs LoRA data for TORGO, in tts_finetune's splits.json / refs_ft.json format.

    python prep_tts.py            # all 8 speakers -> /scratch/pa2753/torgo_aug/higgs/<spk>/{data/splits.json, refs/}

Only ASR-train clips (items.csv split == train) of the speaker are used, so the TTS never hears an ASR val or test
clip or sentence (the patient study's val contamination). tts_val = 10% of the speaker's ASR-train texts, used only for
the per-epoch val loss log; the adapter epoch is fixed in advance (B recipe, epoch 8), never chosen on val.
Generation reference: ft_stitch5 = 5 distinct-text tts_train clips (sentences first, 1.5-12 s, <= 45 s total).
Then, per speaker: HIGGS_FT_DATA=<spk>/data python tts_finetune/prep_data.py tokenize.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import random
import sys

DATA = "/scratch/pa2753/torgo_aug/data"
OUT = "/scratch/pa2753/torgo_aug/higgs"
SEED = 20261015
sys.path.insert(0, "/scratch/pa2753/higgs_gen")


def main() -> None:
    from make_refs import stitch

    items = list(csv.DictReader(open(f"{DATA}/items.csv")))
    summary = {}
    for spk in sorted({i["speaker"] for i in items}):
        rows = [i for i in items if i["speaker"] == spk and i["split"] == "train"]
        clips = {}
        for i in rows:
            u = f"{spk}_{i['session']}_{os.path.basename(i['wav'])[:-4]}"
            clips[u] = {"audio_path": i["wav"], "transcript": i["prompt_raw"].strip(), "norm": i["text"],
                        "duration_sec": float(i["duration"]), "type": i["type"]}
        rng = random.Random(f"{SEED}-{spk}")
        texts = sorted({c["norm"] for c in clips.values()})
        rng.shuffle(texts)
        val_texts = set(texts[: max(2, round(0.1 * len(texts)))])
        split = {"tts_train": sorted(u for u, c in clips.items() if c["norm"] not in val_texts),
                 "tts_val": sorted(u for u, c in clips.items() if c["norm"] in val_texts), "tts_test": []}
        h = hashlib.sha1(json.dumps(split).encode()).hexdigest()[:12]
        tr = split["tts_train"]
        log = {"speaker": spk, "clips": {k: len(v) for k, v in split.items()}, "split_seed": SEED, "split_hash": h,
               "train_minutes": round(sum(clips[u]["duration_sec"] for u in tr) / 60, 2),
               "train_sentence_clips": sum(clips[u]["type"] == "sentence" for u in tr)}
        os.makedirs(f"{OUT}/{spk}/data", exist_ok=True)
        os.makedirs(f"{OUT}/{spk}/refs", exist_ok=True)
        os.makedirs(f"{OUT}/{spk}/runs", exist_ok=True)
        json.dump({**split, "clips": clips, "log": log, "asr_dropped_utt_ids": []},
                  open(f"{OUT}/{spk}/data/splits.json", "w"), indent=1)
        # generation reference: sentences first, then words, 1.5-12 s each, distinct texts, <= 45 s
        cands = [u for u in tr if 1.5 <= clips[u]["duration_sec"] <= 12.0]
        rng.shuffle(cands)
        cands.sort(key=lambda u: clips[u]["type"] != "sentence")
        picked, norms, total = [], set(), 0.0
        for u in cands:
            if clips[u]["norm"] in norms or total + clips[u]["duration_sec"] > 45.0:
                continue
            picked.append(u)
            norms.add(clips[u]["norm"])
            total += clips[u]["duration_sec"]
            if len(picked) == 5:
                break
        assert len(picked) == 5, (spk, len(picked))
        out = f"{OUT}/{spk}/refs/ft_stitch5.wav"
        st = stitch([clips[u]["audio_path"] for u in picked], out)
        refs = {"ft_stitch5": {"audio_path": out, "transcript": " ".join(clips[u]["transcript"] for u in picked),
                               "source_utt_ids": picked, "source_audio_paths": [clips[u]["audio_path"] for u in picked],
                               "duration_sec": st["duration_sec"], "origin": "TORGO ASR-train only"}}
        json.dump({"refs": refs, "skipped": {}, "split_hash": h}, open(f"{OUT}/{spk}/refs/refs_ft.json", "w"), indent=1)
        log["ref_sec"] = round(st["duration_sec"], 1)
        log["ref_sentences"] = sum(clips[u]["type"] == "sentence" for u in picked)
        summary[spk] = log
        print(json.dumps(log))
    # hard check: nothing outside ASR train
    nontrain = {i["wav"] for i in items if i["split"] != "train"}
    for spk in summary:
        sp = json.load(open(f"{OUT}/{spk}/data/splits.json"))
        assert not ({c["audio_path"] for c in sp["clips"].values()} & nontrain)
        r = json.load(open(f"{OUT}/{spk}/refs/refs_ft.json"))["refs"]["ft_stitch5"]
        assert not (set(r["source_audio_paths"]) & nontrain)
    json.dump(summary, open(f"{OUT}/prep_summary.json", "w"), indent=1)
    print("OK: all TTS clips and references are ASR-train only")


if __name__ == "__main__":
    main()
