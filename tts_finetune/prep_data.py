"""Phase 1 data prep for the Higgs LoRA finetune (patient bingge, TTS only).

    python prep_data.py split      # stdlib only: splits.json (leakage guard, held-out sentences)
    python prep_data.py split --asr-holdout test   # reproduce the frozen 2026-10-06 split (B_r64 adapter)
    python prep_data.py tokenize   # audio tokenizer (GPU if available): data/codes/<utt_id>.pt
    python prep_data.py refs       # fixed inference references built only from tts_train clips

Leakage guard: every clip whose sentence is in the ASR val or test split is dropped (default --asr-holdout val,test).
ASR val selects checkpoints and filter thresholds, so a TTS model trained on it makes val easier for synthetic arms
(asr_aug audit, 2026-10-14). The frozen B_r64 adapter used --asr-holdout test; 37/40 ASR val clips were in its tts_train.

Only READS /scratch/th3482 (patient manifest, ASR split files); everything is written to OUT.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from higgs_ft_lib import normalize_text  # noqa: E402

TH_BASE = "/scratch/th3482/TORGO/runs/patient_higgs_whisper_20260702_072656"
MANIFEST = f"{TH_BASE}/manifests/patient_real_manifest.csv"
ASR_SPLIT = "/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1/split_assignments.csv"
SENTENCES = "/scratch/pa2753/gen_runs/sentences.json"        # the 308 sentences every zero-shot run generated
BASE_REFS = "/scratch/pa2753/gen_runs/_refs/refs.json"
OUT = os.environ.get("HIGGS_FT_DATA", "/scratch/pa2753/higgs_ft/data")
REFS_OUT = os.environ.get("HIGGS_FT_REFS", "/scratch/pa2753/higgs_ft/refs")

SPLIT_SEED = 20261005
N_TEST, N_VAL = 40, 20
WC_BINS = [(1, 3), (4, 5), (6, 8), (9, 99)]


def _wc_bin(norm: str) -> int:
    n = len(norm.split())
    return next(i for i, (lo, hi) in enumerate(WC_BINS) if lo <= n <= hi)


def _allocate(sizes: list[int], total: int) -> list[int]:
    """Largest-remainder proportional allocation of `total` slots over bins of the given sizes."""
    s = sum(sizes)
    raw = [total * x / s for x in sizes]
    alloc = [min(int(r), x) for r, x in zip(raw, sizes)]
    order = sorted(range(len(sizes)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    for i in order * 3:
        if sum(alloc) >= total:
            break
        if alloc[i] < sizes[i]:
            alloc[i] += 1
    return alloc


def do_split(asr_holdout: set[str], force: bool = False) -> None:
    rows = list(csv.DictReader(open(MANIFEST)))
    clips = {}
    for r in rows:
        clips[r["utt_id"]] = {"audio_path": r["audio_path"], "transcript": r["transcript"].strip(),
                              "norm": normalize_text(r["transcript"]), "duration_sec": float(r["duration_sec"]),
                              "manifest_split": r["split"]}
    log = {"manifest_clips": len(clips), "manifest_unique_sentences": len({c["norm"] for c in clips.values()})}

    # 1. ASR-leakage guard: no clip whose sentence is in a held-out ASR split (val and test by default).
    asr = list(csv.DictReader(open(ASR_SPLIT)))
    asr_test = {normalize_text(r["transcript"]) for r in asr if r["split"] in asr_holdout}
    log["asr_holdout"] = sorted(asr_holdout)
    log["asr_holdout_sentences"] = {f"{s}/{d}": len({normalize_text(r["transcript"]) for r in asr
                                                     if r["split"] == s and r["domain"] == d})
                                    for s in sorted(asr_holdout) for d in sorted({r["domain"] for r in asr})}
    dropped = sorted(u for u, c in clips.items() if c["norm"] in asr_test)
    log["asr_holdout_clips_dropped"] = len(dropped)
    log["asr_holdout_unmatched_sentences"] = sorted(asr_test - {c["norm"] for c in clips.values()})
    live = {u: c for u, c in clips.items() if u not in set(dropped)}

    # 2. held-out pool: sentences the zero-shot runs generated (free paired baselines), minus the
    #    source sentences of every reference clip.
    gen_norms = {s["norm"] for s in json.load(open(SENTENCES))}
    by_path = {c["audio_path"]: c["norm"] for c in clips.values()}
    ref_src = set()
    for k, v in json.load(open(BASE_REFS)).items():
        if isinstance(v, dict):
            ref_src |= {by_path[p] for p in v.get("source_audio_paths", []) if p in by_path}
    log["ref_source_sentences_excluded"] = len(ref_src)
    pool = sorted({c["norm"] for c in live.values()} & gen_norms - ref_src)
    log["heldout_pool_sentences"] = len(pool)

    rng = random.Random(SPLIT_SEED)
    bins: dict[int, list[str]] = collections.defaultdict(list)
    for n in pool:
        bins[_wc_bin(n)].append(n)
    ids = sorted(bins)
    for b in ids:
        rng.shuffle(bins[b])
    sizes = [len(bins[b]) for b in ids]
    a_test = _allocate(sizes, N_TEST)
    a_val = _allocate([s - t for s, t in zip(sizes, a_test)], N_VAL)
    test_n, val_n = [], []
    for b, t, v in zip(ids, a_test, a_val):
        test_n += bins[b][:t]
        val_n += bins[b][t : t + v]
    log["word_count_bins"] = {f"{WC_BINS[b][0]}-{WC_BINS[b][1]}": {"pool": len(bins[b]), "test": t, "val": v}
                              for b, t, v in zip(ids, a_test, a_val)}
    assert len(test_n) == N_TEST and len(val_n) == N_VAL, (len(test_n), len(val_n))
    held = set(test_n) | set(val_n)

    # 3. split clips: every repeat of a held-out sentence leaves training.
    split = {"tts_train": [], "tts_val": [], "tts_test": []}
    for u in sorted(live):
        n = live[u]["norm"]
        split["tts_test" if n in test_n else "tts_val" if n in val_n else "tts_train"].append(u)
    clips_out = {u: live[u] for u in sorted(live)}

    # 4. hard checks
    norms = {k: {clips_out[u]["norm"] for u in v} for k, v in split.items()}
    assert not (norms["tts_train"] & norms["tts_val"]) and not (norms["tts_train"] & norms["tts_test"]) \
        and not (norms["tts_val"] & norms["tts_test"]), "split overlap"
    assert not any(norms[k] & asr_test for k in norms), "held-out ASR sentence leaked into TTS splits"
    assert not any(clips_out[u]["norm"] in ref_src for u in split["tts_test"] + split["tts_val"])
    assert held <= gen_norms

    h = hashlib.sha1(json.dumps({k: sorted(v) for k, v in split.items()}).encode()).hexdigest()[:12]
    log.update({"clips": {k: len(v) for k, v in split.items()},
                "unique_sentences": {k: len(norms[k]) for k in norms},
                "train_minutes": round(sum(clips_out[u]["duration_sec"] for u in split["tts_train"]) / 60, 2),
                "split_seed": SPLIT_SEED, "split_hash": h})
    old = f"{OUT}/splits.json"
    if os.path.exists(old) and json.load(open(old))["log"].get("split_hash") != h and not force:
        sys.exit(f"{old} exists with a different split (hash {h} new); set HIGGS_FT_DATA to a new dir or pass --force")
    os.makedirs(OUT, exist_ok=True)
    json.dump({**split, "clips": clips_out, "log": log, "asr_dropped_utt_ids": dropped},
              open(f"{OUT}/splits.json", "w"), indent=1)
    print(json.dumps(log, indent=1))


def _splits() -> dict:
    return json.load(open(f"{OUT}/splits.json"))


def do_tokenize() -> None:
    import torch
    from transformers import AutoProcessor

    from higgs_ft_lib import MODEL, encode_wave, load_wave

    sp = _splits()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    processor.audio_tokenizer.to(device)
    processor.audio_tokenizer.eval()
    os.makedirs(f"{OUT}/codes", exist_ok=True)
    index = {}
    ids = sp["tts_train"] + sp["tts_val"] + sp["tts_test"]
    for i, u in enumerate(ids, 1):
        p = f"{OUT}/codes/{u}.pt"
        if os.path.exists(p):
            codes = torch.load(p)
        else:
            codes = encode_wave(processor, load_wave(sp["clips"][u]["audio_path"]))
            torch.save(codes.to(torch.int16), p + ".tmp")
            os.replace(p + ".tmp", p)
        index[u] = int(codes.shape[1])
        if i % 50 == 0:
            print(f"  tokenized {i}/{len(ids)} on {device}", flush=True)
    json.dump(index, open(f"{OUT}/codes_index.json", "w"))
    fr = [index[u] for u in sp["tts_train"]]
    print(f"codes for {len(index)} clips; train frames min/median/max = {min(fr)}/{sorted(fr)[len(fr)//2]}/{max(fr)} "
          f"(25 fps)", flush=True)


def do_refs() -> None:
    """Inference references built only from tts_train: one new 5-clip stitch, plus any existing reference
    (stitch5/stitch10/lowpitch*/median*) whose source clips all lie in tts_train."""
    sys.path.insert(0, os.environ.get("HIGGS_GEN", "/scratch/pa2753/higgs_gen"))
    from make_refs import stitch

    sp = _splits()
    clips, train = sp["clips"], sp["tts_train"]
    train_paths = {clips[u]["audio_path"] for u in train}
    os.makedirs(REFS_OUT, exist_ok=True)
    refs: dict = {}

    rng = random.Random(SPLIT_SEED + 5)
    cands = [u for u in train if 4.0 <= clips[u]["duration_sec"] <= 12.0]
    rng.shuffle(cands)
    picked, norms, total = [], set(), 0.0
    for u in cands:
        c = clips[u]
        if c["norm"] in norms or total + c["duration_sec"] > 45.0:
            continue
        picked.append(u)
        norms.add(c["norm"])
        total += c["duration_sec"]
        if len(picked) == 5:
            break
    assert len(picked) == 5 and set(picked) <= set(train), picked
    out = f"{REFS_OUT}/ft_stitch5.wav"
    st = stitch([clips[u]["audio_path"] for u in picked], out)
    refs["ft_stitch5"] = {"audio_path": out, "transcript": " ".join(clips[u]["transcript"] for u in picked),
                          "source_utt_ids": picked, "source_audio_paths": [clips[u]["audio_path"] for u in picked],
                          "duration_sec": st["duration_sec"], "origin": "new, tts_train only"}

    skipped = {}
    for k, v in json.load(open(BASE_REFS)).items():
        if not isinstance(v, dict) or "source_audio_paths" not in v or k.startswith("M04") or k == "original_1clip":
            continue
        miss = [p for p in v["source_audio_paths"] if p not in train_paths]
        if miss:
            skipped[k] = f"{len(miss)}/{len(v['source_audio_paths'])} source clips not in tts_train"
        else:
            refs[k] = {**{x: v[x] for x in ("audio_path", "transcript", "source_audio_paths", "duration_sec")},
                       "origin": "existing zero-shot reference, all sources in tts_train"}
    json.dump({"refs": refs, "skipped": skipped, "split_hash": sp["log"]["split_hash"]},
              open(f"{REFS_OUT}/refs_ft.json", "w"), indent=1)
    for k, v in refs.items():
        print(f"{k:11s} {v['duration_sec']:6.1f}s  {len(v['source_audio_paths'])} clips  [{v['origin']}]")
    for k, why in skipped.items():
        print(f"{k:11s} SKIPPED: {why}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["split", "tokenize", "refs"])
    ap.add_argument("--asr-holdout", default="val,test", help="ASR splits whose sentences TTS never trains on")
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits.json with a different split")
    a = ap.parse_args()
    if a.stage == "split":
        do_split(set(a.asr_holdout.split(",")), a.force)
    else:
        {"tokenize": do_tokenize, "refs": do_refs}[a.stage]()
