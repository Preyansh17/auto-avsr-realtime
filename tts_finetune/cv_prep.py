"""5-fold cross-fitting splits for the larger-held-out-set experiment (README, "Follow-up experiments", experiment K).

    python cv_prep.py            # writes data/cv/splits_cv{0..4}.json and data/cv/cv_folds.json

Pool = every sentence of a live (non-ASR-test) clip that the zero-shot runs generated (paired baselines exist) and that
is not the source sentence of ANY inference reference, i.e. every reference in gen_runs/_refs/refs.json AND in
refs/refs_ft.json (ft_stitch5), so every reference clip is in every training set and never in a held-out fold, minus the
tts_val sentences (they chose the epochs, so they stay in training). Folds are split by SENTENCE (all repeats of a
sentence together), stratified by word-count bin, seed 20261007. Each splits_cv<k>.json has the splits.json schema
(tts_train / tts_val / tts_test / clips / log), so train_lora.py (--splits) and build_plan_ft.py (--splits-file) read it
unchanged. For fold k: tts_test = clips of fold k's sentences, tts_train = every other live clip, tts_val = up to 22
of fold k's own clips used ONLY for the diagnostic validation-loss log (nothing is selected on it).
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_data as P  # noqa: E402

K = 5
CV_SEED = 20261007
N_LOG_VAL = 22


def main() -> None:
    sp = json.load(open(f"{P.OUT}/splits.json"))
    clips = sp["clips"]  # live clips only (ASR-test sentences already dropped by prep_data split)
    norms = {u: c["norm"] for u, c in clips.items()}
    asr_dropped = set(sp["asr_dropped_utt_ids"])
    assert not (set(clips) & asr_dropped), "ASR-test clip present among live clips"

    gen_norms = {s["norm"] for s in json.load(open(P.SENTENCES))}
    by_path = {c["audio_path"]: c["norm"] for c in clips.values()}
    ref_src: set[str] = set()
    ref_clips: set[str] = set()  # source clip ids of EVERY inference reference (they must be in every training set)
    ref_files = [P.BASE_REFS, f"{P.REFS_OUT}/refs_ft.json"]  # refs_ft.json holds ft_stitch5 (not in BASE_REFS): reviewer-found leak
    for rf in ref_files:
        rj = json.load(open(rf))
        for _, v in (rj.get("refs") or rj).items():
            if isinstance(v, dict):
                paths = [q for q in v.get("source_audio_paths", []) if q in by_path]
                ref_src |= {by_path[q] for q in paths}
                ref_clips |= {u for u, c in clips.items() if c["audio_path"] in set(paths)}
    assert "ft_stitch5" in (json.load(open(ref_files[1])).get("refs") or json.load(open(ref_files[1]))), "ft_stitch5 missing"
    pool = sorted(set(norms.values()) & gen_norms - ref_src)
    orig_pool = len(sp["log"]["heldout_pool_sentences"]) if isinstance(sp["log"]["heldout_pool_sentences"], list) else sp["log"]["heldout_pool_sentences"]
    assert len(pool) <= orig_pool, (len(pool), orig_pool)
    print(f"pool {len(pool)} sentences (original pool {orig_pool}; {orig_pool - len(pool)} more removed because they are sources of refs_ft.json references)")
    val_norms = {norms[u] for u in sp["tts_val"]}
    pool_cv = sorted(set(pool) - val_norms)
    assert len(pool_cv) == len(pool) - len(val_norms)

    rng = random.Random(CV_SEED)
    bins: dict[int, list[str]] = collections.defaultdict(list)
    for n in pool_cv:
        bins[P._wc_bin(n)].append(n)
    order: list[str] = []
    for b in sorted(bins):
        rng.shuffle(bins[b])
        order += bins[b]
    fold_of = {n: i % K for i, n in enumerate(order)}  # stratified round-robin over the bin-sorted, shuffled list
    folds = [sorted(n for n, f in fold_of.items() if f == k) for k in range(K)]
    assert sum(len(f) for f in folds) == len(pool_cv) and len({n for f in folds for n in f}) == len(pool_cv)

    out_dir = f"{P.OUT}/cv"
    os.makedirs(out_dir, exist_ok=True)
    summary = {"seed": CV_SEED, "k": K, "pool_sentences": len(pool), "val_sentences_kept_in_training": len(val_norms),
               "cv_sentences": len(pool_cv), "folds": {}}
    for k in range(K):
        held = set(folds[k])
        test = sorted(u for u in clips if norms[u] in held)
        train = sorted(u for u in clips if norms[u] not in held)
        val = test[:N_LOG_VAL]
        # hard checks: leakage, ASR guard, references, completeness
        tn, trn = {norms[u] for u in test}, {norms[u] for u in train}
        assert not (tn & trn), f"fold {k}: sentence overlap between train and test"
        assert not (set(test) & set(train)) and not (set(test + train) & asr_dropped)
        assert not (tn & ref_src), f"fold {k}: a reference-source sentence is held out"
        assert ref_clips <= set(train), f"fold {k}: a reference source clip is not in the training set"
        assert not (ref_clips & set(test)), f"fold {k}: a reference source clip is in the held-out fold"
        assert tn == held and tn <= gen_norms
        assert all(norms[u] in trn for u in sp["tts_val"]), f"fold {k}: tts_val clips must be in training"
        h = hashlib.sha1(json.dumps({"tts_train": train, "tts_test": test}).encode()).hexdigest()[:12]
        log = {"cv_fold": k, "cv_seed": CV_SEED, "split_hash": h, "n_train": len(train), "n_test": len(test),
               "n_test_sentences": len(held), "train_minutes": round(sum(clips[u]["duration_sec"] for u in train) / 60, 2),
               "parent_split_hash": sp["log"]["split_hash"],
               "note": "tts_val here is a diagnostic subset of the held-out fold (val-loss log only, no selection)"}
        json.dump({"tts_train": train, "tts_val": val, "tts_test": test, "clips": clips, "log": log,
                   "asr_dropped_utt_ids": sp["asr_dropped_utt_ids"]}, open(f"{out_dir}/splits_cv{k}.json", "w"), indent=1)
        summary["folds"][str(k)] = {**log, "sentences": folds[k]}
        print(f"fold {k}: {len(held)} sentences, {len(test)} test clips, {len(train)} train clips "
              f"({log['train_minutes']} min), hash {h}")
    # every pool_cv sentence is held out in exactly one fold
    cnt = collections.Counter(n for f in folds for n in f)
    assert set(cnt) == set(pool_cv) and set(cnt.values()) == {1}
    json.dump(summary, open(f"{out_dir}/cv_folds.json", "w"), indent=1)
    print(f"{len(pool_cv)} sentences over {K} folds -> {out_dir}")


if __name__ == "__main__":
    main()
