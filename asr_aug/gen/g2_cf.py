"""G2 Seed-VC counterfactuals (asr_aug/gen/README.md, amendment 3).

    TORGO pairs: the repo's own build_pairs_new + build_pairs_old (mode both, seed 1234), unchanged.
    Patient pairs (our adaptation; the text comes from the content source, so no TORGO prompt match is needed):
      (a) <X>_converted_from_<semg>/<v>_<uid>.wav: content = patient tts_train clip, timbre ref = a random >= 3-word
          utterance of timbre target X (the repo's 10 TARGETS_NEW); label = patient class (6 semg1, 7 semg2)
      (b) <semg>_converted/<v>_<uid>.wav: content = a random >= 3-word TORGO control utterance, timbre ref = the
          patient clip; label 0
    Patient clips: /scratch/pa2753/gen/g1/<v>/{wavs,manifest.tsv} (the Higgs adapters' tts_train), v in old, s1, s2.
    Leak guard: a patient pair whose text is a held-out sentence (check_leak.held_out) is dropped. 45 old clips have
    old_legal_val / old_legacy_val text, so their 450 (a) pairs are not synthesised; the real clips stay in training.

    python g2_cf.py --pairs-only                 # write cf_patient.jsonl and pair counts, no models
    python g2_cf.py --shard k --nshard N [--max-items M]   # convert (run from third_party/seed-vc)

Resumable: a pair whose output exists is skipped; each clip is written to a temp file and renamed. VC settings are the
repo's (30 diffusion steps, cfg 0.7, length 1.0, no f0, fp16).
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

G = "/scratch/pa2753/gen"
SVC = f"{G}/code/Controllable-Dysarthric-Speech-Synthesis/third_party/seed-vc"
sys.path.insert(0, SVC)
import generate_data as gd  # noqa: E402

sys.path.insert(0, f"{G}/code")
from check_leak import held_out, norm  # noqa: E402

TORGO = f"{G}/data/TORGO"
OUT = f"{G}/g2/converted"
CLASS = {"old": ("semg1", 6), "s1": ("semg1", 6), "s2": ("semg2", 7)}
MIN_WORDS = 3


def patient_clips():
    out = []
    for v in ("old", "s1", "s2"):
        for line in open(f"{G}/g1/{v}/manifest.tsv"):
            uid, text = line.rstrip("\n").split("\t")
            out.append({"voice": v, "uid": uid, "wav": f"{G}/g1/{v}/wavs/{uid}.wav", "text": text})
    return out


def build_pairs():
    rows = gd.scan_torgo(TORGO)
    rng = np.random.default_rng(1234)  # the repo's default seed and order (new, then old)
    torgo = [(s, r, o, None) for s, r, o in gd.build_pairs_new(rows, OUT, rng) + gd.build_pairs_old(rows, OUT, rng)]
    prng = np.random.default_rng(20261102)
    long_by_spk = {}
    for r in rows:
        if len(r["text"].split()) >= MIN_WORDS:
            long_by_spk.setdefault(r["spk"], []).append(r)
    controls = [r for s in gd.CONTROL for r in long_by_spk[s]]
    patient = []
    for c in patient_clips():
        semg, k = CLASS[c["voice"]]
        name = f"{c['voice']}_{c['uid']}.wav"
        for x in gd.TARGETS_NEW:
            ref = long_by_spk[x][int(prng.integers(0, len(long_by_spk[x])))]
            patient.append((c["wav"], ref["wav"], f"{OUT}/{x}_converted_from_{semg}/{name}",
                            {"text": c["text"], "label": k, "dir": f"{x}_converted_from_{semg}"}))
        src = controls[int(prng.integers(0, len(controls)))]
        patient.append((src["wav"], c["wav"], f"{OUT}/{semg}_converted/{name}",
                        {"text": src["text"], "label": 0, "dir": f"{semg}_converted"}))
    held = held_out()
    kept = [p for p in patient if norm(p[3]["text"]) not in held]
    print(f"leak guard: dropped {len(patient) - len(kept)} patient pairs with held-out text", flush=True)
    return torgo, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-only", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-items", type=int, default=0)
    a = ap.parse_args()
    torgo, patient = build_pairs()
    print(f"pairs: TORGO {len(torgo)}, patient {len(patient)}", flush=True)
    if a.pairs_only:
        os.makedirs(f"{G}/g2", exist_ok=True)
        with open(f"{G}/g2/cf_patient.jsonl", "w") as f:
            for s, r, o, m in patient:
                f.write(json.dumps({"src": s, "ref": r, "out": o, **m}) + "\n")
        return
    pairs = torgo + patient
    todo = [p for p in pairs[a.shard::a.nshard] if not os.path.exists(p[2])]
    if a.max_items:
        todo = todo[:a.max_items]
    print(f"shard {a.shard}/{a.nshard}: {len(todo)} to do", flush=True)
    import torch
    import inference as inf
    gd.device = inf.device = torch.device("cuda")
    inf.fp16 = True
    model, semantic_fn, _, vocoder_fn, campplus, mel_fn, _ = inf.load_models(
        types.SimpleNamespace(f0_condition=False, checkpoint=None, config=None, fp16=True))
    t0, failed = time.time(), 0
    for i, (src, ref, out, _) in enumerate(todo, 1):
        tmp = out[:-4] + ".tmp.wav"
        try:
            gd.convert_one(model, semantic_fn, vocoder_fn, campplus, mel_fn, gd.device, True, src, ref, tmp)
            os.replace(tmp, out)
        except Exception as e:
            failed += 1
            print(f"FAILED src={src} ref={ref} err={e!r}", flush=True)
        if i % 200 == 0:
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {el / i:.2f}s/clip eta {el / i * (len(todo) - i) / 60:.0f}min", flush=True)
    print(f"GEN_DONE failed={failed}", flush=True)


if __name__ == "__main__":
    main()
