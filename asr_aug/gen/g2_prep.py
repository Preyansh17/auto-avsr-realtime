"""G2 training features (asr_aug/gen/README.md, amendment 3): the repo's scripts/prepare_torgo.py, extended with our
patient dirs and classes 6 (semg1) / 7 (semg2).

    dirs: the repo's 70 (15 real TORGO, 5 X_converted, 50 X_converted_from_Y; same prompt filter and labels)
        + semg1 (real old + s1 tts_train, label 6), semg2 (real s2 tts_train, label 7)
        + the g2_cf.py patient dirs from g2/cf_patient.jsonl (<X>_converted_from_semg{1,2} -> 6/7, semg{1,2}_converted -> 0)
    python g2_prep.py --shard k --nshard N    # GPU: features + per-dir 2% valid split for dirs[k::N] (resumable)
    python g2_prep.py --merge                 # CPU: speaker_info.json + pathology_embedding/mean_..._{0..7}.npy
A dir is finalised (info.json) only when all of its wavs exist, so prep can run before the counterfactuals finish.

Same per-dir logic as prepare_torgo.main (feature extractor, 0.2-20 s, 2% valid, medoid). One difference, so that dirs
can be processed in parallel: each dir's valid split uses its own rng (seed 91 + crc32 of the dir name) rather than one
rng shared across dirs in order.
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

G = "/scratch/pa2753/gen"
REPO = f"{G}/code/Controllable-Dysarthric-Speech-Synthesis"
sys.path.insert(0, f"{REPO}/scripts")
import prepare_torgo as pt  # noqa: E402

TORGO = f"{G}/data/TORGO"
CONV = f"{G}/g2/converted"
OUT = f"{G}/g2/prepared"
MODEL = f"{G}/models/IndexTTS-1.5"
CFG = f"{REPO}/configs/controllable_dysarthric_speech_synthesis.yaml"
NUM_CLASSES = 8


def all_dirs():
    """[(dir_name, label, [(key, wav, text)])] in a fixed order."""
    prompts = {spk: pt.load_prompts(TORGO, spk) for spk in pt.REAL_SPEAKERS}
    repo = [(s, f"{TORGO}/{s}", s) for s in pt.REAL_SPEAKERS]
    repo += [(f"{x}_converted", f"{CONV}/{x}_converted", x) for x in pt.CONTENT_SOURCES]
    repo += [(f"{x}_converted_from_{y}", f"{CONV}/{x}_converted_from_{y}", x)
             for x in pt.TIMBRE_TARGETS for y in pt.CONTENT_SOURCES]
    out = []
    for d, root, pspk in repo:
        ents = []
        if os.path.isdir(root):
            for session, mic, uid, wav in pt.iter_wavs(root):
                text = prompts[pspk].get((session, uid))
                if text is not None:
                    ents.append((f"{session}_{mic}_{uid}", wav, text))
        out.append((d, pt.pathology_of_dir(d), ents))
    real = {"semg1": [], "semg2": []}
    for v, semg in (("old", "semg1"), ("s1", "semg1"), ("s2", "semg2")):
        for line in open(f"{G}/g1/{v}/manifest.tsv"):
            uid, text = line.rstrip("\n").split("\t")
            real[semg].append((f"{v}_{uid}", f"{G}/g1/{v}/wavs/{uid}.wav", text))
    out += [("semg1", 6, real["semg1"]), ("semg2", 7, real["semg2"])]
    cf = {}
    for line in open(f"{G}/g2/cf_patient.jsonl"):
        r = json.loads(line)
        cf.setdefault((r["dir"], r["label"]), []).append((os.path.basename(r["out"])[:-4], r["out"], r["text"]))
    out += [(d, k, ents) for (d, k), ents in sorted(cf.items())]
    return out


def prep_dir(fx, d, label, ents):
    out_spk = f"{OUT}/{d}"
    if os.path.exists(f"{out_spk}/info.json"):
        return
    missing = sum(not os.path.exists(w) for _, w, _ in ents)
    if not ents or missing:
        print(f"[{d}] not ready: {missing} of {len(ents)} wavs missing, skipped", flush=True)
        return
    os.makedirs(f"{out_spk}/feats", exist_ok=True)
    items = []
    for key, wav, text in ents:
        p = {s: f"{out_spk}/feats/{key}_{s}" for s in ("mel.npy", "codes.npy", "condition.npy", "dur.json")}
        if all(os.path.exists(x) for x in p.values()):
            duration = json.load(open(p["dur.json"]))["duration"]
        else:
            try:
                r = fx.extract(wav)
            except Exception as e:
                print(f"extract failed {wav}: {e}", flush=True)
                continue
            if r is None:
                continue
            mel, codes, cond, duration = r
            np.save(p["mel.npy"], mel)
            np.save(p["codes.npy"], codes)
            np.save(p["condition.npy"], cond)
            json.dump({"duration": duration}, open(p["dur.json"], "w"))
        items.append({"text": text, "codes": p["codes.npy"], "mels": p["mel.npy"], "condition": p["condition.npy"],
                      "duration": duration, "pathology_label": label, "patient_id": d, "wav": wav})
    if not items:
        print(f"no items for {d}", flush=True)
        return
    rng = np.random.default_rng(91 + zlib.crc32(d.encode()))
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_valid = 0 if len(items) == 1 else min(len(items) - 1, max(1, int(round(len(items) * 0.02))))
    valid = set(idx[:n_valid].tolist())
    with open(f"{out_spk}/metadata_train.jsonl", "w") as ftr, open(f"{out_spk}/metadata_valid.jsonl", "w") as fva:
        for i, it in enumerate(items):
            (fva if i in valid else ftr).write(json.dumps(it, ensure_ascii=False) + "\n")
    conds = np.stack([np.load(it["condition"])[0] for it in items[::max(1, len(items) // 200)]])
    np.save(f"{out_spk}/medoid_condition.npy", conds.mean(axis=0, keepdims=True).astype(np.float32))
    info = {"speaker": d, "train_jsonl": f"{out_spk}/metadata_train.jsonl", "valid_jsonl": f"{out_spk}/metadata_valid.jsonl",
            "medoid_condition": f"{out_spk}/medoid_condition.npy", "pathology_label": label, "n_items": len(items)}
    json.dump(info, open(f"{out_spk}/info.tmp.json", "w"))
    os.replace(f"{out_spk}/info.tmp.json", f"{out_spk}/info.json")
    print(f"[{d}] items={len(items)} of {len(ents)} pathology={label}", flush=True)


def merge(dirs, partial=False):
    info = []
    for d, _, ents in dirs:
        f = f"{OUT}/{d}/info.json"
        assert partial or os.path.exists(f) or not ents, f"missing {f}"
        if os.path.exists(f):
            info.append(json.load(open(f)))
    info.sort(key=lambda s: s["speaker"])
    json.dump(info, open(f"{OUT}/speaker_info.json", "w"), indent=2, ensure_ascii=False)
    sums, counts = {}, {k: 0 for k in range(NUM_CLASSES)}
    for si in info:
        for line in open(si["train_jsonl"]):
            it = json.loads(line)
            k, c = int(it["pathology_label"]), np.load(it["condition"])[0]
            sums[k] = c if k not in sums else sums[k] + c
            counts[k] += 1
    os.makedirs(f"{OUT}/pathology_embedding", exist_ok=True)
    for k in range(NUM_CLASSES):
        assert counts[k], f"class {k} empty"
        np.save(f"{OUT}/pathology_embedding/mean_pathology_condition_{k}.npy", (sums[k] / counts[k]).astype(np.float32)[None])
    print(f"dirs {len(info)}, items {sum(s['n_items'] for s in info)}, train items per class {counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--partial", action="store_true", help="merge only the dirs that are ready (training smoke only)")
    ap.add_argument("--only-real", action="store_true",
                    help="real-speaker dirs only (15 TORGO + semg1/2); converted TORGO dirs list only the wavs that "
                         "exist, so they must not be prepared before every g2_cf job has finished")
    ap.add_argument("--list", action="store_true", help="print dirs and entry counts, no features")
    a = ap.parse_args()
    dirs = all_dirs()
    if a.only_real:
        dirs = [x for x in dirs if x[0] in pt.REAL_SPEAKERS or x[0] in ("semg1", "semg2")]
    if a.list:
        for d, k, ents in dirs:
            print(d, k, len(ents))
        print("total", sum(len(e) for _, _, e in dirs))
        return
    if a.merge:
        return merge(dirs, a.partial)
    fx = pt.FeatureExtractor(MODEL, CFG)
    for d, k, ents in dirs[a.shard::a.nshard]:
        prep_dir(fx, d, k, ents)
    print("PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
