"""A13 step 0.1: training pool, eval overlap check, 30 held-out reconstruction sentences, devcrop manifest.

CPU only. Reads Tianyu's split / raw-video files read-only; writes under --out (/scratch/pa2753/synvid/t2m2/pool).
Pool = semg3_rcv_20261109_v1 roles dc_s3_train, dc_s3ov_train, dc_v34_train, dc_real_all (the deployed-chain CV split).
Every transcript goes through the deliverable's eval normaliser (dysarthric_asr.evaluation.metrics.normalize).
"""
import argparse, collections, glob, json, os, random, re, sys

TH = "/scratch/th3482/Dysarthric_ASR/data/patients"
R = f"{TH}/splits/semg3_rcv_20261109_v1/roles"
LC = f"{TH}/splits/LATEST_combined/roles"
DASR = "/scratch/th3482/projects/auto-avsr/deliverables/semg3_phone_v1.8_pa_20261105/code/hpc/third_party/dasr_metrics"
SESS = f"{TH}/semg3/sessions"
PKG = f"{TH}/semg3/packages/SEMG3_HPC_20261107"
REAL = f"{TH}/semg3_convey_2026-10-22/raw_v1/Convey-semg3-real-15Pro-20261022-reviewed"
V34IDX = f"{SESS}/2026-10-28_openvocab-v3.4/derived/aligned_v1/index"
POOL = dict(s3="dc_s3_train", s3ov="dc_s3ov_train", v34="dc_v34_train", real="dc_real_all")
EVAL = ["dc_s3_val", "dc_s3ov_val", "dc_s3_test", "dc_s3_oov", "dc_s3ov_test", "dc_v34_val", "dc_v34_test",
        "dc_v34_val_abl", "dc_v34_test_abl", "dc_s3_rainbow", "dc_home_all", "dc_home_dev", "dc_home_test", "dc_home_val"]
EVAL_LC = ["s3_val", "s3ov_val", "s3_test", "s3_oov", "s3ov_test", "s3_rainbow"]
HELD = dict(s3=8, s3ov=9, v34=9, real=4)  # 30, roughly proportional to 240/262/298/116
SEED = 20261113

sys.path.insert(0, DASR)
from dysarthric_asr.evaluation.metrics import normalize  # noqa: E402


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


import csv
REAL_LAB = {x["\ufeffindex"] if "\ufeffindex" in x else x["index"]: x["label_text"]
            for x in csv.DictReader(open(f"{REAL}/labels.csv", encoding="utf-8"))}


def raw_index():
    """take_id -> raw source video, per source."""
    idx = {}
    for f in glob.glob(f"{SESS}/2026-10-15_w50s350/raw/*.mp4"):  # s3: <text>_<pos>-350_repeat1_<date>_<hhmmss>.mp4
        m = re.search(r"_(\d+)-350_repeat\d+_\d{8}_(\d{6})(?:_redo)?\.mp4$", f)
        if m: idx.setdefault(("s3", f"p{int(m.group(1)):03d}_{m.group(2)}"), []).append(f)
    for f in glob.glob(f"{SESS}/2026-10-22_openvocab-v3.3/raw/*.mp4"):  # s3ov: keyed by time of day
        m = re.search(r"_\d{8}_(\d{6})\.mp4$", f)
        if m: idx.setdefault(("s3ov_time", m.group(1)), []).append(f)
    for f in sorted(glob.glob(f"{V34IDX}/*.jsonl")):
        for x in jl(f):
            idx.setdefault(("v34", x["sample_id"]), []).append(f"{PKG}/{x['phone_video']}")
    return idx


def source_video(src, r, idx):
    if src == "s3":
        c = idx.get(("s3", r["take_id"]), [])
    elif src == "s3ov":
        c = idx.get(("s3ov_time", r["take_id"].split("_")[1]), [])
        if not c:  # no raw file for this take: the session's 25 fps CFR copy (same take)
            c = [f"{SESS}/2026-10-22_openvocab-v3.3/processed/v1/_normalized_25p/{r['take_id']}.mp4"]
    elif src == "v34":
        c = idx.get(("v34", r["take_id"]), [])
    else:
        c = [f"{REAL}/videos/{r['take_id'].split('_')[1]}.mov"]  # rl_NNN = labels.csv index NNN
    c = [p for p in c if os.path.exists(p)]
    if src == "real" and c:
        lab = REAL_LAB.get(r["take_id"].split("_")[1])
        if lab is None or normalize(lab) != normalize(r["transcript"]):
            return None, -1
    return (c[0] if len(c) == 1 else None), len(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = collections.OrderedDict()

    # eval sets: clip identities (take_id, devcrop_id, video basename) and normalised texts
    ev_ids, ev_txt, ev_rows = set(), {}, 0
    for role in EVAL:
        for x in jl(f"{R}/{role}.jsonl"):
            ev_rows += 1
            for k in ("devcrop_id", "sample_id", "phrase_id"):
                if x.get(k): ev_ids.add(str(x[k]))
            ev_ids.add(f"{x.get('domain')}|{x.get('take_id')}")
            ev_ids.add(os.path.basename(x["video_path"]).rsplit(".", 1)[0])
            ev_txt.setdefault(normalize(x["transcript"]), set()).add(role)
    for role in EVAL_LC:
        for x in jl(f"{LC}/{role}.jsonl"):
            ev_ids.add(f"{x['domain']}|{x['take_id']}"); ev_txt.setdefault(normalize(x["transcript"]), set()).add("LC_" + role)
    rep["eval_rows_read"] = ev_rows

    idx = raw_index()
    pool, drop = [], collections.Counter()
    rep["listed"] = {}
    for src, role in POOL.items():
        rows = jl(f"{R}/{role}.jsonl"); rep["listed"][src] = len(rows)
        for x in rows:
            ids = {str(x.get(k)) for k in ("devcrop_id", "sample_id") if x.get(k)} | {f"{x.get('domain')}|{x['take_id']}"}
            ids.add(os.path.basename(x["video_path"]).rsplit(".", 1)[0])
            n = normalize(x["transcript"])
            rec = dict(id=f"{src}__{x['take_id']}", src=src, take_id=x["take_id"], transcript=x["transcript"], norm=n,
                       tianyu_crop=x["video_path"])
            if ids & ev_ids:
                drop[f"{src}:clip_in_eval"] += 1; rec["drop"] = "clip_in_eval"
            elif n in ev_txt:
                drop[f"{src}:text_equals_eval"] += 1; rec["drop"] = "text_equals_eval:" + ",".join(sorted(ev_txt[n]))
            elif not n:
                drop[f"{src}:empty_text"] += 1; rec["drop"] = "empty_text"
            v, nc = source_video(src, x, idx)
            rec["video"] = v
            if "drop" not in rec and v is None:
                drop[f"{src}:video_{ {0: 'missing', -1: 'label_mismatch'}.get(nc, 'ambiguous')}"] += 1; rec["drop"] = "no_unique_video"
            elif "drop" not in rec and not os.access(v, os.R_OK):
                drop[f"{src}:video_unreadable"] += 1; rec["drop"] = "video_unreadable"
            pool.append(rec)
    keep = [p for p in pool if "drop" not in p]
    rep["dropped"] = dict(drop)
    rep["kept"] = dict(collections.Counter(p["src"] for p in keep)); rep["kept_total"] = len(keep)

    # 30 held-out reconstruction sentences: >= 3 words, text unique within its source, spread by fixed quota
    rng = random.Random(SEED); held = []
    tc = collections.Counter(p["norm"] for p in keep)
    for src, k in HELD.items():
        c = sorted({p["norm"] for p in keep if p["src"] == src and len(p["norm"].split()) >= 3})
        held += [(src, t) for t in rng.sample(c, k)]
    ht = {t for _, t in held}
    for p in keep:
        p["role"] = "heldout" if p["norm"] in ht else "gen_train"
    rep["heldout_sentences"] = len(held)
    rep["heldout_clips_excluded"] = dict(collections.Counter(p["src"] for p in keep if p["role"] == "heldout"))
    rep["gen_train_clips"] = dict(collections.Counter(p["src"] for p in keep if p["role"] == "gen_train"))
    rep["repeated_texts_in_pool"] = sum(1 for t, c in tc.items() if c > 1)

    with open(f"{a.out}/pool.jsonl", "w") as f:
        for p in pool: f.write(json.dumps(p) + "\n")
    with open(f"{a.out}/heldout30.txt", "w") as f:
        for s, t in held: f.write(f"{s}\t{t}\n")
    with open(f"{a.out}/devcrop_manifest.jsonl", "w") as f:
        for p in keep: f.write(json.dumps(dict(id=p["id"], video=p["video"])) + "\n")
    json.dump(rep, open(f"{a.out}/pool_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
