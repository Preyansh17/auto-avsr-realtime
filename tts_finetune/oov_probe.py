"""Generalization probe for the Higgs LoRA finetune: does the adapter still say NEW text?

    python oov_probe.py build   # 50 sentences outside the patient set -> data/oov_sentences.json (+ _stats.json)
    python oov_probe.py plan    # build_plan_ft wrapper: the SAME sentences with the adapter and with the pretrained model
    python oov_probe.py score   # ramble rate, duration per word, PathBench PER vs the TARGET text; FT vs zero-shot

There is no ground truth for these sentences (the patient never said them), so every number is measured against the
text we asked for, and the finetuned run is always read next to the pretrained model on identical text and reference.

Text source. The plan names manifests/librispeech_prompts.csv of patient_higgs_whisper_20260702_072656, but that file is
byte-identical to patient_tts_prompts.csv: the patient's OWN 308 sentences, not LibriSpeech. The real LibriSpeech
prompts are in the TORGO run (14,855 rows / 3,688 unique) and the paper-style patient run; those are used here. They are
also the text the downstream ASR augmentation will actually synthesise, so the probe measures the intended use.

Selection (seeded): unique by normalize_text; plain ASCII letters/apostrophes; 4-12 words; not equal to ANY patient
sentence (patient manifest, every clip in splits.json, every ASR split row of any split/domain); every word is a single
token of the Higgs tokenizer (a proxy for "common English word": it removes names and foreign words that recur in a few
books); at least --min-oov-frac of the tokens unseen in the tts_train vocabulary (73 words), so the sentences are outside
the VOCABULARY and not merely outside the sentence set; stratified by word count (patient distribution, floor 10 on
the longest bin). Text is written in sentence case like the patient's own prompts (all-caps would be read oddly).

Ramble (defined from the patient's real speech, no ground truth needed). Fit real duration on tts_train as
    expected(n_words) = a + b * n_words          (ordinary least squares over the tts_train clips)
and call a synthetic clip a RAMBLE when duration > 2 x expected. This is the plan's ">2x real length" rule with the
paired real clip replaced by its expectation; on the 308-sentence zero-shot runs it agrees with the paired rule within
2 points (R1_stitch5_s1: 13.6% vs 15.3%). Also reported: SHORT (< 0.5 x expected: truncated / early stop), SILENT
(rms < 0.005; the quietest zero-shot clip has rms 0.011), the share within +-20% of expected, and a sensitivity
version of the ramble rate that fits expected duration on characters (long LibriSpeech words are not the patient's).

PER is PathBench's PEREvaluator exactly as pb_eval/pb_score_items.py runs it (no --dartp), with the clip's own target text
as the reference. It needs the PathBench environment: `source /scratch/pa2753/pb_eval/pb_env.sh` first (oov_probe.sbatch
does this), or pass --per-from <items_scored.csv from pb_score_items.py>, or --per off.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import random
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from higgs_ft_lib import normalize_text, stable_id  # noqa: E402
from prep_data import ASR_SPLIT, MANIFEST  # noqa: E402

ROOT = os.environ.get("HIGGS_FT_ROOT", "/scratch/pa2753/higgs_ft")
DATA = os.environ.get("HIGGS_FT_DATA", f"{ROOT}/data")
GEN = "/scratch/pa2753/gen_runs"
PB_EVAL = "/scratch/pa2753/pb_eval"
TH_RUNS = "/scratch/th3482/TORGO/runs"  # READ-ONLY
PROMPT_SOURCES = [
    f"{TH_RUNS}/torgo_higgs_whisper_20260702_021056/manifests/librispeech_prompts.csv",
    f"{TH_RUNS}/patient_higgs_whisper_paperstyle_20260702_080650/manifests/librispeech_prompts.csv",
]
TOKENIZER_GLOB = "{hf}/hub/models--bosonai--higgs-audio-v2-generation-3B-base/snapshots/*/tokenizer.json"

BUILD_SEED = 20261006
N_SENTENCES = 50
WC_BINS = [(4, 5), (6, 8), (9, 12)]
MIN_PER_BIN = 10
MIN_OOV_FRAC = 0.6
MIN_OOV_TOKENS = 2

RAMBLE_X, SHORT_X, SILENT_RMS, WITHIN = 2.0, 0.5, 0.005, 0.2
# verdict limits (a screening probe at n=50: one clip = 2 points)
MAX_RAMBLE_EXCESS = 0.04   # FT ramble rate may exceed zero-shot's by at most 2 clips of 50
MAX_COLLAPSE = 0.04        # SHORT + SILENT clips of the finetuned run
MAX_PER_DELTA = 0.15       # mean paired PER(FT) - PER(zero-shot) on the new text
BOOT, BOOT_SEED = 5000, 20261007


# ============================================================================ build
def _wc_bin(n: int) -> int | None:
    return next((i for i, (lo, hi) in enumerate(WC_BINS) if lo <= n <= hi), None)


def _allocate(weights: list[float], total: int, floor: int) -> list[int]:
    alloc = [max(floor, round(total * w / sum(weights))) for w in weights]
    while sum(alloc) > total:
        alloc[max(range(len(alloc)), key=lambda i: alloc[i])] -= 1
    while sum(alloc) < total:
        alloc[max(range(len(alloc)), key=lambda i: weights[i] / alloc[i])] += 1
    return alloc


def _sentence_case(norm: str) -> str:
    t = re.sub(r"\bi\b", "I", norm)
    return t[0].upper() + t[1:]


def _vocab_stats(entries: list[dict], vocab: set[str]) -> dict:
    toks = [w for e in entries for w in e["norm"].split()]
    per = [sum(w not in vocab for w in e["norm"].split()) / len(e["norm"].split()) for e in entries]
    unseen = {w for w in toks if w not in vocab}
    return {"sentences": len(entries), "tokens": len(toks), "unique_words": len(set(toks)),
            "oov_token_frac": round(sum(w not in vocab for w in toks) / len(toks), 4),
            "oov_frac_per_sentence_mean": round(statistics.mean(per), 4),
            "oov_frac_per_sentence_min": round(min(per), 4),
            "sentences_with_no_seen_word": sum(p == 1.0 for p in per),
            "unique_oov_words": len(unseen), "vocab_size": len(vocab)}


def patient_norms(sp: dict) -> set[str]:
    """Every sentence the patient said that we know of: the manifest, every clip in splits.json, and every row of the
    ASR split file (any split, any domain)."""
    out = {normalize_text(r["transcript"]) for r in csv.DictReader(open(MANIFEST))}
    out |= {c["norm"] for c in sp["clips"].values()}
    out |= {normalize_text(r["transcript"]) for r in csv.DictReader(open(ASR_SPLIT))}
    return out


def _single_token(words: list[str]) -> dict[str, bool]:
    from tokenizers import Tokenizer

    hits = [h for hf in (os.environ.get("HF_HOME"), "/scratch/pa2753/hf_cache") if hf
            for h in glob.glob(TOKENIZER_GLOB.format(hf=hf))]
    if not hits:
        raise SystemExit("Higgs tokenizer.json not found in the HF cache; pass --no-tokenizer-filter to skip the "
                         "common-word filter")
    tk = Tokenizer.from_file(hits[0])
    return {w: len(tk.encode(" " + w, add_special_tokens=False).ids) == 1 for w in words}


def do_build(args) -> None:
    sp = json.load(open(f"{DATA}/splits.json"))
    clips = sp["clips"]
    exclude = patient_norms(sp)
    train_vocab = {w for u in sp["tts_train"] for w in clips[u]["norm"].split()}
    all_vocab = {w for n in exclude for w in n.split()}

    pool: dict[str, dict] = {}
    raw_rows = 0
    for src in PROMPT_SOURCES:
        for r in csv.DictReader(open(src)):
            raw_rows += 1
            norm = normalize_text(r["text"])
            pool.setdefault(norm, {"raw": r["text"], "source": src.split("/runs/")[1].split("/")[0]})
    funnel = {"prompt_rows_read": raw_rows, "unique_texts": len(pool)}

    overlap = set(pool) & exclude
    funnel["equal_to_patient_or_asr_sentence"] = len(overlap)
    cand = {n: v for n, v in pool.items() if n not in exclude}
    cand = {n: v for n, v in cand.items() if re.fullmatch(r"[A-Za-z' ]+", v["raw"]) and 4 <= len(n.split()) <= 12}
    funnel["ascii_4_to_12_words_not_patient"] = len(cand)
    if args.no_tokenizer_filter:
        funnel["tokenizer_filter"] = "skipped"
    else:
        single = _single_token(sorted({w for n in cand for w in n.split()}))
        cand = {n: v for n, v in cand.items() if all(single[w] for w in n.split())}
        funnel["all_words_single_token"] = len(cand)
    for n, v in cand.items():
        ws = n.split()
        v["oov_frac"] = sum(w not in train_vocab for w in ws) / len(ws)
        v["oov_words"] = sum(w not in train_vocab for w in ws)
    pool_median_oov = statistics.median(v["oov_frac"] for v in cand.values())
    cand = {n: v for n, v in cand.items() if v["oov_frac"] >= args.min_oov_frac and v["oov_words"] >= MIN_OOV_TOKENS}
    funnel[f"oov_frac>={args.min_oov_frac}"] = len(cand)

    patient_wc = collections.Counter(_wc_bin(len(clips[u]["norm"].split())) for u in sp["tts_train"])
    weights = [max(1, patient_wc[i]) for i in range(len(WC_BINS))]
    alloc = _allocate(weights, args.n, MIN_PER_BIN)
    rng = random.Random(BUILD_SEED)
    chosen = []
    for i, k in enumerate(alloc):
        members = sorted(n for n in cand if _wc_bin(len(n.split())) == i)
        if len(members) < k:
            raise SystemExit(f"bin {WC_BINS[i]}: only {len(members)} candidates for {k} slots; lower --min-oov-frac")
        rng.shuffle(members)
        chosen += members[:k]
    chosen.sort(key=lambda n: stable_id("gen", n, length=12))

    entries = []
    for n in chosen:
        text = _sentence_case(n)
        assert normalize_text(text) == n and text.isascii()
        assert n not in exclude, "probe sentence equals a patient/ASR sentence"
        entries.append({"prompt_id": stable_id("gen", n, length=12), "text": text, "norm": n,
                        "n_words": len(n.split()), "oov_frac": round(cand[n]["oov_frac"], 3),
                        "oov_words": cand[n]["oov_words"], "source": cand[n]["source"]})
    assert len(entries) == args.n and len({e["norm"] for e in entries}) == args.n
    assert all(4 <= e["n_words"] <= 12 for e in entries)

    stats = {"seed": BUILD_SEED, "funnel": funnel, "word_count_bins": {f"{lo}-{hi}": a for (lo, hi), a in zip(WC_BINS, alloc)},
             "patient_word_count_weights_tts_train": dict(zip([f"{lo}-{hi}" for lo, hi in WC_BINS], weights)),
             "candidate_median_oov_frac_before_threshold": round(pool_median_oov, 3),
             "vs_tts_train_vocab": _vocab_stats(entries, train_vocab),
             "vs_all_patient_vocab_incl_val_test_asr": _vocab_stats(entries, all_vocab),
             "patient_sentences_checked": len(exclude), "overlap_with_patient_or_asr": 0,
             "sources": dict(collections.Counter(e["source"] for e in entries))}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(entries, open(args.out, "w"), indent=1)
    stats_path = os.path.splitext(args.out)[0] + "_stats.json"
    json.dump(stats, open(stats_path, "w"), indent=1)
    print(json.dumps(stats, indent=1))
    for e in entries[:8]:
        print(f"  [{e['n_words']:2d}w oov {e['oov_frac']:.2f}] {e['text']}")
    print(f"wrote {args.out} and {stats_path}")


# ============================================================================ plan
def do_plan(args) -> None:
    import build_plan_ft as bp

    if not os.path.exists(args.sentences_file):
        raise SystemExit(f"{args.sentences_file} missing: run `oov_probe.py build` first")
    if args.adapter != "none" and not os.path.isdir(args.adapter):
        raise SystemExit(f"adapter dir {args.adapter} does not exist")
    tag = args.ft_tag or "_".join(os.path.normpath(args.adapter).split(os.sep)[-2:])
    bp.GEN = args.gen_root  # build_plan_ft hardcodes the output root; a dry run points it at scratch
    argv = ["build_plan_ft.py", "--out", args.out, "--sentences-file", args.sentences_file, "--ref", args.ref,
            "--seeds", *map(str, args.seeds), "--adapter", f"{tag}={args.adapter}", "--adapter", f"{args.zs_tag}=none"]
    if args.ref_json:
        argv += ["--ref-json", args.ref_json]
    old, sys.argv = sys.argv, argv
    try:
        bp.main()
    finally:
        sys.argv = old

    plan = json.load(open(args.out))
    sents = {s["prompt_id"] for s in json.load(open(args.sentences_file))}
    arms: dict[str, set] = collections.defaultdict(set)
    for u in plan["units"]:
        arms[u["variant"]] |= {p["prompt_id"] for p in u["prompts"]}
    assert all(ids == sents for ids in arms.values()), "arms do not cover the same prompt ids"
    n_clips = sum(len(u["prompts"]) for u in plan["units"])
    print(f"{len(sents)} sentences x {len(arms)} arms = {n_clips} clips in {len(plan['units'])} units")
    for v in sorted(arms):
        adapter = next(u["adapter"] for u in plan["units"] if u["variant"] == v)
        print(f"  {'zero-shot' if adapter == 'none' else 'finetuned'}  {args.gen_root}/{v}")
    print(f"next: sbatch oov_probe.sbatch  (PLAN={args.out}; split the {len(plan['units'])} units with NTASKS/TASK), "
          f"then `oov_probe.py score --plan {args.out}`")


# ============================================================================ score
def _read_arm(run_dir: str) -> dict[str, dict]:
    rows = {}
    for f in sorted(glob.glob(f"{run_dir}/synthetic/metadata/*.csv")):
        for r in csv.DictReader(open(f)):
            rows[r["prompt_id"]] = r
    return rows


def _arms_from_plan(plan_path: str) -> list[dict]:
    seen = {}
    for u in json.load(open(plan_path))["units"]:
        seen.setdefault(u["variant"], {"label": u["variant"], "run_dir": u["out_root"],
                                       "role": "zs" if u.get("adapter") in (None, "", "none") else "ft"})
    return list(seen.values())


def _seed_of(label: str) -> str:
    m = re.search(r"_s(\d+)$", label)
    return m.group(1) if m else "0"


def fit_expected(sp: dict) -> dict:
    cl = [sp["clips"][u] for u in sp["tts_train"]]
    n = np.array([len(c["norm"].split()) for c in cl], float)
    ch = np.array([len(c["norm"].replace(" ", "")) for c in cl], float)
    d = np.array([c["duration_sec"] for c in cl])
    bw, aw = np.polyfit(n, d, 1)
    bc, ac = np.polyfit(ch, d, 1)
    return {"words": (float(aw), float(bw)), "chars": (float(ac), float(bc)), "n_clips": len(cl),
            "sec_per_word_ratio_of_sums": float(d.sum() / n.sum()), "resid_sd_words": float(np.std(d - (aw + bw * n)))}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p, den = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / den, (c + h) / den


def compute_per(items: list[tuple[str, str]]) -> dict[str, tuple[float, str]]:
    sys.path.insert(0, PB_EVAL)
    try:
        import pb_score as pb
        from pathbench import PEREvaluator
    except ImportError as e:
        raise SystemExit(f"PathBench is not importable ({e}). Run under `source {PB_EVAL}/pb_env.sh` (nemo_asr python), "
                         "or pass --per-from <items_scored.csv> / --per off.")
    ev = PEREvaluator(language=pb.LANG)
    out = {}
    for i, (path, text) in enumerate(items):
        out[path] = pb.score_one(ev, True, f"u{i}", path, text)
        if (i + 1) % 25 == 0:
            print(f"  PER {i + 1}/{len(items)}", flush=True)
    return out


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    ramble = sum(r["ramble"] for r in rows)
    ratio = np.array([r["dur_ratio"] for r in rows])
    spw = np.array([r["sec_per_word"] for r in rows])
    per = np.array([r["per"] for r in rows if r["per"] is not None and math.isfinite(r["per"])])
    lo, hi = wilson(ramble, n)
    return {"n": n, "ramble": ramble, "ramble_rate": ramble / n, "ramble_ci": (lo, hi),
            "ramble_chars": sum(r["ramble_chars"] for r in rows), "short": sum(r["short"] for r in rows),
            "silent": sum(r["silent"] for r in rows), "within20": float(np.mean(np.abs(ratio - 1) <= WITHIN)),
            "ratio_med": float(np.median(ratio)), "ratio_p10": float(np.percentile(ratio, 10)),
            "ratio_p90": float(np.percentile(ratio, 90)), "spw_mean": float(spw.mean()), "spw_med": float(np.median(spw)),
            "spw_p10": float(np.percentile(spw, 10)), "spw_p90": float(np.percentile(spw, 90)),
            "dur_max": max(r["dur_sec"] for r in rows), "per_n": len(per),
            "per_mean": float(per.mean()) if len(per) else float("nan"),
            "per_median": float(np.median(per)) if len(per) else float("nan"),
            "per_sd": float(per.std(ddof=1)) if len(per) > 1 else float("nan")}


def _paired_delta(ft: dict[str, dict], zs: dict[str, dict]) -> dict:
    ids = [p for p in ft if p in zs and ft[p]["per"] is not None and zs[p]["per"] is not None
           and math.isfinite(ft[p]["per"]) and math.isfinite(zs[p]["per"])]
    if len(ids) < 5:
        return {"n": len(ids), "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    d = np.array([ft[p]["per"] - zs[p]["per"] for p in ids])
    rng = np.random.default_rng(BOOT_SEED)
    means = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
    return {"n": len(ids), "mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5))}


def _fmt(x: float, f: str = ".3f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, f)


def do_score(args) -> None:
    sp = json.load(open(f"{DATA}/splits.json"))
    sents = {s["prompt_id"]: s for s in json.load(open(args.sentences_file))}
    train_vocab = {w for u in sp["tts_train"] for w in sp["clips"][u]["norm"].split()}
    for s in sents.values():
        s["norm"] = s.get("norm") or normalize_text(s["text"])
        s["n_words"] = len(s["norm"].split())
        s.setdefault("oov_frac", sum(w not in train_vocab for w in s["norm"].split()) / s["n_words"])
    fit = fit_expected(sp)
    known = patient_norms(sp)
    overlap = sorted(s["norm"] for s in sents.values() if s["norm"] in known)

    if args.ft_run or args.zs_run:
        arms = [{"label": os.path.basename(p.rstrip("/")), "run_dir": p, "role": "ft"} for p in args.ft_run] + \
               [{"label": os.path.basename(p.rstrip("/")), "run_dir": p, "role": "zs"} for p in args.zs_run]
    elif args.plan:
        arms = _arms_from_plan(args.plan)
    else:
        raise SystemExit("give --plan or --ft-run/--zs-run")
    if not any(a["role"] == "ft" for a in arms) or not any(a["role"] == "zs" for a in arms):
        raise SystemExit("need at least one finetuned and one zero-shot arm")

    # ---- per-clip rows
    rows, missing = [], collections.defaultdict(list)
    for arm in arms:
        meta = _read_arm(arm["run_dir"])
        for pid, s in sents.items():
            m = meta.get(pid)
            if m is None or not os.path.exists(m["audio_path"]):
                missing[arm["label"]].append(pid)
                continue
            dur, n = float(m["duration_sec"]), s["n_words"]
            exp_w = fit["words"][0] + fit["words"][1] * n
            exp_c = fit["chars"][0] + fit["chars"][1] * len(s["norm"].replace(" ", ""))
            rows.append({"prompt_id": pid, "norm": s["norm"], "n_words": n, "oov_frac": s["oov_frac"],
                         "arm": arm["label"], "role": arm["role"], "seed": _seed_of(arm["label"]),
                         "audio_path": m["audio_path"], "transcript": m["transcript"], "dur_sec": dur,
                         "exp_dur_sec": exp_w, "dur_ratio": dur / exp_w, "sec_per_word": dur / n,
                         "ramble": int(dur > RAMBLE_X * exp_w), "ramble_chars": int(dur > RAMBLE_X * exp_c),
                         "short": int(dur < SHORT_X * exp_w), "silent": int(float(m["rms"]) < SILENT_RMS),
                         "per": None, "per_err": ""})
    complete = not missing

    # ---- PER vs the clip's own target text
    per_mode = "off" if args.per == "off" else ("from-csv" if args.per_from else "computed")
    if per_mode != "off" and rows:
        if args.per_from:
            scored = {}
            for f in args.per_from:
                for r in csv.DictReader(open(f)):
                    if r.get("per") not in (None, ""):
                        scored[r["audio_path"]] = float(r["per"])
            got = {r["audio_path"]: (scored.get(r["audio_path"], float("nan")), "" if r["audio_path"] in scored else "not in --per-from")
                   for r in rows}
        else:
            got = compute_per(sorted({(r["audio_path"], r["transcript"]) for r in rows}))
        for r in rows:
            v, err = got[r["audio_path"]]
            r["per"], r["per_err"] = (v if math.isfinite(v) else None), err

    # ---- outputs
    first_ft = next(a["label"] for a in arms if a["role"] == "ft")
    out_dir = args.out_dir or f"{ROOT}/oov_probe/{re.sub(r'^FT_', '', first_ft)}"
    os.makedirs(out_dir, exist_ok=True)
    cols = ["prompt_id", "arm", "role", "seed", "n_words", "oov_frac", "dur_sec", "exp_dur_sec", "dur_ratio", "sec_per_word",
            "ramble", "ramble_chars", "short", "silent", "per", "per_err", "audio_path", "transcript"]
    with open(f"{out_dir}/oov_clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(f"{out_dir}/oov_pb_items.csv", "w", newline="") as f:  # same format pb_score_items.py reads
        w = csv.DictWriter(f, fieldnames=["source", "speaker", "severity", "audio_path", "transcript"])
        w.writeheader()
        w.writerows({"source": "synthetic", "speaker": "bingge", "severity": r["arm"], "audio_path": r["audio_path"],
                     "transcript": r["transcript"]} for r in rows)

    by_arm = {a["label"]: [r for r in rows if r["arm"] == a["label"]] for a in arms}
    agg = {k: _agg(v) for k, v in by_arm.items() if v}
    verdicts, summary = [], {"definitions": {"ramble_x": RAMBLE_X, "short_x": SHORT_X, "silent_rms": SILENT_RMS},
                             "expected_duration_fit": fit, "per_mode": per_mode, "arms": {}, "pairs": []}
    for k, a in agg.items():
        summary["arms"][k] = {x: y for x, y in a.items()}
    ft_arms = [a for a in arms if a["role"] == "ft" and a["label"] in agg]
    zs_by_seed = {_seed_of(a["label"]): a for a in arms if a["role"] == "zs" and a["label"] in agg}
    for fa in ft_arms:
        za = zs_by_seed.get(_seed_of(fa["label"])) or next(iter(zs_by_seed.values()), None)
        if za is None:
            continue
        A, Z = agg[fa["label"]], agg[za["label"]]
        pd = _paired_delta({r["prompt_id"]: r for r in by_arm[fa["label"]]}, {r["prompt_id"]: r for r in by_arm[za["label"]]})
        n = A["n"]
        ok_r = A["ramble_rate"] <= Z["ramble_rate"] + MAX_RAMBLE_EXCESS
        ok_c = (A["short"] + A["silent"]) / n <= MAX_COLLAPSE
        ok_p = None if not math.isfinite(pd["mean"]) else pd["mean"] <= MAX_PER_DELTA
        status = "INVALID" if overlap else "INCOMPLETE" if (not complete or n < len(sents)) else \
            "FAIL" if (not ok_r or not ok_c or ok_p is False) else ("PASS" if ok_p else "PASS (PER not scored)")
        v = (f"OOV PROBE {status}: n={n}/{len(sents)} new sentences, seed {_seed_of(fa['label'])} | "
             f"ramble FT {A['ramble']}/{n} vs zero-shot {Z['ramble']}/{Z['n']} ({'ok' if ok_r else 'WORSE'}, limit +{MAX_RAMBLE_EXCESS:.0%}) | "
             f"short+silent FT {A['short'] + A['silent']}/{n} ({'ok' if ok_c else 'COLLAPSE'}, limit {MAX_COLLAPSE:.0%}) | "
             f"PER FT {_fmt(A['per_mean'])} vs zero-shot {_fmt(Z['per_mean'])}, paired delta {_fmt(pd['mean'], '+.3f')} "
             f"[{_fmt(pd['lo'], '+.3f')}, {_fmt(pd['hi'], '+.3f')}] ({'n/a' if ok_p is None else 'ok' if ok_p else 'DEGRADED'}, limit +{MAX_PER_DELTA})")
        verdicts.append(v)
        summary["pairs"].append({"ft": fa["label"], "zs": za["label"], "per_delta": pd, "ok_ramble": ok_r, "ok_collapse": ok_c,
                                 "ok_per": ok_p, "status": status, "verdict": v})
    if overlap:
        verdicts.append(f"INVALID: {len(overlap)}/{len(sents)} sentences are patient/ASR sentences (e.g. {overlap[0]!r}); "
                        "this is not an out-of-set probe")
        summary["overlap_with_patient_or_asr"] = overlap
    if missing:
        verdicts.append("MISSING clips: " + "; ".join(f"{k}: {len(v)}" for k, v in missing.items()))
        summary["missing"] = {k: v for k, v in missing.items()}
    json.dump(summary, open(f"{out_dir}/oov_summary.json", "w"), indent=1)

    # ---- markdown
    labels = list(agg)
    L = [f"# OOV generalization probe: {first_ft}", "",
         *[f"**{v}**" for v in verdicts], "",
         f"Sentences: {len(sents)} from `{args.sentences_file}`; {len(overlap)} of them equal a patient or ASR sentence "
         f"(checked against {len(known)} known patient sentences). PER mode: {per_mode}. Mean fraction of tokens "
         f"unseen in the {len(train_vocab)}-word training vocabulary: {statistics.mean(s['oov_frac'] for s in sents.values()):.2f}.", "",
         "## Definitions", "",
         f"- Expected duration = {fit['words'][0]:.2f} + {fit['words'][1]:.3f} x words (OLS on {fit['n_clips']} real tts_train clips; "
         f"residual sd {fit['resid_sd_words']:.2f} s; real sec/word ratio-of-sums {fit['sec_per_word_ratio_of_sums']:.2f}).",
         f"- RAMBLE: duration > {RAMBLE_X:g} x expected. SHORT: < {SHORT_X:g} x expected. SILENT: rms < {SILENT_RMS}. "
         f"`ramble (chars)` is the same rule with expected duration fit on characters (sensitivity to longer words).",
         "- PER: PathBench PEREvaluator against the clip's OWN target text (there is no ground truth for new sentences).",
         f"- Verdict limits: ramble <= zero-shot + {MAX_RAMBLE_EXCESS:.0%}; short+silent <= {MAX_COLLAPSE:.0%}; "
         f"mean paired PER(FT) - PER(zero-shot) <= +{MAX_PER_DELTA}.", "",
         "## Arms (same text, same reference, 1 seed)", "",
         "| metric | " + " | ".join(f"{k} [{next(a['role'] for a in arms if a['label'] == k)}]" for k in labels) + " |",
         "|---|" + "---:|" * len(labels)]

    def line(name, fn):
        L.append(f"| {name} | " + " | ".join(fn(agg[k]) for k in labels) + " |")

    line("clips", lambda a: str(a["n"]))
    line("ramble (>2x expected)", lambda a: f"{a['ramble']} ({a['ramble_rate']:.0%}; 95% CI {a['ramble_ci'][0]:.0%}-{a['ramble_ci'][1]:.0%})")
    line("ramble (chars)", lambda a: str(a["ramble_chars"]))
    line("short (<0.5x)", lambda a: str(a["short"]))
    line("silent", lambda a: str(a["silent"]))
    line("within +-20% of expected", lambda a: f"{a['within20']:.0%}")
    line("duration / expected: median [p10, p90]", lambda a: f"{a['ratio_med']:.2f} [{a['ratio_p10']:.2f}, {a['ratio_p90']:.2f}]")
    line("sec per word: mean (median) [p10, p90]", lambda a: f"{a['spw_mean']:.2f} ({a['spw_med']:.2f}) [{a['spw_p10']:.2f}, {a['spw_p90']:.2f}]")
    line("longest clip (s)", lambda a: f"{a['dur_max']:.1f}")
    line("PER mean (median, sd)", lambda a: f"{_fmt(a['per_mean'])} ({_fmt(a['per_median'])}, {_fmt(a['per_sd'], '.2f')}) n={a['per_n']}")
    for pr in summary["pairs"]:
        d = pr["per_delta"]
        L.append(f"\nPaired PER delta {pr['ft']} - {pr['zs']}: {_fmt(d['mean'], '+.3f')} "
                 f"(bootstrap 95% CI [{_fmt(d['lo'], '+.3f')}, {_fmt(d['hi'], '+.3f')}], {d['n']} sentences).")

    def group_table(title, keyfn, order):
        L.extend(["", f"## {title}", "", "| group | n | " + " | ".join(f"{k}: ramble / PER" for k in labels) + " |",
                  "|---|---:|" + "---:|" * len(labels)])
        for g in order:
            cells, n_g = [], 0
            for k in labels:
                sub = [r for r in by_arm[k] if keyfn(r) == g]
                n_g = len(sub)
                per = [r["per"] for r in sub if r["per"] is not None]
                cells.append(f"{sum(r['ramble'] for r in sub)}/{len(sub)} / {_fmt(float(np.mean(per)) if per else float('nan'), '.2f')}")
            L.append(f"| {g} | {n_g} | " + " | ".join(cells) + " |")

    names = [f"{lo}-{hi} words" for lo, hi in WC_BINS]  # open-ended outer bins so any sentence file scores
    bin_name = lambda r: next((nm for nm, (_, hi) in zip(names, WC_BINS) if r["n_words"] <= hi), names[-1])  # noqa: E731
    group_table("By word count", bin_name, names)
    order = sorted(sents, key=lambda p: (sents[p]["oov_frac"], p))
    low = set(order[: len(order) // 2])
    group_table("By share of unseen words (median split)", lambda r: "lower-OOV half" if r["prompt_id"] in low else "higher-OOV half",
                ["lower-OOV half", "higher-OOV half"])
    L += ["", "## Caveats", "",
          f"- One seed, {len(sents)} sentences: one clip is {100 / len(sents):.0f} points of ramble rate and the PER delta CI is wide. This is a screening probe for "
          "memorisation/collapse, not a ranking; the headline comparison is the 3-seed tts_test evaluation.",
          "- PER here is an ASR-phoneme mismatch against the target text, so it also rises with harder (rarer, longer) words; "
          "read it as FT relative to zero-shot on identical text, not against the patient's real PER (0.665 on in-domain text).",
          "- Ramble is duration-only. A fluent but wrong utterance (a memorised patient sentence said instead of the target) "
          "shows up as high PER, not as ramble: inspect the highest-PER FT clips by ear.",
          f"- Per-clip table: `{out_dir}/oov_clips.csv`; PathBench items (for pb_score_items.py cross-check): `{out_dir}/oov_pb_items.csv`."]
    open(f"{out_dir}/oov_probe.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out_dir}/oov_probe.md, oov_summary.json, oov_clips.csv, oov_pb_items.csv")
    # a FAIL verdict is a result (exit 0); a probe that did not measure what it claims is an error (exit 3)
    if overlap or missing or any(p["status"] in ("INCOMPLETE", "INVALID") for p in summary["pairs"]):
        sys.exit(3)


# ============================================================================ cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--out", default=f"{DATA}/oov_sentences.json")
    b.add_argument("--n", type=int, default=N_SENTENCES)
    b.add_argument("--min-oov-frac", type=float, default=MIN_OOV_FRAC)
    b.add_argument("--no-tokenizer-filter", action="store_true")
    b.set_defaults(fn=do_build)

    p = sub.add_parser("plan")
    p.add_argument("--adapter", required=True, help="selected adapter dir, e.g. runs/A_r16/epoch_08")
    p.add_argument("--ft-tag", help="run-name tag for the adapter (default: <run>_<epoch dir>)")
    p.add_argument("--zs-tag", default="zs")
    p.add_argument("--ref", default="ft_stitch5")
    p.add_argument("--ref-json")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--sentences-file", default=f"{DATA}/oov_sentences.json")
    p.add_argument("--out", default=f"{DATA}/plan_oov.json")
    p.add_argument("--gen-root", default=GEN)
    p.set_defaults(fn=do_plan)

    s = sub.add_parser("score")
    s.add_argument("--plan", help="plan written by `plan` (arms are read from its units)")
    s.add_argument("--ft-run", action="append", default=[], help="finetuned run dir (instead of --plan)")
    s.add_argument("--zs-run", action="append", default=[], help="zero-shot run dir (instead of --plan)")
    s.add_argument("--sentences-file", default=f"{DATA}/oov_sentences.json")
    s.add_argument("--out-dir")
    s.add_argument("--per", choices=["compute", "off"], default="compute")
    s.add_argument("--per-from", action="append", default=[], help="items_scored.csv from pb_score_items.py (column per)")
    s.set_defaults(fn=do_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
