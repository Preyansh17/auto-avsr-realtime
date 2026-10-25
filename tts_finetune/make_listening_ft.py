"""Blind listening test for the Higgs LoRA finetune: the patient's real clips vs the finetuned model vs the best zero-shot run.

    python make_listening_ft.py build --ft-run /scratch/pa2753/gen_runs/FT_<tag>_ft_stitch5_tts_test_s1 \
        [--zs-run /scratch/pa2753/gen_runs/LP_low5_s1] [--n-sentences 4]
    python make_listening_ft.py summarize --key <name>_key.json --responses <listener>.json [more.json ...]

Design (12 clips = 4 sentences x {real, finetuned, zero-shot}, all on the SAME text, so only the voice/pace/rhythm differ):
  * Sentences are drawn from tts_test (data/splits.json) with a fixed seed, one from each quantile of word count, from the
    sentences BOTH synthetic runs generated. By default a sentence is only eligible if both synthetic clips are within
    0.5-2x the length of the real clip (--max-dur-ratio 0 disables). This makes the test a "typical case" comparison
    that is KINDER to the synthetic speech: a 30 s ramble would identify its condition by length alone and with only
    12 clips it would dominate the listener's impression. Rambles are measured on the full test set (ramble rate in
    the final eval and oov_probe.py), not here. The number of excluded sentences is written to the key.
  * Each trial is one clip. The listener hears a reference (a real patient clip with a different sentence, drawn from
    tts_train) and then the clip to judge, and answers the same questions as the earlier tests in /scratch/pa2753/tts_eval
    (same speaker? voice / pace / rhythm similarity 1-5) plus one directional rating added here, because the zero-shot
    failure modes are opposite (too clean vs over-impaired): "clearer / same / more impaired than the reference".
  * Trial order is a seeded shuffle with no two clips of the same sentence adjacent. Every clip, real or synthetic, goes
    through the same pipeline (mono 16 kHz, peak-normalised to 0.95, same codec), is named by an opaque id, and the HTML
    (checked by an assertion) contains no run name, path or condition. Real clips are copied from the read-only th3482 data
    by reading them, never writing there.
  * Output: <out-dir>/<name>/index.html + audio/<id>.<ext> (relative paths, open the folder anywhere) and the key, kept
    OUTSIDE that folder so that sharing the folder does not share the answers: <out-dir>/<name>_key.json.
The page CSS is imported from tts_eval/make_listening_test.py so the UI stays the same as the earlier tests.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import html
import json
import os
import random
import re
import statistics
import sys

import numpy as np

ROOT = os.environ.get("HIGGS_FT_ROOT", "/scratch/pa2753/higgs_ft")
DATA = os.environ.get("HIGGS_FT_DATA", f"{ROOT}/data")
TTS_EVAL = "/scratch/pa2753/tts_eval"
DEFAULT_ZS = "/scratch/pa2753/gen_runs/LP_low5_s1"
SR = 16000          # as make_listening_test.py
PEAK = 0.95         # level-match every clip, as make_listening_test.py (loudness alone must not identify a condition)
SEED = 20261008
CONDITIONS = ("real", "ft", "zs")

IMPAIRMENT = [(1, "much clearer"), (2, "somewhat clearer"), (3, "about the same"),
              (4, "somewhat more impaired"), (5, "much more impaired")]

PAGE_JS = """
function collect() {
  const rows = [];
  document.querySelectorAll('.trial').forEach(function (el) {
    const idx = el.dataset.index;
    const same = el.querySelector('input[name="same' + idx + '"]:checked');
    const get = function (k) {
      const n = el.querySelector('input[name="' + k + idx + '"]:checked');
      return n ? Number(n.value) : null;
    };
    rows.push({ index: Number(idx), clip_id: el.dataset.clip, same_speaker: same ? same.value : null,
                voice: get('voice'), pace: get('pace'), rhythm: get('rhythm'), impairment: get('imp') });
  });
  const listener = document.getElementById('listener').value.trim();
  document.getElementById('out').value = JSON.stringify({ listener: listener, responses: rows }, null, 1);
  document.getElementById('out').scrollIntoView({ behavior: 'smooth' });
}
"""


def opaque(seed: int, kind: str, key: str) -> str:
    return kind + hashlib.sha1(f"{seed}|{kind}|{key}".encode()).hexdigest()[:8]


def read_arm(run_dir: str) -> dict[str, dict]:
    """norm_transcript -> metadata row, from <run>/synthetic/metadata/*.csv (the layout gen_higgs/gen_ft write)."""
    rows = {}
    for f in sorted(glob.glob(f"{run_dir}/synthetic/metadata/*.csv")):
        for r in csv.DictReader(open(f)):
            rows[r["norm_transcript"]] = r
    if not rows:
        raise SystemExit(f"no metadata under {run_dir}/synthetic/metadata")
    return rows


def pick_codec() -> tuple[str, str, dict]:
    import io

    import soundfile as sf

    try:
        sf.write(io.BytesIO(), np.zeros(1600, dtype="float32"), SR, format="OGG", subtype="VORBIS")
        return "ogg", "OGG", {"subtype": "VORBIS"}
    except Exception:
        return "wav", "WAV", {"subtype": "PCM_16"}


def encode(src: str, dst: str, fmt: str, kw: dict) -> float:
    import librosa
    import soundfile as sf

    wav, _ = librosa.load(src, sr=SR, mono=True)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = wav * (PEAK / peak)
    sf.write(dst, wav, SR, format=fmt, **kw)
    return len(wav) / SR


def choose(args) -> tuple[list[dict], dict]:
    rng = random.Random(args.seed)
    sp = json.load(open(f"{DATA}/splits.json"))
    clips = sp["clips"]
    real_by_norm = collections.defaultdict(list)
    for u in sp[args.split]:
        real_by_norm[clips[u]["norm"]].append(u)
    ft, zs = read_arm(args.ft_run), read_arm(args.zs_run)

    pool, dropped = [], collections.Counter()
    for norm in sorted(real_by_norm):
        if norm not in ft or norm not in zs:
            dropped["not generated by both runs"] += 1
            continue
        u = rng.choice(sorted(real_by_norm[norm]))
        real_dur = clips[u]["duration_sec"]
        d_ft, d_zs = float(ft[norm]["duration_sec"]), float(zs[norm]["duration_sec"])
        if real_dur < 1.0:
            dropped["real clip < 1 s"] += 1
        elif args.max_dur_ratio and not all(1 / args.max_dur_ratio <= d / real_dur <= args.max_dur_ratio for d in (d_ft, d_zs)):
            dropped[f"a synthetic clip outside {1 / args.max_dur_ratio:g}-{args.max_dur_ratio:g}x the real length"] += 1
        elif not (os.path.exists(ft[norm]["audio_path"]) and os.path.exists(zs[norm]["audio_path"])):
            dropped["audio file missing"] += 1
        else:
            pool.append({"norm": norm, "text": clips[u]["transcript"], "n_words": len(norm.split()), "real_utt": u,
                         "real_path": clips[u]["audio_path"], "real_dur": real_dur,
                         "ft_path": ft[norm]["audio_path"], "ft_dur": d_ft, "zs_path": zs[norm]["audio_path"], "zs_dur": d_zs})
    if len(pool) < args.n_sentences:
        raise SystemExit(f"only {len(pool)} eligible sentences for {args.n_sentences}; dropped: {dict(dropped)}")
    # one sentence from each word-count quantile, so short and long sentences are both represented
    pool.sort(key=lambda p: (p["n_words"], p["norm"]))
    q = [pool[i * len(pool) // args.n_sentences:(i + 1) * len(pool) // args.n_sentences] for i in range(args.n_sentences)]
    return [rng.choice(g) for g in q], {"eligible": len(pool), "dropped": dict(dropped), "split_clips": len(sp[args.split]),
                                        "split_hash": sp["log"]["split_hash"]}


def shuffle_no_adjacent(items: list[dict], rng: random.Random) -> list[dict]:
    for _ in range(2000):
        order = items[:]
        rng.shuffle(order)
        if all(a["norm"] != b["norm"] for a, b in zip(order, order[1:])):
            return order
    raise SystemExit("could not find an order without adjacent same-sentence clips")


def render(trials: list[dict], css: str) -> str:
    esc = html.escape
    parts = [
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Patient speech listening test</title>",
        f"<style>{css}</style>",
        "<h1>Patient speech listening test</h1>",
        '<p class="muted">Each trial plays a reference clip of the patient, then a clip to judge. Some clips to judge are the '
        "patient, some are synthesised. Several clips say the same sentence: rate each one on its own, do not try to match "
        "them up. Judge the voice, pace and rhythm, not the content. Headphones recommended.</p>",
        '<p class="q"><span class="label">Your name or initials:</span> '
        '<input id="listener" style="font:inherit;padding:4px 8px"></p>',
    ]
    for t in trials:
        i = t["index"]
        parts.append(f'<div class="trial" data-index="{i}" data-clip="{t["clip_id"]}">')
        parts.append(f"<h3>Trial {i + 1} of {len(trials)}</h3>")
        parts.append('<div class="label">Reference &mdash; the patient</div>')
        parts.append(f'<audio controls preload="none" src="{t["anchor_src"]}"></audio>')
        parts.append(f'<div class="muted">&ldquo;{esc(t["anchor_text"])}&rdquo;</div>')
        parts.append('<div class="label" style="margin-top:12px">Clip to judge</div>')
        parts.append(f'<audio controls preload="none" src="{t["src"]}"></audio>')
        parts.append(f'<div class="muted">&ldquo;{esc(t["text"])}&rdquo;</div>')
        parts.append('<div class="q">Is the clip to judge the same speaker as the reference?</div><div class="opts">')
        for val, lab in (("same", "Same speaker"), ("different", "Different / synthetic"), ("unsure", "Cannot tell")):
            parts.append(f'<label><input type="radio" name="same{i}" value="{val}">{lab}</label>')
        parts.append("</div>")
        for key, question in (("voice", "Voice quality and tone similarity"), ("pace", "Speaking pace similarity"),
                              ("rhythm", "Rhythm, pausing and stress similarity")):
            parts.append(f'<div class="q">{question} <span class="muted">(1 = totally different, 5 = indistinguishable)</span></div>')
            parts.append('<div class="scale">' + "".join(
                f'<label><input type="radio" name="{key}{i}" value="{v}">{v}</label>' for v in range(1, 6)) + "</div>")
        parts.append('<div class="q">Compared with the reference, the speech in the clip to judge is&hellip;</div>')
        parts.append('<div class="scale">' + "".join(
            f'<label><input type="radio" name="imp{i}" value="{v}">{v}&nbsp;{lab}</label>' for v, lab in IMPAIRMENT) + "</div>")
        parts.append("</div>")
    parts.append('<h2>Finish</h2><p class="muted">Click below, then copy the text box and send it back. '
                 "Nothing is uploaded from this page.</p>")
    parts.append('<button onclick="collect()">Collect my responses</button>')
    parts.append('<textarea id="out" readonly placeholder="Your responses appear here."></textarea>')
    parts.append(f"<script>{PAGE_JS}</script>")
    return "\n".join(parts)


def do_build(args) -> None:
    sys.path.insert(0, TTS_EVAL)
    try:
        import make_listening_test as base
    except ImportError as e:
        raise SystemExit(f"cannot import the existing listening-test generator from {TTS_EVAL}: {e}")

    sp = json.load(open(f"{DATA}/splits.json"))
    clips = sp["clips"]
    sentences, info = choose(args)
    rng = random.Random(args.seed + 1)

    # anchors: real tts_train clips of a DIFFERENT sentence, 3-9 s, each used once
    anchor_pool = sorted(u for u in sp["tts_train"] if 3.0 <= clips[u]["duration_sec"] <= 9.0)
    rng.shuffle(anchor_pool)
    trials = []
    for s in sentences:
        for cond, key in (("real", "real_path"), ("ft", "ft_path"), ("zs", "zs_path")):
            anchor = next(u for u in anchor_pool if clips[u]["norm"] != s["norm"])
            anchor_pool.remove(anchor)
            trials.append({"cond": cond, "norm": s["norm"], "text": s["text"], "src_path": s[key],
                           "dur": s["real_dur"] if cond == "real" else s[f"{cond}_dur"], "anchor_utt": anchor})
    trials = shuffle_no_adjacent(trials, rng)

    name = args.name or re.sub(r"^FT_", "", os.path.basename(args.ft_run.rstrip("/"))) + "_blind12"
    page_dir = f"{args.out_dir}/{name}"
    ext, fmt, kw = pick_codec()
    os.makedirs(f"{page_dir}/audio", exist_ok=True)
    ids = set()
    for i, t in enumerate(trials):
        t["index"] = i
        t["clip_id"] = opaque(args.seed, "c", t["src_path"] + "|" + t["cond"])
        a_id = opaque(args.seed, "a", clips[t["anchor_utt"]]["audio_path"] + f"|{i}")
        assert t["clip_id"] not in ids and a_id not in ids, "opaque id collision"
        ids |= {t["clip_id"], a_id}
        t["src"], t["anchor_src"] = f"audio/{t['clip_id']}.{ext}", f"audio/{a_id}.{ext}"
        t["anchor_text"] = clips[t["anchor_utt"]]["transcript"]
        t["dur_encoded"] = encode(t["src_path"], f"{page_dir}/{t['src']}", fmt, kw)
        encode(clips[t["anchor_utt"]]["audio_path"], f"{page_dir}/{t['anchor_src']}", fmt, kw)

    page = render(trials, base.PAGE_CSS)
    leaks = [x for x in (os.path.basename(args.ft_run.rstrip("/")), os.path.basename(args.zs_run.rstrip("/")), "/scratch", "th3482",
                         "gen_runs", "FT_", "zero-shot", "finetun", "synthetic_split") if x.lower() in page.lower()]
    # the page itself says "synthesised" (as the earlier tests do) but never which clip is which
    assert not leaks, f"HTML leaks the condition/run: {leaks}"
    open(f"{page_dir}/index.html", "w").write(page)

    key_path = f"{args.out_dir}/{name}_key.json"
    json.dump({"name": name, "seed": args.seed, "split": args.split, "ft_run": args.ft_run, "zs_run": args.zs_run,
               "page": f"{page_dir}/index.html", "max_dur_ratio": args.max_dur_ratio, **info,
               "sentences": [{k: s[k] for k in ("norm", "text", "n_words", "real_utt", "real_dur", "ft_dur", "zs_dur")}
                             for s in sentences],
               "answer_format": {"listener": "str", "responses": [{"index": "trial", "clip_id": "opaque id",
                                 "same_speaker": "same|different|unsure", "voice": "1-5", "pace": "1-5", "rhythm": "1-5",
                                 "impairment": "1 much clearer .. 3 same .. 5 much more impaired"}]},
               "key": [{"index": t["index"], "clip_id": t["clip_id"], "condition": t["cond"], "norm": t["norm"],
                        "text": t["text"], "source_path": t["src_path"], "duration_sec": round(t["dur"], 3),
                        "anchor_utt_id": t["anchor_utt"], "anchor_path": clips[t["anchor_utt"]]["audio_path"]}
                       for t in trials]}, open(key_path, "w"), indent=1)

    print(f"{len(trials)} clips = {len(sentences)} sentences x {len(CONDITIONS)} conditions "
          f"({info['eligible']} eligible sentences; dropped {info['dropped']})")
    for s in sorted(sentences, key=lambda s: s["n_words"]):
        print(f"  {s['n_words']:2d}w  real {s['real_dur']:.1f}s  ft {s['ft_dur']:.1f}s  zs {s['zs_dur']:.1f}s  {s['text']}")
    print(f"page: {page_dir}/index.html ({len(os.listdir(page_dir + '/audio'))} audio files, {ext})")
    print(f"key (keep private): {key_path}")


def do_summarize(args) -> None:
    key = json.load(open(args.key))
    cond = {k["clip_id"]: k["condition"] for k in key["key"]}
    acc: dict[str, dict[str, list]] = {c: collections.defaultdict(list) for c in CONDITIONS}
    listeners = []
    for f in args.responses:
        d = json.load(open(f))
        listeners.append(d.get("listener") or os.path.basename(f))
        for r in d["responses"]:
            c = cond[r["clip_id"]]
            for k in ("voice", "pace", "rhythm", "impairment", "same_speaker"):
                if r.get(k) is not None:
                    acc[c][k].append(r[k])
    print(f"listeners: {', '.join(listeners)}  |  key {args.key}\n")
    print("| condition | rated | said same speaker | voice | pace | rhythm | impairment (1 clearer .. 5 more impaired) "
          "| clearer / same / more impaired |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in CONDITIONS:
        a = acc[c]
        imp = a["impairment"]
        same = a["same_speaker"]
        m = lambda k: f"{statistics.mean(a[k]):.2f}" if a[k] else "n/a"  # noqa: E731
        print(f"| {c} | {max(len(imp), len(same))} | {sum(x == 'same' for x in same)}/{len(same)} | {m('voice')} | {m('pace')} | "
              f"{m('rhythm')} | {m('impairment')} | {sum(x < 3 for x in imp)} / {sum(x == 3 for x in imp)} / {sum(x > 3 for x in imp)} |")
    print("\nconditions: real = the patient's own recording, ft = finetuned, zs = best zero-shot. 12 clips per listener: "
          "descriptive only, no significance claim.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ft-run", required=True, help="finetuned run dir on tts_test (gen_runs/FT_<tag>_<ref>_tts_test_s1)")
    b.add_argument("--zs-run", default=DEFAULT_ZS, help="best zero-shot run dir (default: lowpitch5, seed 1)")
    b.add_argument("--split", default="tts_test", choices=["tts_test", "tts_val"])
    b.add_argument("--n-sentences", type=int, default=4)
    b.add_argument("--max-dur-ratio", type=float, default=2.0)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--name")
    b.add_argument("--out-dir", default=f"{ROOT}/listening")
    b.set_defaults(fn=do_build)
    s = sub.add_parser("summarize")
    s.add_argument("--key", required=True)
    s.add_argument("--responses", nargs="+", required=True)
    s.set_defaults(fn=do_summarize)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
