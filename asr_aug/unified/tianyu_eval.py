"""Score a NeMo RNNT model on Tianyu's semg2 package sets with his own scorer (PLAN.md 3.3 item 0, v2).

    python tianyu_eval.py --model best.nemo --tag U2r_seed1 --mode offline
    python tianyu_eval.py --model best.nemo --tag U2r_seed1 --mode stream --chunk 0.08 --right 0.08   # buffered (unified)
    python tianyu_eval.py --model best.nemo --tag N2r_seed1 --mode cache --att 70,1                   # cache-aware (nemotron_en)

Sets and clips come from the package's phone_streaming_nemotron/data/eval_*.csv (read only, extracted to --pkg);
references and metrics are the package's tools/score.py (WER, exact, le1, new-word recall), so numbers compare 1:1 with
its README section 4. conv34 = conv26 + conv8 references pooled, as in its summary.json.
Writes <out>/<tag>_<cfg>_<set>.tsv (clip, raw hyp) and <out>/<tag>_<cfg>_scores.json.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import soundfile as sf

PKG = "/scratch/pa2753/tianyu_semg2_best_v1/semg2_best_v1_20261029"
SETS = {"val25": ["eval_s2ov_val"], "val30": ["eval_s2_val"], "test30": ["eval_s2_test"], "oov50": ["eval_s2_oov"],
        "test_open38": ["eval_s2ov_test"], "conv34": ["eval_conv34"]}
REF_SETS = {"conv34": ["conv26", "conv8"]}  # score.py has per-sentence refs for the two halves only


def stem(f): return os.path.splitext(os.path.basename(f))[0]  # refs are keyed p013_123353.mp4, the CSVs .wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["offline", "stream", "cache"], default="offline")
    ap.add_argument("--left", type=float, default=5.6)
    ap.add_argument("--chunk", type=float, default=0.08)
    ap.add_argument("--right", type=float, default=0.08)
    ap.add_argument("--att", default="70,1", help="cache-aware att_context_size (70,1 = 80 ms look-ahead, 70,13 = 1.04 s)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--nemo-repo", default="/scratch/pa2753/unified/Speech")  # main clone = the nemo31 env
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mref = f"model_path={a.model}" if a.model.endswith(".nemo") else f"pretrained_name={a.model}"
    sys.path.insert(0, f"{a.pkg}/tools")
    import score as tscore  # Tianyu's scorer, stdlib only
    os.makedirs(a.out, exist_ok=True)
    mdir = f"{a.out}/manifests"; os.makedirs(mdir, exist_ok=True)
    cfg = {"offline": "offline", "stream": f"L{a.left:g}_C{a.chunk:g}_R{a.right:g}",
           "cache": f"cache{a.att.replace(',', '_')}"}[a.mode]
    if os.path.exists(f"{a.out}/{a.tag}_{cfg}_scores.json"):  # done before a preemption: skip, so requeues resume
        print(f"skip {a.tag} {cfg}: scores exist"); return

    model = lang = None
    if a.mode == "offline":
        import nemo.collections.asr as nemo_asr
        load = nemo_asr.models.ASRModel.restore_from if a.model.endswith(".nemo") else nemo_asr.models.ASRModel.from_pretrained
        model = load(a.model).cuda().eval()
        pd = model.cfg.get("model_defaults", {}).get("prompt_dictionary") or {}
        lang = {"target_lang": a.lang if a.lang in pd else next(k for k in ("en-US", "en") if k in pd)} if pd else {}

    scores = {}
    for s, csvs in SETS.items():
        rows = [r for c in csvs for r in csv.DictReader(open(f"{a.pkg}/phone_streaming_nemotron/data/{c}.csv"))]
        mpath = f"{mdir}/{s}.json"
        with open(mpath, "w") as f:
            for r in rows:
                f.write(json.dumps({"audio_filepath": r["torch_path"], "duration": sf.info(r["torch_path"]).duration,
                                    "text": r["text"]}) + "\n")
        if a.mode == "offline":
            out = model.transcribe([r["torch_path"] for r in rows], batch_size=8, **lang)
            hyps = [o.text if hasattr(o, "text") else str(o) for o in out]
        elif a.mode == "stream":
            ofile = f"{a.out}/{a.tag}_{cfg}_{s}_stream.json"
            subprocess.run([sys.executable, f"{a.nemo_repo}/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py",
                            mref, f"dataset_manifest={mpath}", f"output_filename={ofile}",
                            f"left_context_secs={a.left}", f"chunk_secs={a.chunk}", f"right_context_secs={a.right}",
                            "att_context_size_as_chunk=true", f"target_lang={a.lang}", "batch_size=8"], check=True)
            pred = [json.loads(l) for l in open(ofile)]
            assert [p["audio_filepath"] for p in pred] == [r["torch_path"] for r in rows], "stream output order differs"
            hyps = [p["pred_text"] for p in pred]
        else:
            odir = f"{a.out}/{a.tag}_{cfg}_{s}_cache"
            subprocess.run([sys.executable, f"{a.nemo_repo}/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py",
                            mref, f"dataset_manifest={mpath}", f"output_path={odir}",
                            f"att_context_size=[{a.att}]", "online_normalization=true", "batch_size=8"], check=True)
            pred = [json.loads(l) for f in os.listdir(odir) if f.endswith(".json") for l in open(f"{odir}/{f}")]
            assert len(pred) == len(rows), "cache-aware output length differs"
            hyps = [p["pred_text"] for p in pred]
        H = {}
        with open(f"{a.out}/{a.tag}_{cfg}_{s}.tsv", "w") as f:
            for r, h in zip(rows, hyps):
                f.write(f"{r['file']}\t{h.strip()}\n"); H[stem(r["file"])] = tscore.norm(h)
        R = {}
        for rs in REF_SETS.get(s, [s]):
            R.update({stem(k): v for k, v in tscore.refs(rs).items()})
        assert set(R) == set(H), f"{s}: clip sets differ ({len(R)} refs vs {len(H)} hyps)"
        scores[s] = tscore.score([(R[k], H[k]) for k in sorted(R)])
        print(f"{a.tag} {cfg} {s} WER {scores[s]['wer']:.2f} exact {scores[s]['exact']} new-word {scores[s]['new_word_recall']}", flush=True)
    json.dump(scores, open(f"{a.out}/{a.tag}_{cfg}_scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
