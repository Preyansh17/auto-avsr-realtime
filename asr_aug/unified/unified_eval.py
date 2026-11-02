"""Zero-shot benchmark of NeMo RNNT models on the shared s1/s2 val/test/oov rows (PLAN.md 3.3 item 0).

    python unified_eval.py --model nvidia/nemotron-labs-unified-asr-0.6b --tag unified --mode offline
    python unified_eval.py --model ... --tag unified --mode stream --chunk 0.56 --right 0.56   # buffered streaming

Manifests come from Tianyu's role files (read only); refs are the role transcripts, scored with the
report_stack.py normaliser (lowercase, punctuation stripped). Offline uses model.transcribe; streaming runs NeMo's
examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py from a NeMo Speech checkout (--nemo-repo).
Writes <out>/<tag>_<cfg>_<set>_hyps.tsv (take_id, ref, hyp) and prints WER per set.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import soundfile as sf

ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
SETS = [f"{p}_{s}" for p in ("s1", "s2") for s in ("val", "test", "oov")] + ["s2ov_test"]  # s2ov_test = Tianyu's semg2 test-open (38)


def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def manifest(s, mdir):
    path = f"{mdir}/{s}.json"
    rows = [json.loads(l) for l in open(f"{ROLES}/{s}.jsonl")]
    if not os.path.exists(path):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps({"audio_filepath": r["audio_path"], "duration": sf.info(r["audio_path"]).duration,
                                    "text": r["transcript"], "take_id": r["take_id"]}) + "\n")
    return path, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["offline", "stream"], default="offline")
    ap.add_argument("--left", type=float, default=5.6)
    ap.add_argument("--chunk", type=float, default=0.56)
    ap.add_argument("--right", type=float, default=0.56)
    ap.add_argument("--lang", default="en", help="target_lang for language-prompted models; ignored otherwise")
    ap.add_argument("--nemo-repo", default="/scratch/pa2753/unified/Speech_v300")  # v3.0.0 tag = installed NeMo
    ap.add_argument("--manifests", default="/scratch/pa2753/unified/manifests")
    ap.add_argument("--out", default="/scratch/pa2753/unified/out")
    a = ap.parse_args()
    os.makedirs(a.manifests, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    cfg = "offline" if a.mode == "offline" else f"L{a.left:g}_C{a.chunk:g}_R{a.right:g}"

    model = None
    if a.mode == "offline":
        import nemo.collections.asr as nemo_asr
        load = nemo_asr.models.ASRModel.restore_from if a.model.endswith(".nemo") else nemo_asr.models.ASRModel.from_pretrained
        model = load(a.model).cuda().eval()
        # NeMo 3.0.0's prompt model takes target_lang via **prompt; keys are its prompt_dictionary ("en", "en-US", ...)
        pd = model.cfg.get("model_defaults", {}).get("prompt_dictionary") or {}
        lang = {}
        if pd:
            key = a.lang if a.lang in pd else next(k for k in ("en-US", "en") if k in pd)
            lang = {"target_lang": key}
            print("prompt languages:", [k for k in pd if k.startswith("en")], "->", key, pd[key])
        print("transcribe kwargs:", lang)

    for s in SETS:
        mpath, rows = manifest(s, a.manifests)
        if a.mode == "offline":
            out = model.transcribe([r["audio_path"] for r in rows], batch_size=8, **lang)
            hyps = [o.text if hasattr(o, "text") else str(o) for o in out]
        else:
            ofile = f"{a.out}/{a.tag}_{cfg}_{s}_stream.json"
            cmd = [sys.executable, f"{a.nemo_repo}/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py",
                   (f"model_path={a.model}" if a.model.endswith(".nemo") else f"pretrained_name={a.model}"), f"dataset_manifest={mpath}", f"output_filename={ofile}",
                   f"left_context_secs={a.left}", f"chunk_secs={a.chunk}", f"right_context_secs={a.right}",
                   "att_context_size_as_chunk=true", f"target_lang={a.lang}", "batch_size=8"]
            subprocess.run(cmd, check=True)
            pred = [json.loads(l) for l in open(ofile)]
            assert [p["audio_filepath"] for p in pred] == [r["audio_path"] for r in rows], "stream output order differs"
            hyps = [p["pred_text"] for p in pred]
        e = w = 0
        with open(f"{a.out}/{a.tag}_{cfg}_{s}_hyps.tsv", "w") as f:
            f.write("take_id\tref\thyp\n")
            for r, h in zip(rows, hyps):
                ref, hn = norm(r["transcript"]), norm(h)
                e += ed(ref.split(), hn.split()); w += len(ref.split())
                f.write(f"{r['take_id']}\t{ref}\t{h.strip()}\n")
        print(f"{a.tag} {cfg} {s} WER {100 * e / w:.2f}", flush=True)


if __name__ == "__main__":
    main()
