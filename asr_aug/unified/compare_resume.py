"""Compare the kill-and-resume test runs (test_resume.sbatch): B vs A against the A2 vs A floor.

    python compare_resume.py /scratch/pa2753/unified/resume_test
"""
import glob
import io
import json
import os
import re
import sys
import tarfile

import torch

T = sys.argv[1]


def maxdiff(a, b):
    assert a.keys() == b.keys(), "different parameter sets"
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a if a[k].is_floating_point())


def whisper(run):
    ck = max(glob.glob(f"{T}/{run}/checkpoint-*"), key=lambda d: int(d.rsplit("-", 1)[1]))
    h = json.load(open(f"{ck}/trainer_state.json"))["log_history"]
    from safetensors.torch import load_file
    w = {}
    for f in sorted(glob.glob(f"{T}/{run}/best/*.safetensors")):
        w.update(load_file(f))
    assert w, f"{run}: no best/ weights (training did not finish)"
    return ({x["step"]: x["loss"] for x in h if "loss" in x}, [round(x["eval_wer"], 6) for x in h if "eval_wer" in x], w)


def nemo(run):
    txt = "".join(open(p).read() for p in sorted(glob.glob(f"{T}/{run}.log*")))
    val = {}
    for e, w in re.findall(r"\[val\] epoch=(\d+) val_wer=([\d.]+)", txt):
        val[int(e)] = float(w)  # a resumed run re-reports epochs after its checkpoint; the later value wins
    with tarfile.open(f"{T}/{run}/nemotron_patient.nemo") as t:
        m = next(x for x in t.getmembers() if x.name.endswith("model_weights.ckpt"))
        w = torch.load(io.BytesIO(t.extractfile(m).read()), map_location="cpu")
    return val, w


for name, fn, runs in (("whisper", whisper, ("wA", "wA2", "wB")), ("nemotron", nemo, ("nA", "nA2", "nB"))):
    try:
        A, A2, B = (fn(r) for r in runs)
    except Exception as e:  # a run that failed (no best/ or .nemo) must not hide the other trainer's result
        print(f"== {name}: could not load all runs ({type(e).__name__}: {e}), skipped")
        continue
    print(f"== {name}")
    if name == "whisper":
        for lbl, X in (("A2 vs A", A2), ("B vs A", B)):
            steps = sorted(set(A[0]) & set(X[0]))
            dl = max(abs(A[0][s] - X[0][s]) for s in steps) if steps else float("nan")
            print(f"  {lbl}: logged-loss max |diff| {dl:.3g} over {len(steps)} logs; eval_wer {X[1]} vs {A[1]}; "
                  f"best weights max |diff| {maxdiff(A[2], X[2]):.3g}")
    else:
        for lbl, X in (("A2 vs A", A2), ("B vs A", B)):
            print(f"  {lbl}: val_wer by epoch {[X[0].get(e) for e in sorted(A[0])]} vs {[A[0][e] for e in sorted(A[0])]}; "
                  f"final weights max |diff| {maxdiff(A[1], X[1]):.3g}")
