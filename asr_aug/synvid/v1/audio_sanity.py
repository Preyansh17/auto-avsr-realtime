"""V1 step 0: does the init checkpoint's AUDIO branch read healthy TTS speech?
Greedy CTC (ctc_a) on the first N Higgs s3_train drive clips (drive/train, whisper-checked). Prints WER.
If the audio branch is untrained in vsr_huge_hr.pth (VSR-only fine-tune), WER will be near 100% and V1 needs another init."""
import argparse, json, sys, re
from pathlib import Path
import torch, torchaudio
sys.path.insert(0, "/scratch/pa2753/synvid/v0/code")
import usr2_ft
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="/scratch/pa2753/synvid/models/usr2/vsr_huge_hr.pth"); ap.add_argument("--n", type=int, default=60)
a = ap.parse_args(); a.backbone = "resnet_transformer_huge"
dev = torch.device("cuda"); model, units = usr2_ft.build(a, dev); model.eval()
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower())).strip().split()
def ed(x, y):
    d = list(range(len(y) + 1))
    for i in range(1, len(x) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(y) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x[i - 1] != y[j - 1]))
    return d[len(y)]
rows = [json.loads(l) for l in open("/scratch/pa2753/synvid/drive/train/whisper.jsonl")][: a.n]
E = N = 0
for i, r in enumerate(rows):
    wav, sr = torchaudio.load(r["wav"]); wav = wav.mean(0); assert sr == 16000
    wav = torch.nn.functional.pad(wav, (0, (-wav.numel()) % 640))
    with torch.no_grad():
        feat = model.encoder(xs_a=wav.view(1, -1, 1).to(dev))
        ids = model.ctc_a.log_softmax(feat).argmax(-1)[0].tolist()
    toks, prev = [], None
    for t in ids:
        if t != prev and t != 0: toks.append(units[t])
        prev = t
    hyp = "".join(toks).replace("<eos>", "").replace("▁", " ")
    e = ed(norm(r["text"]), norm(hyp)); E += e; N += len(norm(r["text"]))
    if i < 6: print(r["text"], "|", hyp, flush=True)
print(json.dumps({"n": len(rows), "audio_greedy_wer": round(100 * E / N, 2)}))
