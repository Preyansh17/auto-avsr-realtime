"""A13 step 0.3a: video CTC forced alignment of each pool clip's known transcript on its devcrop npz.

Same method as align_words.py `video` (USR 2.0 Huge VSR CTC head, greedy unigram1000 unit split, torchaudio
forced_align, one encoder frame = one 25 fps frame); only the reader differs: devcrop gray96 (0..255) -> /255 ->
centre 88 -> Normalize(0.421, 0.165), which equals judge.py's transform on a grey crop. Run with cwd = usr2 checkout.
Out JSONL: {id, frames, missing, words: [[word, start, end_excl], ...]} (or error).
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/scratch/pa2753/synvid/code")
from align_words import ctc_align, done_ids, norm, tokenize_units, word_spans  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True); ap.add_argument("--crops", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from utils.utils import UNIGRAM1000_LIST as TOK
    units = {u: i for i, u in enumerate(TOK)}
    OmegaConf.register_new_resolver("len", len, replace=True)
    with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "conf"), version_base="1.3"):
        cfg = compose("config", overrides=["model/backbone=resnet_transformer_huge"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = E2E(len(TOK), cfg.model.backbone)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if any(k.startswith("_orig_mod.") for k in ck):
        ck = {k.replace("_orig_mod.", "", 1): v for k, v in ck.items()}
    if any(k.startswith("model.backbone.") for k in ck):
        ck = {k.replace("model.backbone.", "", 1): v for k, v in ck.items() if k.startswith("model.backbone.")}
    model.load_state_dict(ck); model.eval().to(dev)
    rows = [json.loads(l) for l in open(a.pool) if l.strip()]
    rows = [r for r in rows if "drop" not in r]
    done = done_ids(a.out, "id")
    with open(a.out, "a") as fo, torch.no_grad():
        for r in rows:
            if r["id"] in done:
                continue
            rec = dict(id=r["id"], ref=r["norm"])
            try:
                z = np.load(f"{a.crops}/{r['id']}.npz")
                g = torch.from_numpy(z["gray96"].astype(np.float32)) / 255.0
                g = g[:, 4:92, 4:92]  # CenterCrop(88) of 96
                x = ((g - 0.421) / 0.165).unsqueeze(0).to(dev)  # (1,T,88,88)
                feat = model.encoder(xs_v=x)
                logp = model.ctc_v.log_softmax(feat)[0]
                words = norm(r["norm"]).split()
                ids, wot = tokenize_units(words, units)
                sp = word_spans(ctc_align(logp, ids, blank=0), wot)
                rec.update(frames=int(logp.shape[0]), missing=int(z["missing"].sum()), units=len(ids),
                           words=[[w, s, e] for w, (s, e) in zip(words, sp)])
            except Exception as e:
                rec.update(error=f"{type(e).__name__}: {e}"[:300])
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(rec["id"], rec.get("error", len(rec.get("words", []))), flush=True)


if __name__ == "__main__":
    main()
