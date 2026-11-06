"""Zero-shot USR 2.0 lip-reading judge for mouth-crop mp4s.

Input: a role JSONL (rows with video_path, transcript, take_id) or a TSV "path<TAB>reference".
Output: JSONL, one row per clip: take_id, path, ref, hyp, errors, ref_words, wer, wer_cap1.

Crops are fed exactly as usr2/demo.py does after its own mouth cropping (commit recorded in phase1.md):
  cv2 read BGR -> RGB uint8 (T,H,W,3) -> float (C,T,H,W) -> /255 -> CenterCrop(88) -> torchvision Grayscale
  (ITU-R 601 luma) -> Normalize(mean 0.421, std 0.165) -> (T,88,88). No resampling: the crops are 25 fps already.
  Encoder: model.encoder(xs_v=...) (video only). Decoding: usr2 BatchBeamSearch, defaults beam 40, ctc 0.1.
Scoring: asr_aug/shared_eval/score_shared.py norm (lowercase, hyphen -> space, drop chars outside [a-z' ]).

Run from the usr2 checkout:  cd $USR2 && python $SV/code/judge.py --rows X.jsonl --ckpt M.pth --backbone huge --out O.jsonl
"""
import argparse, json, os, re, sys, time

import cv2
import numpy as np
import torch
from torchvision.transforms import CenterCrop, Compose, Grayscale, Lambda


def norm(t):  # identical to asr_aug/shared_eval/score_shared.py
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def load_rows(path):
    rows = []
    for l in open(path):
        l = l.rstrip("\n")
        if not l:
            continue
        if path.endswith(".jsonl"):
            x = json.loads(l)
            rows.append(dict(take_id=x.get("take_id", os.path.basename(x["video_path"])[:-4]),
                             path=x["video_path"], ref=x["transcript"]))
        else:
            p, r = l.split("\t", 1)
            rows.append(dict(take_id=os.path.basename(p)[:-4], path=p, ref=r))
    return rows


def read_crop(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not fr:
        raise RuntimeError(f"no frames: {path}")
    return torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float()  # (C,T,H,W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="huge", choices=["base", "baseplus", "large", "huge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=40)
    ap.add_argument("--ctc", type=float, default=0.1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()

    usr2 = os.getcwd()
    sys.path.insert(0, usr2)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from data.transforms import NormalizeVideo
    from espnet.asr.asr_utils import parse_hypothesis
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from espnet.nets.scorers.length_bonus import LengthBonus
    from utils.utils import UNIGRAM1000_LIST as TOK

    OmegaConf.register_new_resolver("len", len, replace=True)
    with initialize_config_dir(config_dir=os.path.join(usr2, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=[f"model/backbone=resnet_transformer_{a.backbone}",
                                           f"decode.beam_size={a.beam}", f"decode.ctc_weight={a.ctc}"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = E2E(len(TOK), cfg.model.backbone)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if any(k.startswith("_orig_mod.") for k in ck):
        ck = {k.replace("_orig_mod.", "", 1): v for k, v in ck.items()}
    if any(k.startswith("model.backbone.") for k in ck):
        ck = {k.replace("model.backbone.", "", 1): v for k, v in ck.items() if k.startswith("model.backbone.")}
    model.load_state_dict(ck)
    model.eval().to(dev)
    sc = model.scorers()
    sc["length_bonus"] = LengthBonus(len(TOK))
    bs = BatchBeamSearch(beam_size=cfg.decode.beam_size, vocab_size=len(TOK),
                         weights=dict(decoder=1.0 - cfg.decode.ctc_weight, ctc=cfg.decode.ctc_weight,
                                      length_bonus=cfg.decode.penalty),
                         scorers=sc, sos=len(TOK) - 1, eos=len(TOK) - 1, token_list=TOK,
                         pre_beam_score_key=None if cfg.decode.ctc_weight == 1.0 else "decoder").to(dev)
    tf = Compose([Lambda(lambda x: x / 255.0), CenterCrop(88), Lambda(lambda x: x.transpose(0, 1)), Grayscale(),
                  Lambda(lambda x: x.transpose(0, 1)), NormalizeVideo(mean=(0.421,), std=(0.165,)),
                  Lambda(lambda x: x.squeeze(0))])

    rows = load_rows(a.rows)[a.shard::a.nshard]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["take_id"] for l in open(a.out) if l.strip()}
    with open(a.out, "a") as fo, torch.no_grad():
        for r in rows:
            if r["take_id"] in done:
                continue
            t0 = time.time()
            v = tf(read_crop(r["path"])).unsqueeze(0).to(dev)
            feat = model.encoder(xs_v=v)
            hy = bs(x=feat.squeeze(0), modality="v", maxlenratio=cfg.decode.maxlenratio,
                    minlenratio=cfg.decode.minlenratio)
            txt, _, _, _ = parse_hypothesis(hy[0].asdict(), TOK)
            txt = txt.replace("<eos>", "").replace("▁", " ").strip()
            R, H = norm(r["ref"]).split(), norm(txt).split()
            e = ed(R, H)
            w = e / max(1, len(R))
            fo.write(json.dumps(dict(take_id=r["take_id"], path=r["path"], ref=r["ref"], hyp=txt, errors=e,
                                     ref_words=len(R), wer=w, wer_cap1=min(1.0, w), frames=int(v.shape[1]),
                                     sec=round(time.time() - t0, 2))) + "\n")
            fo.flush()


if __name__ == "__main__":
    main()
