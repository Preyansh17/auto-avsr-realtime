"""G-LSft training pairs (A9): word timings on the real lip video and on a healthy TTS clip, then a piecewise retime of
the TTS clip so its words land on the video's word timings.

Subcommands
  video   CTC forced alignment of the transcript on the mouth crop with the USR 2.0 judge's video CTC head
          (env envs/usr2, cwd = the usr2 checkout). The transcript is cut into the judge's unigram1000 units by greedy
          longest match (word-initial units carry the U+2581 prefix). One encoder frame = one 25 fps video frame.
          Out: JSONL {take_id, frames, words: [[word, start_frame, end_frame_exclusive], ...]}.
  audio   CTC forced alignment of the transcript on a 16 kHz wav with facebook/hubert-large-ls960-ft (character CTC,
          20 ms frames, env nemo_asr, HF cache offline). Out: JSONL {id, sec, words: [[word, start_s, end_s], ...]}.
  retime  Cut points are speech start, the midpoints of the gaps between consecutive words, and speech end, on both
          sides. Each audio segment between two cut points is tempo-changed (ffmpeg atempo, pitch kept) to the length
          of the matching video segment. Leading and trailing silence are set to the video's. The output wav is
          exactly frames * 640 samples at 16 kHz (25 fps). Out: wav per pair + JSONL with per-segment tempo factors.
          Pairs whose word counts differ, or whose any tempo factor is outside [0.25, 4], are skipped and logged.
"""
import argparse, json, os, re, subprocess, sys, tempfile

import numpy as np
import torch


def norm(t):  # asr_aug/shared_eval/score_shared.py
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def done_ids(path, key):
    s = set()
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                s.add(json.loads(l)[key])
    return s


def word_spans(path_tokens, word_of_token):
    """path_tokens: per-frame aligned token index into the target list (or -1 for blank).
    Returns per word [start_frame, end_frame_exclusive] from first to last frame of its tokens."""
    spans = {}
    for t, k in enumerate(path_tokens):
        if k < 0:
            continue
        w = word_of_token[k]
        s = spans.setdefault(w, [t, t + 1])
        s[1] = t + 1
    return [spans[i] for i in range(max(word_of_token) + 1)]


def ctc_align(logp, targets, blank):
    """logp (T,V) log-probs, targets list[int]. Returns per-frame index into targets (-1 = blank)."""
    import torchaudio.functional as AF
    ali, _ = AF.forced_align(logp.unsqueeze(0).float().cpu(), torch.tensor([targets], dtype=torch.int32), blank=blank)
    ali = ali[0].tolist()
    out, k, prev = [], -1, blank
    for a in ali:  # map the label path back to target positions (repeats collapse, blank separates)
        if a == blank:
            out.append(-1)
        else:
            if a != prev:  # identical consecutive targets are always separated by a blank frame
                k += 1
            out.append(k)
        prev = a
    assert k == len(targets) - 1, (k, len(targets))
    return out


# ---------------------------------------------------------------------- video
def tokenize_units(words, units):
    """greedy longest match into unigram1000 units; first piece of each word carries U+2581."""
    ids, wot = [], []
    for wi, w in enumerate(words):
        w = w.upper()
        pos, first = 0, True
        while pos < len(w):
            best = None
            for e in range(len(w), pos, -1):
                piece = ("▁" if first else "") + w[pos:e]
                if piece in units:
                    best = (piece, e)
                    break
            if best is None:
                if first and "▁" in units:  # bare word-boundary unit, then continue inside the word
                    ids.append(units["▁"]); wot.append(wi); first = False
                    continue
                ids.append(units["<unk>"]); wot.append(wi); pos += 1; first = False
                continue
            ids.append(units[best[0]]); wot.append(wi); pos = best[1]; first = False
    return ids, wot


def cmd_video(a):
    usr2 = os.getcwd()
    sys.path.insert(0, usr2)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from torchvision.transforms import CenterCrop, Compose, Grayscale, Lambda
    from data.transforms import NormalizeVideo
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from utils.utils import UNIGRAM1000_LIST as TOK
    from judge import load_rows, read_crop

    units = {u: i for i, u in enumerate(TOK)}
    OmegaConf.register_new_resolver("len", len, replace=True)
    with initialize_config_dir(config_dir=os.path.join(usr2, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=[f"model/backbone=resnet_transformer_{a.backbone}"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = E2E(len(TOK), cfg.model.backbone)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if any(k.startswith("_orig_mod.") for k in ck):
        ck = {k.replace("_orig_mod.", "", 1): v for k, v in ck.items()}
    if any(k.startswith("model.backbone.") for k in ck):
        ck = {k.replace("model.backbone.", "", 1): v for k, v in ck.items() if k.startswith("model.backbone.")}
    model.load_state_dict(ck)
    model.eval().to(dev)
    tf = Compose([Lambda(lambda x: x / 255.0), CenterCrop(88), Lambda(lambda x: x.transpose(0, 1)), Grayscale(),
                  Lambda(lambda x: x.transpose(0, 1)), NormalizeVideo(mean=(0.421,), std=(0.165,)),
                  Lambda(lambda x: x.squeeze(0))])  # identical to judge.py
    rows = load_rows(a.rows)
    done = done_ids(a.out, "take_id")
    with open(a.out, "a") as fo, torch.no_grad():
        for r in rows:
            if r["take_id"] in done:
                continue
            words = norm(r["ref"]).split()
            rec = dict(take_id=r["take_id"], path=r["path"], ref=r["ref"])
            try:
                x = tf(read_crop(r["path"])).unsqueeze(0).to(dev)  # (1,T,88,88), as in judge.py
                feat = model.encoder(xs_v=x)
                logp = model.ctc_v.log_softmax(feat)[0]  # (T,V)
                ids, wot = tokenize_units(words, units)
                path = ctc_align(logp, ids, blank=0)
                sp = word_spans(path, wot)
                rec.update(frames=int(logp.shape[0]), units=len(ids),
                           words=[[w, s, e] for w, (s, e) in zip(words, sp)])
            except Exception as e:
                rec.update(error=f"{type(e).__name__}: {e}"[:300])
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(rec["take_id"], rec.get("error", len(rec.get("words", []))), flush=True)


# ---------------------------------------------------------------------- audio
def cmd_audio(a):
    import soundfile as sf
    from transformers import AutoProcessor, HubertForCTC
    name = "facebook/hubert-large-ls960-ft"
    proc = AutoProcessor.from_pretrained(name)
    model = HubertForCTC.from_pretrained(name).eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    vocab = proc.tokenizer.get_vocab()
    blank = proc.tokenizer.pad_token_id
    done = done_ids(a.out, "id")
    with open(a.out, "a") as fo, torch.no_grad():
        for l in open(a.items):
            iid, wav, text = l.rstrip("\n").split("\t", 2)
            if iid in done:
                continue
            rec = dict(id=iid, wav=wav, text=text)
            try:
                y, sr = sf.read(wav, dtype="float32")
                assert sr == 16000, sr
                y = y if y.ndim == 1 else y.mean(1)
                inp = proc(y, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
                logp = torch.log_softmax(model(inp).logits[0], -1)
                words = norm(text).split()
                ids, wot = [], []
                for wi, w in enumerate(words):
                    for ch in w.upper():
                        if ch in vocab:
                            ids.append(vocab[ch]); wot.append(wi)
                path = ctc_align(logp, ids, blank=blank)
                fr = len(y) / 16000 / logp.shape[0]
                sp = word_spans(path, wot)
                rec.update(sec=len(y) / 16000, frame_s=fr, words=[[w, s * fr, e * fr] for w, (s, e) in zip(words, sp)])
            except Exception as e:
                rec.update(error=f"{type(e).__name__}: {e}"[:300])
            fo.write(json.dumps(rec) + "\n"); fo.flush()


# ---------------------------------------------------------------------- retime
def cut_points(spans):
    """spans [[s,e],...] in seconds -> [speech start, gap midpoints..., speech end]"""
    c = [spans[0][0]]
    for (s0, e0), (s1, e1) in zip(spans[:-1], spans[1:]):
        c.append((e0 + s1) / 2)
    c.append(spans[-1][1])
    return c


def atempo_chain(f):
    parts = []
    while f > 2.0:
        parts.append(2.0); f /= 2.0
    while f < 0.5:
        parts.append(0.5); f /= 0.5
    parts.append(f)
    return ",".join(f"atempo={p:.6f}" for p in parts)


def stretch(seg, target_n, tmp):
    """tempo-change seg (float32 16k) to exactly target_n samples with ffmpeg atempo (pitch kept)."""
    import soundfile as sf
    if target_n <= 0:
        return np.zeros(0, np.float32)
    if len(seg) < 160:  # too short to stretch meaningfully: resample length by padding/cutting
        out = np.zeros(target_n, np.float32); out[:min(len(seg), target_n)] = seg[:target_n]; return out
    f = len(seg) / target_n
    i, o = os.path.join(tmp, "i.wav"), os.path.join(tmp, "o.wav")
    sf.write(i, seg, 16000)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", i, "-filter:a", atempo_chain(f), "-ar", "16000", o],
                   check=True)
    y, _ = sf.read(o, dtype="float32")
    out = np.zeros(target_n, np.float32)
    out[:min(len(y), target_n)] = y[:target_n]
    return out


def cmd_retime(a):
    import soundfile as sf
    vid = {json.loads(l)["take_id"]: json.loads(l) for l in open(a.video_align) if l.strip()}
    aud = {json.loads(l)["id"]: json.loads(l) for l in open(a.audio_align) if l.strip()}
    os.makedirs(a.out_dir, exist_ok=True)
    done = done_ids(a.out, "take_id")
    with open(a.out, "a") as fo, tempfile.TemporaryDirectory() as tmp:
        for l in open(a.pairs):
            tid, aid = l.split()[:2]
            if tid in done:
                continue
            v, au = vid.get(tid), aud.get(aid)
            rec = dict(take_id=tid, audio_id=aid)
            if v is None or au is None or "error" in v or "error" in au:
                rec["skip"] = "missing_or_error"
            elif len(v["words"]) != len(au["words"]):
                rec["skip"] = f"word_count {len(v['words'])} vs {len(au['words'])}"
            else:
                vs = [[s / 25.0, e / 25.0] for _, s, e in v["words"]]
                as_ = [[s, e] for _, s, e in au["words"]]
                cv, ca = cut_points(vs), cut_points(as_)
                y, _ = sf.read(au["wav"], dtype="float32")
                n_total = v["frames"] * 640
                tempos = [(ca[i + 1] - ca[i]) / max(cv[i + 1] - cv[i], 1e-3) for i in range(len(cv) - 1)]
                if min(tempos) < 0.25 or max(tempos) > 4:
                    rec.update(skip="tempo_range", tempos=tempos)
                else:
                    out = [np.zeros(int(round(cv[0] * 16000)), np.float32)]
                    for i in range(len(cv) - 1):
                        seg = y[int(round(ca[i] * 16000)):int(round(ca[i + 1] * 16000))]
                        n = int(round(cv[i + 1] * 16000)) - int(round(cv[i] * 16000))
                        out.append(stretch(seg, n, tmp))
                    z = np.concatenate(out)
                    z = np.pad(z, (0, max(0, n_total - len(z))))[:n_total]
                    p = os.path.join(a.out_dir, f"{tid}.wav")
                    sf.write(p, z, 16000)
                    rec.update(wav=p, frames=v["frames"], tempos=[round(t, 4) for t in tempos],
                               video_cuts=cv, audio_cuts=ca)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(tid, rec.get("skip", "ok"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("video"); v.add_argument("--rows", required=True); v.add_argument("--ckpt", required=True)
    v.add_argument("--backbone", default="huge"); v.add_argument("--out", required=True)
    au = sp.add_parser("audio"); au.add_argument("--items", required=True, help="TSV id<TAB>wav16k<TAB>text")
    au.add_argument("--out", required=True)
    r = sp.add_parser("retime"); r.add_argument("--video-align", required=True); r.add_argument("--audio-align", required=True)
    r.add_argument("--pairs", required=True, help="lines: take_id audio_id"); r.add_argument("--out-dir", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(video=cmd_video, audio=cmd_audio, retime=cmd_retime)[a.cmd](a)


if __name__ == "__main__":
    main()
