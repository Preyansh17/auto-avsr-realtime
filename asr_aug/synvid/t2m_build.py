"""Text-to-mouth step 1 (README A11): splice 96x96 semg3 mouth clips for new text by unit selection from real s3_train
mouth crops. CPU only, no generator.

Inputs (all from step 1.4 / A9, s3_train only)
  crops      p1/crop240/crops/<take_id>.mp4      96x96 colour mouth crops, 25 fps
  landmarks  p1/crop240/landmarks/<take_id>.npy  FAN 68 points per frame (aperture as in lip_activity.py)
  lip        p1/lip_activity_s3_train.json       per-clip active span (frozen threshold)
  align      glsft/video_align.jsonl             one CTC peak frame per word (judge's video CTC head)
  cmudict    nltk cmudict file                   first pronunciation, stress removed
Held-out: random.Random(20261110).sample(sorted take_ids, 24) -> t2m/heldout.txt; their clips never enter the inventory.
Word span: cut at the midpoints between consecutive word peaks; first word from the lip span start (or its peak if
earlier), last word to the lip span end (or its peak if later).
Phones: spread inside the word span in proportion to AVG_DUR. Diphone = phone midpoint to next phone midpoint (also across
words); the first/last diphone of a target word is extended to the source phone's start/end.
Selection (Viterbi): target cost whole word 0, exact diphone 0.5, viseme-class diphone 1, fallback 2; join cost
= mean |gray48 a - gray48 b| / 255 + 5 |aperture a - aperture b| + 2 |mean brightness a - b| / 255 at the boundary
frames. At most 30 candidates per slot (random.Random(seed) per slot). Cross-fade 2 frames (1 inside a diphone word).
6-frame real rest windows (outside the lip span) at start and end.
Sets: a = 24 held-out sentences, b = the 100 D2 pilot sentences (pilot/pilot.jsonl).
Out: t2m/<set>/clips/<sid>.mp4, t2m/<set>/ap/<sid>.npy (carried aperture), t2m/<set>/rows.jsonl, t2m/build_report.json
"""
import argparse, json, os, random, re, subprocess, sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lip_activity import aperture  # noqa: E402

SV = "/scratch/pa2753/synvid"
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
FFMPEG = f"{SV}/envs/latentsync/bin/ffmpeg"
K = 30
REST = 6
AVG_DUR = dict(AA=.12, AE=.12, AH=.07, AO=.12, AW=.15, AY=.14, EH=.09, ER=.11, EY=.12, IH=.07, IY=.10, OW=.12, OY=.16,
               UH=.07, UW=.10, B=.06, CH=.09, D=.05, DH=.04, F=.09, G=.06, HH=.06, JH=.08, K=.07, L=.06, M=.07, N=.06,
               NG=.07, P=.07, R=.06, S=.10, SH=.11, T=.06, TH=.09, V=.06, W=.06, Y=.06, Z=.08, ZH=.08)
VIS = {}
for cls, ps in dict(bilab="P B M", labdent="F V", dent="TH DH", alv="T D N L S Z", post="SH ZH CH JH",
                    vel="K G NG HH", r="R", w="W", y="Y", round="UW UH OW AO OY AW", open="AA AE AH AY",
                    spread="IY IH EY EH", er="ER").items():
    for p in ps.split():
        VIS[p] = cls


def norm(t):  # asr_aug/shared_eval/score_shared.py
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def load_cmu(path):
    d = {}
    for l in open(path, encoding="latin-1"):
        p = l.split()
        if len(p) >= 3 and p[0].lower() not in d:
            d[p[0].lower()] = [re.sub(r"\d", "", x) for x in p[2:]]
    return d


def read_frames(path):
    cap, fr = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return np.stack(fr) if fr else None


class Clip:
    def __init__(self, tid, frames, ap):
        self.tid, self.f, self.ap = tid, frames, ap
        self.g48 = np.stack([cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (48, 48)).astype(np.float32) for x in frames])


def build_inventory(rows, clips, align, lip, cmu, held):
    words, di, mono, rest = defaultdict(list), defaultdict(list), defaultdict(list), []
    used, skipped = 0, defaultdict(int)
    for r in rows:
        tid = r["take_id"]
        if tid in held:
            continue
        if tid not in clips or tid not in align or tid not in lip:
            skipped["missing"] += 1; continue
        c, al, lp = clips[tid], align[tid], lip[tid]
        n = len(c.f)
        ws = [w[0] for w in al["words"]]
        if ws != norm(r["transcript"]).split():
            skipped["word_mismatch"] += 1; continue
        pk = [w[1] for w in al["words"]]
        b = [min(lp["start"], pk[0])] + [int(round((pk[i - 1] + pk[i]) / 2)) for i in range(1, len(pk))] \
            + [min(n, max(lp["end"] + 1, pk[-1] + 1))]
        if any(b[i + 1] <= b[i] for i in range(len(b) - 1)):
            skipped["bad_bounds"] += 1; continue
        used += 1
        for i, w in enumerate(ws):
            words[w].append((tid, b[i], b[i + 1]))
        seq = []  # (phone, start, end) floats, broken at words without a pronunciation
        for i, w in enumerate(ws):
            ph = cmu.get(w)
            if not ph:
                seq.append(None); continue
            wt = np.array([AVG_DUR.get(p, .08) for p in ph]); cut = b[i] + (b[i + 1] - b[i]) * np.concatenate([[0], np.cumsum(wt) / wt.sum()])
            for k, p in enumerate(ph):
                seq.append((p, cut[k], cut[k + 1]))
                mono[p].append((tid, cut[k], cut[k + 1]))
        for x, y in zip(seq[:-1], seq[1:]):
            if x and y:
                di[(x[0], y[0])].append((tid, x[1], (x[1] + x[2]) / 2, (y[1] + y[2]) / 2, y[2]))
        for s0, s1 in ((0, b[0]), (b[-1], n)):
            for k in range(s0, s1 - REST + 1, 2):
                rest.append((tid, k, k + REST))
    return dict(words=words, di=di, mono=mono, rest=rest, used=used, skipped=dict(skipped))


def slot_candidates(slot, inv, rng):
    kind = slot["kind"]
    if kind == "rest":
        c = [(t, s, e, 0.0) for t, s, e in inv["rest"]]
    elif kind == "word":
        c = [(t, s, e, 0.0) for t, s, e in inv["words"][slot["word"]]]
    elif kind == "mono":
        p = slot["ph"]
        c = [(t, int(round(s)), int(round(e)), 0.5) for t, s, e in inv["mono"].get(p, [])]
        if not c:
            c = [(t, int(round(s)), int(round(e)), 1.0) for q, L in inv["mono"].items() if VIS.get(q) == VIS.get(p) for t, s, e in L]
    else:
        a, b, first, last = slot["a"], slot["b"], slot["first"], slot["last"]

        def mk(L, cost):
            out = []
            for t, sa, ma, mb, eb in L:
                s, e = int(round(sa if first else ma)), int(round(eb if last else mb))
                out.append((t, s, max(e, s + 1), cost))
            return out
        c = mk(inv["di"].get((a, b), []), 0.5)
        if not c:
            c = mk([x for (p, q), L in inv["di"].items() if VIS.get(p) == VIS.get(a) and VIS.get(q) == VIS.get(b) for x in L], 1.0)
        if not c:
            c = mk([x for (p, q), L in inv["di"].items() if VIS.get(q) == VIS.get(b) for x in L], 2.0)
    if len(c) > K:
        c = rng.sample(c, K)
    return c


def synth(text, inv, clips, cmu, seed):
    ws = norm(text).split()
    slots, unknown, invocab = [dict(kind="rest", grp="r0")], [], 0
    for i, w in enumerate(ws):
        if w in inv["words"]:
            slots.append(dict(kind="word", word=w, grp=f"w{i}")); invocab += 1
        elif w in cmu:
            ph = cmu[w]
            if len(ph) == 1:
                slots.append(dict(kind="mono", ph=ph[0], grp=f"w{i}"))
            for k in range(len(ph) - 1):
                slots.append(dict(kind="di", a=ph[k], b=ph[k + 1], first=k == 0, last=k == len(ph) - 2, grp=f"w{i}"))
        else:
            unknown.append(w)
    slots.append(dict(kind="rest", grp="r1"))
    cands = []
    for j, s in enumerate(slots):
        c = slot_candidates(s, inv, random.Random(seed * 1000 + j))
        c = [x for x in c if x[2] <= len(clips[x[0]].f) and x[2] > x[1]]
        if not c:
            return None, dict(unknown=unknown, empty_slot=s)
        cands.append(c)

    def edge(x, last):
        cl = clips[x[0]]; k = x[2] - 1 if last else x[1]
        return cl.g48[k], cl.ap[k], float(cl.g48[k].mean())

    # Viterbi
    cost = [np.array([x[3] for x in cands[0]])]
    back = []
    for j in range(1, len(cands)):
        prev_e = [edge(x, True) for x in cands[j - 1]]
        cur_e = [edge(x, False) for x in cands[j]]
        J = np.array([[np.abs(pg - cg).mean() / 255 + 5 * abs(pa - ca) + 2 * abs(pb - cb) / 255
                       for (cg, ca, cb) in cur_e] for (pg, pa, pb) in prev_e])
        tot = cost[-1][:, None] + J + np.array([x[3] for x in cands[j]])[None, :]
        back.append(tot.argmin(0)); cost.append(tot.min(0))
    k = int(cost[-1].argmin()); path = [k]
    for bk in reversed(back):
        k = int(bk[k]); path.append(k)
    path = path[::-1]
    out_f, out_a, units = [], [], []
    for j, (s, k) in enumerate(zip(slots, path)):
        t, a0, a1, tc = cands[j][k]
        U = clips[t].f[a0:a1].astype(np.float32); A = clips[t].ap[a0:a1].astype(np.float64)
        c = 1 if j > 0 and s["kind"] in ("di", "mono") and slots[j - 1]["grp"] == s["grp"] else 2
        ce = min(c, len(out_f) - 1, len(U) - 1) if out_f else 0
        for m in range(max(0, ce)):
            al = (m + 1) / (ce + 1)
            out_f[-ce + m] = (1 - al) * out_f[-ce + m] + al * U[m]
            out_a[-ce + m] = (1 - al) * out_a[-ce + m] + al * A[m]
        out_f += list(U[max(0, ce):]); out_a += list(A[max(0, ce):])
        units.append(dict(kind=s["kind"], tid=t, s=int(a0), e=int(a1), tcost=tc))
    return (np.clip(np.stack(out_f), 0, 255).astype(np.uint8), np.array(out_a)), dict(
        unknown=unknown, invocab=invocab, words=len(ws), units=units, cost=float(cost[-1].min()))


def write_mp4(frames, path):
    p = subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "96x96", "-r", "25",
                        "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", path + ".tmp.mp4"],
                       input=frames.tobytes(), check=True)
    os.replace(path + ".tmp.mp4", path)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default=f"{SV}/t2m")
    ap_.add_argument("--cmudict", default="/home/pa2753/nltk_data/corpora/cmudict/cmudict")
    a = ap_.parse_args()
    rows = [json.loads(l) for l in open(f"{ROLES}/s3_train.jsonl") if l.strip()]
    for r in rows:
        r.setdefault("take_id", os.path.basename(r["video_path"])[:-4])
    tids = sorted(r["take_id"] for r in rows)
    held = set(random.Random(20261110).sample(tids, 24))
    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/heldout.txt", "w").write("\n".join(sorted(held)) + "\n")
    cmu = load_cmu(a.cmudict)
    align = {}
    for l in open(f"{SV}/glsft/video_align.jsonl"):
        x = json.loads(l)
        if x.get("words"):
            align[x["take_id"]] = x
    lip = {c["take_id"]: c for c in json.load(open(f"{SV}/p1/lip_activity_s3_train.json"))["per_clip"] if c.get("start") is not None}
    clips = {}
    for r in rows:
        t = r["take_id"]
        if t in held:
            continue
        f = read_frames(f"{SV}/p1/crop240/crops/{t}.mp4")
        lm = np.load(f"{SV}/p1/crop240/landmarks/{t}.npy")
        apx = aperture(lm) if f is not None else None
        if f is None or apx is None:
            continue
        n = min(len(f), len(apx))
        clips[t] = Clip(t, f[:n], apx[:n])
    inv = build_inventory(rows, clips, align, lip, cmu, held)
    rep = dict(heldout=len(held), clips_loaded=len(clips), inventory_clips=inv["used"], skipped=inv["skipped"],
               vocab=len(inv["words"]), diphone_types=len(inv["di"]), rest_windows=len(inv["rest"]))
    sets = dict(a=[dict(sid="a_" + r["take_id"], text=r["transcript"], src=r["take_id"]) for r in rows if r["take_id"] in held],
                b=[dict(sid=p["pid"], text=p["text"]) for p in map(json.loads, open(f"{SV}/pilot/pilot.jsonl")) ])
    train_sents = {norm(r["transcript"]) for r in rows if r["take_id"] not in held}
    for name, items in sets.items():
        d = f"{a.out}/{name}"
        os.makedirs(f"{d}/clips", exist_ok=True); os.makedirs(f"{d}/ap", exist_ok=True)
        nw = niv = 0
        with open(f"{d}/rows.jsonl", "w") as fo:
            for i, it in enumerate(items):
                res, info = synth(it["text"], inv, clips, cmu, seed=20261111 + i)
                ws = norm(it["text"]).split()
                nw += len(ws); niv += sum(w in inv["words"] for w in ws)
                row = dict(it, words=len(ws), in_vocab=sum(w in inv["words"] for w in ws),
                           sentence_in_train=norm(it["text"]) in train_sents, unknown=info.get("unknown", []))
                if res is None:
                    row.update(valid=False, reason=str(info.get("empty_slot")))
                else:
                    fr, apc = res
                    p = f"{d}/clips/{it['sid']}.mp4"
                    write_mp4(fr, p); np.save(f"{d}/ap/{it['sid']}.npy", apc)
                    row.update(valid=not info["unknown"], path=p, frames=int(len(fr)), units=info["units"], cost=info["cost"])
                fo.write(json.dumps(row) + "\n")
        rep[f"set_{name}"] = dict(n=len(items), words=nw, in_vocab_word_rate=niv / max(1, nw),
                                  sentence_in_train=sum(norm(x["text"]) in train_sents for x in items))
    json.dump(rep, open(f"{a.out}/build_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
