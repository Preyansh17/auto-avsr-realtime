"""Phase 3b (plan section 11, items 3b.1 and 3b.2): driving audio for three G-LSft arms on the D2 pilot pairs.

Every arm keeps the D2 pilot's pid, template and target sentence (pilot/pilot.jsonl); only the driving wav changes.
  rate_a  (3b.1a) normal-rate Higgs audio: the pool clip's speech span (first word start .. last word end, the same
          audio CTC word timings drive_final.py used), unstretched, with the same lead-in and tail silence as the
          D2 driving clip.
  rate_b  (3b.1b) the same normal-rate speech span, cut at the midpoints of the gaps between consecutive words, with
          equal digital silence inserted at every cut so the speech span is as long as the D2 (1.47x stretched)
          span; same lead-in and tail, so the clip length equals the D2 clip. One-word clips get the silence after the
          word.
  own     (3b.2) each template driven by healthy Higgs TTS of the template's own transcript, retimed piecewise onto
          the real clip's word timings (glsft/retimed/<take_id>.wav from the A9 prep). Target sentence = template
          sentence, so template leak is not defined. Templates without a retimed wav (whisper fail or tempo out of
          range in the A9 prep) are left out. Note: these are G-LSft's own training pairs.
Out: <out-root>/<arm>/rows.jsonl (pilot.jsonl format, drive_wav/drive_sec replaced) and <out-root>/<arm>/audio/*.wav.
"""
import argparse, json, os

import numpy as np
import soundfile as sf

SV = "/scratch/pa2753/synvid"
SR = 16000


def sil(sec):
    return np.zeros(int(round(max(0.0, sec) * SR)), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=f"{SV}/pilot/pilot.jsonl")
    ap.add_argument("--drive", default=f"{SV}/data/drive.json")
    ap.add_argument("--align", default=f"{SV}/drive/pool/audio_align.jsonl")
    ap.add_argument("--retimed", default=f"{SV}/glsft/retimed")
    ap.add_argument("--out-root", default=f"{SV}/phase3b")
    a = ap.parse_args()
    pilot = [json.loads(l) for l in open(a.pilot) if l.strip()]
    drive = {d["id"]: d for d in json.load(open(a.drive))}
    al = {}
    for l in open(a.align):
        if l.strip():
            x = json.loads(l)
            al[x["id"]] = x
    out = {k: [] for k in ("rate_a", "rate_b", "own")}
    stats = {k: [] for k in out}
    for k in out:
        os.makedirs(f"{a.out_root}/{k}/audio", exist_ok=True)
    for p in pilot:
        d, x = drive[p["id"]], al[p["id"]]
        y, sr = sf.read(x["wav"], dtype="float32")
        assert sr == SR, (x["wav"], sr)
        w = x["words"]
        s, e = w[0][1], w[-1][2]
        span = y[int(s * SR):int(e * SR)]
        lead, tail = sil(d["lead_s"]), sil(d["tail_s"])
        # 3b.1a: unstretched span
        za = np.concatenate([lead, span, tail])
        # 3b.1b: unstretched words, silence inserted at gap midpoints to reach the D2 span length
        target_span = p["drive_sec"] - d["lead_s"] - d["tail_s"]
        extra = max(0.0, target_span - (e - s))
        cuts = [int(((w[i][2] + w[i + 1][1]) / 2 - s) * SR) for i in range(len(w) - 1)]
        pieces = np.split(span, cuts) if cuts else [span]
        gap = sil(extra / max(1, len(pieces) - 1))
        if len(pieces) == 1:
            zb = np.concatenate([lead, span, sil(extra), tail])
        else:
            zb = np.concatenate([lead] + [q for i, pc in enumerate(pieces) for q in ((pc, gap) if i < len(pieces) - 1 else (pc,))] + [tail])
        for k, z in (("rate_a", za), ("rate_b", zb)):
            f = f"{a.out_root}/{k}/audio/{p['pid']}.wav"
            sf.write(f, z, SR)
            out[k].append(dict(p, drive_wav=f, drive_sec=round(len(z) / SR, 3)))
            stats[k].append(dict(sec=len(z) / SR, spw=(len(z) - len(lead) - len(tail)) / SR / len(w)))
        # 3b.2: own sentence
        r = f"{a.retimed}/{p['template_tid']}.wav"
        if os.path.exists(r):
            n = sf.info(r).frames
            out["own"].append(dict(p, text=p["template_text"], words=len(p["template_text"].split()), drive_wav=r,
                                   drive_sec=round(n / SR, 3), voice="retimed_tts", pingpong=False))
            stats["own"].append(dict(sec=n / SR))
    rep = {}
    for k, rows in out.items():
        with open(f"{a.out_root}/{k}/rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        rep[k] = dict(n=len(rows), sec_median=float(np.median([s["sec"] for s in stats[k]])))
        if "spw" in stats[k][0]:
            rep[k]["span_sec_per_word_median"] = float(np.median([s["spw"] for s in stats[k]]))
    rep["d2_stretched_sec_median"] = float(np.median([p["drive_sec"] for p in pilot]))
    json.dump(rep, open(f"{a.out_root}/audio_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
