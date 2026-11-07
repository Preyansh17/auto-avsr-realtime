"""Phase 2.3: rate match, jitter and padding of the driving clips that passed the whisper check (plan 3.2 step 4, A5).

Inputs
  --whisper   drive/pool/whisper.jsonl (make_drive.py check)
  --align     word timings of those clips (align_words.py audio)
  --lip       p1/lip_activity_s3_train.json (lip_activity.py on the real s3_train crops, step 1.4)
Steps, per clip that passed whisper (WER <= 0.10) and aligned:
  1 speech span = first word start .. last word end (audio CTC timings); active sec/word = span / words.
  2 one global factor g = median real active sec/word / median driving active sec/word (frozen, written to the report).
  3 per clip a jitter j ~ U[0.9, 1.1] (random.Random(20261107), in fixed clip order); the speech span is time-stretched
    with pitch kept (ffmpeg atempo) to span * g * j.
  4 lead-in and tail silence are drawn independently from the real s3_train per-clip lead_s / tail_s values (A5)
    and added as digital silence; the original leading/trailing audio outside the speech span is dropped.
Out: <out-dir>/<id>.wav (16 kHz) and data/drive.json [{id, text, voice, wav, sec, g, jitter, lead_s, tail_s}],
plus drive_report.json.
"""
import argparse, json, os, random, subprocess, tempfile

import numpy as np
import soundfile as sf


def atempo_chain(f):
    parts = []
    while f > 2.0:
        parts.append(2.0); f /= 2.0
    while f < 0.5:
        parts.append(0.5); f /= 0.5
    parts.append(f)
    return ",".join(f"atempo={p:.6f}" for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--lip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.out_dir = os.path.abspath(a.out_dir)  # paths in drive.json must work from any cwd
    os.makedirs(a.out_dir, exist_ok=True)
    wh = [json.loads(l) for l in open(a.whisper) if l.strip()]
    al = {json.loads(l)["id"]: json.loads(l) for l in open(a.align) if l.strip()}
    lip = json.load(open(a.lip))
    real = [c for c in lip["per_clip"] if c.get("active_spw") is not None]
    leads = [c["lead_s"] for c in real]
    tails = [c["tail_s"] for c in real]
    real_spw = float(np.median([c["active_spw"] for c in real]))

    clips = []
    for r in sorted(wh, key=lambda x: x["id"]):
        x = al.get(r["id"])
        if not r["pass"] or x is None or "error" in x or not x["words"]:
            continue
        s, e = x["words"][0][1], x["words"][-1][2]
        if e - s <= 0.1:
            continue
        clips.append(dict(r, s=s, e=e, spw=(e - s) / len(x["words"])))
    g = real_spw / float(np.median([c["spw"] for c in clips]))
    rng = random.Random(20261107)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for c in clips:
            j = rng.uniform(0.9, 1.1)
            lead, tail = rng.choice(leads), rng.choice(tails)
            p = os.path.join(a.out_dir, f"{c['id']}.wav")
            if not os.path.exists(p):
                y, sr = sf.read(c["wav"], dtype="float32")
                seg = y[int(c["s"] * sr):int(c["e"] * sr)]
                i, o = os.path.join(tmp, "i.wav"), os.path.join(tmp, "o.wav")
                sf.write(i, seg, sr)
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", i, "-filter:a",
                                atempo_chain(1.0 / (g * j)), "-ar", "16000", o], check=True)
                z, _ = sf.read(o, dtype="float32")
                z = np.concatenate([np.zeros(int(round(lead * 16000)), np.float32), z,
                                    np.zeros(int(round(tail * 16000)), np.float32)])
                sf.write(p + ".tmp.wav", z, 16000)
                os.replace(p + ".tmp.wav", p)
            n = sf.info(p).frames
            out.append(dict(id=c["id"], text=c["text"], voice=c["voice"], wav=p, sec=round(n / 16000, 3),
                            g=round(g, 4), jitter=round(j, 4), lead_s=round(lead, 3), tail_s=round(tail, 3),
                            whisper_wer=c["wer"]))
    json.dump(out, open(a.out, "w"), indent=1)
    rep = dict(n_whisper=len(wh), n_pass=sum(r["pass"] for r in wh), n_final=len(out), g=g,
               real_active_spw_median=real_spw, drive_active_spw_median=float(np.median([c["spw"] for c in clips])),
               sec_median=float(np.median([o["sec"] for o in out])), sec_max=float(max(o["sec"] for o in out)),
               per_voice={v: sum(o["voice"] == v for o in out) for v in sorted({o["voice"] for o in out})})
    json.dump(rep, open(os.path.join(os.path.dirname(a.out), "drive_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
