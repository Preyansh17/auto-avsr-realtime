"""Phase 2.2 / 2.3 driving audio (plan 3.2 steps 2-4, amendment A5).

  gen     Higgs v2 base, zero-shot voice cloning from 3 healthy stock voice prompts (boson-ai/higgs-audio
          examples/voice_prompts: en_woman, en_man, belinda; Apache-2.0), round-robin by item index. Never a patient
          voice, no adapters. One clip per item, seed = 1000 + index. Items: data/text.json (generation pool) and,
          with --set train, the 240 s3_train transcripts (G-LSft training pairs, A9). Writes <out>/<id>.wav (24 kHz)
          and <id>_16k.wav. Sharded (--shard/--nshard) and resumable (existing wavs are skipped).
  check   whisper-large-v3 zero-shot (HF transformers, English, greedy) on the 16 kHz wavs; WER after score_shared
          normalisation. Pass if WER <= 0.10. Out: JSONL {id, text, hyp, wer, pass}.
Env: /scratch/pa2753/envs/nemo_asr python, PYTHONPATH=/scratch/pa2753/higgs_site, HF_HOME=/scratch/pa2753/hf_cache,
HF_HUB_OFFLINE=1.
"""
import argparse, json, os, re, sys

sys.path.insert(0, "/scratch/pa2753/higgs_gen")
VOICES = ["en_woman", "en_man", "belinda"]
VDIR = "/scratch/pa2753/synvid/voices"
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def items(which):
    if which == "pool":
        return [(r["id"], r["text"]) for r in json.load(open("/scratch/pa2753/synvid/data/text.json"))]
    out = []
    for l in open(f"{ROLES}/s3_train.jsonl"):
        if l.strip():
            x = json.loads(l)
            out.append((x.get("take_id", os.path.basename(x["video_path"])[:-4]), x["transcript"]))
    return out


def cmd_gen(a):
    import librosa, soundfile as sf
    from gen_higgs import Client, display_text
    its = items(a.set)
    os.makedirs(a.out, exist_ok=True)
    cl = Client("bosonai/higgs-audio-v2-generation-3B-base")
    params = dict(temperature=0.7, top_k=50, top_p=0.95, max_new_tokens=1024, ras_win_len=7, ras_win_max_num_repeat=2)
    refs = {v: (f"{VDIR}/{v}.wav", open(f"{VDIR}/{v}.txt").read().strip()) for v in VOICES}
    meta = open(os.path.join(a.out, f"gen_{a.shard}.jsonl"), "a")
    for i, (iid, text) in enumerate(its):
        if i % a.nshard != a.shard:
            continue
        p16 = os.path.join(a.out, f"{iid}_16k.wav")
        if os.path.exists(p16):
            continue
        v = VOICES[i % len(VOICES)]
        p = os.path.join(a.out, f"{iid}.wav")
        try:
            cl.generate(display_text(text), refs[v][0], refs[v][1], p, 1000 + i, params,
                        "Generate audio following instruction")
            y, _ = librosa.load(p, sr=16000, mono=True)
            sf.write(p16 + ".tmp.wav", y, 16000)
            os.replace(p16 + ".tmp.wav", p16)
            rec = dict(id=iid, text=text, voice=v, seed=1000 + i, wav=p16, sec=round(len(y) / 16000, 3))
        except Exception as e:
            rec = dict(id=iid, text=text, voice=v, error=f"{type(e).__name__}: {e}"[:300])
        meta.write(json.dumps(rec) + "\n"); meta.flush()
        print(json.dumps(rec), flush=True)


def cmd_check(a):
    import glob, torch, soundfile as sf
    from transformers import pipeline
    meta = {}
    for f in glob.glob(os.path.join(a.dir, "gen_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if "error" not in r:
                meta[r["id"]] = r
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16,
                   device="cuda:0")
    todo = [r for k, r in sorted(meta.items()) if k not in done]
    with open(a.out, "a") as fo:
        for b in range(0, len(todo), 16):
            batch = todo[b:b + 16]
            ys = [sf.read(r["wav"], dtype="float32")[0] for r in batch]
            hyps = asr([{"raw": y, "sampling_rate": 16000} for y in ys], batch_size=16,
                       generate_kwargs=dict(language="en", task="transcribe"))
            for r, h in zip(batch, hyps):
                R, H = norm(r["text"]).split(), norm(h["text"]).split()
                w = ed(R, H) / max(1, len(R))
                fo.write(json.dumps(dict(id=r["id"], text=r["text"], voice=r["voice"], wav=r["wav"], sec=r["sec"],
                                         hyp=h["text"], wer=round(w, 4), **{"pass": w <= 0.10})) + "\n")
            fo.flush()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen"); g.add_argument("--set", choices=["pool", "train"], default="pool")
    g.add_argument("--out", required=True); g.add_argument("--shard", type=int, default=0)
    g.add_argument("--nshard", type=int, default=1)
    c = sp.add_parser("check"); c.add_argument("--dir", required=True); c.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(gen=cmd_gen, check=cmd_check)[a.cmd](a)


if __name__ == "__main__":
    main()
