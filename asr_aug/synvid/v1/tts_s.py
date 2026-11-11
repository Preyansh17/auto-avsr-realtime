"""V1 TTS: the 169 V0 support sentences (S) x 3 healthy Higgs stock voices (en_woman, en_man, belinda) = 507 clips.
Reuses make_drive.py gen/check unchanged (seed 1000 + item index, whisper-large-v3 pass if WER <= 0.10).
TTS input text = the S row's spoken-form `norm`; training target stays the row's raw_sentence (same target as V0 arm A1).
  gen:      python tts_s.py gen --shard i --nshard n
  check:    python tts_s.py check
  manifest: python tts_s.py manifest   -> /scratch/pa2753/synvid/v1/tts/v1_S_audio.jsonl (passing clips, audio_only rows)"""
import json, sys
sys.path.insert(0, "/scratch/pa2753/synvid/code")
import make_drive
S = [json.loads(l) for l in open("/scratch/pa2753/synvid/v0/split/roles/v0_S.jsonl") if l.strip()]
OUT = "/scratch/pa2753/synvid/v1/tts"
ITEMS = [(f"{r['id']}__v{j}", r["norm"]) for r in S for j in range(3)]  # index i -> voice VOICES[i % 3] = j
make_drive.items = lambda which: ITEMS
if sys.argv[1] == "manifest":
    raw = {r["id"]: r for r in S}
    ok = [json.loads(l) for l in open(f"{OUT}/whisper.jsonl") if l.strip()]
    ok = [r for r in ok if r["pass"]]
    with open(f"{OUT}/v1_S_audio.jsonl", "w") as f:
        for r in ok:
            s = raw[r["id"].rsplit("__v", 1)[0]]
            f.write(json.dumps({"id": r["id"], "audio_only": True, "audio_path": r["wav"], "raw_sentence": s["raw_sentence"],
                                "transcript": s["transcript"], "video_path": None, "voice": r["voice"]}) + "\n")
    cov = {r["id"].rsplit("__v", 1)[0] for r in ok}
    print(json.dumps({"clips": len(ITEMS), "pass": len(ok), "sentences_with_audio": len(cov), "sentences": len(S)}))
else:
    sys.argv = ["make_drive.py"] + sys.argv[1:] + (["--out", OUT] if sys.argv[1] == "gen" else ["--dir", OUT, "--out", f"{OUT}/whisper.jsonl"])
    make_drive.main()
