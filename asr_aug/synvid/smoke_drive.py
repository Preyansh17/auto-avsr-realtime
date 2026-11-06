"""Step 1.5 driving clip for the generator smoke tests: one healthy-voice Higgs clip (no patient voice, no adapter).

Text: the first pool_B sentence (3-8 words, A4) that matches no sentence of any semg3 role after normalisation.
Voice: Higgs v2 base with no reference audio (the model's own default voice). If the processor refuses a prompt
without audio, falls back to a healthy TORGO control speaker reference (FC/MC) and says so in the JSON.
Output: <out>/drive_smoke.wav (Higgs 24 kHz) + drive_smoke_16k.wav + drive_smoke.json.
Env: /scratch/pa2753/envs/nemo_asr python with PYTHONPATH=/scratch/pa2753/higgs_site, HF_HOME=/scratch/pa2753/hf_cache.
"""
import copy, glob, json, os, re, sys

sys.path.insert(0, "/scratch/pa2753/higgs_gen")
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def main(out):
    os.makedirs(out, exist_ok=True)
    used = set()
    for f in glob.glob(f"{ROLES}/s3*.jsonl"):
        for l in open(f):
            if l.strip():
                used.add(norm(json.loads(l)["transcript"]))
    pool = json.load(open("/scratch/pa2753/asr_aug/data/pool_B.json"))
    pool = pool if isinstance(pool, list) else pool.get("sentences", list(pool.values())[0])
    pool = [p if isinstance(p, str) else p.get("text") for p in pool]
    text = next(s for s in pool if 3 <= len(norm(s).split()) <= 8 and norm(s) not in used)

    from gen_higgs import Client, normalize_for_higgs, display_text
    import soundfile as sf, librosa, torch
    cl = Client("bosonai/higgs-audio-v2-generation-3B-base")
    params = dict(temperature=0.7, top_k=50, top_p=0.95, max_new_tokens=1024, ras_win_len=7, ras_win_max_num_repeat=2)
    wav_path = os.path.join(out, "drive_smoke.wav")
    mode = "no_reference"
    msgs = [{"role": "system", "content": "Generate audio following instruction"},
            {"role": "user", "content": normalize_for_higgs(display_text(text))}]
    try:
        prompt = cl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        with torch.inference_mode():
            inputs = cl.processor(text=prompt, return_tensors="pt").to(cl.device)
            torch.manual_seed(1)
            gc = copy.deepcopy(cl.model.generation_config)
            gc.do_sample, gc.temperature, gc.top_k, gc.top_p = True, 0.7, 50, 0.95
            gc.max_new_tokens, gc.ras_win_len, gc.ras_win_max_num_repeat = 1024, 7, 2
            ids = cl.model.generate(**inputs, generation_config=gc)
            cl.processor.save_audio(cl.processor.decode(ids), wav_path, sampling_rate=cl.sr)
    except Exception as e:  # fallback: healthy TORGO control speaker as reference
        mode = f"torgo_control_reference (no-ref failed: {type(e).__name__}: {str(e)[:200]})"
        refs = sorted(glob.glob("/scratch/pa2753/gen/data/torgo_dl/**/FC*/**/wav_arrayMic/*.wav", recursive=True))
        ref = refs[0]
        ptxt = ref.replace("wav_arrayMic", "prompts")[:-4] + ".txt"
        rtext = open(ptxt).read().strip() if os.path.exists(ptxt) else ""
        cl.generate(display_text(text), ref, rtext, wav_path, 1, params, "Generate audio following instruction")
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    sf.write(os.path.join(out, "drive_smoke_16k.wav"), y, 16000)
    json.dump(dict(text=text, mode=mode, sec=len(y) / 16000), open(os.path.join(out, "drive_smoke.json"), "w"), indent=1)
    print(json.dumps(dict(text=text, mode=mode, sec=len(y) / 16000)))


if __name__ == "__main__":
    main(sys.argv[1])
