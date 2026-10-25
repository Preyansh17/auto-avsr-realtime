"""Phase 0 feasibility gates for the Higgs LoRA finetune. Run on a GPU node (see gates.sbatch).

  1  forward/loss support + module tree + LoRA target match
  2  teacher-forcing sanity: matched vs shuffled vs mis-shifted labels, builder parity with the processor
  3  audio tokenizer round trip + delay-pattern inverse
  4  peft wrap: fwd+bwd in bf16, adapter is a no-op at init (byte-identical generate), generate with adapter,
     adapter save/load round trip, seconds per training step

Writes <out>/gates.json and prints one PASS/FAIL line per gate.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import higgs_ft_lib as L  # noqa: E402

DATA = os.environ.get("HIGGS_FT_DATA", "/scratch/pa2753/higgs_ft/data")
LN_V = math.log(1026)
RES: dict = {}


def gate(name):
    def deco(fn):
        def run(*a, **k):
            print(f"\n===== {name} =====", flush=True)
            try:
                out = fn(*a, **k)
                RES[name] = {"pass": bool(out.pop("pass")), **out}
            except Exception:
                RES[name] = {"pass": False, "error": traceback.format_exc()}
                print(RES[name]["error"], flush=True)
            print(f"{name}: {'PASS' if RES[name]['pass'] else 'FAIL'}", flush=True)
        return run
    return deco


def tgt_codes(proc, clips, u):
    p = f"{DATA}/codes/{u}.pt"
    if os.path.exists(p):
        return torch.load(p).long()
    return L.encode_wave(proc, L.load_wave(clips[u]["audio_path"]))


def mean_ce(model, ex, dev, labels_pos=None):
    s, c = L.example_ce(model, ex, dev, labels_pos)
    return float(s.sum() / c.sum().clamp(min=1)), (s / c.clamp(min=1)).tolist(), int(c.sum())


@gate("gate1_forward_and_modules")
def gate1(model, proc, sp):
    cfg = model.config
    sig = inspect.signature(model.forward)
    names = [n for n, _ in model.named_modules()]
    matched = [n for n in names if re.fullmatch(L.LORA_TARGET_REGEX, n)]
    print("top-level:", [n for n, _ in model.named_children()], "| model.*:", [n for n, _ in model.model.named_children()])
    print("layer0 modules:", [n.split("layers.0.")[1] for n in names if "layers.0." in n and n.count(".") >= 3][:24])
    print(f"audio_lm_head {tuple(model.audio_lm_head.weight.shape)}  text_lm_head={model.text_lm_head}")
    print(f"codebooks={cfg.num_codebooks} codebook_size={cfg.codebook_size} stream_bos={cfg.audio_stream_bos_id} "
          f"stream_eos={cfg.audio_stream_eos_id} audio_token={cfg.audio_token_id} delay_token={cfg.audio_delay_token_id}")
    train = sp["tts_train"]
    u, v = train[0], train[1]
    ex = L.build_example(proc, sp["clips"][v]["transcript"], tgt_codes(proc, sp["clips"], v),
                         sp["clips"][u]["transcript"], tgt_codes(proc, sp["clips"], u))
    ids = ex["input_ids"][0]
    tk = proc.tokenizer
    print("prompt head:", repr(tk.decode(ids[:14])))
    a = (ids == cfg.audio_token_id) | (ids == cfg.audio_delay_token_id)
    first, last = int(a.nonzero()[0]), int(a.nonzero()[-1])
    print("prompt tail:", repr(tk.decode(ids[last - 2 :])), f"| seq len {ids.shape[0]}, audio frames {int(a.sum())}")
    return {"pass": "audio_labels" in sig.parameters and len(matched) == 28 * 10,
            "lora_target_modules_matched": len(matched), "codebook_size": cfg.codebook_size,
            "has_audio_labels_arg": "audio_labels" in sig.parameters}


@gate("gate2_teacher_forcing")
def gate2(model, proc, sp, dev):
    clips, train = sp["clips"], sp["tts_train"]
    pool = L.RefPool(clips, train)
    rng = random.Random(0)
    ids = sorted(train)[:: max(1, len(train) // 10)][:10]
    codes = {u: tgt_codes(proc, clips, u) for u in ids}
    refs = {}
    for i, u in enumerate(ids):
        w, t, src = pool.sample(rng, u, k=[1, 3, 5][i % 3])
        refs[u] = (w, t, L.encode_wave(proc, w))

    def ex_of(ref_u, text_u, audio_u=None, eos=False):
        _, rt, rc = refs[ref_u]
        return L.build_example(proc, rt, rc, clips[text_u]["transcript"], codes[audio_u or text_u], eos_labels=eos)

    # (a) builder parity vs the stock processor on two examples, and HF-loss parity
    par = []
    for u in ids[:2]:
        rw, rt, rc = refs[u]
        ex = ex_of(u, u)
        msgs = [{"role": "system", "content": L.SYSTEM},
                {"role": "user", "content": L.normalize_for_higgs(rt)},
                {"role": "assistant", "content": [{"type": "audio"}]},
                {"role": "user", "content": L.normalize_for_higgs(clips[u]["transcript"])},
                {"role": "assistant", "content": [{"type": "audio"}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        tw = L.load_wave(clips[u]["audio_path"])
        ref = proc(text=prompt, audio=[rw, tw], sampling_rate=L.SR, return_tensors="pt", output_labels=True)
        ref = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in ref.items()}  # processor tokenizes on the GPU
        same_ids = bool(torch.equal(ref["input_ids"], ex["input_ids"]))
        fr_ref, fr_me = ref["audio_input_ids"][0], ex["audio_input_ids"][0]
        eq = float((fr_ref == fr_me).float().mean()) if fr_ref.shape == fr_me.shape else -1.0
        n_ref = ex["n_ref_frames"]
        lab_ref = ref["audio_labels"][0][n_ref:]
        lab_me = ex["audio_labels"][0][n_ref:]
        # processor masks BOS and EOS; mine (eos_labels=False) masks BOS and EOS too -> identical on the target span
        lab_same = bool(torch.equal(lab_ref, lab_me))
        hf = model(**{k: ex[k].to(dev) for k in L.MODEL_KEYS}, audio_labels=ex["audio_labels"].to(dev)).loss
        s, c = L.example_ce(model, ex, dev)
        mine = float((s / c.clamp(min=1)).sum())
        par.append({"same_input_ids": same_ids, "frame_equal_frac": eq, "target_labels_equal": lab_same,
                    "hf_loss": float(hf), "my_sum_of_codebook_means": mine})
        print("parity:", par[-1], flush=True)
    parity_ok = all(p["same_input_ids"] and p["frame_equal_frac"] > 0.98 and p["target_labels_equal"]
                    and abs(p["hf_loss"] - p["my_sum_of_codebook_means"]) < 0.05 * max(1.0, p["hf_loss"])
                    for p in par)

    # (b) matched vs controls
    rows = []
    with torch.no_grad():
        for i, u in enumerate(ids):
            j = ids[(i + 1) % len(ids)]
            m, per_cb, ntok = mean_ce(model, ex_of(u, u), dev)
            sh, _, _ = mean_ce(model, L.build_example(proc, refs[u][1], refs[u][2], clips[u]["transcript"], codes[j], False), dev)
            tx, _, _ = mean_ce(model, L.build_example(proc, refs[u][1], refs[u][2], clips[j]["transcript"], codes[u], False), dev)
            exm = ex_of(u, u)
            lp = exm["labels_pos"]
            s1, _, _ = mean_ce(model, exm, dev, torch.roll(lp, 1, 0))
            s2, _, _ = mean_ce(model, exm, dev, torch.roll(lp, -1, 0))
            exe = L.build_example(proc, refs[u][1], refs[u][2], clips[u]["transcript"], codes[u], True)
            eos_only = torch.where(exe["labels_pos"] == L.EOS_ID, exe["labels_pos"], torch.full_like(exe["labels_pos"], L.IGNORE))
            eos_ce = mean_ce(model, exe, dev, eos_only)[0]
            rows.append({"utt": u, "matched": m, "audio_shuffled": sh, "text_shuffled": tx, "shift+1": s1,
                         "shift-1": s2, "eos_ce": eos_ce, "tokens": ntok, "per_codebook": per_cb})
            print(f"{u[:8]} matched={m:.3f} audio_shuf={sh:.3f} text_shuf={tx:.3f} shift+1={s1:.3f} shift-1={s2:.3f} "
                  f"eos_ce={eos_ce:.3f}  n={ntok}", flush=True)
    avg = {k: float(np.mean([r[k] for r in rows])) for k in
           ("matched", "audio_shuffled", "text_shuffled", "shift+1", "shift-1", "eos_ce")}
    pcb = np.mean([r["per_codebook"] for r in rows], axis=0).round(3).tolist()
    print("MEAN:", {k: round(v, 3) for k, v in avg.items()}, f"| ln(1026)={LN_V:.3f}", "| per-codebook matched:", pcb)
    ok = (avg["matched"] < 0.75 * LN_V and avg["matched"] < avg["audio_shuffled"] - 0.3
          and avg["matched"] < min(avg["shift+1"], avg["shift-1"]) - 0.3)
    return {"pass": ok and parity_ok, "mean": avg, "per_codebook_matched": pcb, "parity": par, "rows": rows,
            "ln_codebook_size": LN_V, "builder_parity_ok": parity_ok}


@gate("gate3_tokenizer_roundtrip")
def gate3(proc, sp, dev, out):
    import librosa
    import soundfile as sf

    clips = sp["clips"]
    ids = sorted(sp["tts_train"])[5:: max(1, len(sp["tts_train"]) // 3)][:3]
    os.makedirs(f"{out}/roundtrip", exist_ok=True)
    orig, dec, rows = {}, {}, []
    for u in ids:
        w = L.load_wave(clips[u]["audio_path"])
        codes = L.encode_wave(proc, w)
        back = L.revert_delay_pattern(L.build_delay_pattern(codes))
        with torch.inference_mode():
            wav = proc.audio_tokenizer.decode(codes[None].to(dev)).audio_values.cpu().float().squeeze().numpy()
        sf.write(f"{out}/roundtrip/{u[:8]}_orig.wav", w, L.SR)
        sf.write(f"{out}/roundtrip/{u[:8]}_dec.wav", wav, L.SR)
        orig[u], dec[u] = w, wav
        rows.append({"utt": u, "orig_sec": len(w) / L.SR, "dec_sec": len(wav) / L.SR, "frames": int(codes.shape[1]),
                     "delay_roundtrip_ok": bool(torch.equal(back, codes))})

    def mel(x):
        return np.log(librosa.feature.melspectrogram(y=x, sr=L.SR, n_mels=64) + 1e-6)

    for r, u in zip(rows, ids):
        other = next(v for v in ids if v != u)
        a, b, o = mel(orig[u]), mel(dec[u]), mel(orig[other])
        n = min(a.shape[1], b.shape[1], o.shape[1])
        r["mel_l1_own"], r["mel_l1_other"] = float(np.abs(a[:, :n] - b[:, :n]).mean()), float(np.abs(o[:, :n] - b[:, :n]).mean())
        print(r, flush=True)
    ok = all(abs(r["orig_sec"] - r["dec_sec"]) < 0.1 and r["delay_roundtrip_ok"]
             and r["mel_l1_own"] < 0.7 * r["mel_l1_other"] for r in rows)
    return {"pass": ok, "rows": rows, "wavs": f"{out}/roundtrip (listen: *_orig.wav vs *_dec.wav)"}


@gate("gate4_peft")
def gate4(model, proc, sp, dev, out):
    from peft import LoraConfig, PeftModel, get_peft_model

    clips, train = sp["clips"], sp["tts_train"]
    pool = L.RefPool(clips, train)
    rng = random.Random(1)
    tgt = train[3]
    t0 = time.time()
    rw, rt, src = pool.sample(rng, tgt, k=3)
    rc = L.encode_wave(proc, rw)
    ref_secs = time.time() - t0
    ex = L.build_example(proc, rt, rc, clips[tgt]["transcript"], tgt_codes(proc, clips, tgt))

    # generate with the base model first (seeded, short), then again with a fresh (zero-B) adapter attached
    client = L.make_client(proc, model, dev)
    ref_wav = f"{out}/ref_for_gen.wav"
    import soundfile as sf
    sf.write(ref_wav, rw, L.SR)
    params = {"temperature": 1.0, "top_k": 50, "top_p": 0.95, "max_new_tokens": 60, "ras_win_len": 7,
              "ras_win_max_num_repeat": 2}
    text = clips[train[7]]["transcript"]
    client.generate(text, ref_wav, rt, f"{out}/gen_base.wav", 123, params, L.SYSTEM)

    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=L.LORA_TARGET_REGEX, bias="none")
    pm = get_peft_model(model, cfg)
    n_tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in pm.parameters())
    print(f"trainable {n_tr/1e6:.2f}M / {n_all/1e6:.1f}M ({100*n_tr/n_all:.2f}%)", flush=True)
    lora_dtypes = sorted({str(p.dtype) for p in pm.parameters() if p.requires_grad})
    client.model = pm
    pm.eval()
    client.generate(text, ref_wav, rt, f"{out}/gen_adapter0.wav", 123, params, L.SYSTEM)
    identical0 = open(f"{out}/gen_base.wav", "rb").read() == open(f"{out}/gen_adapter0.wav", "rb").read()
    print("generate with zero-init adapter byte-identical to base:", identical0, flush=True)

    # fwd+bwd, then a few steps overfitting one example: loss must fall, B must receive gradient
    pm.train()
    params_tr = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params_tr, lr=3e-4, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()
    losses, step_secs = [], []
    for step in range(6):
        t0 = time.time()
        s, c = L.example_ce(pm, ex, dev)
        loss = s.sum() / c.sum()
        loss.backward()
        if step == 0:
            gB = [p.grad.abs().max().item() for n, p in pm.named_parameters() if p.requires_grad and "lora_B" in n and p.grad is not None]
            n_grad = len(gB)
            gmax = max(gB) if gB else 0.0
        gn = torch.nn.utils.clip_grad_norm_(params_tr, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_secs.append(time.time() - t0)
        losses.append(float(loss.detach()))
        print(f"  step {step} loss={float(loss):.4f} gnorm={float(gn):.3f} {step_secs[-1]:.2f}s", flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30

    pm.eval()
    client.generate(text, ref_wav, rt, f"{out}/gen_adapter_trained.wav", 123, params, L.SYSTEM)
    differs = open(f"{out}/gen_base.wav", "rb").read() != open(f"{out}/gen_adapter_trained.wav", "rb").read()

    # save / reload round trip inside the process
    pm.save_pretrained(f"{out}/adapter_test")
    with torch.no_grad():
        before = mean_ce(pm, ex, dev)[0]
        pm.load_adapter(f"{out}/adapter_test", adapter_name="re")
        pm.set_adapter("re")
        after = mean_ce(pm, ex, dev)[0]
    sz = sum(os.path.getsize(f"{out}/adapter_test/{f}") for f in os.listdir(f"{out}/adapter_test")) / 2**20
    print(f"save/load: loss before {before:.4f} after reload {after:.4f}; adapter {sz:.0f} MiB; peak GPU {peak_gb:.1f} GiB; "
          f"{np.median(step_secs[1:]):.2f}s/step; ref build+encode {ref_secs:.2f}s", flush=True)
    ok = (identical0 and n_grad > 0 and gmax > 0 and all(math.isfinite(x) for x in losses) and losses[-1] < losses[0] - 0.05
          and differs and abs(before - after) < 1e-3)
    return {"pass": ok, "trainable_params_M": n_tr / 1e6, "lora_dtypes": lora_dtypes, "losses": losses,
            "sec_per_step_median": float(np.median(step_secs[1:])), "peak_gpu_gib": peak_gb,
            "zero_init_generate_identical": identical0, "trained_generate_differs": differs,
            "grad_on_lora_B": n_grad, "save_reload_loss": [before, after], "adapter_mib": sz,
            "ref_build_encode_sec": ref_secs, "seq_len": int(ex["input_ids"].shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}; torch {torch.__version__}", flush=True)
    sp = json.load(open(f"{DATA}/splits.json"))
    proc, model = L.load_models(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    gate1(model, proc, sp)
    gate2(model, proc, sp, dev)
    gate3(proc, sp, dev, args.out)
    gate4(model, proc, sp, dev, args.out)
    json.dump(RES, open(f"{args.out}/gates.json", "w"), indent=1, default=str)
    print("\n" + "\n".join(f"{k}: {'PASS' if v['pass'] else 'FAIL'}" for k, v in RES.items()), flush=True)
    sys.exit(0 if all(v["pass"] for v in RES.values()) else 1)


if __name__ == "__main__":
    main()
