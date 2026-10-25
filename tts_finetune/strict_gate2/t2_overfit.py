"""T2: overfit a throwaway LoRA on K=4 clips with one fixed reference, then free-run those targets through gen_ft.

    python t2_overfit.py <out_dir> --arm aligned|shift_m1|shift_p1|noeos [--asr_only]

aligned  : labels as built by build_example
shift_m1 : labels_pos rolled -1 (logits at p learn frame p+2: the double-shift bug)
shift_p1 : labels_pos rolled +1 (logits at p learn frame p itself: the no-shift bug)
noeos    : eos_labels False
The aligned arm also free-runs the base model (no adapter) as the floor.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FT_DIR, SEED_BASE, GenCapture, L, asr_whisper, cached_codes, load_refs, load_splits, wer  # noqa: E402

sys.path.insert(0, FT_DIR)
DEV = "cuda:0"
K, MAX_STEPS, TARGET_CE, MAX_NEW = 4, 400, 0.5, 750
GEN = {"temperature": 0.3, "top_k": 50, "top_p": 0.95, "max_new_tokens": MAX_NEW, "ras_win_len": 7, "ras_win_max_num_repeat": 2}


def pick_targets(sp, ref):
    clips = sp["clips"]
    src = set(ref["source_utt_ids"])
    src_norm = {clips[u]["norm"] for u in src}
    cand = [u for u in sorted(sp["tts_train"]) if u not in src and clips[u]["norm"] not in src_norm
            and 2.0 <= clips[u]["duration_sec"] <= 5.0]
    random.Random(20261005).shuffle(cand)
    out, norms = [], set()
    for u in cand:
        if clips[u]["norm"] not in norms:
            out.append(u)
            norms.add(clips[u]["norm"])
        if len(out) == K:
            break
    return out


def labels_for(ex, arm):
    lp = ex["labels_pos"]
    if arm == "shift_m1":
        return torch.roll(lp, -1, 0)
    if arm == "shift_p1":
        return torch.roll(lp, 1, 0)
    return lp


@torch.no_grad()
def tf_ce(model, exs, arm):
    model.eval()
    s = c = 0.0
    for ex in exs:
        ss, cc = L.example_ce(model, ex, DEV, labels_for(ex, arm))
        s += float(ss.sum())
        c += float(cc.sum())
    return s / max(c, 1.0)


def train(out, arm, sp, ref, targets):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    proc, model = L.load_models(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=L.LORA_TARGET_REGEX, bias="none"))
    params = [p for p in pm.parameters() if p.requires_grad]
    dtypes = sorted({str(p.dtype) for p in params})
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    clips = sp["clips"]
    rc = L.encode_wave(proc, L.load_wave(ref["audio_path"]))
    exs = [L.build_example(proc, ref["transcript"], rc, clips[u]["transcript"], cached_codes(u), eos_labels=(arm != "noeos"))
           for u in targets]
    n_tok = sum(int((labels_for(e, arm)[1:] != L.IGNORE).sum()) for e in exs)
    warmup = max(1, int(0.05 * MAX_STEPS))
    log = [{"step": 0, "ce_own": tf_ce(pm, exs, arm), "ce_aligned": tf_ce(pm, exs, "aligned")}]
    print(f"[{arm}] lora dtypes {dtypes}; step 0 CE own={log[0]['ce_own']:.3f}", flush=True)
    t0, reached = time.time(), None
    for step in range(MAX_STEPS):
        pm.train()
        lr = L.cosine_lr(step, MAX_STEPS, warmup, 1e-4)
        for pg in opt.param_groups:
            pg["lr"] = lr
        for ex in exs:
            s, _ = L.example_ce(pm, ex, DEV, labels_for(ex, arm))
            (s.sum() / n_tok).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0:
            rec = {"step": step + 1, "lr": lr, "ce_own": tf_ce(pm, exs, arm), "ce_aligned": tf_ce(pm, exs, "aligned"),
                   "min": (time.time() - t0) / 60}
            log.append(rec)
            print(f"[{arm}] {json.dumps(rec)}", flush=True)
            if rec["ce_own"] < TARGET_CE:
                reached = step + 1
                break
    pm.save_pretrained(f"{out}/adapter")
    del pm, model, opt, params, exs
    torch.cuda.empty_cache()
    return {"lora_dtypes": dtypes, "train_log": log, "reached_ce_below_0.5_at_step": reached,
            "final_ce_own": log[-1]["ce_own"], "final_ce_aligned": log[-1]["ce_aligned"]}


def best_lag_match(gen, true, max_lag=5):
    T = true.shape[0]
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        j = np.arange(T)
        g = j + lag
        ok = (g >= 0) & (g < gen.shape[0])
        best = max(best, float((gen[g[ok]] == true[j[ok]]).sum()) / T)
    return best


def generate(out, arm, sp, ref, targets, adapter, tag):
    from gen_ft import FTClient

    client = FTClient(L.MODEL)
    clips = sp["clips"]
    res = []
    for u in targets:
        unit = {"name": tag, "out_root": f"{out}/gen_{tag}", "speaker": "bingge", "variant": f"t2_{arm}_{tag}",
                "seed_base": SEED_BASE, "ref_wav": ref["audio_path"], "ref_text": ref["transcript"], "params": GEN,
                "system_prompt": L.SYSTEM, "prompts": [{"prompt_id": u, "text": clips[u]["transcript"]}], "adapter": adapter}
        with GenCapture(client.model, client.processor) as cap:
            ok, _ = client.run(unit)
        n_ref = L.build_delay_pattern(L.encode_wave(client.processor, L.load_wave(ref["audio_path"]))).shape[0]
        G = cap.frames[0][n_ref:].long()
        finished = bool((G[-1] == L.EOS_ID).all())
        true = cached_codes(u)
        # generated codes: diagonal read of the delay pattern, cut at each codebook's first EOS
        n = G.shape[0]
        Lc = n - L.NUM_CB + 1
        diag = torch.stack([G[c:c + Lc, c] for c in range(L.NUM_CB)])[:, 1:]  # drop BOS
        e0 = (diag[0] == L.EOS_ID).nonzero()
        Tg = int(e0[0]) if len(e0) else diag.shape[1]
        gc = diag[:, :Tg]
        wav = f"{out}/gen_{tag}/synthetic/audio/bingge/train/{u}.wav"
        r = {"utt": u, "tag": tag, "ok": ok, "finished": finished, "gen_frames": n, "T_gen": Tg, "T_true": int(true.shape[1]),
             "dur_ratio": Tg / true.shape[1], "cb0_match_lag": best_lag_match(gc[0].numpy(), true[0].numpy()),
             "cb0_match_pos": float((gc[0, :min(Tg, true.shape[1])] == true[0, :min(Tg, true.shape[1])]).sum()) / true.shape[1],
             "allcb_match_lag": float(np.mean([best_lag_match(gc[c].numpy(), true[c].numpy()) for c in range(L.NUM_CB)])),
             "wav": wav, "text": clips[u]["transcript"]}
        res.append(r)
        print(f"[{arm}/{tag}] {u[:8]} finished={finished} T_gen={Tg} T_true={r['T_true']} ratio={r['dur_ratio']:.2f} "
              f"cb0_match={r['cb0_match_lag']:.3f}", flush=True)
    # decode the true codes once (ASR reference for the consistency metric)
    os.makedirs(f"{out}/true_decoded", exist_ok=True)
    import soundfile as sf

    for u in targets:
        p = f"{out}/true_decoded/{u}.wav"
        if not os.path.exists(p):
            with torch.inference_mode():
                w = client.processor.audio_tokenizer.decode(cached_codes(u)[None].to(DEV)).audio_values
            sf.write(p, w.float().cpu().squeeze().numpy(), L.SR)
    del client
    torch.cuda.empty_cache()
    return res


def asr_block(out, sp, targets, gens):
    clips = sp["clips"]
    paths = [g["wav"] for g in gens] + [f"{out}/true_decoded/{u}.wav" for u in targets] + [clips[u]["audio_path"] for u in targets]
    hyp = asr_whisper(paths, DEV)
    n = len(gens)
    true_dec = dict(zip(targets, hyp[n:n + len(targets)]))
    real = dict(zip(targets, hyp[n + len(targets):]))
    for g, h in zip(gens, hyp[:n]):
        g.update({"asr_gen": h, "asr_true_decoded": true_dec[g["utt"]], "asr_real": real[g["utt"]],
                  "wer_gen_vs_text": wer(g["text"], h), "wer_gen_vs_asr_true_decoded": wer(true_dec[g["utt"]], h),
                  "wer_real_vs_text": wer(g["text"], real[g["utt"]]), "wer_true_decoded_vs_text": wer(g["text"], true_dec[g["utt"]])})
    return gens


def summarize(gens):
    d = np.array([g["dur_ratio"] for g in gens])
    s = {"n": len(gens), "all_finished": all(g["finished"] for g in gens), "n_finished": sum(g["finished"] for g in gens),
         "dur_ratios": d.round(3).tolist(), "n_dur_in_range": int(((d >= 0.8) & (d <= 1.25)).sum()),
         "mean_cb0_match": float(np.mean([g["cb0_match_lag"] for g in gens])),
         "mean_cb0_match_pos": float(np.mean([g["cb0_match_pos"] for g in gens])),
         "mean_allcb_match": float(np.mean([g["allcb_match_lag"] for g in gens]))}
    if "wer_gen_vs_asr_true_decoded" in gens[0]:
        s["mean_wer_consistency"] = float(np.mean([g["wer_gen_vs_asr_true_decoded"] for g in gens]))
        s["mean_wer_gen_vs_text"] = float(np.mean([g["wer_gen_vs_text"] for g in gens]))
        s["mean_wer_real_vs_text"] = float(np.mean([g["wer_real_vs_text"] for g in gens]))
    s["crit2_stop_and_duration"] = s["all_finished"] and s["n_dur_in_range"] >= 3
    s["crit4_asr"] = s.get("mean_wer_consistency", 9) <= 0.5
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--arm", required=True, choices=["aligned", "shift_m1", "shift_p1", "noeos"])
    ap.add_argument("--asr_only", action="store_true")
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    sp, ref = load_splits(), load_refs()["ft_stitch5"]
    targets = pick_targets(sp, ref)
    print(f"[{a.arm}] targets {targets}", flush=True)
    state_p = f"{out}/t2_{a.arm}.json"
    if a.asr_only:
        res = json.load(open(state_p))
    else:
        res = {"arm": a.arm, "targets": targets, "gen_params": GEN}
        res["train"] = train(out, a.arm, sp, ref, targets)
        res["gen"] = generate(out, a.arm, sp, ref, targets, f"{out}/adapter", "adapter")
        if a.arm == "aligned":
            res["gen_base"] = generate(out, a.arm, sp, ref, targets, "none", "base")
        json.dump(res, open(state_p, "w"), indent=1)
    res["gen"] = asr_block(out, sp, targets, res["gen"])
    res["summary"] = summarize(res["gen"])
    if "gen_base" in res:
        res["gen_base"] = asr_block(out, sp, targets, res["gen_base"])
        res["summary_base"] = summarize(res["gen_base"])
    json.dump(res, open(state_p, "w"), indent=1)
    print(json.dumps({k: res[k] for k in res if k.startswith("summary")}, indent=1))
    print(json.dumps({"reached": res["train"]["reached_ce_below_0.5_at_step"], "final_ce_own": res["train"]["final_ce_own"],
                      "final_ce_aligned": res["train"]["final_ce_aligned"]}), flush=True)


if __name__ == "__main__":
    main()
