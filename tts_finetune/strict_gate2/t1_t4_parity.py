"""T1 (training-path vs inference-path logit parity on the model's own samples) and T4 (frame-level label audit).

    python t1_t4_parity.py <out_dir>

T1: generate N tts_val sentences through gen_ft.FTClient (adapter none, ref ft_stitch5, build_plan_ft default sampling),
capturing the raw audio_lm_head logits of every generation step and the generated frames; rebuild the same sequence with
higgs_ft_lib.build_example and compare per (frame, codebook). Controls: training rows shifted +-1 frame, different ref.
Noise floor: bf16 vs fp32 training path, full vs truncated training path. T4 runs between the bf16 and fp32 parts.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (FT_DIR, GEN_PARAMS, SEED_BASE, GenCapture, L, cached_codes, cell_stats, first_eos,  # noqa: E402
                    load_refs, load_splits, pool)

sys.path.insert(0, FT_DIR)
from gen_ft import FTClient  # noqa: E402

N_CLIPS = 10
MAX_NEW = 1000  # keeps ref (~700 frames) + target + text under the 2048 positions of the model
DEV = "cuda:0"


def tgt_rows(ex, proc):
    ids = ex["input_ids"][0]
    apos = ((ids == proc.audio_token_id) | (ids == proc.audio_delay_token_id)).nonzero().squeeze(1)
    t = apos[ex["n_ref_frames"]:]
    return t, bool((t[1:] - t[:-1] == 1).all())


@torch.no_grad()
def train_logits(model, ex):
    return L.forward_logits(model, ex, DEV).float().cpu()


def truncated(ex, cut):
    """The example cut to its first `cut` text positions (audio frames cut accordingly)."""
    ids = ex["input_ids"][:, :cut]
    n_audio = int(((ids == AUDIO_ID) | (ids == DELAY_ID)).sum())
    return {"input_ids": ids, "attention_mask": ex["attention_mask"][:, :cut],
            "audio_input_ids": ex["audio_input_ids"][:, :n_audio], "audio_input_ids_mask": ex["audio_input_ids_mask"][:, :n_audio]}


def main():
    global AUDIO_ID, DELAY_ID
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sp, refs = load_splits(), load_refs()
    clips = sp["clips"]
    ref, ref2 = refs["ft_stitch5"], refs["stitch5"]
    seen, val = set(), []
    for u in sorted(sp["tts_val"]):
        if clips[u]["norm"] not in seen:
            seen.add(clips[u]["norm"])
            val.append(u)
    val = val[:N_CLIPS]

    client = FTClient(L.MODEL)
    proc, model = client.processor, client.model
    AUDIO_ID, DELAY_ID = proc.audio_token_id, proc.audio_delay_token_id
    ref_codes = L.encode_wave(proc, L.load_wave(ref["audio_path"]))
    ref2_codes = L.encode_wave(proc, L.load_wave(ref2["audio_path"]))
    ref_frames = L.build_delay_pattern(ref_codes)
    params = {**GEN_PARAMS, "max_new_tokens": MAX_NEW}

    # ------------------------------------------------------------------ generate through the real path
    samples = []
    for u in val:
        unit = {"name": "t1", "out_root": f"{out}/gen", "speaker": "bingge", "variant": "strict_t1", "seed_base": SEED_BASE,
                "ref_wav": ref["audio_path"], "ref_text": ref["transcript"], "params": params, "system_prompt": L.SYSTEM,
                "prompts": [{"prompt_id": u, "text": clips[u]["transcript"]}], "adapter": "none"}
        with GenCapture(model, proc) as cap:
            ok, bad = client.run(unit)
        if not ok or cap.frames is None:
            print(f"{u}: generation failed", flush=True)
            continue
        lg = torch.stack(cap.logits)
        fr = cap.frames[0]
        n = fr.shape[0] - ref_frames.shape[0]  # generated frames; one lm-head call per generated frame expected
        samples.append({"utt": u, "text": clips[u]["transcript"], "I": lg[-n:].reshape(n, L.NUM_CB, -1),
                        "G": fr[-n:].long(), "R": fr[:-n].long(), "n_head_calls": int(lg.shape[0])})
        print(f"{u}: {n} generated frames, {lg.shape[0]} lm-head calls, total frames {fr.shape[0]}", flush=True)

    # generation prompt ids, built exactly like gen_higgs.Client.generate
    def gen_prompt_ids(text):
        msgs = [{"role": "system", "content": L.SYSTEM}, {"role": "user", "content": L.normalize_for_higgs(ref["transcript"])},
                {"role": "assistant", "content": [{"type": "audio"}]}, {"role": "user", "content": L.normalize_for_higgs(text)}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=prompt, audio=[client._ref(ref["audio_path"])], sampling_rate=client.sr, return_tensors="pt")
        return enc["input_ids"][0].cpu(), enc["audio_input_ids"][0].cpu()

    # ------------------------------------------------------------------ rebuild + compare (bf16)
    res = {"clips": [], "params": params, "ref": "ft_stitch5", "control_ref": "stitch5"}
    matched, sh_p, sh_m, oref, bos_cells, eos_cells, valid = [], [], [], [], [], [], []
    for s in samples:
        G, I = s["G"], s["I"]
        n = G.shape[0]
        finished = bool((G[-1] == L.EOS_ID).all())
        starts_bos = bool((G[0] == L.BOS_ID).all())
        codes = L.revert_delay_pattern(G)
        well_formed = finished and starts_bos and bool(torch.equal(L.build_delay_pattern(codes), G))
        rec = {"utt": s["utt"], "n_gen_frames": n, "n_head_calls": s["n_head_calls"], "finished": finished, "starts_bos": starts_bos, "well_formed": well_formed,
               "ref_frames_equal": bool(s["R"].shape == ref_frames.shape and torch.equal(s["R"], ref_frames))}
        if not well_formed:
            rec["cb_first_eos"] = [first_eos(G[:, c]) for c in range(L.NUM_CB)]
            res["clips"].append(rec)
            print("invalid clip", rec, flush=True)
            continue
        ex = L.build_example(proc, ref["transcript"], ref_codes, s["text"], codes, eos_labels=True)
        rows, contiguous = tgt_rows(ex, proc)
        gp_ids, gp_audio = gen_prompt_ids(s["text"])
        rec.update({"T_codes": int(codes.shape[1]), "rows_contiguous": contiguous, "n_rows": int(rows.numel()),
                    "prompt_prefix_equal": bool(torch.equal(ex["input_ids"][0][: gp_ids.numel()], gp_ids)),
                    "gen_prompt_len": int(gp_ids.numel()), "first_row": int(rows[0]) - 1,
                    "first_row_is_last_prompt_token": int(rows[0]) - 1 == gp_ids.numel() - 1,
                    "last_prompt_token": int(gp_ids[-1]), "audio_bos_token_id": int(proc.audio_bos_token_id),
                    "processor_ref_frames_equal_encode_wave": bool(gp_audio.shape == ref_frames.shape and torch.equal(gp_audio, ref_frames)),
                    "cb_first_eos_gen": [first_eos(G[:, c]) for c in range(L.NUM_CB)]})
        if rows.numel() != n:
            rec["error"] = "target rows != generated frames"
            res["clips"].append(rec)
            continue
        TL = train_logits(model, ex)
        T = TL[rows - 1]
        st = cell_stats(I, T, G)
        matched.append(st)
        e0 = first_eos(G[:, 0])
        bos_cells.append({k: v[0:1] for k, v in st.items()})
        eos_cells.append({k: v[e0:e0 + 1] for k, v in st.items()})
        sh_p.append(cell_stats(I[:-1], TL[rows[1:] - 1], G[:-1]))   # training rows one frame later
        sh_m.append(cell_stats(I[1:], TL[rows[:-1] - 1], G[1:]))    # one frame earlier
        ex2 = L.build_example(proc, ref2["transcript"], ref2_codes, s["text"], codes, eos_labels=True)
        rows2, _ = tgt_rows(ex2, proc)
        oref.append(cell_stats(I, train_logits(model, ex2)[rows2 - 1], G))
        # N2: same training path, truncated after the middle target frame
        k = n // 2
        cut = int(rows[k])  # positions [0, rows[k]) -> rows-1 for frames 0..k
        TLc = train_logits(model, truncated(ex, cut))
        n2 = cell_stats(T[:k + 1], TLc[rows[:k + 1] - 1], G[:k + 1])
        rec.update({"eos_step": e0, "matched": pool([st]), "bos_step": pool([bos_cells[-1]]), "eos_step_stats": pool([eos_cells[-1]])})
        res["clips"].append(rec)
        valid.append({"s": s, "ex": ex, "rows": rows, "T": T, "n2": n2, "codes": codes})
        print(f"{s['utt']}: matched median meanD={rec['matched']['median_meanD']:.4f} agree={rec['matched']['agree']:.4f} "
              f"| N2 median meanD={pool([n2])['median_meanD']:.4f}", flush=True)

    # ------------------------------------------------------------------ T4 (still bf16, before the fp32 cast)
    t4 = run_t4(proc, sp, ref, ref_codes, valid)
    res["T4"] = t4

    # ------------------------------------------------------------------ N1: fp32 training path
    model.float()
    n1 = []
    for v in valid:
        T32 = train_logits(model, v["ex"])[v["rows"] - 1]
        n1.append(cell_stats(v["T"], T32, v["s"]["G"]))
    N1, N2 = pool(n1), pool([v["n2"] for v in valid])
    NF = {k: max(N1[k], N2[k]) for k in ("median_meanD", "p99_meanD", "median_maxD", "mean_dLP")}
    NF["agree"] = min(N1["agree"], N2["agree"])
    res.update({"n_valid": len(valid), "N1_bf16_vs_fp32": N1, "N2_full_vs_truncated": N2, "NF": NF,
                "matched": pool(matched) if matched else None, "shift_plus1": pool(sh_p) if sh_p else None,
                "shift_minus1": pool(sh_m) if sh_m else None, "other_ref": pool(oref) if oref else None,
                "bos_step": pool(bos_cells) if bos_cells else None, "eos_step": pool(eos_cells) if eos_cells else None})

    # ------------------------------------------------------------------ verdict (THRESHOLDS.md T1)
    v = {}
    if len(valid) < 6:
        v["verdict"] = "INCONCLUSIVE"
    else:
        M = res["matched"]
        c = res["clips"]
        okc = [r for r in c if r.get("well_formed") and "error" not in r]
        v["structure"] = all(r["n_head_calls"] == r["n_gen_frames"] and r["prompt_prefix_equal"] and r["ref_frames_equal"] and r["rows_contiguous"]
                             and r["first_row_is_last_prompt_token"] for r in okc) and len(okc) == len(valid)
        v["parity"] = (M["median_meanD"] <= 3 * NF["median_meanD"] and M["p99_meanD"] <= 3 * NF["p99_meanD"]
                       and M["median_maxD"] <= 3 * NF["median_maxD"] and M["mean_dLP"] <= 3 * NF["mean_dLP"]
                       and M["agree"] >= NF["agree"] - 0.02)
        v["bos_eos_steps"] = all(r["bos_step"]["max_meanD"] <= 3 * NF["p99_meanD"]
                                 and r["eos_step_stats"]["max_meanD"] <= 3 * NF["p99_meanD"] for r in okc)
        for name in ("shift_plus1", "shift_minus1", "other_ref"):
            C = res[name]
            v[f"control_{name}_fails_clearly"] = C["median_meanD"] >= 9 * NF["median_meanD"] and C["agree"] < NF["agree"] - 0.02
        v["verdict"] = "PASS" if all(v.values()) else "FAIL"
    res["T1_checks"] = v
    json.dump(res, open(f"{out}/t1_t4.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_valid", "NF", "matched", "shift_plus1", "shift_minus1", "other_ref",
                                           "bos_step", "eos_step", "T1_checks")}, indent=1), flush=True)
    print("T4:", json.dumps(t4["checks"], indent=1), flush=True)


def run_t4(proc, sp, ref, ref_codes, valid):
    clips = sp["clips"]
    ids = sorted(sp["tts_train"])[3:: max(1, len(sp["tts_train"]) // 6)][:6]
    rows, ok_all = [], True
    tok = proc.audio_tokenizer
    for u in ids:
        codes = cached_codes(u)
        T = codes.shape[1]
        r = {"utt": u, "T": T}
        for eos in (True, False):
            ex = L.build_example(proc, ref["transcript"], ref_codes, clips[u]["transcript"], codes, eos_labels=eos)
            nr = ex["n_ref_frames"]
            lab = ex["audio_labels"][0]
            tgt_frames = ex["audio_input_ids"][0][nr:]
            tl = lab[nr:]
            masked = tl == L.IGNORE
            if eos:
                r["ref_labels_all_ignored"] = bool((lab[:nr] == L.IGNORE).all())
                r["masked_cells_are_exactly_bos"] = bool(torch.equal(masked, tgt_frames == L.BOS_ID))
                filled = tl.clone()
                filled[masked] = L.BOS_ID
                rc = L.revert_delay_pattern(filled)
                r["codes_equal"] = bool(torch.equal(rc, codes))
                with torch.inference_mode():
                    w_cached = tok.decode(codes[None].to(DEV)).audio_values.float().cpu()
                    w_cached2 = tok.decode(codes[None].to(DEV)).audio_values.float().cpu()
                    w_rec = tok.decode(rc[None].to(DEV)).audio_values.float().cpu()
                r["decode_deterministic"] = bool(torch.equal(w_cached, w_cached2))
                r["decode_bit_identical"] = bool(torch.equal(w_cached, w_rec))
                r["decode_samples"] = int(w_cached.numel())
                valid_cells = ~masked
                r["first_label_frame_per_cb"] = [int(valid_cells[:, c].nonzero()[0]) for c in range(L.NUM_CB)]
                r["first_eos_label_frame_per_cb"] = [first_eos(tl[:, c]) for c in range(L.NUM_CB)]
                r["last_label_frame"] = int(valid_cells.any(1).nonzero()[-1])
                r["n_target_frames"] = int(tl.shape[0])
                # labels_pos: loss pairs logits at position p with label at p+1; the BOS frame's position predicts frame 1
                lp = ex["labels_pos"]
                ids_ = ex["input_ids"][0]
                apos = ((ids_ == proc.audio_token_id) | (ids_ == proc.audio_delay_token_id)).nonzero().squeeze(1)[nr:]
                r["labels_pos_matches_audio_labels"] = bool(torch.equal(lp[apos], tl))
                r["no_labels_off_target"] = int((lp != L.IGNORE).any(1).sum()) == int((tl != L.IGNORE).any(1).sum())
                r["expect"] = {"first_label": [1 + c for c in range(8)], "first_eos": [T + 1 + c for c in range(8)], "last": T + 8}
                good = (r["ref_labels_all_ignored"] and r["masked_cells_are_exactly_bos"] and r["codes_equal"]
                        and r["decode_bit_identical"] and r["first_label_frame_per_cb"] == r["expect"]["first_label"]
                        and r["first_eos_label_frame_per_cb"] == r["expect"]["first_eos"] and r["last_label_frame"] == T + 8
                        and r["labels_pos_matches_audio_labels"] and r["no_labels_off_target"])
            else:
                r["noeos_eos_labels_left"] = int((tl == L.EOS_ID).sum())
                r["noeos_label_cells"] = int((tl != L.IGNORE).sum())
                good = r["noeos_eos_labels_left"] == 0
            r["ok_eos" if eos else "ok_noeos"] = good
            ok_all &= good
        rows.append(r)
        print("T4", r, flush=True)
    gen = []
    for v in valid:
        G, codes = v["s"]["G"], v["codes"]
        Tg = codes.shape[1]
        cb_eos = [first_eos(G[:, c]) for c in range(L.NUM_CB)]
        lab = v["ex"]["audio_labels"][0][v["ex"]["n_ref_frames"]:]
        g = {"utt": v["s"]["utt"], "T_gen": Tg, "cb_first_eos_gen": cb_eos, "cb0_first_eos_label": first_eos(lab[:, 0]),
             "cb0_last_speech_frame": Tg}
        g["ok"] = cb_eos == [Tg + 1 + c for c in range(8)] and g["cb0_first_eos_label"] == Tg + 1
        ok_all &= g["ok"]
        gen.append(g)
    checks = {"cached_clips_ok": all(r["ok_eos"] and r["ok_noeos"] for r in rows), "gen_eos_ok": all(g["ok"] for g in gen),
              "n_gen_checked": len(gen), "verdict": "PASS" if ok_all else "FAIL"}
    return {"cached": rows, "generated": gen, "checks": checks}


if __name__ == "__main__":
    main()
