"""Item 8: SGEM test-time adaptation for a finetuned Whisper (design in asr_aug/tta/README.md).

    python sgem_whisper.py --model /scratch/pa2753/stack/runs/E2_seed1/best --scope ln --out-dir <run>/tta/ln

Episodic per utterance: restore the checkpoint, then N steps of beam-5 pseudo-label -> teacher-forced logits ->
Renyi entropy (GEM) + negative sampling (NS) loss -> AdamW + cosine step. The adapted model is decoded with beam 10
after steps 0 (no TTA), 1, 3, 5, 10. Writes <out-dir>/step<k>_<set>_hyps.tsv (path, ref, hyp) and, per utterance,
the pseudo-labels and losses in <out-dir>/<set>.jsonl. Resumable per utterance.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation.utils import GenerationMixin

PROJECT_ROOT = "/home/pa2753/auto-avsr-realtime"
sys.path.insert(0, PROJECT_ROOT)
from asr_baselines.patient_audio import load_audio_examples, load_waveform  # noqa: E402

SETS = [f"{p}_{s}" for p in ("s1", "s2") for s in ("val", "test", "oov")]
EVAL_STEPS = (0, 1, 3, 5, 10)
N_STEPS, TEMP, PSEUDO_BEAMS, EVAL_BEAMS = 10, 2.5, 5, 10
# scope -> (lr start, lr end, renyi alpha, lambda_NS); per-scope values of Wang et al. (NCMMSC 2026, Table 3 settings)
SCOPES = {"ln": (2e-4, 2e-5, 1.5, 1.0), "feat": (4e-5, 2e-5, 1.5, 1.0)}


def scope_params(model, scope):
    enc = model.model.encoder
    if scope == "feat":
        return list(enc.conv1.parameters()) + list(enc.conv2.parameters())
    return [p for m in enc.modules() if isinstance(m, torch.nn.LayerNorm) for p in m.parameters()]


def renyi(logits, alpha):
    """Mean Renyi entropy of softmax(logits) over rows (logits already divided by T)."""
    logp = torch.log_softmax(logits, dim=-1)
    return (torch.logsumexp(alpha * logp, dim=-1) / (1 - alpha)).mean()


def neg_sampling(logits, tau):
    """SGEM ns3l: penalise the tempered mass on tokens whose untempered probability is below tau."""
    mask = (torch.softmax(logits, dim=-1) < tau).float()
    mass = (mask * torch.softmax(logits / TEMP, dim=-1)).sum(-1)
    return -torch.log((1 - mass).clamp_min(1e-8)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scope", required=True, choices=list(SCOPES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--limit", type=int, default=0, help="first N utterances per set (smoke)")
    ap.add_argument("--root-dir", default="/scratch/pa2753/stack/root")
    ap.add_argument("--labels", default="/scratch/pa2753/multi_patient/labels")
    ap.add_argument("--sp-model-path", default=os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model"))
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    proc = WhisperProcessor.from_pretrained(a.model, language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.model).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = scope_params(model, a.scope)
    for p in params:
        p.requires_grad_(True)
    init = [p.detach().clone() for p in params]
    lr0, lr1, alpha, lam = SCOPES[a.scope]
    tok, gc = proc.tokenizer, model.generation_config
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    prefix = [gc.decoder_start_token_id, tok.convert_tokens_to_ids("<|en|>"), tok.convert_tokens_to_ids("<|transcribe|>"),
              gc.no_timestamps_token_id]
    tau = 0.4 / model.config.vocab_size
    print(f"scope {a.scope}: {sum(p.numel() for p in params)} params, lr {lr0}->{lr1}, alpha {alpha}, lambda_NS {lam}, "
          f"tau {tau:.3g}", flush=True)

    def generate(feats, beams):
        with torch.no_grad():
            g = GenerationMixin.generate(model, input_features=feats, decoder_input_ids=torch.tensor([prefix], device=dev),
                                         num_beams=beams, max_new_tokens=128)
        y = g[0].tolist()[len(prefix):]
        return y[:y.index(eot) + 1] if eot in y else y

    def text(y):
        return tok.decode([t for t in y if t != eot], skip_special_tokens=True).strip()

    def adapt_step(feats, opt, sch):
        y = generate(feats, PSEUDO_BEAMS)
        if not y or y == [eot]:
            return None, y
        full = prefix + y
        logits = model(input_features=feats, decoder_input_ids=torch.tensor([full[:-1]], device=dev)).logits[0]
        o = logits[len(prefix) - 1:].float()
        content = torch.tensor([t != eot for t in y], device=dev)
        gem = renyi(o[content] / TEMP, alpha) if content.any() else o.sum() * 0
        ns = neg_sampling(o, tau)
        loss = gem + lam * ns
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        return (float(gem), float(ns)), y

    os.makedirs(a.out_dir, exist_ok=True)
    for s in a.sets:
        if all(os.path.exists(f"{a.out_dir}/step{k}_{s}_hyps.tsv") for k in EVAL_STEPS) and not a.limit:
            print(f"skip {s}: done", flush=True)
            continue
        exs = load_audio_examples(a.root_dir, f"{a.labels}/{s}_spm1023.csv", a.sp_model_path)
        if a.limit:
            exs = exs[:a.limit]
        jl = f"{a.out_dir}/{s}.jsonl"
        done = {}
        if os.path.exists(jl):
            for line in open(jl):
                r = json.loads(line)
                done[r["path"]] = r
        t0 = time.time()
        with open(jl, "a") as f:
            for i, ex in enumerate(exs):
                if ex.path in done:
                    continue
                feats = proc.feature_extractor(load_waveform(ex.path).numpy(), sampling_rate=16000,
                                               return_tensors="pt").input_features.to(dev)
                with torch.no_grad():
                    for p, v in zip(params, init):
                        p.copy_(v)
                opt = torch.optim.AdamW(params, lr=lr0)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=lr1)
                r = {"path": ex.path, "ref": ex.text, "hyp": {}, "loss": [], "pseudo": []}
                r["hyp"]["0"] = text(generate(feats, EVAL_BEAMS))
                for k in range(1, N_STEPS + 1):
                    lo, y = adapt_step(feats, opt, sch)
                    r["loss"].append(lo)
                    r["pseudo"].append(text(y))
                    if k in EVAL_STEPS:
                        r["hyp"][str(k)] = text(generate(feats, EVAL_BEAMS))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                done[ex.path] = r
                print(f"{s} {i + 1}/{len(exs)} {time.time() - t0:.0f}s | step0 '{r['hyp']['0'][:40]}' | "
                      f"step10 '{r['hyp']['10'][:40]}'", flush=True)
        for k in EVAL_STEPS:
            rows = [done[ex.path] for ex in exs]
            with open(f"{a.out_dir}/step{k}_{s}_hyps.tsv", "w", encoding="utf-8") as g:
                g.write("path\tref\thyp\n" + "".join(f"{r['path']}\t{r['ref']}\t{r['hyp'][str(k)]}\n" for r in rows))
        print(f"{s} done {(time.time() - t0) / 60:.1f} min", flush=True)
    with torch.no_grad():
        for p, v in zip(params, init):
            p.copy_(v)
    print("TTA_DONE", flush=True)


if __name__ == "__main__":
    main()
