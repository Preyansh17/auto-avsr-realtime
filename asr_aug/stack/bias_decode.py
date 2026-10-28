"""Item 1: contextual biasing decodes for a finetuned Whisper (PLAN.md 3.3, design in asr_aug/stack/README.md).

    python bias_decode.py --model <run>/best --out-dir <run>/bias     # all configs, see main()

Methods (beam 10, lang_task prefix):
  beam    plain beam search (the reference for the other two)
  prompt  Whisper prompt biasing: the bias word list as <|startofprev|> prompt tokens (no LM)
  ngram   shallow fusion: + lam * log P_lm(token | 3 previous tokens) on every beam step. P_lm is an interpolated
          token 4-gram over Whisper BPE ids of the bias sentences (each ends in <|endoftext|>), mixed with 1% uniform.
Bias conditions (bias_text.py writes them):
  clean   text the models already train on: patient s1/s2 train sentences, old split_v1 train, pool A/B (minus held text);
          prompt words = the training vocabulary.
  words   clean + the s1/s2 OOV words as a bare word list (one word per line; no OOV sentence), the "caregiver supplies
          the vocabulary" ceiling. Reported apart from clean, never as the claim.
Every decode also writes its 10-best (text, sequence score), the input of llm_rescore.py.
"""
import argparse
import collections
import json
import math
import os
import sys

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.generation.utils import GenerationMixin

PROJECT_ROOT = "/home/pa2753/auto-avsr-realtime"
sys.path.insert(0, PROJECT_ROOT)
from asr_baselines.patient_audio import load_audio_examples, load_waveform  # noqa: E402

BIAS = "/scratch/pa2753/stack/data/bias"
ORDER, UNIFORM = 4, 0.01
WEIGHTS = [0.1, 0.2, 0.3, 0.4]  # orders 1..4; renormalised over the orders whose history was seen


class TokenNgram:
    def __init__(self, seqs, vocab):
        self.V = vocab
        self.c = [collections.defaultdict(collections.Counter) for _ in range(ORDER)]
        for s in seqs:
            for i, w in enumerate(s):
                for k in range(ORDER):
                    if i - k < 0:
                        break
                    self.c[k][tuple(s[i - k:i])][w] += 1

    def logprob_vec(self, hist, device):
        p = torch.zeros(self.V, device=device)
        used = []
        for k in range(ORDER):
            h = tuple(hist[len(hist) - k:]) if k else ()
            if k > len(hist) or h not in self.c[k]:
                continue
            used.append((k, self.c[k][h]))
        tot_w = sum(WEIGHTS[k] for k, _ in used)
        for k, cnt in used:
            n = sum(cnt.values())
            idx = torch.tensor(list(cnt.keys()), device=device)
            val = torch.tensor([v / n for v in cnt.values()], device=device)
            p.index_add_(0, idx, (1 - UNIFORM) * WEIGHTS[k] / tot_w * val)
        return torch.log(p + UNIFORM / self.V)


class Fusion(LogitsProcessor):
    def __init__(self, lm, lam, start_id):
        self.lm, self.lam, self.start = lm, lam, start_id

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            hist = ids[len(ids) - 1 - ids[::-1].index(self.start) + 1:] if self.start in ids else ids
            scores[b] = scores[b] + self.lam * self.lm.logprob_vec(hist[-(ORDER - 1):], scores.device)
        return scores


def load(model_dir, dev):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_dir, language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(dev).eval()
    return proc, model


def bias_kw(proc, model, method, bias, lam, beams, dev, lms):
    tok = proc.tokenizer
    nots, eot = model.generation_config.no_timestamps_token_id, tok.convert_tokens_to_ids("<|endoftext|>")
    # Base GenerationMixin.generate with an explicit decoder prefix: Whisper's own generate() expands the batch by
    # num_return_sequences AND num_beams (100 sequences for a 10-best) and runs out of memory on a 44 GB L40S.
    gc = model.generation_config
    prefix = [gc.decoder_start_token_id, tok.convert_tokens_to_ids("<|en|>"), tok.convert_tokens_to_ids("<|transcribe|>"), nots]
    if method == "prompt":
        words = open(f"{BIAS}/{bias}_words.txt").read().split()
        ids = proc.get_prompt_ids(" ".join(words)).tolist()
        assert len(ids) <= 224, len(ids)
        prefix = ids + prefix
    kw = {"decoder_input_ids": torch.tensor([prefix], device=dev), "num_beams": beams, "num_return_sequences": beams,
          "return_dict_in_generate": True, "output_scores": True, "max_new_tokens": 128}
    if method == "ngram":
        if bias not in lms:
            seqs = [tok.encode(" " + l.strip(), add_special_tokens=False) + [eot]
                    for l in open(f"{BIAS}/{bias}_text.txt") if l.strip()]
            lms[bias] = TokenNgram(seqs, model.config.vocab_size)
        kw["logits_processor"] = LogitsProcessorList([Fusion(lms[bias], lam, nots)])
    return kw


def decode(proc, model, examples, kw, dev):
    """Returns rows [(path, ref, hyp)] and the n-best {take_id: [{text, score}]}."""
    nots = model.generation_config.no_timestamps_token_id
    rows, nbest = [], {}
    for ex, feats in examples:
        with torch.inference_mode():
            g = GenerationMixin.generate(model, input_features=feats.to(dev), **kw)
        texts = []
        for s in g.sequences.tolist():
            if nots in s:
                s = s[s.index(nots) + 1:]
            texts.append(proc.tokenizer.decode(s, skip_special_tokens=True).strip())
        sc = g.sequences_scores.tolist() if getattr(g, "sequences_scores", None) is not None else [0.0] * len(texts)
        rows.append((ex.path, ex.text, texts[0]))
        nbest[os.path.splitext(os.path.basename(ex.path))[0]] = [{"text": t, "score": x} for t, x in zip(texts, sc)]
    return rows, nbest


def write(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("path\tref\thyp\n" + "".join(f"{p}\t{r}\t{h}\n" for p, r, h in rows))


def wer(rows):
    import re
    n = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip().split()
    e = w = 0
    for _, r, h in rows:
        d = list(range(len(n(h)) + 1))
        for i, x in enumerate(n(r), 1):
            p, d[0] = d[0], i
            for j, y in enumerate(n(h), 1):
                p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
        e += d[-1]; w += len(n(r))
    return 100 * e / w


LAMS = [0.1, 0.2, 0.3, 0.5]
SETS = [f"{p}_{s}" for p in ("s1", "s2") for s in ("val", "test", "oov")]


def main():
    """All configs for one model. Phase 1: every config on s1_val + s2_val. lam per (ngram, bias) = lowest pooled val
    WER (ties -> smaller lam). Phase 2: beam, prompt x bias, ngram x bias at the chosen lam, on test and oov.
    The full lam grid is also decoded on oov (descriptive only: it shows what val selection left on the table)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root-dir", default="/scratch/pa2753/stack/root")
    ap.add_argument("--labels", default="/scratch/pa2753/multi_patient/labels")
    ap.add_argument("--sp-model-path", default=os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model"))
    ap.add_argument("--beams", type=int, default=10)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load(a.model, dev)
    os.makedirs(a.out_dir, exist_ok=True)
    ex = {}
    for s in SETS:
        ex[s] = [(e, proc.feature_extractor(load_waveform(e.path).numpy(), sampling_rate=16000, return_tensors="pt").input_features)
                 for e in load_audio_examples(a.root_dir, f"{a.labels}/{s}_spm1023.csv", a.sp_model_path)]
    lms, val = {}, {}
    cfgs = [("beam", "none", 0.0)] + [("prompt", b, 0.0) for b in ("clean", "words")] + \
           [("ngram", b, l) for b in ("clean", "words") for l in LAMS]

    def run(cfg, s):
        m, b, l = cfg
        tag = f"{m}_{b}_{l:g}_{s}"
        rows, nb = decode(proc, model, ex[s], bias_kw(proc, model, m, b, l, a.beams, dev, lms), dev)
        write(rows, f"{a.out_dir}/{tag}_hyps.tsv")
        json.dump(nb, open(f"{a.out_dir}/{tag}_nbest.json", "w"))
        return rows

    for cfg in cfgs:
        val[cfg] = run(cfg, "s1_val") + run(cfg, "s2_val")
    chosen = {b: min(LAMS, key=lambda l: (wer(val["ngram", b, l]), l)) for b in ("clean", "words")}
    sel = {"lam": chosen, "val_wer": {f"{m}_{b}_{l:g}": round(wer(r), 3) for (m, b, l), r in val.items()}}
    json.dump(sel, open(f"{a.out_dir}/selection.json", "w"), indent=1)
    print(json.dumps(sel))
    final = [("beam", "none", 0.0), ("prompt", "clean", 0.0), ("prompt", "words", 0.0)] + \
            [("ngram", b, chosen[b]) for b in ("clean", "words")]
    for s in ("s1_test", "s1_oov", "s2_test", "s2_oov"):
        for cfg in final:
            print(s, cfg, f"{wer(run(cfg, s)):.2f}")
    for b in ("clean", "words"):
        for l in LAMS:
            if l != chosen[b]:
                for s in ("s1_oov", "s2_oov"):
                    run(("ngram", b, l), s)


if __name__ == "__main__":
    main()
