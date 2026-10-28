"""Item 1, third method: rescore the plain-beam 10-best (bias_decode.py beam_none_0_<set>_nbest.json) with Qwen2.5-1.5B.

    python llm_rescore.py --bias-dir <run>/bias

score = Whisper sequence score (length-normalised log-prob) + mu * LLM mean token log-prob of the candidate, where the
LLM sees a prefix naming the bias vocabulary: clean = training vocabulary (clean_words.txt), words = + OOV words
(words_words.txt). mu per condition = lowest pooled s1_val + s2_val WER over MUS (ties -> smaller mu); then test/oov.
Writes llm_<bias>_<mu>_<set>_hyps.tsv (take_id, ref, hyp) and llm_selection.json.
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM = "Qwen/Qwen2.5-1.5B"
BIAS = "/scratch/pa2753/stack/data/bias"
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
MUS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]  # amendment 1 (README): finer low end
SETS = [f"{p}_{s}" for p in ("s1", "s2") for s in ("val", "test", "oov")]


def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias-dir", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(LLM)
    lm = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.bfloat16).cuda().eval()
    refs = {s: {json.loads(l)["take_id"]: norm(json.loads(l)["transcript"]) for l in open(f"{ROLES}/{s}.jsonl")} for s in SETS}
    nb = {s: json.load(open(f"{a.bias_dir}/beam_none_0_{s}_nbest.json")) for s in SETS}

    @torch.inference_mode()
    def llm_score(prefix, texts):
        pre = tok(prefix, return_tensors="pt").input_ids[0]
        out = []
        for t in texts:
            body = tok(" " + (norm(t) or "."), add_special_tokens=False).input_ids
            ids = torch.cat([pre, torch.tensor(body)]).unsqueeze(0).cuda()
            lp = torch.log_softmax(lm(ids).logits[0, :-1].float(), -1)
            tgt = ids[0, 1:]
            tl = lp[torch.arange(len(tgt)), tgt][len(pre) - 1:]
            out.append(float(tl.mean()))
        return out

    sel = {}
    for bias in ("clean", "words"):
        words = open(f"{BIAS}/{bias}_words.txt").read().split()
        prefix = f"Words this speaker often uses: {', '.join(words)}.\nThe speaker said:"
        llm = {s: {k: llm_score(prefix, [c["text"] for c in v]) for k, v in nb[s].items()} for s in SETS}

        def pick(s, mu):
            return {k: max(range(len(v)), key=lambda i: v[i]["score"] + mu * llm[s][k][i]) for k, v in nb[s].items()}

        def wer(s, mu):
            ch = pick(s, mu)
            e = sum(ed(refs[s][k].split(), norm(nb[s][k][i]["text"]).split()) for k, i in ch.items())
            return 100 * e / sum(len(refs[s][k].split()) for k in ch)

        vw = {mu: (wer("s1_val", mu) + wer("s2_val", mu)) / 2 for mu in MUS}
        mu = min(MUS, key=lambda m: (vw[m], m))
        sel[bias] = {"mu": mu, "val_wer_mean": {str(m): round(v, 3) for m, v in vw.items()}}
        for s in SETS:
            for m in MUS:
                ch = pick(s, m)
                with open(f"{a.bias_dir}/llm_{bias}_{m:g}_{s}_hyps.tsv", "w") as f:
                    f.write("take_id\tref\thyp\n" + "".join(f"{k}\t{refs[s][k]}\t{nb[s][k][i]['text']}\n" for k, i in ch.items()))
            print(bias, s, f"mu={mu}", f"{wer(s, mu):.2f}")
    json.dump(sel, open(f"{a.bias_dir}/llm_selection.json", "w"), indent=1)
    print(json.dumps(sel))


if __name__ == "__main__":
    main()
