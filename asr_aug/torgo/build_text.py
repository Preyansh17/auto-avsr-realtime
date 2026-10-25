"""Synthetic text for TORGO generation: per speaker N_WORDS single words + N_SENTS sentences.

    python build_text.py   # -> /scratch/pa2753/torgo_aug/data/text_<spk>.json (build_plan_ft --sentences-file format)

Sources: words = (TORGO ASR-train vocabulary U patient pool A/B vocabulary); sentences = patient pool B (everyday
open-vocab sentences). Leakage rules, asserted: no text equals a TORGO val/test text and no sentence is within 1 word edit of a
val/test sentence;
no text contains a "forbidden" word (in TORGO val/test, never in TORGO train), so unseen test words stay unseen.
Word/sentence mix ~ TORGO's (about 3:1). Texts are distinct within a speaker; speakers draw independently.
"""
from __future__ import annotations

import csv
import json
import random

DATA = "/scratch/pa2753/torgo_aug/data"
POOLS = "/scratch/pa2753/asr_aug/data"
SEED = 20261015
N_WORDS, N_SENTS = 160, 60


def edit_le1(a: list[str], b: list[str]) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def main() -> None:
    items = list(csv.DictReader(open(f"{DATA}/items.csv")))
    train_v = {w for i in items if i["split"] == "train" for w in i["text"].split()}
    held = {i["text"] for i in items if i["split"] != "train"}
    forbidden = {w for t in held for w in t.split()} - train_v
    pool_ab = json.load(open(f"{POOLS}/pool_A.json")) + json.load(open(f"{POOLS}/pool_B.json"))
    words = sorted((train_v | {w for e in pool_ab for w in e["norm"].split()}) - forbidden - held)
    words = [w for w in words if len(w) > 1 or w in ("a", "i")]
    held_tok = [t.split() for t in held if len(t.split()) > 1]  # edit rule: sentences vs held sentences
    sents = sorted({e["norm"] for e in json.load(open(f"{POOLS}/pool_B.json"))
                    if not set(e["norm"].split()) & forbidden and not any(edit_le1(e["norm"].split(), h) for h in held_tok)})
    print(f"word pool {len(words)}, sentence pool {len(sents)}, forbidden {len(forbidden)}")
    for spk in sorted({i["speaker"] for i in items}):
        rng = random.Random(f"{SEED}-text-{spk}")
        pick = [(w, "word") for w in rng.sample(words, N_WORDS)] + [(s, "sentence") for s in rng.sample(sents, N_SENTS)]
        out = []
        for t, typ in pick:
            assert t not in held and not set(t.split()) & forbidden
            assert typ == "word" or not any(edit_le1(t.split(), h) for h in held_tok)
            out.append({"text": t[0].upper() + t[1:] + ("." if typ == "sentence" else ""), "norm": t, "type": typ})
        json.dump(out, open(f"{DATA}/text_{spk}.json", "w"), indent=0)
    json.dump({"seed": SEED, "forbidden_words": sorted(forbidden), "word_pool": len(words), "sentence_pool": len(sents),
               "n_words": N_WORDS, "n_sents": N_SENTS}, open(f"{DATA}/text_audit.json", "w"), indent=1)
    print("OK")


if __name__ == "__main__":
    main()
