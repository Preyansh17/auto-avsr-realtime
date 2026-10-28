"""Bias text for bias_decode.py, two conditions -> /scratch/pa2753/stack/data/bias/{clean,words}_{text,words}.txt

clean_text   s1/s2 train sentences, old split_v1 train (old_legacy/old_legal train roles), pool A/B text (stack/data/text.json,
             already minus every s1/s2 val/test/oov sentence)
clean_words  vocabulary of the s1/s2 train sentences (the prompt-biasing list; must fit Whisper's 224-token prompt)
words_*      clean_* + every s1/s2 OOV word not already in clean_words, as bare words (one per line in words_text)
Asserts that no s1/s2 val/test/oov sentence appears in any text file.
"""
import json
import os
import re

ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
S = "/scratch/pa2753/stack/data"
OUT = f"{S}/bias"


def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def role(r): return [norm(json.loads(l)["transcript"]) for l in open(f"{ROLES}/{r}.jsonl")]


def main():
    os.makedirs(OUT, exist_ok=True)
    train = role("s1_train") + role("s2_train")
    old = role("old_legacy_train") + role("old_legal_train")
    pools = [norm(e["norm"]) for e in json.load(open(f"{S}/text.json"))]
    held = {t for p in ("s1", "s2") for r in ("val", "test", "oov") for t in role(f"{p}_{r}")}
    text = sorted(set(train + old + pools))
    assert not (set(text) & held), sorted(set(text) & held)[:5]
    vocab = sorted({w for t in train for w in t.split()})
    oov_words = sorted({w for p in ("s1", "s2") for t in role(f"{p}_oov") for w in t.split()} - set(vocab))
    open(f"{OUT}/clean_text.txt", "w").write("\n".join(text) + "\n")
    open(f"{OUT}/clean_words.txt", "w").write(" ".join(vocab) + "\n")
    open(f"{OUT}/words_text.txt", "w").write("\n".join(text + oov_words) + "\n")
    open(f"{OUT}/words_words.txt", "w").write(" ".join(vocab + oov_words) + "\n")
    in_text = {w for t in text for w in t.split()}
    print(f"clean text {len(text)} sentences; train vocab {len(vocab)} words; OOV words not in train vocab "
          f"{len(oov_words)}, of which {sum(w in in_text for w in oov_words)} already occur somewhere in clean_text")


if __name__ == "__main__":
    main()
