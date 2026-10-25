"""Sentence pools for Higgs synthetic ASR augmentation, with a leakage audit.

    python asr_aug/build_sentences.py --split-csv split_assignments.csv --out-dir data/ [--n-a 1000 --n-b 1000]

Pool A ("known"): new sentences built only from the split_v1 ASR-train vocabulary (76 words), so they recombine
words the patient already said in training. Pool B ("open"): everyday / AAC-register sentences with open vocabulary.
Both come from small hand-written grammars sampled with a fixed seed (no external text).

Leakage rules (asserted, written to audit.json):
  * no sentence equals, or is within one word edit of, any ASR val or test sentence;
  * no sentence contains a word that is in val/test but not in train (the "unseen-word" test subset stays unseen);
  * pool A contains no exact ASR-train sentence (it is new text) and only train-vocab words.
Output: pool_A.json / pool_B.json (lists of {"text", "norm", "pool"}) and audit.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re

SEED = 20261012

# ---------------------------------------------------------------- pool A grammar (train vocabulary only)
# Agreement is kept safe by only using bare verbs after i/you/we/they or after a modal / "did" / "did not".
A_PRON = ["i", "you", "we", "they"]
A_NP3 = ["my client", "the judge", "my nurse", "my family", "the client"]
A_MODAL = ["will", "can", "did", "will not", "can not", "did not"]
A_PRES_NEG = ["do not"]
A_VT = {  # transitive verb -> objects
    "confirm": ["the case", "the trial", "the schedule", "the appeal", "your signature", "it", "this", "that",
                "the case schedule", "the trial schedule", "my signature"],
    "get": ["the case", "your signature", "my glasses", "my computer", "it", "this", "that", "the schedule",
            "the signature", "them", "my point"],
    "know": ["the judge", "the client", "the case", "the schedule", "my nurse", "it", "this", "that", "them", "us",
             "your point", "about the case", "about the trial", "about the appeal", "about this", "about that",
             "about the litigation", "about the exposure"],
    "have": ["the case", "the schedule", "your signature", "my glasses", "my computer", "faith", "it", "this",
             "the point"],
    "need": ["my glasses", "my nurse", "my computer", "your signature", "the schedule", "it", "this", "them", "you",
             "us", "my family"],
    "like": ["my nurse", "the judge", "my computer", "it", "this", "that", "them", "the schedule"],
    "schedule": ["the trial", "the appeal", "it", "this", "that"],
    "appeal": ["the case", "it", "this", "that"],
    "bring": ["my glasses", "my computer", "the schedule", "the case", "it", "them", "my family"],
    "tell": ["my family", "my nurse", "the judge", "the client", "my client", "them", "us", "you"],
    "clean": ["my glasses", "my computer", "it", "this", "that"],
}
A_TAIL = ["", "", "", "now", "here", "there", "with us", "with them", "with you", "for us", "for you", "for them",
          "for the client", "for my client", "with the judge", "on the schedule", "for the trial", "in the case",
          "in this case", "for the appeal", "in the litigation"]
A_ADJ = ["comfortable", "hungry", "thirsty", "okay", "clean", "good", "very comfortable", "very hungry",
         "very thirsty", "very good", "not comfortable", "not hungry", "not okay", "outside", "here", "there",
         "on trial", "in litigation", "with us", "with them", "right"]
A_WH = ["what", "when", "where", "how"]


def gen_a(rng: random.Random) -> str:
    t = rng.randrange(14)
    pron = rng.choice(A_PRON)
    subj = rng.choice(A_PRON + A_NP3)
    v = rng.choice(list(A_VT))
    o = rng.choice(A_VT[v])
    tail = rng.choice(A_TAIL)
    if t == 0:
        return f"{subj} {rng.choice(A_MODAL)} {v} {o} {tail}"
    if t == 1:
        return f"{pron} {rng.choice(A_PRES_NEG + ['', ''])} {v} {o} {tail}"
    if t == 2:
        return f"can you {v} {o} {tail}"
    if t == 3:
        return f"when will {subj} {v} {o}"
    if t == 4:
        s2 = rng.choice(A_PRON + A_NP3)
        return f"{pron} {rng.choice(['think', 'know', 'hope', 'do not think', 'do not know'])} {s2} {rng.choice(['will', 'can', 'did', 'will not', 'did not'])} {v} {o}"
    if t == 5:
        return f"do you know {rng.choice(A_WH)} {subj} {rng.choice(['will', 'can'])} {v} {o}"
    if t == 6:
        be = {"i": "am", "you": "are", "we": "are", "they": "are"}.get(subj, "is")
        return f"{subj} {be} {rng.choice(A_ADJ)} {rng.choice(['', '', 'now', 'here'])}"
    if t == 7:
        verb = "feel" if subj in A_PRON else rng.choice(["will feel", "did not feel", "can feel"])
        return f"{subj} {verb} {rng.choice(['comfortable', 'hungry', 'thirsty', 'okay', 'good', 'very comfortable', 'very good', 'very hungry', 'very thirsty'])}"
    if t == 8:
        return f"please {v} {o} {tail}"
    if t == 9:
        be = {"i": "am", "you": "are", "we": "are", "they": "are"}.get(subj, "is")
        return f"{rng.choice(['where', 'how'])} {be} {subj} {rng.choice(['', 'now'])}"
    if t == 10:
        return f"{rng.choice(['what', 'when'])} {rng.choice(['will', 'can', 'did'])} {subj} do {rng.choice(['', 'now', 'about it', 'about the case', 'with it', 'with them'])}"
    if t == 11:
        return f"is {rng.choice(['the case', 'the trial', 'the appeal', 'the schedule', 'my computer', 'my glasses', 'it', 'this', 'that'])} {rng.choice(['okay', 'right', 'good', 'clean', 'here', 'there', 'on the schedule', 'for us', 'for you'])}"
    if t == 12:
        return f"if {pron} {rng.choice(['can', 'will', 'did'])} {v} {o} {rng.choice(['tell us', 'tell them', 'tell my family', 'confirm with us', 'confirm it'])}"
    return f"{rng.choice(['yes', 'no', 'okay', 'so'])} {pron} {rng.choice(A_MODAL + ['', ''])} {v} {o} {tail}"


# ---------------------------------------------------------------- pool B grammar (open vocabulary, AAC register)
B_ITEM = ["some water", "a drink", "my medicine", "a blanket", "another pillow", "my phone", "the remote",
          "my wheelchair", "some coffee", "a cup of tea", "some juice", "my book", "the newspaper", "a towel",
          "my shoes", "my jacket", "a tissue", "some ice", "my headphones", "the charger", "my wallet",
          "a pen", "some soup", "my toothbrush", "a straw", "my hearing aid", "the menu", "my keys"]
B_PERSON = ["the doctor", "my wife", "my husband", "my son", "my daughter", "my brother", "my sister", "my friend",
            "my mother", "my father", "the nurse", "the therapist", "my neighbor", "my lawyer", "the driver",
            "my grandson", "my granddaughter", "the pharmacist"]
B_ACT = ["open the window", "close the door", "turn on the light", "turn off the light", "turn on the television",
         "turn off the television", "raise the bed", "lower the bed", "move my leg", "move my arm",
         "help me sit up", "help me stand up", "fix my pillow", "charge my phone", "read this to me",
         "call my daughter", "call my son", "call the doctor", "close the curtains", "open the curtains",
         "turn the volume down", "turn the volume up", "change the channel", "push my chair closer",
         "pass me the cup", "wipe my mouth", "adjust my glasses", "check my temperature", "help me eat",
         "take me to the bathroom", "take me for a walk", "set an alarm", "write this down", "speak more slowly",
         "repeat that please", "wait a minute"]
B_FEEL = ["cold", "warm", "hot", "sleepy", "dizzy", "sore", "fine", "much better", "a little better", "worried",
          "happy", "sad", "nervous", "full", "uncomfortable", "itchy", "short of breath", "in pain", "relaxed",
          "bored", "lonely", "excited", "grateful"]
B_BODY = ["my back", "my neck", "my head", "my leg", "my arm", "my shoulder", "my stomach", "my chest", "my throat",
          "my hand", "my foot", "my knee", "my eyes"]
B_TIME = ["", "", "today", "tomorrow", "tonight", "this morning", "this afternoon", "later", "soon", "right now",
          "after lunch", "after dinner", "before bed", "on monday", "on friday", "next week", "this weekend"]
B_PLACE = ["the kitchen", "the garden", "the living room", "the bedroom", "the hospital", "the store", "the park",
           "the clinic", "the bathroom", "church", "the porch", "the car"]
B_FOOD = ["soup", "pasta", "rice", "chicken", "fish", "eggs", "toast", "a sandwich", "fruit", "yogurt", "pudding",
          "oatmeal", "pancakes", "ice cream", "salad", "mashed potatoes"]
B_THING = ["the game", "the news", "a movie", "the weather", "the photos", "the letter", "my email", "the show",
           "the baseball game", "the cooking show", "the birds outside"]


def gen_b(rng: random.Random) -> str:
    t = rng.randrange(16)
    tm = rng.choice(B_TIME)
    if t == 0:
        return f"{rng.choice(['i need', 'i want', 'can i have', 'please give me', 'could you get me'])} {rng.choice(B_ITEM)} {rng.choice(['', '', 'please'])}"
    if t == 1:
        return f"{rng.choice(['can you', 'could you', 'please', 'would you'])} {rng.choice(B_ACT)}"
    if t == 2:
        return f"i feel {rng.choice(B_FEEL)} {rng.choice(['', '', 'today', 'tonight', 'this morning', 'right now', 'now'])}"
    if t == 3:
        return f"{rng.choice(B_BODY)} {rng.choice(['hurts', 'is sore', 'feels stiff', 'is itchy', 'feels better', 'is cold'])} {rng.choice(['', '', 'today', 'tonight', 'this morning', 'right now', 'again', 'a lot'])}"
    if t == 4:
        if rng.random() < 0.4:
            return f"{rng.choice(['is', 'when is', 'why is'])} {rng.choice(B_PERSON)} {rng.choice(['coming', 'here', 'coming today', 'on the way', 'still here'])}"
        return f"{rng.choice(['please call', 'i want to see', 'did you talk to', 'can you ask', 'please find'])} {rng.choice(B_PERSON)} {rng.choice(['', 'today', 'about this', 'for me'])}"
    if t == 5:
        return f"i would like {rng.choice(B_FOOD)} for {rng.choice(['breakfast', 'lunch', 'dinner', 'a snack'])}"
    if t == 6:
        return f"{rng.choice(['let us watch', 'can we watch', 'i missed', 'what time is'])} {rng.choice(['the game', 'the news', 'a movie', 'the show', 'the baseball game', 'the cooking show'])} {tm}"
    if t == 7:
        return f"{rng.choice(['i want to go to', 'take me to', 'can we go to', 'when can i go to'])} {rng.choice(B_PLACE)} {tm}"
    if t == 8:
        return f"{rng.choice(B_PERSON)} {rng.choice(['is coming', 'will visit', 'called me', 'said hi', 'needs to know'])} {tm}"
    if t == 9:
        return f"{rng.choice(['thank you for', 'i appreciate', 'sorry about'])} {rng.choice(['your help', 'everything', 'the wait', 'the trouble', 'the visit', 'the flowers', 'the card', 'dinner'])}"
    if t == 10:
        return f"{rng.choice(['what time is it', 'what day is it', 'is it raining outside', 'is it cold outside', 'who is at the door', 'where is my phone', 'when is my appointment', 'when do i take my pills', 'who called', 'what did the doctor say'])}"
    if t == 11:
        return f"{rng.choice(['i am', 'i was', 'i will be'])} {rng.choice(['ready', 'done', 'awake', 'busy', 'hungry', 'thirsty', 'fine', 'comfortable', 'cold', 'warm'])} {tm}"
    if t == 12:
        return f"{rng.choice(['please wait for', 'i will talk to', 'say thank you to', 'give this to', 'tell'])} {rng.choice(B_PERSON)} {rng.choice(['', 'later', 'for me', 'please'])}"
    if t == 13:
        return f"{rng.choice(['i do not want', 'i would rather have', 'no more', 'just a little'])} {rng.choice(B_FOOD + ['coffee', 'tea', 'water', 'juice', 'milk'])}"
    if t == 14:
        return f"{rng.choice(['my', 'the'])} {rng.choice(['phone', 'computer', 'television', 'light', 'heater', 'fan', 'bed', 'chair', 'oxygen machine', 'feeding pump'])} {rng.choice(['is not working', 'is too loud', 'is broken', 'needs charging', 'is beeping', 'works now'])}"
    return f"{rng.choice(['i love you', 'good morning', 'good night', 'see you tomorrow', 'have a nice day', 'that sounds great', 'i agree with you', 'i do not understand', 'say that again please', 'give me a minute', 'i will try again', 'that is funny', 'i miss my dog', 'the food was great today'])}"


# ---------------------------------------------------------------- shared
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", s.lower())).strip()


def edit_le1(a: list[str], b: list[str]) -> bool:
    """Word-level Levenshtein distance <= 1."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", required=True, help="split_v1 split_assignments.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-a", type=int, default=1000)
    ap.add_argument("--n-b", type=int, default=1000)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words-a", type=int, default=9)
    ap.add_argument("--max-words-b", type=int, default=10)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.split_csv)))
    by = collections.defaultdict(set)
    for r in rows:
        by[r["split"]].add(norm(r["normalized_transcript"]))
    train_vocab = {w for s in by["train"] for w in s.split()}
    heldout = by["val"] | by["test"]
    heldout_words = [s.split() for s in heldout]
    forbidden = {w for s in heldout for w in s.split()} - train_vocab
    print(f"train vocab {len(train_vocab)}; held-out sentences {len(heldout)}; forbidden words {sorted(forbidden)}")

    rejects = collections.Counter()

    def ok(s: str, pool: str, maxw: int) -> bool:
        ws = s.split()
        if not (args.min_words <= len(ws) <= maxw):
            rejects[f"{pool}:length"] += 1
            return False
        if forbidden & set(ws):
            rejects[f"{pool}:forbidden_word"] += 1
            return False
        if s in heldout or any(edit_le1(ws, h) for h in heldout_words):
            rejects[f"{pool}:near_heldout"] += 1
            return False
        if pool == "A":
            if set(ws) - train_vocab:
                rejects["A:out_of_vocab"] += 1
                return False
            if s in by["train"]:
                rejects["A:exact_train"] += 1
                return False
        return True

    pools = {}
    for pool, gen, n, maxw in (("A", gen_a, args.n_a, args.max_words_a), ("B", gen_b, args.n_b, args.max_words_b)):
        rng = random.Random(f"{SEED}-{pool}")
        out, seen, tries = [], set(), 0
        while len(out) < n and tries < 200 * n:
            tries += 1
            s = norm(gen(rng))
            if s in seen:
                rejects[f"{pool}:duplicate"] += 1
                continue
            seen.add(s)
            if ok(s, pool, maxw):
                out.append({"text": s[0].upper() + s[1:], "norm": s, "pool": pool})
        assert len(out) == n, f"pool {pool}: only {len(out)} of {n} sentences after {tries} draws"
        pools[pool] = out

    # independent re-check of every rule on the final lists
    for pool, lst in pools.items():
        for e in lst:
            ws = e["norm"].split()
            assert e["norm"] not in heldout and not any(edit_le1(ws, h) for h in heldout_words), e
            assert not (forbidden & set(ws)), e
            if pool == "A":
                assert set(ws) <= train_vocab and e["norm"] not in by["train"], e
    assert not ({e["norm"] for e in pools["A"]} & {e["norm"] for e in pools["B"]})

    os.makedirs(args.out_dir, exist_ok=True)
    stats = {}
    for pool, lst in pools.items():
        json.dump(lst, open(os.path.join(args.out_dir, f"pool_{pool}.json"), "w"), indent=1)
        toks = [w for e in lst for w in e["norm"].split()]
        lens = collections.Counter(len(e["norm"].split()) for e in lst)
        stats[pool] = {"n": len(lst), "unique_words": len(set(toks)),
                       "oov_token_frac_vs_train": round(sum(w not in train_vocab for w in toks) / len(toks), 4),
                       "sentences_with_oov_word": sum(bool(set(e["norm"].split()) - train_vocab) for e in lst),
                       "near_train_le1": sum(any(edit_le1(e["norm"].split(), t.split()) for t in by["train"]) for e in lst),
                       "length_hist": dict(sorted(lens.items()))}
    audit = {"seed": SEED, "train_vocab_size": len(train_vocab), "n_heldout_sentences": len(heldout),
             "forbidden_words": sorted(forbidden), "rejects": dict(rejects), "pools": stats,
             "checks": {"exact_or_near_heldout": 0, "forbidden_word_hits": 0, "pool_A_out_of_vocab": 0,
                        "pool_A_exact_train": 0, "A_B_overlap": 0}}
    json.dump(audit, open(os.path.join(args.out_dir, "audit.json"), "w"), indent=1)
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
