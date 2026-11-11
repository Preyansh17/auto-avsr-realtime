"""Build G4 sentence sets (A16) as phonemized item lists for gen.py sample.

Sets (all under /scratch/pa2753/synvid/t2m2/g4/<set>/items.jsonl):
  a  30 held-out reconstruction sentences (pool heldout30; real crops exist for reference)
  b  100 new sentences from DailyDialog pool (3-10 words, seed 20261117), not in pool or any eval text
  d  169 V0 support (S) sentences: contain W words the generator never saw on video
Phones: CMUdict first pronunciation (stress removed), g2p_en fallback.
"""
import json, os, random, re, sys

SV = "/scratch/pa2753/synvid"
OUT = f"{SV}/t2m2/g4"


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def load_cmu():
    d = {}
    for l in open("/home/pa2753/nltk_data/corpora/cmudict/cmudict", encoding="latin-1"):
        if l.startswith(";;;"):
            continue
        w, *ph = l.split()
        w = re.sub(r"\(\d+\)$", "", w).lower()
        if w not in d:
            d[w] = [q for q in (re.sub(r"\d", "", p) for p in ph) if q.isalpha()]  # nltk format: "word N PH..."
    return d


CMU = load_cmu()
_g2p = None


def phones(word):
    global _g2p
    if word in CMU:
        return CMU[word], "cmu"
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    return [re.sub(r"\d", "", p) for p in _g2p.predict(word) if re.sub(r"\d", "", p).isalpha()], "g2p"


def norm(t):
    return re.sub(r"[^a-z' ]", " ", t.lower()).split()


def item(sid, text):
    ws = norm(text)
    out, src = [], []
    for w in ws:
        p, s = phones(w)
        out.append([w, p]); src.append(s)
    return dict(sid=sid, text=" ".join(ws), phones=out, g2p_words=sum(s == "g2p" for s in src))


def main():
    pool = jl(f"{SV}/t2m2/pool/pool.jsonl")
    sets = {}
    sets["a"] = [item(f"a_{i:02d}", r["norm"]) | dict(real_id=r["id"]) for i, r in
                 enumerate(sorted((r for r in pool if r.get("role") == "heldout"), key=lambda r: r["id"]))]
    sets["d"] = [item(f"d_{i:03d}", r["norm"]) | dict(real_id=r["id"]) for i, r in
                 enumerate(jl(f"{SV}/v0/split/roles/v0_S.jsonl"))]
    seen = {r["norm"] for r in pool}
    for role in os.listdir(f"{SV}/v0/split/roles"):
        seen |= {r["norm"] for r in jl(f"{SV}/v0/split/roles/{role}")}
    cand = [r["norm"] for r in jl(f"{SV}/t2m2/text/dd_pool.jsonl")
            if 3 <= len(r["norm"].split()) <= 10 and r["norm"] not in seen]
    cand = sorted(set(cand))
    random.Random(20261117).shuffle(cand)
    sets["b"] = [item(f"b_{i:03d}", t) for i, t in enumerate(cand[:100])]
    rep = {}
    for k, v in sets.items():
        os.makedirs(f"{OUT}/{k}", exist_ok=True)
        with open(f"{OUT}/{k}/items.jsonl", "w") as f:
            for it in v:
                f.write(json.dumps(it) + "\n")
        rep[k] = dict(n=len(v), words=sum(len(it["phones"]) for it in v), g2p_words=sum(it["g2p_words"] for it in v),
                      empty=sum(any(not p for _, p in it["phones"]) for it in v))
    json.dump(rep, open(f"{OUT}/phonemize_report.json", "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
