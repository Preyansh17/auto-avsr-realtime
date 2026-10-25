"""Split the unique s1_oov sentences into 2 folds (fixed seed) and write sents_fold{A,B}.json for build_plan_ft.py."""
import json, random, re
W = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/w50s350_v1/roles"
D = "/scratch/pa2753/asr_aug/data"
def norm(t): return re.sub(r"\s+", " ", t.lower().replace("-", " ")).strip()
seen = {}
for l in open(f"{W}/s1_oov.jsonl"):
    t = json.loads(l)["transcript"]; seen.setdefault(norm(t), t)
keys = sorted(seen); random.Random("oov-2fold").shuffle(keys)
for name, ks in (("A", keys[0::2]), ("B", keys[1::2])):
    json.dump([{"text": seen[k], "norm": k, "pool": "F"} for k in sorted(ks)], open(f"{D}/sents_fold{name}.json", "w"), indent=1)
    print(name, len(ks), sorted(ks)[:4])
