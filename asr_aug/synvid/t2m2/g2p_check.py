"""A13 step 0.3: does g2p_en run offline here? Phonemises the pool's CMUdict-OOV words (from step0_stats.json)."""
import json, sys
from g2p_en import G2p
d = json.load(open("/scratch/pa2753/synvid/t2m2/step0_stats.json"))
w0 = d.get("timings", {}).get("cmudict_oov_words", {}).get("words", [])
w = sorted(set(w0) | {"jasper", "jiujitsu", "chatgpt"})
g = G2p()
# G2p.__call__ needs the NLTK tagger "averaged_perceptron_tagger_eng" (absent offline; only used for homographs).
# Single OOV words go straight to the neural predictor, which needs no tagger.
res = {x: g.predict(x) for x in w}
print(json.dumps(dict(ok=True, n=len(res), sample=dict(list(res.items())[:15])), indent=1))
json.dump(res, open("/scratch/pa2753/synvid/t2m2/align/g2p_oov.json", "w"), indent=1)
