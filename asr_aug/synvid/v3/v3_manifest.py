"""A17 V3: training rows for the 169 generated S clips (G4 set d, generator seed 1), one per V0 S sentence, same targets as the real S clips."""
import json
from pathlib import Path
SV = Path("/scratch/pa2753/synvid")
S = [json.loads(l) for l in (SV / "v0/split/roles/v0_S.jsonl").read_text().splitlines() if l.strip()]
D = {r["sid"]: r for r in map(json.loads, (SV / "t2m2/g4/d/rows.jsonl").read_text().splitlines())}
out = []
for i, r in enumerate(S):
    g = D[f"d_{i:03d}"]
    assert g["valid"] and g["transcript"] == r["norm"], (i, g["transcript"], r["norm"])
    out.append(dict(id=r["id"] + "__gen", take_id=r["id"] + "__gen", rel_path=r["id"] + "__gen.npz", domain="gen",
                    raw_sentence=r["raw_sentence"], transcript=r["transcript"], norm=r["norm"], video_path=g["npz_path"]))
(SV / "v3").mkdir(exist_ok=True)
(SV / "v3/v3_S_gen.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
print(len(out), "rows")
