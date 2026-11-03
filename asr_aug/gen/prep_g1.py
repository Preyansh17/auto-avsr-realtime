"""G1 data and generation plans (asr_aug/gen/README.md, amendment 1: match Higgs exactly). One IndexTTS-2 LoRA per
Higgs adapter, trained on the same tts_train clips, regenerating the same (text, reference) rows:
    old: higgs_ft/data/splits.json (the ONS adapter B_r64, split_v1 data) -> asr_aug/data/plan_gen.json (4 refs x 500)
    s1:  stack/higgs/s1/data/splits.json (Oct 1 adapter)                 -> stack/data/plan_gen.json, FT_ST_s1_* units
    s2:  stack/higgs/s2/data/splits.json                                 -> stack/data/plan_gen.json, FT_ST_s2_* units
Writes /scratch/pa2753/gen/g1/<v>/{wavs/ (symlinks), manifest.tsv, plan.json}. Each plan row's `rel` is the Higgs clip's
path relative to gen_runs, so a G1 clip sits at gen/root/g1/<rel> and replaces the Higgs row with the same rel.
Fails if any plan text matches a held-out sentence (check_leak.py).

    python3 prep_g1.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_leak import held_out, norm  # noqa: E402

G = "/scratch/pa2753/gen/g1"
VOICES = {
    "old": ("/scratch/pa2753/higgs_ft/data/splits.json", "/scratch/pa2753/asr_aug/data/plan_gen.json", None),
    "s1": ("/scratch/pa2753/stack/higgs/s1/data/splits.json", "/scratch/pa2753/stack/data/plan_gen.json", "FT_ST_s1_"),
    "s2": ("/scratch/pa2753/stack/higgs/s2/data/splits.json", "/scratch/pa2753/stack/data/plan_gen.json", "FT_ST_s2_"),
}


def main():
    held = held_out()
    for v, (splits, plan, prefix) in VOICES.items():
        d = f"{G}/{v}"
        os.makedirs(f"{d}/wavs", exist_ok=True)
        sp = json.load(open(splits))
        with open(f"{d}/manifest.tsv", "w") as f:
            for u in sp["tts_train"]:
                c = sp["clips"][u]
                link = f"{d}/wavs/{u}.wav"
                if not os.path.lexists(link):
                    os.symlink(c["audio_path"], link)
                f.write(f"{u}\t{c['transcript'].strip()}\n")
        rows = []
        for unit in json.load(open(plan))["units"]:
            if prefix and not unit["variant"].startswith(prefix):
                continue
            for p in unit["prompts"]:
                rows.append({"rel": f"{unit['variant']}/synthetic/audio/{unit['speaker']}/train/{p['prompt_id']}.wav",
                             "ref_wav": unit["ref_wav"], "text": p["text"]})
        assert len({r["rel"] for r in rows}) == len(rows), f"{v}: duplicate rel paths"
        # The ONS plan (old) predates the s1/s2 oov sets and has 2 rows whose text is now s1/s2 oov ("I feel hot now",
        # "The computer is not working"). make_mix.py already drops them from every E2 draw; drop them here so they are
        # never generated.
        leaks = [r for r in rows if norm(r["text"]) in held]
        rows = [r for r in rows if norm(r["text"]) not in held]
        assert not [r for r in rows if norm(r["text"]) in held]
        json.dump(rows, open(f"{d}/plan.json", "w"), indent=0)
        print(f"{v}: {len(sp['tts_train'])} train clips, {len(rows)} plan rows, "
              f"{len({r['ref_wav'] for r in rows})} refs, dropped {len(leaks)} held-out rows {[r['text'] for r in leaks]}")


if __name__ == "__main__":
    main()
