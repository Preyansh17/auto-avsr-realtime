"""semg1 (oct1 session) / semg2 / pooled label CSVs in the repo format (dataset_name,rel_path,frames,token_ids)
from the combined_v5 role JSONLs (token_ids in the roles are NOT spm1023, so transcripts are re-encoded). Read-only on th3482; writes /scratch/pa2753/multi_patient.

    python3 make_labels.py
"""
import json
import os
import re

import sentencepiece as spm

S = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined"
W = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/w50s350_v1"
M = "/scratch/pa2753/multi_patient"
SP = spm.SentencePieceProcessor(model_file="/home/pa2753/auto-avsr-realtime/spm/spm_unigram_1023.model")
SETS = {
    "s2_train": ["s2_train"], "s2_val": ["s2_val"], "s2_test": ["s2_test"], "s2_oov": ["s2_oov"],
    "s1_train": ["s1_train"], "s1_val": ["s1_val"], "s1_test": ["s1_test"], "s1_oov": ["s1_oov"],
    "pool12_train": ["s1_train", "s2_train"], "pool12_val": ["s1_val", "s2_val"],
}


def rows(role):
    out, miss = [], 0
    for line in open(f"{S}/roles/{role}.jsonl"):
        x = json.loads(line)
        wav = f"{M}/root/{x['dataset_name']}/" + x["rel_path"].rsplit(".", 1)[0] + ".wav"
        miss += not os.path.exists(wav)
        # the role token_ids use another tokenizer; re-encode the transcript with spm1023 (lowercase, hyphen = space)
        text = re.sub(r"\s+", " ", x["transcript"].lower().replace("-", " ")).strip()
        ids = SP.encode(text)
        assert SP.decode(ids) == text, text
        out.append(f"{x['dataset_name']},{x['rel_path']},{x['frame_count']},{' '.join(map(str, ids))}")
    return out, miss


def main():
    os.makedirs(f"{M}/labels", exist_ok=True)
    os.makedirs(f"{M}/root", exist_ok=True)
    for d in ("semg1_oct1_retinaface", "semg2_oct1_retinaface"):
        if not os.path.islink(f"{M}/root/{d}"):
            os.symlink(f"{W}/{d}", f"{M}/root/{d}")
    for name, roles in SETS.items():
        r, m = [], 0
        for role in roles:
            a, b = rows(role)
            r += a
            m += b
        open(f"{M}/labels/{name}_spm1023.csv", "w").write("\n".join(r) + "\n")
        print(name, len(r), "missing wav", m)


if __name__ == "__main__":
    main()
