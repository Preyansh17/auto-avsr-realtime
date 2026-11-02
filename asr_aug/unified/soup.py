"""Uniform weight average ("soup") of fine-tuned .nemo checkpoints from one base (PLAN.md 3.3 item 0, v3).

    python soup.py --out soup.nemo a.nemo b.nemo c.nemo

Tianyu's phone model is this: a uniform soup of three seeds (package recipe/soup_info.json). Float tensors are
averaged; integer buffers (e.g. BatchNorm num_batches_tracked) are taken from the first ingredient. BN running stats
are frozen during our fine-tunes, so they are identical across ingredients anyway.
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("ingredients", nargs="+")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(a.ingredients[0], map_location="cpu")
    acc = {k: v.clone().double() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    for p in a.ingredients[1:]:
        sd = nemo_asr.models.ASRModel.restore_from(p, map_location="cpu").state_dict()
        assert sd.keys() == acc.keys(), f"{p}: parameter names differ"
        for k, v in sd.items():
            if v.is_floating_point():
                acc[k] += v.double()
    n = len(a.ingredients)
    ref = model.state_dict()
    model.load_state_dict({k: (v / n).to(ref[k].dtype) if ref[k].is_floating_point() else v for k, v in acc.items()})
    model.save_to(a.out)
    print(f"soup of {n} -> {a.out}")


if __name__ == "__main__":
    main()
