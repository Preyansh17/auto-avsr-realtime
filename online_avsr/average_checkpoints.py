import glob
import os
from typing import List, Optional

import torch


def average_checkpoints(paths: List[str]) -> dict:
    """Average the state_dicts of several Lightning checkpoints.

    Ported from pytorch/audio examples/avsr/average_checkpoints.py.
    """
    avg = None
    for path in paths:
        states = torch.load(path, map_location="cpu")["state_dict"]
        if avg is None:
            avg = {k: v.clone() for k, v in states.items()}
        else:
            for k in avg.keys():
                avg[k] += states[k]
    for k in avg.keys():
        if avg[k].is_floating_point():
            avg[k] /= len(paths)
        else:
            avg[k] //= len(paths)
    return avg


def ensemble(run_dir: str, last_n: int = 10, out_name: str = "model_avg.pth") -> Optional[str]:
    """Average the most recent epoch checkpoints in a run directory.

    Unlike the upstream recipe (fixed epoch={n}.ckpt names), this globs
    whatever epoch checkpoints ModelCheckpoint produced and averages the
    newest last_n of them. Returns the output path, or None if fewer than
    two checkpoints exist.
    """
    candidates = sorted(
        (p for p in glob.glob(os.path.join(run_dir, "*.ckpt")) if "last" not in os.path.basename(p)),
        key=os.path.getmtime,
    )
    if len(candidates) < 2:
        return None
    paths = candidates[-last_n:]
    out_path = os.path.join(run_dir, out_name)
    torch.save({"state_dict": average_checkpoints(paths), "averaged_from": paths}, out_path)
    return out_path
