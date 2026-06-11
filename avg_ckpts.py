import os
import re
from pathlib import Path

import torch


def average_checkpoints(last):
    if not last:
        raise ValueError("No checkpoints provided for averaging.")

    avg = None
    for path in last:
        states = torch.load(path, map_location=lambda storage, loc: storage)[
            "state_dict"
        ]
        states = {k[6:]: v for k, v in states.items() if k.startswith("model.")}
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= len(last)
            else:
                avg[k] //= len(last)
    return avg


def _select_last_checkpoints(exp_path, max_count=10):
    exp_dir = Path(exp_path)
    ckpts = []
    pattern = re.compile(r"^epoch=(\d+)\.ckpt$")
    for p in exp_dir.glob("*.ckpt"):
        m = pattern.match(p.name)
        if m:
            ckpts.append((int(m.group(1)), str(p)))

    ckpts.sort(key=lambda x: x[0])
    selected = [p for _, p in ckpts[-max_count:]]

    if not selected:
        last_ckpt = exp_dir / "last.ckpt"
        if last_ckpt.exists():
            selected = [str(last_ckpt)]

    return selected


def ensemble(args):
    exp_path = os.path.join(args.exp_dir, args.exp_name)
    last = _select_last_checkpoints(exp_path, max_count=10)
    if not last:
        raise FileNotFoundError(
            f"No checkpoint found under {exp_path}. "
            "Expected files like epoch=<n>.ckpt or last.ckpt"
        )

    print(f"[avg_ckpts] Averaging {len(last)} checkpoint(s)")
    model_path = os.path.join(
        args.exp_dir, args.exp_name, f"model_avg_10.pth"
    )
    torch.save(average_checkpoints(last), model_path)
    return model_path
