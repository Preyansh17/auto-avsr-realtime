"""LoRA finetune of Higgs Audio v2 on the patient's own clips (plain PyTorch loop, batch 1 x grad-accum).

Each step: target = one tts_train clip, reference = 1-5 other-sentence tts_train clips stitched like
make_refs.py (fresh draw every epoch), prompt identical to inference. Loss = CE on the target
assistant audio tokens only, all 8 codebooks. The adapter is saved every epoch; checkpoints are chosen
later on generation metrics, never on this loss.

    python train_lora.py --run A_r16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import higgs_ft_lib as L  # noqa: E402

ROOT = os.environ.get("HIGGS_FT_ROOT", "/scratch/pa2753/higgs_ft")
DATA = f"{ROOT}/data"


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def sha1_file(p: str) -> str:
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]


class Data:
    def __init__(self, proc, max_target_sec: float, splits_path: str | None = None):
        self.sp = json.load(open(splits_path or f"{DATA}/splits.json"))
        self.clips = self.sp["clips"]
        index = json.load(open(f"{DATA}/codes_index.json"))
        keep = lambda u: not max_target_sec or index[u] / 25.0 <= max_target_sec  # noqa: E731
        self.train = [u for u in self.sp["tts_train"] if keep(u)]
        self.val = list(self.sp["tts_val"])
        self.proc = proc
        self.pool = L.RefPool(self.clips, self.sp["tts_train"])
        self._codes: dict[str, torch.Tensor] = {}

    def codes(self, u: str):
        if u not in self._codes:
            self._codes[u] = torch.load(f"{DATA}/codes/{u}.pt").long()
        return self._codes[u]

    def example(self, rng: random.Random, target: str, eos_labels: bool, k: int | None = None):
        for attempt in range(4):  # a long target + long stitched reference can exceed the context: redraw smaller
            rw, rt, _ = self.pool.sample(rng, target, k=k or (rng.randint(1, 5) if attempt == 0 else rng.randint(1, 2)))
            ex = L.build_example(self.proc, rt, L.encode_wave(self.proc, rw), self.clips[target]["transcript"],
                                 self.codes(target), eos_labels)
            if ex["input_ids"].shape[1] <= L.MAX_SEQ_LEN:
                return ex
        return None


def n_labelled(ex) -> int:
    return int((ex["labels_pos"][1:] != L.IGNORE).sum())


@torch.no_grad()
def val_loss(model, val_examples, dev) -> float:
    model.eval()
    s = c = 0.0
    for ex in val_examples:
        ss, cc = L.example_ce(model, ex, dev)
        s += float(ss.sum())
        c += float(cc.sum())
    return s / max(c, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss_weighting", choices=["token", "clip"], default="token",
                    help="token: mean over all target tokens in the accumulation group; clip: mean per clip, then over clips")
    ap.add_argument("--no_eos_labels", action="store_true", help="mask stream-EOS labels (default: train them)")
    ap.add_argument("--max_target_sec", type=float, default=0.0, help="drop target clips longer than this (0 = keep all)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--splits", default=None, help="splits json (default data/splits.json); cross-fitting folds: data/cv/splits_cv<k>.json")
    ap.add_argument("--stop_after", type=int, default=0, help="stop after saving this epoch (the cosine schedule still spans --epochs)")
    args = ap.parse_args()

    out = f"{ROOT}/runs/{args.run}"
    if os.path.exists(f"{out}/config.json") and not args.resume:
        sys.exit(f"{out} already holds a run; pick a new --run name or pass --resume (metrics.jsonl is appended to)")
    os.makedirs(out, exist_ok=True)
    dev = "cuda:0"
    seed_all(args.seed)
    proc, model = L.load_models(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict

    pm = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                          target_modules=L.LORA_TARGET_REGEX, bias="none"))
    params = [p for p in pm.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params)/1e6:.2f}M params", flush=True)

    data = Data(proc, args.max_target_sec, args.splits)
    eos = not args.no_eos_labels
    steps_per_epoch = math.ceil(len(data.train) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup_frac * total_steps))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    cfg = {**vars(args), "git_hash": os.environ.get("GIT_HASH", "unknown"), "split_hash": data.sp["log"]["split_hash"],
           "n_train_clips": len(data.train), "n_val_clips": len(data.val), "steps_per_epoch": steps_per_epoch,
           "total_steps": total_steps, "warmup_steps": warmup, "torch": torch.__version__,
           "script_sha1": {f: sha1_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                           for f in ("train_lora.py", "higgs_ft_lib.py")},
           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    try:
        import peft, transformers
        cfg.update(peft=peft.__version__, transformers=transformers.__version__)
    except Exception:
        pass
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)

    start_epoch, step = 0, 0
    if args.resume and os.path.exists(f"{out}/last.pt"):
        st = torch.load(f"{out}/last.pt", map_location="cpu")
        start_epoch, step = st["epoch"], st["step"]
        res = set_peft_model_state_dict(pm, load_peft_weights(f"{out}/epoch_{start_epoch:02d}"))
        bad = list(getattr(res, "unexpected_keys", []) or []) + [k for k in (getattr(res, "missing_keys", []) or []) if "lora_" in k]
        assert not bad, f"adapter checkpoint does not match the model: {bad[:5]}"
        opt.load_state_dict(st["opt"])
        print(f"resumed after epoch {start_epoch} (step {step})", flush=True)
        if args.stop_after and start_epoch >= args.stop_after:
            print("TRAIN_DONE (already at --stop_after)", flush=True)
            return

    # fixed validation examples (references drawn once, from tts_train only) so val loss is comparable across epochs
    vrng = random.Random(1234)
    val_ex = [e for e in (data.example(vrng, u, eos, k=[1, 3, 5][i % 3]) for i, u in enumerate(data.val)) if e is not None]
    print(f"val_loss before training: {val_loss(pm, val_ex, dev):.4f} ({len(val_ex)} clips)", flush=True)

    log = open(f"{out}/metrics.jsonl", "a")
    t_start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        pm.train()
        rng = random.Random(args.seed * 1000 + epoch)
        order = list(data.train)
        rng.shuffle(order)
        ep_sum = ep_tok = 0.0
        t_ep = time.time()
        for g in range(0, len(order), args.accum):
            group = [e for e in (data.example(rng, u, eos) for u in order[g : g + args.accum]) if e is not None]
            n_group = sum(n_labelled(e) for e in group)
            lr = L.cosine_lr(step, total_steps, warmup, args.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr
            for ex in group:
                s, c = L.example_ce(pm, ex, dev)
                loss = s.sum() / n_group if args.loss_weighting == "token" else (s.sum() / c.sum()) / len(group)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}")
                loss.backward()
                ep_sum += float(s.sum())
                ep_tok += float(c.sum())
            gn = torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0:
                print(f"  ep {epoch} step {step}/{total_steps} lr={lr:.2e} loss={ep_sum/max(ep_tok,1):.4f} gnorm={float(gn):.2f} "
                      f"[{(time.time()-t_ep)/60:.1f} min]", flush=True)
        vl = val_loss(pm, val_ex, dev)
        rec = {"epoch": epoch, "train_loss": ep_sum / max(ep_tok, 1.0), "val_loss": vl, "lr": lr, "step": step,
               "epoch_min": (time.time() - t_ep) / 60, "elapsed_min": (time.time() - t_start) / 60}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        pm.save_pretrained(f"{out}/epoch_{epoch:02d}")
        torch.save({"epoch": epoch, "step": step, "opt": opt.state_dict()}, f"{out}/last.pt")
        if args.stop_after and epoch >= args.stop_after:
            break
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
