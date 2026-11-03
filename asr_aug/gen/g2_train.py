"""G2 training (asr_aug/gen/README.md, amendment 3): the repo's train.py Trainer, made resumable for preemptible nodes.

    python g2_train.py --model-dir <run>/model --data-dir /scratch/pa2753/gen/g2/prepared \
        --config <run>/g2.yaml --epochs 20 --batch-size 2 --num-workers 4 [--max-train-batches N --skip-validation]

Unchanged from the repo: model, LoRA/LoRA+ optimizer, cosine schedule, losses, GRL ramp, per-batch step, validation and
best-by-val-mel selection (gpt_best.pth). Changes, all for resuming:
  - each epoch's order is a fixed permutation (seed 91 + epoch) instead of DataLoader shuffle, so a resumed epoch
    continues where it stopped;
  - every --save-every batches and at each epoch end, resume.pt holds the trainable (unmerged) parameters, optimizer,
    scheduler, position, step counters, best val loss and RNG states (written to a temp file, then renamed);
  - the per-epoch merged gpt_epoch_<k>.pth copies are not written (only gpt_best.pth and gpt_finetuned.pth are used).
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO = "/scratch/pa2753/gen/code/Controllable-Dysarthric-Speech-Synthesis"
sys.path.insert(0, REPO)
os.chdir(REPO)
import train as T  # noqa: E402


class ResumableTrainer(T.Trainer):
    def _resume_path(self):
        return os.path.join(self.checkpoint_dir, "resume.pt")

    def _save_resume(self, epoch, next_batch):
        state = {
            "trainable": {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad},
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "epoch": epoch, "next_batch": next_batch, "global_step": self.global_step,
            "update_steps": self.update_steps, "best_val_loss": self.best_val_loss,
            "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()),
        }
        tmp = self._resume_path() + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self._resume_path())

    def _load_resume(self):
        s = torch.load(self._resume_path(), map_location="cpu", weights_only=False)
        missing = set(s["trainable"]) - {n for n, _ in self.model.named_parameters()}
        assert not missing, f"resume has unknown params: {sorted(missing)[:5]}"
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, v in s["trainable"].items():
                params[n].copy_(v.to(params[n].device))
        self.optimizer.load_state_dict(s["optimizer"])
        self.scheduler.load_state_dict(s["scheduler"])
        self.global_step, self.update_steps, self.best_val_loss = s["global_step"], s["update_steps"], s["best_val_loss"]
        random.setstate(s["rng"][0]), np.random.set_state(s["rng"][1]), torch.set_rng_state(s["rng"][2])
        torch.cuda.set_rng_state_all(s["rng"][3])
        logger.info(f"Resumed at epoch {s['epoch'] + 1}, batch {s['next_batch']}, global_step {self.global_step}")
        return s["epoch"], s["next_batch"]

    def train_resumable(self, train_set, valid_ds, batch_size, num_workers, save_every, max_train_batches=None,
                        skip_validation=False):
        train_cfg = self.config.train
        batches_per_epoch = (len(train_set) + batch_size - 1) // batch_size
        if max_train_batches is not None:
            batches_per_epoch = min(batches_per_epoch, max_train_batches)
        self._setup_optimizer_and_scheduler(num_training_steps=max(1, batches_per_epoch * train_cfg.epochs))
        logger.info(f"Starting training for {train_cfg.epochs} epochs, {batches_per_epoch} batches per epoch.")
        self.global_step = 0
        start_epoch, start_batch = self._load_resume() if os.path.exists(self._resume_path()) else (0, 0)
        text_weight = train_cfg.text_weight
        alpha = float(getattr(self.config.pathology, "alpha", 1.0))
        beta = float(getattr(self.config.pathology, "beta", 1.0))
        for epoch in range(start_epoch, train_cfg.epochs):
            first = start_batch if epoch == start_epoch else 0
            perm = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(train_cfg.seed + epoch)).tolist()
            loader = DataLoader(train_set, batch_size=batch_size, sampler=perm[first * batch_size:],
                                collate_fn=T.collate_finetune_fn, num_workers=num_workers)
            logger.info(f"EPOCH {epoch + 1}/{train_cfg.epochs} started at batch {first}" + "=" * 30)
            for batch_idx, batch in enumerate(loader, start=first):
                if batch_idx >= batches_per_epoch:
                    break
                data_batch = tuple(x.to(self.device) if torch.is_tensor(x) else x for x in batch)
                loss_text, loss_mel, sum_cls_loss, spk_cls_loss, acc = self._train_step(data_batch)
                weighted_loss = (text_weight * loss_text + (1.0 - text_weight) * loss_mel
                                 + alpha * sum_cls_loss + beta * spk_cls_loss)
                if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                    logger.warning(f"NaN or Inf loss at epoch {epoch}, batch {batch_idx}. Skipping.")
                    continue
                self.optimizer.zero_grad()
                weighted_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.update_steps += 1
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{batches_per_epoch} | "
                                f"text_loss={loss_text.item():.4f}, mel_loss={loss_mel.item():.4f}, "
                                f"sum_cls={sum_cls_loss.item():.4f}, spk_cls={spk_cls_loss.item():.4f}, "
                                f"acc@1={acc['acc_1']:.2f}%, grad_norm={grad_norm.item():.2f}")
                self.global_step += 1
                if self.save_checkpoints and (batch_idx + 1) % save_every == 0:
                    self._save_resume(epoch, batch_idx + 1)
            if skip_validation:
                val_text_loss, val_mel_loss = float(loss_text.item()), float(loss_mel.item())
            else:
                val_text_loss, val_mel_loss, _, _, _ = self._validate_epoch(valid_ds, epoch)
            logger.info(f"Epoch {epoch + 1} val text_loss={val_text_loss:.4f} mel_loss={val_mel_loss:.4f}")
            if self.save_checkpoints and val_mel_loss < self.best_val_loss[2]:
                logger.info(f"New best validation mel_loss: {val_mel_loss:.4f}. Saving best model.")
                self.best_val_loss = (epoch, val_text_loss, val_mel_loss)
                self._save_checkpoint("gpt_best.pth", merge_lora=True, unload_after_merge=True)
            if self.save_checkpoints:
                self._save_resume(epoch + 1, 0)
            T.clear_torch_cache()
        logger.info("Training finished.")
        if self.save_checkpoints and not os.path.exists(os.path.join(self.checkpoint_dir, "gpt_finetuned.pth")):
            self._save_checkpoint("gpt_finetuned.pth", merge_lora=True, unload_after_merge=True)
        logger.info(f"Best validation loss at epoch {self.best_val_loss[0] + 1}: mel_loss {self.best_val_loss[2]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--max-train-batches", type=int)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    config = OmegaConf.load(a.config)
    config.train.finetune_model_dir = os.path.abspath(a.model_dir)
    config.train.data_path = os.path.abspath(a.data_dir)
    config.pathology.embedding_dir = os.path.join(config.train.data_path, "pathology_embedding")
    config.train.epochs = a.epochs
    bpe = os.path.join(config.train.finetune_model_dir, config.dataset.bpe_model)
    train_set, valid_set = T.load_finetune_datasets_pathology(config, bpe)
    valid_ds = DataLoader(valid_set, batch_size=min(a.batch_size, 8), shuffle=False, collate_fn=T.collate_finetune_fn,
                          num_workers=min(a.num_workers, 2))
    logger.info(f"train items {len(train_set)}, valid items {len(valid_set)}")
    trainer = ResumableTrainer(config, save_checkpoints=not a.no_save)
    trainer.train_resumable(train_set, valid_ds, a.batch_size, a.num_workers, a.save_every, a.max_train_batches,
                            a.skip_validation)


if __name__ == "__main__":
    main()
