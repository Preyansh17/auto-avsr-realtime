import math

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine decay, stepped per optimizer step.

    Ported from pytorch/audio examples/avsr/schedulers.py.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: float,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self._step_count / max(self.warmup_steps, 1) * base_lr for base_lr in self.base_lrs]
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        return [
            0.5 * base_lr * (1 + math.cos(math.pi * (self._step_count - self.warmup_steps) / decay_steps))
            for base_lr in self.base_lrs
        ]
