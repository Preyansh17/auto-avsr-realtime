"""LoRA adapters for the streaming Emformer RNN-T.

Adapted from the offline repo's modules/lora.py (which targeted espnet
Conformer scopes). Linear layers inside the chosen scopes are wrapped with
low-rank adapters and everything else is frozen; merge_lora() folds the
adapters back into plain Linear weights so the result is a standard
checkpoint usable by eval.py / demo_realtime.py.

Scopes (friendly name -> module prefix):
  encoder         model.transcriber   (Emformer attention + ffn + in/out linears)
  predictor       model.predictor     (LSTM gate projections + output linear)
  joiner          model.joiner
  fusion          fusion
  video_frontend  video_frontend      (a Linear only in the device architecture)
  audio_frontend  audio_frontend      (conv-only: no Linear layers, no-op)
"""

import math
from typing import Iterable, List, Tuple

import torch
from torch import nn

SCOPE_PREFIXES = {
    "encoder": "model.transcriber",
    "predictor": "model.predictor",
    "joiner": "model.joiner",
    "fusion": "fusion",
    "video_frontend": "video_frontend",
    "audio_frontend": "audio_frontend",
}


class LoRALinear(nn.Module):
    """Drop-in Linear with a frozen base weight and trainable low-rank update:
    y = x W^T + b + scale * x A^T B^T."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.bias_param = (
            nn.Parameter(base.bias.data.clone(), requires_grad=False) if base.bias is not None else None
        )
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(x, self.weight, self.bias_param)
        lora = nn.functional.linear(nn.functional.linear(self.dropout(x), self.lora_A), self.lora_B)
        return result + self.scaling * lora

    def merged_linear(self) -> nn.Linear:
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias_param is not None)
        with torch.no_grad():
            linear.weight.copy_(self.weight + self.scaling * (self.lora_B @ self.lora_A))
            if self.bias_param is not None:
                linear.bias.copy_(self.bias_param)
        return linear


def resolve_scopes(scopes: Iterable[str]) -> List[str]:
    prefixes = []
    for scope in scopes:
        if scope == "all":
            return list(SCOPE_PREFIXES.values())
        if scope not in SCOPE_PREFIXES:
            raise ValueError(f"Unknown LoRA scope {scope!r}; choose from {sorted(SCOPE_PREFIXES)} or 'all'")
        prefixes.append(SCOPE_PREFIXES[scope])
    return prefixes


def inject_lora(
    module: nn.Module,
    scopes: Iterable[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> Tuple[int, int, int]:
    """Wrap Linears in the given scopes and freeze all non-LoRA parameters.

    Returns (replaced_layers, trainable_params, total_params).
    """
    prefixes = resolve_scopes(scopes)
    replaced = 0
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if not any(name == p or name.startswith(p + ".") for p in prefixes):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        replaced += 1
    if replaced == 0:
        raise ValueError(f"No Linear layers found in LoRA scopes {list(scopes)}")

    for name, param in module.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return replaced, trainable, total


def merge_lora(module: nn.Module) -> int:
    """Fold every LoRALinear back into a plain nn.Linear (in place)."""
    merged = 0
    for name, child in list(module.named_modules()):
        if not isinstance(child, LoRALinear):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, child_name, child.merged_linear())
        merged += 1
    for param in module.parameters():
        param.requires_grad = True
    return merged
