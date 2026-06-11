import math
import re
from typing import Iterable, List, Optional

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Drop-in Linear with LoRA adapters.

    W_out x = (W + scale * A @ B) x, where A: in->r, B: r->out.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = base.bias is not None
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        self.bias_param = None
        if self.bias:
            self.bias_param = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros((r, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)))
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization as in LoRA paper
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.nn.functional.linear(x, self.weight, self.bias_param)
        lora = self.lora_B @ (self.lora_A @ self.dropout(x).transpose(-1, -2))
        lora = lora.transpose(-1, -2)
        return result + self.scaling * lora


def _qualname(root: nn.Module, module: nn.Module) -> Optional[str]:
    for name, m in root.named_modules():
        if m is module:
            return name
    return None


def inject_lora(
    model: nn.Module,
    target_scopes: Iterable[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    name_patterns: Optional[List[str]] = None,
    pattern_scopes: Optional[List[str]] = None,
) -> int:
    """Replace Linear layers within target scopes by LoRALinear.

    Args:
        model: root module
        target_scopes: e.g., ["encoder", "aux_encoder", "decoder"]
        r, alpha, dropout: LoRA hparams
        name_patterns: optional list of regex to further filter linear names
        pattern_scopes: scopes to apply patterns to (if None, apply to all target_scopes)
    Returns:
        count of layers replaced
    """
    replaced = 0
    patterns = [re.compile(p) for p in (name_patterns or [])]
    pattern_scopes_set = set(pattern_scopes) if pattern_scopes else set(target_scopes)

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        qual = name
        # Check if in target scope
        if not any(qual.startswith(scope) for scope in target_scopes):
            continue
        # Apply pattern filtering only to specific scopes
        if patterns:
            in_pattern_scope = any(qual.startswith(scope) for scope in pattern_scopes_set)
            if in_pattern_scope and not any(p.search(qual) for p in patterns):
                continue
        # Find parent to set attribute
        parent_name = qual.rsplit(".", 1)[0] if "." in qual else ""
        child_name = qual.split(".")[-1]
        parent = model if parent_name == "" else dict(model.named_modules())[parent_name]
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced += 1
    return replaced


