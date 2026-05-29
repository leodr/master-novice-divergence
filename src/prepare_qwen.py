from __future__ import annotations

import time

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .disentangled_linear import DisentangledLinear
except ImportError:
    from disentangled_linear import DisentangledLinear


MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_RANK_FRACTION = 0.1


def _replace_linear_descendants(module: nn.Module, rank_fraction: float) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, DisentangledLinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(
                module,
                child_name,
                DisentangledLinear.from_linear(child, rank_fraction=rank_fraction),
            )
            replaced += 1
        else:
            replaced += _replace_linear_descendants(child, rank_fraction)
    return replaced


def replace_mlp_linears_with_disentangled(
    model: nn.Module,
    rank_fraction: float = DEFAULT_RANK_FRACTION,
) -> int:
    """Replace Linear layers below decoder MLP blocks with DisentangledLinear."""
    replaced = 0
    for module_name, module in list(model.named_modules()):
        if module_name == "mlp" or module_name.endswith(".mlp"):
            replaced += _replace_linear_descendants(module, rank_fraction)
    return replaced


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    replaced = replace_mlp_linears_with_disentangled(
        model, rank_fraction=DEFAULT_RANK_FRACTION
    )

    print(f"Replaced {replaced} MLP linear layers with DisentangledLinear.")

    first_disentangled = next(
        module for module in model.modules() if isinstance(module, DisentangledLinear)
    )
    start = time.perf_counter()
    first_disentangled.disentangle(log_prefix="first_mlp_linear", log_every=1, steps=3)
    elapsed = time.perf_counter() - start
    print(f"Disentangled first MLP linear layer in {elapsed:.2f}s.")
