"""Save/load a disentangled "Novice" model as Q/U/s/Vh factors.

A disentangled ``DisentangledLinear`` stores its factors (Q, U, s, Vh) as
parameters but keeps ``shape_4d`` as a plain Python attribute, and swaps its
``weight`` parameter for ``None`` — so a vanilla ``state_dict`` round-trip is not
enough to reconstruct it. These helpers persist exactly what is needed and
rebuild the module structure on load.

Only the disentangled factors are saved; the untouched (non-MLP) weights are
restored by re-loading the base model from its original checkpoint, so the file
stays as small as the factors require.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

try:
    from .disentangled_linear import DisentangledLinear
    from .prepare_qwen import replace_mlp_linears_with_disentangled
except ImportError:
    from disentangled_linear import DisentangledLinear
    from prepare_qwen import replace_mlp_linears_with_disentangled


def save_disentangled(model: nn.Module, path: str, *, model_name: str, rank_fraction: float) -> int:
    """Save every disentangled DisentangledLinear's factors to ``path``.

    Returns the number of disentangled modules written.
    """
    modules: dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, DisentangledLinear) and module.is_disentangled:
            modules[name] = {
                "shape_4d": tuple(int(d) for d in module.shape_4d),
                "in_features": module.in_features,
                "out_features": module.out_features,
                "Q": module.Q.detach().cpu(),
                "U": module.U.detach().cpu(),
                "s": module.s.detach().cpu(),
                "Vh": module.Vh.detach().cpu(),
                "bias": None if module.bias is None else module.bias.detach().cpu(),
            }
    checkpoint = {
        "format": "novice_disentangled_v1",
        "model_name": model_name,
        "rank_fraction": rank_fraction,
        "modules": modules,
    }
    torch.save(checkpoint, path)
    return len(modules)


def materialize_dense_model(
    model: nn.Module,
    rank_fraction: float | None = None,
    progress_callback: Callable[[], None] | None = None,
) -> int:
    """Replace every disentangled DisentangledLinear with a plain ``nn.Linear``
    holding its truncated weight.

    Disentangling is for live experimentation, but its ``forward`` recomputes the
    full ``Qᵀ·A⁻¹(U s Vh)`` reconstruction on *every* call — far too slow for
    token-by-token generation. Baking the chosen rank into a static weight makes
    inference a normal matmul. Returns the number of layers baked.
    """
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, DisentangledLinear) and module.is_disentangled:
            if rank_fraction is not None:
                module.rank_fraction = rank_fraction
            weight = module.materialize_weight(truncate=True).detach()
            dense = nn.Linear(
                module.in_features, module.out_features,
                bias=module.bias is not None, device=weight.device, dtype=weight.dtype,
            )
            with torch.no_grad():
                dense.weight.copy_(weight)
                if module.bias is not None:
                    dense.bias.copy_(module.bias.detach())
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, attr, dense))
            if progress_callback is not None:
                progress_callback()
    for parent, attr, dense in replacements:
        setattr(parent, attr, dense)
    return len(replacements)


def load_disentangled(
    path: str,
    *,
    device: torch.device | str = "cpu",
    trust_remote_code: bool = True,
) -> nn.Module:
    """Rebuild a disentangled Novice model from a ``save_disentangled`` file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "novice_disentangled_v1":
        raise ValueError(f"Unexpected checkpoint format: {checkpoint.get('format')!r}")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint["model_name"], trust_remote_code=trust_remote_code
    )
    replace_mlp_linears_with_disentangled(model, rank_fraction=checkpoint["rank_fraction"])

    modules_by_name = dict(model.named_modules())
    for name, data in checkpoint["modules"].items():
        module = modules_by_name[name]
        if not isinstance(module, DisentangledLinear):
            raise RuntimeError(f"Expected DisentangledLinear at {name!r}, got {type(module)}")
        module.shape_4d = tuple(data["shape_4d"])
        module.rank_fraction = checkpoint["rank_fraction"]
        # Switch from the dense-weight slot to the factored representation.
        module.register_parameter("weight", None)
        module.Q = nn.Parameter(data["Q"], requires_grad=False)
        module.U = nn.Parameter(data["U"], requires_grad=False)
        module.s = nn.Parameter(data["s"], requires_grad=False)
        module.Vh = nn.Parameter(data["Vh"], requires_grad=False)
        if data["bias"] is not None:
            module.bias = nn.Parameter(data["bias"], requires_grad=False)

    return model.to(device)
