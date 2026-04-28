from pathlib import Path
import hashlib
from typing import Any, Literal
import re
import math
from datetime import datetime, UTC
import json

import torch
import gymnasium as gym


import classes.rl_models as CR
import classes.disentanglement as CD

def utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def stable_hash(payload: dict[str, Any]) -> str:
    
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checkpoint_output_name(checkpoint_path: str | Path) -> str:
    checkpoint_path = Path(checkpoint_path).resolve()
    stem = checkpoint_path.parent.name
    digest = hashlib.sha256(str(checkpoint_path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"

def factor_pair(value: int) -> tuple[int, int]:
    if value <= 0:
        raise ValueError(f"Value must be positive, got {value}")
    root = int(math.isqrt(value))
    for left in range(root, 0, -1):
        if value % left == 0:
            return left, value // left
    return 1, value

def choose_tensor_shape(matrix_2d: torch.Tensor) -> CD.TensorShape4D:
    if matrix_2d.ndim != 2:
        raise ValueError(f"Expected 2d matrix, got shape {tuple(matrix_2d.shape)}")
    rows, cols = matrix_2d.shape
    left_outer, left_inner = factor_pair(int(rows))
    right_outer, right_inner = factor_pair(int(cols))
    return CD.TensorShape4D(left_outer, left_inner, right_outer, right_inner)


def matrix_to_tensor4d(
    matrix_2d: torch.Tensor, shape_4d: CD.TensorShape4D | None = None
) -> torch.Tensor:
    if matrix_2d.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {tuple(matrix_2d.shape)}")
    shape_4d = shape_4d or choose_tensor_shape(matrix_2d)
    if tuple(matrix_2d.shape) != shape_4d.matrix_shape:
        raise ValueError(
            f"Matrix shape {tuple(matrix_2d.shape)} does not match requested 4D shape {shape_4d.matrix_shape}"
        )
    return matrix_2d.reshape(
        shape_4d.left_outer,
        shape_4d.left_inner,
        shape_4d.right_outer,
        shape_4d.right_inner,
    )


def restore_observation_normalization(
    envs: gym.vector.VectorEnv, normalization_state: dict[str, Any]
) -> gym.vector.VectorEnv:
    if "obs_rms" not in normalization_state:
        return envs
    envs = gym.wrappers.vector.NormalizeObservation(envs)
    obs_rms = normalization_state["obs_rms"]
    envs.obs_rms.mean[...] = obs_rms["mean"]
    envs.obs_rms.var[...] = obs_rms["var"]
    envs.obs_rms.count = obs_rms["count"]
    return envs


def max_model_difference(
    checkpoint: dict[str, Any],
    modified_state_dict: dict[str, torch.Tensor],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    base_model = build_model_from_checkpoint(checkpoint, device)
    modified_checkpoint = dict(checkpoint)
    modified_checkpoint["model_state_dict"] = modified_state_dict
    modified_model = build_model_from_checkpoint(modified_checkpoint, device)

    torch.manual_seed(seed)
    observations = torch.randn(
        32, checkpoint["config"]["env_info"]["observation_dim"], device=device
    )
    with torch.no_grad():
        base_logits = base_model.get_policy_logits(observations)
        mod_logits = modified_model.get_policy_logits(observations)
        base_values = base_model.get_value(observations)
        mod_values = modified_model.get_value(observations)
    return {
        "max_abs_logit_diff": float(
            torch.max(torch.abs(base_logits - mod_logits)).item()
        ),
        "max_abs_value_diff": float(
            torch.max(torch.abs(base_values - mod_values)).item()
        ),
    }

def mark_layer(
    layer_report: dict[str, Any],
    layer_name: str,
    status: str,
    operation: dict[str, Any],
) -> None:
    layer_report.setdefault(layer_name, {"status": "skipped", "operations": []})
    layer_report[layer_name]["status"] = status
    layer_report[layer_name]["operations"].append(operation)


def mlp_linear_prefixes(state_dict: dict[str, torch.Tensor]) -> list[str]:
    prefixes: list[tuple[int, str]] = []
    for key in state_dict:
        match = re.fullmatch(r"trunk\.(\d+)\.weight", key)
        if match:
            prefixes.append((int(match.group(1)), f"trunk.{match.group(1)}"))
    return [prefix for _, prefix in sorted(prefixes)]


def _row_permute(weight: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return weight.index_select(0, permutation)


def _col_permute(weight: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return weight.index_select(1, permutation)

def default_attention_heads(width: int) -> int:
    for heads in (8, 4, 2, 1):
        if width % heads == 0:
            return heads
    return 1


def attention_block_indices(state_dict: dict[str, torch.Tensor]) -> list[int]:
    indices = set()
    for key in state_dict:
        match = re.fullmatch(r"blocks\.(\d+)\..+", key)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)

def restore_normalization_state(envs: gym.vector.VectorEnv, state: dict[str, Any]) -> None:
    current = envs
    while hasattr(current, "env"):
        if "obs_rms" in state and hasattr(current, "obs_rms"):
            current.obs_rms.mean[...] = state["obs_rms"]["mean"]
            current.obs_rms.var[...] = state["obs_rms"]["var"]
            current.obs_rms.count = state["obs_rms"]["count"]
        if "return_rms" in state and hasattr(current, "return_rms"):
            current.return_rms.mean[...] = state["return_rms"]["mean"]
            current.return_rms.var[...] = state["return_rms"]["var"]
            current.return_rms.count = state["return_rms"]["count"]
        current = current.env