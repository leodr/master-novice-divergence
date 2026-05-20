import math

import torch

import tensor_disentanglement_code.classes.disentanglement as CD

from tensor_disentanglement_code.utils.helpers import choose_tensor_shape, matrix_to_tensor4d

def permuted_matricization(tensor_4d: torch.Tensor) -> torch.Tensor:
    if tensor_4d.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(tensor_4d.shape)}")
    a, b, c, d = tensor_4d.shape
    return tensor_4d.permute(0, 3, 1, 2).reshape(a * d, b * c)


def inverse_permuted_matricization(
    matrix: torch.Tensor, shape_4d: CD.TensorShape4D
) -> torch.Tensor:
    expected_shape = (
        shape_4d.left_outer * shape_4d.right_inner,
        shape_4d.left_inner * shape_4d.right_outer,
    )
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {tuple(matrix.shape)}")
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(
            f"Matrix shape {tuple(matrix.shape)} does not match expected permuted shape {expected_shape}"
        )
    return matrix.reshape(
        shape_4d.left_outer,
        shape_4d.right_inner,
        shape_4d.left_inner,
        shape_4d.right_outer,
    ).permute(0, 2, 3, 1)

def spectrum_stats_from_permuted_matrix(matrix: torch.Tensor) -> CD.SpectrumStats:
    singular_values = torch.linalg.svdvals(matrix)
    energies = singular_values.square()
    probabilities = energies / energies.sum().clamp_min(1e-12)
    entropy = float(
        (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
    )
    return CD.SpectrumStats(
        entropy=entropy,
        effective_rank=float(math.exp(entropy)),
        singular_values=[
            float(value) for value in singular_values.detach().cpu().tolist()
        ],
    )


