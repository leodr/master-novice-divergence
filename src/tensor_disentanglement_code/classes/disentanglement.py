from dataclasses import dataclass, asdict
from typing import Any

import torch

@dataclass(frozen=True)
class TensorShape4D:
    left_outer: int
    left_inner: int
    right_outer: int
    right_inner: int

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return (self.left_outer * self.left_inner, self.right_outer * self.right_inner)
    
@dataclass(frozen=True)
class SpectrumStats:
    entropy: float
    effective_rank: float
    singular_values: list[float]

@dataclass(frozen=True)
class Tensor4DDisentanglementResult:
    tensor_shape_4d: TensorShape4D
    initial_stats: SpectrumStats
    best_stats: SpectrumStats
    final_stats: SpectrumStats
    best_step: int
    history: list[dict[str, float | int]]
    best_q: torch.Tensor
    best_tensor_4d: torch.Tensor
    best_matrix_2d: torch.Tensor

    def jsonable(self) -> dict[str, Any]:
        return {
            "tensor_shape_4d": asdict(self.tensor_shape_4d),
            "initial_stats": asdict(self.initial_stats),
            "best_stats": asdict(self.best_stats),
            "final_stats": asdict(self.final_stats),
            "best_step": self.best_step,
            "history": self.history,
        }

@dataclass(frozen=True)
class Matrix2DDisentanglementResult:
    matrix_shape_2d: tuple[int, int]
    tensor_shape_4d: TensorShape4D
    inner: Tensor4DDisentanglementResult

    def jsonable(self) -> dict[str, Any]:
        return {
            "matrix_shape_2d": self.matrix_shape_2d,
            "tensor_shape_4d": asdict(self.tensor_shape_4d),
            "inner": self.inner.jsonable(),
        }
    
@dataclass(frozen=True)
class PermutationSearchResult:
    permutation: list[int]
    initial_stats: SpectrumStats
    best_stats: SpectrumStats
    history: list[dict[str, float | int]]
    tensor_shape_4d: TensorShape4D

    def jsonable(self) -> dict[str, Any]:
        return {
            "permutation_length": len(self.permutation),
            "tensor_shape_4d": asdict(self.tensor_shape_4d),
            "initial_stats": asdict(self.initial_stats),
            "best_stats": asdict(self.best_stats),
            "history": self.history,
        }
