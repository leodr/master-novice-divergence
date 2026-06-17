import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensor_disentanglement_code.disentanglement_base as DB


class DisentangledLinear(nn.Module):
    """Linear layer that can live-truncate a disentangled weight in forward."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank_fraction: float = 0.1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rank_fraction = rank_fraction
        self.shape_4d: tuple[int, int, int, int] | None = None
        self.history = None

        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.register_parameter("Q", None)
        self.register_parameter("U", None)
        self.register_parameter("s", None)
        self.register_parameter("Vh", None)

        self.reset_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank_fraction: float = 0.1,
    ) -> "DisentangledLinear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank_fraction=rank_fraction,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    @property
    def is_disentangled(self) -> bool:
        return self.Q is not None

    @property
    def max_rank(self) -> int:
        if not self.is_disentangled:
            return min(self.out_features, self.in_features)
        return int(self.s.numel())

    @property
    def keep_rank(self) -> int:
        if self.shape_4d is None:
            full_rank = self.max_rank
            return max(1, math.ceil(max(0.0, min(1.0, self.rank_fraction)) * full_rank))
        return DB._default_keep_rank_from_shape(self.shape_4d, self.rank_fraction)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def disentangle(
        self,
        *,
        steps: int = 100,
        learning_rate: float = 0.001,
        seed: int = 0,
        device: str | torch.device | None = None,
        log_prefix: str | None = None,
        log_every: int = 0,
    ) -> "DisentangledLinear":
        """Disentangle the current dense weight without truncating stored factors."""
        if self.is_disentangled:
            dense_weight = self.materialize_weight(truncate=False).detach()
        else:
            dense_weight = self.weight.detach()

        parameter_device = dense_weight.device
        parameter_dtype = dense_weight.dtype
        optimization_device = torch.device(device) if device is not None else torch.device("cpu")
        optimization_weight = dense_weight.to(device=optimization_device)

        result = DB.disentangle_matrix_via_reshape(
            optimization_weight,
            steps=steps,
            learning_rate=learning_rate,
            seed=seed,
            device=str(optimization_device),
            optimizer_type="adam_lie",
            rank_selection="fixed_heuristic",
            rank_fraction=self.rank_fraction,
            log_prefix=log_prefix,
            log_every=log_every,
        )

        shape_4d = result.tensor_shape_4d
        shape_tuple = (
            shape_4d.left_outer,
            shape_4d.left_inner,
            shape_4d.right_outer,
            shape_4d.right_inner,
        )

        q = result.inner.best_q.to(device=optimization_device)
        q_weight = q @ optimization_weight.to(dtype=q.dtype)
        permuted = DB._apply_A(q_weight, shape_tuple)
        u, s, vh = torch.linalg.svd(permuted, full_matrices=False)

        self.shape_4d = shape_tuple
        self.history = result.inner.history
        self.register_parameter("weight", None)
        self.Q = nn.Parameter(q.to(device=parameter_device, dtype=parameter_dtype))
        self.U = nn.Parameter(u.to(device=parameter_device, dtype=parameter_dtype))
        self.s = nn.Parameter(s.to(device=parameter_device, dtype=parameter_dtype))
        self.Vh = nn.Parameter(vh.to(device=parameter_device, dtype=parameter_dtype))
        return self

    def materialize_weight(self, *, truncate: bool = True) -> torch.Tensor:
        if not self.is_disentangled:
            return self.weight
        if self.shape_4d is None:
            raise RuntimeError("Disentangled factors are missing their 4D shape.")

        keep_rank = self.keep_rank if truncate else self.max_rank
        permuted = (self.U[:, :keep_rank] * self.s[:keep_rank]) @ self.Vh[:keep_rank]
        weight = DB._apply_A_inv(permuted, self.shape_4d)
        return self.Q.t() @ weight

    def export_factors(self) -> dict:
        """Serialize this layer's disentangled factors to plain CPU tensors.

        A vanilla ``state_dict`` can't round-trip a disentangled layer (``weight``
        is ``None`` and ``shape_4d`` is a plain attribute), so callers persist this
        dict instead and restore it with :meth:`load_factors`.
        """
        if not self.is_disentangled:
            raise RuntimeError("Layer is not disentangled; nothing to export.")
        return {
            "shape_4d": tuple(int(d) for d in self.shape_4d),
            "in_features": self.in_features,
            "out_features": self.out_features,
            "Q": self.Q.detach().cpu(),
            "U": self.U.detach().cpu(),
            "s": self.s.detach().cpu(),
            "Vh": self.Vh.detach().cpu(),
            "bias": None if self.bias is None else self.bias.detach().cpu(),
        }

    def load_factors(self, data: dict) -> "DisentangledLinear":
        """Populate the disentangled factors from an :meth:`export_factors` dict,
        switching this layer out of its dense-weight representation."""
        self.shape_4d = tuple(data["shape_4d"])
        self.register_parameter("weight", None)
        self.Q = nn.Parameter(data["Q"], requires_grad=False)
        self.U = nn.Parameter(data["U"], requires_grad=False)
        self.s = nn.Parameter(data["s"], requires_grad=False)
        self.Vh = nn.Parameter(data["Vh"], requires_grad=False)
        if data["bias"] is not None:
            self.bias = nn.Parameter(data["bias"], requires_grad=False)
        return self

    def to_dense_linear(self, *, truncate: bool = True) -> nn.Linear:
        """Bake the (truncated) reconstruction into a plain ``nn.Linear`` for fast
        inference — a normal matmul instead of recomputing the factors each call."""
        weight = self.materialize_weight(truncate=truncate).detach()
        dense = nn.Linear(
            self.in_features, self.out_features,
            bias=self.bias is not None, device=weight.device, dtype=weight.dtype,
        )
        with torch.no_grad():
            dense.weight.copy_(weight)
            if self.bias is not None:
                dense.bias.copy_(self.bias.detach())
        return dense

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.materialize_weight(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, rank_fraction={self.rank_fraction}, "
            f"is_disentangled={self.is_disentangled}"
        )

    def effective_rank_history(self) -> list[tuple[int, float]]:
        """``(step, effective_rank)`` trace of ``A(Q·W)`` from the disentangle run.

        Returns the raw data so callers (notebooks, plotting helpers) can render it
        however they like; this keeps visualization dependencies out of the layer.
        """
        if not self.is_disentangled:
            raise RuntimeError("Layer is not disentangled; no history to report.")
        return [(h["step"], h["effective_rank"]) for h in self.history]