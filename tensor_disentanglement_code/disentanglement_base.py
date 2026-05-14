from pathlib import Path
from typing import Any, Callable, Literal
from dataclasses import asdict
import json
import math

import torch
import numpy as np

import classes.disentanglement as CD

from utils.permutation import (
    spectrum_stats_from_permuted_matrix,
)
from utils.helpers import (
    mark_layer,
    choose_tensor_shape,
    matrix_to_tensor4d,
    checkpoint_output_name,
    max_model_difference,
)

#TODO: check if its ok without these

#from utils.evals import evaluate_state_dict
#from visualize.disentangle import visualize_permutation_trajectories


OptimizerType = Literal["adam_lie", "rcg", "rtrn", "alt"]
RankSelectionType = Literal["fixed_heuristic", "binary_search", "rank_free"]


# ---------------------------------------------------------------------------
# Reshaping operators (paper's M, A, A^{-1})
# ---------------------------------------------------------------------------


def tensor4d_to_matrix(tensor_4d: torch.Tensor) -> torch.Tensor:
    """Paper's M: (l, r, b, c) -> (l*r, b*c)."""
    if tensor_4d.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(tensor_4d.shape)}")
    a, b, c, d = tensor_4d.shape
    return tensor_4d.reshape(a * b, c * d)


def _apply_A(matrix: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
    """A = M . P . M^{-1}: (lr, bc) -> (lc, rb), permutation [0, 3, 1, 2]."""
    l, r, b, c = shape
    return matrix.reshape(l, r, b, c).permute(0, 3, 1, 2).reshape(l * c, r * b)


def _apply_A_inv(matrix: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
    """A^{-1}: (lc, rb) -> (lr, bc)."""
    l, r, b, c = shape
    return matrix.reshape(l, c, r, b).permute(0, 2, 3, 1).reshape(l * r, b * c)


def orthogonal_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """Q = exp(B(s)) parameterization for the Adam-Lie baseline."""
    skew = raw - raw.transpose(-1, -2)
    return torch.linalg.matrix_exp(skew)


# ---------------------------------------------------------------------------
# Cost, exact gradient, exact Hessian-vector product
# ---------------------------------------------------------------------------


def _truncation_cost_from_sigma(sigma: torch.Tensor, keep_rank: int) -> float:
    if keep_rank >= sigma.numel():
        return 0.0
    return float((sigma[keep_rank:] ** 2).sum().item())


def _evaluate_cost(
    q: torch.Tensor,
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    keep_rank: int,
) -> float:
    permuted = _apply_A(q @ base_matrix, shape)
    sigma = torch.linalg.svdvals(permuted)
    return _truncation_cost_from_sigma(sigma, keep_rank)


def _euclidean_gradient(
    q: torch.Tensor,
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    keep_rank: int,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper eq. 3.10: grad_Q f(Q) = A^{-1}(U phi'(Sigma) V^T) X^T.

    For phi = chi_mu (eq. 2.5), phi'(sigma_i) = 2*sigma_i for i > k else 0.
    Returns (grad, cost, U, sigma, Vh) so callers can reuse the SVD.
    """
    matrix = _apply_A(q @ base_matrix, shape)
    u, sigma, vh = torch.linalg.svd(matrix, full_matrices=False)
    cost = _truncation_cost_from_sigma(sigma, keep_rank)
    phi_prime = torch.zeros_like(sigma)
    if keep_rank < sigma.numel():
        phi_prime[keep_rank:] = 2.0 * sigma[keep_rank:]
    inner = (u * phi_prime) @ vh
    grad = _apply_A_inv(inner, shape) @ base_matrix.transpose(-1, -2)
    return grad, cost, u, sigma, vh


def _euclidean_hessian_vec(
    q: torch.Tensor,
    e: torch.Tensor,
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    keep_rank: int,
    sigma: torch.Tensor,
    u: torch.Tensor,
    vh: torch.Tensor,
) -> torch.Tensor:
    """D grad_Q f along E using paper's eqs. 4.21-4.23.

    F[i, j] = 1 / (sigma_j^2 - sigma_i^2) for i != j else 0 (eq. 4.23). The
    chi_mu non-smoothness is handled via the diagonal Dphi'(Sigma) mask.
    """
    A_EX = _apply_A(e @ base_matrix, shape)
    UAV = u.transpose(-1, -2) @ A_EX @ vh.transpose(-1, -2)  # A := U^T dM V

    # F[i, j] = 1 / (sigma_j^2 - sigma_i^2) for i != j else 0  (eq. 4.23).
    sigma_sq = sigma ** 2
    diff = sigma_sq.unsqueeze(0) - sigma_sq.unsqueeze(1)
    mask = diff.abs() < 1e-14
    safe = torch.where(mask, torch.ones_like(diff), diff)
    F = torch.where(mask, torch.zeros_like(diff), 1.0 / safe)

    # SVD perturbation: with A := U^T dM V, dSigma_ii = A_ii, and the skew-
    # symmetric coefficients Omega_U = U^T dU, Omega_V = V^T dV satisfy
    # A = Omega_U Sigma + dSigma - Sigma Omega_V. Solving the off-diagonal
    # 2x2 system gives Omega_U_{ij} = F_{ij} (sigma_j A_{ij} + sigma_i A_{ji})
    # and Omega_V_{ij} = F_{ij} (sigma_i A_{ij} + sigma_j A_{ji}).
    sigma_col = sigma.unsqueeze(1)
    sigma_row = sigma.unsqueeze(0)
    omega_U = F * (UAV * sigma_row + sigma_col * UAV.transpose(-1, -2))
    omega_V = F * (sigma_col * UAV + UAV.transpose(-1, -2) * sigma_row)

    safe_sigma = torch.where(sigma.abs() < 1e-14, torch.ones_like(sigma), sigma)
    inv_sigma = torch.where(sigma.abs() < 1e-14, torch.zeros_like(sigma), 1.0 / safe_sigma)

    A_EX_V = A_EX @ vh.transpose(-1, -2)
    proj_left = A_EX_V - u @ (u.transpose(-1, -2) @ A_EX_V)
    DU = u @ omega_U + proj_left * inv_sigma.unsqueeze(0)

    diag_UAV = torch.diagonal(UAV)
    Dphi_prime = torch.zeros_like(sigma)
    if keep_rank < sigma.numel():
        Dphi_prime[keep_rank:] = 2.0 * diag_UAV[keep_rank:]

    V = vh.transpose(-1, -2)
    A_EX_T_U = A_EX.transpose(-1, -2) @ u
    proj_right = A_EX_T_U - V @ (V.transpose(-1, -2) @ A_EX_T_U)
    DV = V @ omega_V + proj_right * inv_sigma.unsqueeze(0)

    phi_prime = torch.zeros_like(sigma)
    if keep_rank < sigma.numel():
        phi_prime[keep_rank:] = 2.0 * sigma[keep_rank:]

    term1 = (DU * phi_prime.unsqueeze(0)) @ vh
    term2 = (u * Dphi_prime.unsqueeze(0)) @ vh
    term3 = (u * phi_prime.unsqueeze(0)) @ DV.transpose(-1, -2)
    return _apply_A_inv(term1 + term2 + term3, shape) @ base_matrix.transpose(-1, -2)


# ---------------------------------------------------------------------------
# Manifold operations
# ---------------------------------------------------------------------------


def _project_to_orthogonal_tangent(q: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Paper eq. 4.16: Proj_Q(Z) = (Z - Q Z^T Q) / 2 == Z - Q sym(Q^T Z)."""
    sym = (q.transpose(-1, -2) @ grad + grad.transpose(-1, -2) @ q) * 0.5
    return grad - q @ sym


def _qr_retraction(q: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    candidate = q + direction
    q_next, r = torch.linalg.qr(candidate)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q_next * signs.unsqueeze(0)


def _transport_tangent(q_next: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return _project_to_orthogonal_tangent(q_next, direction)


def _riemannian_gradient(
    q: torch.Tensor,
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    keep_rank: int,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    eu, cost, u, sigma, vh = _euclidean_gradient(q, base_matrix, shape, keep_rank)
    return _project_to_orthogonal_tangent(q, eu), cost, u, sigma, vh


def _riemannian_hessian_vec(
    q: torch.Tensor,
    e: torch.Tensor,
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    keep_rank: int,
    eu_grad_at_q: torch.Tensor,
    sigma: torch.Tensor,
    u: torch.Tensor,
    vh: torch.Tensor,
) -> torch.Tensor:
    """Hess f(Q)[E] = Proj_Q(D grad f(Q)[E]), eqs. 4.19-4.24."""
    D_eu = _euclidean_hessian_vec(q, e, base_matrix, shape, keep_rank, sigma, u, vh)
    middle = (
        e @ eu_grad_at_q.transpose(-1, -2) @ q
        + q @ D_eu.transpose(-1, -2) @ q
        + q @ eu_grad_at_q.transpose(-1, -2) @ e
    )
    D_grad = (D_eu - middle) / 2.0
    return _project_to_orthogonal_tangent(q, D_grad)


# ---------------------------------------------------------------------------
# Backtracking line search with adaptive trial step
# ---------------------------------------------------------------------------


def _adaptive_line_search(
    q: torch.Tensor,
    direction: torch.Tensor,
    grad: torch.Tensor,
    current_cost: float,
    cost_fn: Callable[[torch.Tensor], float],
    *,
    last_step: float,
    max_step: float = 10.0,
    expand_factor: float = 2.0,
    armijo: float = 1e-4,
    min_step: float = 1e-12,
) -> tuple[torch.Tensor | None, float, float]:
    """Armijo backtracking that starts at expand_factor * last accepted step."""
    directional_derivative = float(torch.sum(grad * direction).item())
    if directional_derivative >= 0.0:
        return None, current_cost, 0.0
    step = min(max_step, max(last_step * expand_factor, min_step))
    while step >= min_step:
        q_candidate = _qr_retraction(q, step * direction)
        cand_cost = cost_fn(q_candidate)
        if cand_cost <= current_cost + armijo * step * directional_derivative:
            return q_candidate, cand_cost, step
        step *= 0.5
    return None, current_cost, 0.0


# ---------------------------------------------------------------------------
# Algorithm 3: Riemannian conjugate gradient (PRP+, projection transport)
# ---------------------------------------------------------------------------


def _rcg(
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    keep_rank: int,
    *,
    steps: int,
    initial_step: float,
    max_step: float = 10.0,
    expand_factor: float = 2.0,
    history_cb: Callable[[int, torch.Tensor, float], None] | None = None,
) -> tuple[torch.Tensor, float]:
    n = base_matrix.shape[0]
    q = torch.eye(n, dtype=base_matrix.dtype, device=base_matrix.device)
    cost_fn = lambda qq: _evaluate_cost(qq, base_matrix, shape, keep_rank)
    grad, cost, *_ = _riemannian_gradient(q, base_matrix, shape, keep_rank)
    direction = -grad
    last_step = initial_step
    if history_cb is not None:
        history_cb(0, q, cost)
    for j in range(1, steps + 1):
        q_new, new_cost, taken = _adaptive_line_search(
            q, direction, grad, cost, cost_fn,
            last_step=last_step, max_step=max_step, expand_factor=expand_factor,
        )
        if q_new is None:
            direction = -grad
            q_new, new_cost, taken = _adaptive_line_search(
                q, direction, grad, cost, cost_fn,
                last_step=last_step, max_step=max_step, expand_factor=expand_factor,
            )
            if q_new is None:
                if history_cb is not None:
                    for k in range(j, steps + 1):
                        history_cb(k, q, cost)
                return q, cost
        q = q_new
        cost = new_cost
        last_step = taken
        grad_next, _, *_ = _riemannian_gradient(q, base_matrix, shape, keep_rank)
        transported_grad = _transport_tangent(q, grad)
        denom = float(torch.sum(grad * grad).item())
        beta = 0.0
        if denom > 0.0:
            num = float(torch.sum(grad_next * (grad_next - transported_grad)).item())
            beta = max(0.0, num / denom)
        transported_dir = _transport_tangent(q, direction)
        direction = -grad_next + beta * transported_dir
        grad = grad_next
        if history_cb is not None:
            history_cb(j, q, cost)
    return q, cost


# ---------------------------------------------------------------------------
# Algorithm 4: Riemannian trust-region Newton with Steihaug-CG
# ---------------------------------------------------------------------------


def _to_boundary(z: torch.Tensor, d: torch.Tensor, radius: float) -> torch.Tensor:
    a = float(torch.sum(d * d).item())
    b = 2.0 * float(torch.sum(z * d).item())
    c = float(torch.sum(z * z).item()) - radius * radius
    disc = max(b * b - 4.0 * a * c, 0.0)
    tau = (-b + math.sqrt(disc)) / (2.0 * a + 1e-30)
    return z + tau * d


def _steihaug_cg(
    hess_vec: Callable[[torch.Tensor], torch.Tensor],
    grad: torch.Tensor,
    radius: float,
    *,
    max_iter: int,
    tol: float,
) -> torch.Tensor:
    z = torch.zeros_like(grad)
    r = grad.clone()
    d = -r.clone()
    r_norm0 = float(r.norm().item())
    if r_norm0 == 0.0:
        return z
    for _ in range(max_iter):
        Hd = hess_vec(d)
        dHd = float(torch.sum(d * Hd).item())
        if dHd <= 0:
            return _to_boundary(z, d, radius)
        r_dot_r = float(torch.sum(r * r).item())
        alpha = r_dot_r / dHd
        z_next = z + alpha * d
        if float(z_next.norm().item()) >= radius:
            return _to_boundary(z, d, radius)
        z = z_next
        r = r + alpha * Hd
        r_norm = float(r.norm().item())
        if r_norm <= tol * r_norm0:
            return z
        beta = float(torch.sum(r * r).item()) / r_dot_r
        d = -r + beta * d
    return z


def _rtrn(
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    keep_rank: int,
    *,
    steps: int,
    initial_radius: float = 1.0,
    max_radius: float = 10.0,
    rmin: float = 0.1,
    regularization: float = 1e-12,
    cg_max_iter: int = 50,
    cg_tol: float = 1e-6,
    history_cb: Callable[[int, torch.Tensor, float], None] | None = None,
) -> tuple[torch.Tensor, float]:
    """Algorithm 4 with regularized Hessian (eq. 4.29)."""
    n = base_matrix.shape[0]
    q = torch.eye(n, dtype=base_matrix.dtype, device=base_matrix.device)
    radius = initial_radius
    grad, cost, u, sigma, vh = _riemannian_gradient(q, base_matrix, shape, keep_rank)
    eu_grad_at_q, *_ = _euclidean_gradient(q, base_matrix, shape, keep_rank)
    if history_cb is not None:
        history_cb(0, q, cost)
    for j in range(1, steps + 1):
        def hess_vec(e: torch.Tensor) -> torch.Tensor:
            e_proj = _project_to_orthogonal_tangent(q, e)
            return _riemannian_hessian_vec(
                q, e_proj, base_matrix, shape, keep_rank, eu_grad_at_q, sigma, u, vh
            ) + regularization * e_proj

        eta = _steihaug_cg(hess_vec, grad, radius, max_iter=cg_max_iter, tol=cg_tol)
        q_trial = _qr_retraction(q, eta)
        trial_cost = _evaluate_cost(q_trial, base_matrix, shape, keep_rank)
        m_red = -float(torch.sum(grad * eta).item()) - 0.5 * float(torch.sum(eta * hess_vec(eta)).item())
        actual_red = cost - trial_cost
        rho = actual_red / m_red if abs(m_red) > 1e-30 else 0.0
        if rho > 0.75 and float(eta.norm().item()) >= 0.95 * radius:
            radius = min(2.0 * radius, max_radius)
        elif rho < 0.25:
            radius = 0.25 * radius
        if rho > rmin:
            q = q_trial
            cost = trial_cost
            grad, cost, u, sigma, vh = _riemannian_gradient(q, base_matrix, shape, keep_rank)
            eu_grad_at_q, *_ = _euclidean_gradient(q, base_matrix, shape, keep_rank)
        if history_cb is not None:
            history_cb(j, q, cost)
        if float(grad.norm().item()) < 1e-10:
            if history_cb is not None:
                for k in range(j + 1, steps + 1):
                    history_cb(k, q, cost)
            break
    return q, cost


# ---------------------------------------------------------------------------
# Algorithm 5: Alternating disentangler
# ---------------------------------------------------------------------------


def _alternating(
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    keep_rank: int,
    *,
    steps: int,
    tol: float = 1e-10,
    history_cb: Callable[[int, torch.Tensor, float], None] | None = None,
) -> tuple[torch.Tensor, float]:
    """Alg. 5: alternate (a) rank-k truncated SVD of A(QX), (b) Procrustes for Q."""
    n = base_matrix.shape[0]
    q = torch.eye(n, dtype=base_matrix.dtype, device=base_matrix.device)
    if history_cb is not None:
        history_cb(0, q, _evaluate_cost(q, base_matrix, shape, keep_rank))
    for j in range(1, steps + 1):
        permuted = _apply_A(q @ base_matrix, shape)
        u, sigma, vh = torch.linalg.svd(permuted, full_matrices=False)
        k = min(keep_rank, sigma.numel())
        sigma_trunc = sigma.clone()
        sigma_trunc[k:] = 0
        m_k = (u * sigma_trunc) @ vh
        a_inv_mk = _apply_A_inv(m_k, shape)
        target = a_inv_mk @ base_matrix.transpose(-1, -2)
        u_p, _, vh_p = torch.linalg.svd(target, full_matrices=False)
        q_new = u_p @ vh_p
        delta = float((q_new - q).norm().item())
        q = q_new
        cost = _evaluate_cost(q, base_matrix, shape, keep_rank)
        if history_cb is not None:
            history_cb(j, q, cost)
        if delta < tol:
            if history_cb is not None:
                for k_pad in range(j + 1, steps + 1):
                    history_cb(k_pad, q, cost)
            break
    return q, _evaluate_cost(q, base_matrix, shape, keep_rank)


# ---------------------------------------------------------------------------
# Adam-Lie baseline
# ---------------------------------------------------------------------------


def _adam_lie(
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    keep_rank: int,
    *,
    steps: int,
    learning_rate: float,
    history_cb: Callable[[int, torch.Tensor, float], None] | None = None,
) -> tuple[torch.Tensor, float]:
    """Q = exp(B(s)) parameterization, Adam updates on s.

    Uses the paper's closed-form Euclidean gradient w.r.t. Q to avoid
    autograd-through-SVD numerical issues, then composes with dQ/d(raw)
    via a surrogate loss.
    """
    n = base_matrix.shape[0]
    raw = torch.zeros(
        (n, n), dtype=base_matrix.dtype, device=base_matrix.device, requires_grad=True
    )
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    if history_cb is not None:
        with torch.no_grad():
            q0 = orthogonal_from_raw(raw)
            history_cb(0, q0, _evaluate_cost(q0, base_matrix, shape, keep_rank))
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        q = orthogonal_from_raw(raw)
        with torch.no_grad():
            eu_grad, _, *_ = _euclidean_gradient(q.detach(), base_matrix, shape, keep_rank)
        # Surrogate s.t. d(surrogate)/d(raw) = (dQ/d(raw))^T eu_grad.
        surrogate = (eu_grad * q).sum()
        surrogate.backward()
        optimizer.step()
        if history_cb is not None:
            with torch.no_grad():
                q_eval = orthogonal_from_raw(raw)
                history_cb(step, q_eval, _evaluate_cost(q_eval, base_matrix, shape, keep_rank))
    with torch.no_grad():
        q_final = orthogonal_from_raw(raw)
        return q_final, _evaluate_cost(q_final, base_matrix, shape, keep_rank)


# ---------------------------------------------------------------------------
# Renyi-1/2 (nuclear norm) surrogate for binary-search initialization
# ---------------------------------------------------------------------------


def _renyi_half_gradient(
    q: torch.Tensor,
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
) -> tuple[torch.Tensor, float]:
    """Nuclear-norm objective f(Q) = sum sigma_i: grad_Q f = A^{-1}(U V^T) X^T."""
    matrix = _apply_A(q @ base_matrix, shape)
    u, sigma, vh = torch.linalg.svd(matrix, full_matrices=False)
    inner = u @ vh
    grad = _apply_A_inv(inner, shape) @ base_matrix.transpose(-1, -2)
    return _project_to_orthogonal_tangent(q, grad), float(sigma.sum().item())


def _rcg_renyi_half(
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    *,
    steps: int,
    initial_step: float,
    max_step: float = 10.0,
    expand_factor: float = 2.0,
) -> torch.Tensor:
    """RCG on the Renyi-1/2 surrogate (Algorithm 6, line 2)."""
    n = base_matrix.shape[0]
    q = torch.eye(n, dtype=base_matrix.dtype, device=base_matrix.device)
    cost_fn = lambda qq: float(torch.linalg.svdvals(_apply_A(qq @ base_matrix, shape)).sum().item())
    grad, cost = _renyi_half_gradient(q, base_matrix, shape)
    direction = -grad
    last_step = initial_step
    for _ in range(steps):
        q_new, new_cost, taken = _adaptive_line_search(
            q, direction, grad, cost, cost_fn,
            last_step=last_step, max_step=max_step, expand_factor=expand_factor,
        )
        if q_new is None:
            direction = -grad
            q_new, new_cost, taken = _adaptive_line_search(
                q, direction, grad, cost, cost_fn,
                last_step=last_step, max_step=max_step, expand_factor=expand_factor,
            )
            if q_new is None:
                return q
        q = q_new
        cost = new_cost
        last_step = taken
        grad_next, _ = _renyi_half_gradient(q, base_matrix, shape)
        transported_grad = _transport_tangent(q, grad)
        denom = float(torch.sum(grad * grad).item())
        beta = 0.0
        if denom > 0.0:
            num = float(torch.sum(grad_next * (grad_next - transported_grad)).item())
            beta = max(0.0, num / denom)
        transported_dir = _transport_tangent(q, direction)
        direction = -grad_next + beta * transported_dir
        grad = grad_next
    return q


def _adam_lie_renyi_half(
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    *,
    steps: int,
    learning_rate: float,
    history_cb: Callable[[int, torch.Tensor, float], None] | None = None,
) -> tuple[torch.Tensor, float]:
    """Adam-on-Lie-algebra against the nuclear-norm objective f(Q) = sum sigma_i.

    This is the rank-free analogue of `_adam_lie`. Uses the closed-form
    gradient grad_Q = A^{-1}(U V^T) X^T (since phi'(t) = 1).
    """
    n = base_matrix.shape[0]
    raw = torch.zeros(
        (n, n), dtype=base_matrix.dtype, device=base_matrix.device, requires_grad=True
    )
    optimizer = torch.optim.Adam([raw], lr=learning_rate)

    def cost_at(qq: torch.Tensor) -> float:
        return float(torch.linalg.svdvals(_apply_A(qq @ base_matrix, shape)).sum().item())

    if history_cb is not None:
        with torch.no_grad():
            q0 = orthogonal_from_raw(raw)
            history_cb(0, q0, cost_at(q0))
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        q = orthogonal_from_raw(raw)
        with torch.no_grad():
            matrix = _apply_A(q.detach() @ base_matrix, shape)
            u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
            inner = u @ vh
            eu_grad = _apply_A_inv(inner, shape) @ base_matrix.transpose(-1, -2)
        surrogate = (eu_grad * q).sum()
        surrogate.backward()
        optimizer.step()
        if history_cb is not None:
            with torch.no_grad():
                q_eval = orthogonal_from_raw(raw)
                history_cb(step, q_eval, cost_at(q_eval))
    with torch.no_grad():
        q_final = orthogonal_from_raw(raw)
        return q_final, cost_at(q_final)


def _rcg_renyi_half_with_history(
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    *,
    steps: int,
    initial_step: float,
    history_cb: Callable[[int, torch.Tensor, float], None] | None = None,
) -> tuple[torch.Tensor, float]:
    """RCG on Renyi-1/2 with the same history callback as the chi_mu optimizers."""
    n = base_matrix.shape[0]
    q = torch.eye(n, dtype=base_matrix.dtype, device=base_matrix.device)
    cost_fn = lambda qq: float(torch.linalg.svdvals(_apply_A(qq @ base_matrix, shape)).sum().item())
    grad, cost = _renyi_half_gradient(q, base_matrix, shape)
    direction = -grad
    last_step = initial_step
    if history_cb is not None:
        history_cb(0, q, cost)
    for j in range(1, steps + 1):
        q_new, new_cost, taken = _adaptive_line_search(
            q, direction, grad, cost, cost_fn, last_step=last_step,
        )
        if q_new is None:
            direction = -grad
            q_new, new_cost, taken = _adaptive_line_search(
                q, direction, grad, cost, cost_fn, last_step=last_step,
            )
            if q_new is None:
                if history_cb is not None:
                    for k in range(j, steps + 1):
                        history_cb(k, q, cost)
                return q, cost
        q = q_new
        cost = new_cost
        last_step = taken
        grad_next, _ = _renyi_half_gradient(q, base_matrix, shape)
        transported_grad = _transport_tangent(q, grad)
        denom = float(torch.sum(grad * grad).item())
        beta = 0.0
        if denom > 0.0:
            num = float(torch.sum(grad_next * (grad_next - transported_grad)).item())
            beta = max(0.0, num / denom)
        transported_dir = _transport_tangent(q, direction)
        direction = -grad_next + beta * transported_dir
        grad = grad_next
        if history_cb is not None:
            history_cb(j, q, cost)
    return q, cost


def _run_optimizer_rank_free(
    optimizer_type: OptimizerType,
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    *,
    steps: int,
    learning_rate: float,
    history_cb: Callable[[int, torch.Tensor, float], None] | None,
) -> tuple[torch.Tensor, float]:
    """Run a rank-agnostic disentangler using the Renyi-1/2 (nuclear norm)
    surrogate. The 'alt' optimizer requires k and is therefore not available
    in rank-free mode (it falls back to RCG with a warning)."""
    if optimizer_type == "adam_lie":
        return _adam_lie_renyi_half(
            base_matrix, shape,
            steps=steps, learning_rate=learning_rate, history_cb=history_cb,
        )
    if optimizer_type in ("rcg", "rtrn"):
        # RTRN's Hessian for the nuclear norm has unbounded curvature near
        # zero singular values; in practice falling back to RCG is robust.
        return _rcg_renyi_half_with_history(
            base_matrix, shape,
            steps=steps, initial_step=learning_rate, history_cb=history_cb,
        )
    if optimizer_type == "alt":
        # Alt is not defined without k; fall back to RCG on the surrogate.
        return _rcg_renyi_half_with_history(
            base_matrix, shape,
            steps=steps, initial_step=learning_rate, history_cb=history_cb,
        )
    raise ValueError(f"Unknown optimizer_type '{optimizer_type}'")


def _smallest_k_within_tol(
    q: torch.Tensor,
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    tolerance: float,
) -> int:
    permuted = _apply_A(q @ base_matrix, shape)
    sigma = torch.linalg.svdvals(permuted)
    sq = sigma ** 2
    total = float(sq.sum().item())
    cum = torch.cumsum(sq, dim=0)
    for k in range(sigma.numel() + 1):
        tail = total - (float(cum[k - 1].item()) if k > 0 else 0.0)
        if tail <= tolerance:
            return k
    return sigma.numel()


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def _default_keep_rank_from_shape(shape: tuple[int, int, int, int], rank_fraction: float) -> int:
    l, r, b, c = shape
    m = min(l * c, r * b)
    if m == 0:
        return 0
    return max(1, int(np.ceil(max(0.0, min(1.0, rank_fraction)) * m)))


def _run_optimizer(
    optimizer_type: OptimizerType,
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    keep_rank: int,
    *,
    steps: int,
    learning_rate: float,
    history_cb: Callable[[int, torch.Tensor, float], None] | None,
) -> tuple[torch.Tensor, float]:
    if optimizer_type == "adam_lie":
        return _adam_lie(
            base_matrix, shape, keep_rank,
            steps=steps, learning_rate=learning_rate, history_cb=history_cb,
        )
    if optimizer_type == "rcg":
        return _rcg(
            base_matrix, shape, keep_rank,
            steps=steps, initial_step=learning_rate, history_cb=history_cb,
        )
    if optimizer_type == "rtrn":
        return _rtrn(
            base_matrix, shape, keep_rank,
            steps=steps, history_cb=history_cb,
        )
    if optimizer_type == "alt":
        return _alternating(
            base_matrix, shape, keep_rank,
            steps=steps, history_cb=history_cb,
        )
    raise ValueError(
        f"Unknown optimizer_type '{optimizer_type}'. "
        "Use one of: 'adam_lie', 'rcg', 'rtrn', 'alt'."
    )


def disentangle_tensor4d(
    tensor_4d: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    seed: int,
    device: str = "cpu",
    log_prefix: str | None = None,
    log_every: int = 0,
    optimizer_type: OptimizerType = "rcg",
    rank_selection: RankSelectionType = "fixed_heuristic",
    rank_fraction: float = 0.1,
    truncation_tolerance: float = 1e-6,
    binary_search_max_iter: int = 12,
) -> CD.Tensor4DDisentanglementResult:
    if tensor_4d.ndim != 4:
        raise ValueError(
            f"Base disentanglement expects a 4D tensor, got shape {tuple(tensor_4d.shape)}"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)

    tensor_4d = tensor_4d.detach().clone().to(dtype=torch.float64, device=device)
    shape_4d = CD.TensorShape4D(*[int(dim) for dim in tensor_4d.shape])
    base_matrix = tensor4d_to_matrix(tensor_4d)
    shape_tuple = (
        shape_4d.left_outer, shape_4d.left_inner,
        shape_4d.right_outer, shape_4d.right_inner,
    )

    history: list[dict[str, float | int]] = []
    initial_stats = spectrum_stats_from_permuted_matrix(_apply_A(base_matrix, shape_tuple))
    best_cost: float = math.inf
    best_q: torch.Tensor | None = None
    best_step: int = 0
    best_stats = initial_stats

    def history_cb(step: int, q: torch.Tensor, cost: float) -> None:
        nonlocal best_cost, best_q, best_step, best_stats
        permuted = _apply_A(q @ base_matrix, shape_tuple)
        stats = spectrum_stats_from_permuted_matrix(permuted)
        history.append({
            "step": step,
            "cost": cost,
            "entropy": stats.entropy,
            "effective_rank": stats.effective_rank,
        })
        if cost < best_cost:
            best_cost = cost
            best_q = q.detach().clone()
            best_step = step
            best_stats = stats
        if log_every and (step == 0 or step % log_every == 0):
            prefix = "" if log_prefix is None else f"[{log_prefix}]"
            print(
                f"{prefix}step={step:04d} cost={cost:.6e} "
                f"entropy={stats.entropy:.6f} effective_rank={stats.effective_rank:.4f}"
            )

    if rank_selection == "fixed_heuristic":
        keep_rank = _default_keep_rank_from_shape(shape_tuple, rank_fraction)
        q_final, final_cost = _run_optimizer(
            optimizer_type, base_matrix, shape_tuple, keep_rank,
            steps=steps, learning_rate=learning_rate, history_cb=history_cb,
        )
        rank_search_log: dict[str, Any] = {
            "method": "fixed_heuristic",
            "keep_rank": keep_rank,
            "rank_fraction": rank_fraction,
        }
    elif rank_selection == "binary_search":
        rank_search_log = _binary_search_rank(
            base_matrix=base_matrix,
            shape=shape_tuple,
            optimizer_type=optimizer_type,
            steps=steps,
            learning_rate=learning_rate,
            tolerance=truncation_tolerance,
            max_iter=binary_search_max_iter,
            history_cb=history_cb,
        )
        keep_rank = rank_search_log["k_opt"]
        q_final = rank_search_log.pop("q_opt")
        final_cost = rank_search_log["c_k_at_opt"]
    elif rank_selection == "rank_free":
        # Minimize the nuclear norm (Renyi-1/2 surrogate). No k is committed
        # to here; the resulting Q concentrates spectral energy as much as
        # possible regardless of any specific truncation rank. Downstream
        # consumers can pick k post-hoc (e.g. by retained-energy fraction).
        q_final, final_cost = _run_optimizer_rank_free(
            optimizer_type, base_matrix, shape_tuple,
            steps=steps, learning_rate=learning_rate, history_cb=history_cb,
        )
        permuted_final = _apply_A(q_final @ base_matrix, shape_tuple)
        sigma_final = torch.linalg.svdvals(permuted_final)
        rank_search_log = {
            "method": "rank_free",
            "objective": "nuclear_norm_renyi_1_2",
            "final_nuclear_norm": float(sigma_final.sum().item()),
            "final_singular_values": [float(x) for x in sigma_final.detach().cpu().tolist()],
            "max_full_rank": int(sigma_final.numel()),
        }
        keep_rank = int(sigma_final.numel())  # no truncation chosen
    else:
        raise ValueError(
            f"Unknown rank_selection '{rank_selection}'. "
            "Use 'fixed_heuristic', 'binary_search', or 'rank_free'."
        )

    if best_q is None:
        best_q = q_final.detach().clone()
        best_step = steps
        permuted = _apply_A(best_q @ base_matrix, shape_tuple)
        best_stats = spectrum_stats_from_permuted_matrix(permuted)
        best_cost = final_cost

    permuted_best = _apply_A(best_q @ base_matrix, shape_tuple)
    best_tensor_4d = (best_q @ base_matrix).reshape(*tensor_4d.shape)
    best_matrix_2d = best_q @ base_matrix
    final_stats = spectrum_stats_from_permuted_matrix(permuted_best)

    result = CD.Tensor4DDisentanglementResult(
        tensor_shape_4d=shape_4d,
        initial_stats=initial_stats,
        best_stats=best_stats,
        final_stats=final_stats,
        best_step=best_step,
        history=history,
        best_q=best_q.detach().cpu(),
        best_tensor_4d=best_tensor_4d.detach().cpu(),
        best_matrix_2d=best_matrix_2d.detach().cpu(),
    )
    object.__setattr__(result, "rank_search", rank_search_log)
    object.__setattr__(result, "best_cost", best_cost)
    object.__setattr__(result, "keep_rank", keep_rank)
    return result


def _binary_search_rank(
    *,
    base_matrix: torch.Tensor,
    shape: tuple[int, int, int, int],
    optimizer_type: OptimizerType,
    steps: int,
    learning_rate: float,
    tolerance: float,
    max_iter: int,
    history_cb: Callable[[int, torch.Tensor, float], None] | None,
) -> dict[str, Any]:
    """Algorithm 6: binary search over k with a Renyi-1/2 surrogate for kr."""
    q_renyi = _rcg_renyi_half(base_matrix, shape, steps=steps, initial_step=learning_rate)
    kr = _smallest_k_within_tol(q_renyi, base_matrix, shape, tolerance)
    permuted = _apply_A(base_matrix, shape)
    m = min(permuted.shape)
    kl = 0
    k_opt = kr
    q_opt = q_renyi
    c_at_opt = _evaluate_cost(q_renyi, base_matrix, shape, k_opt)
    iterations: list[dict[str, Any]] = [{
        "phase": "renyi_seed",
        "kr": kr,
        "renyi_q_norm_diff_from_I": float(
            (q_renyi - torch.eye(q_renyi.shape[0], dtype=q_renyi.dtype)).norm().item()
        ),
    }]
    iter_count = 0
    while kl <= kr and iter_count < max_iter:
        iter_count += 1
        k = (kl + kr) // 2
        q_k, _ = _run_optimizer(
            optimizer_type, base_matrix, shape, k,
            steps=steps, learning_rate=learning_rate, history_cb=history_cb,
        )
        c_k = _evaluate_cost(q_k, base_matrix, shape, k)
        iterations.append({
            "phase": "binary_search",
            "iter": iter_count,
            "kl": kl, "kr": kr, "k": k,
            "c_k": c_k, "tolerance": tolerance,
            "accepted": c_k <= tolerance,
        })
        if c_k <= tolerance:
            k_opt = k
            q_opt = q_k
            c_at_opt = c_k
            kr = k - 1
        else:
            kl = k + 1
    return {
        "method": "binary_search",
        "tolerance": tolerance,
        "max_iter": max_iter,
        "k_opt": k_opt,
        "c_k_at_opt": c_at_opt,
        "max_full_rank": m,
        "iterations": iterations,
        "q_opt": q_opt,
    }


def disentangle_matrix_via_reshape(
    matrix_2d: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    seed: int,
    device: str = "cpu",
    log_prefix: str | None = None,
    log_every: int = 0,
    optimizer_type: OptimizerType = "rcg",
    rank_selection: RankSelectionType = "fixed_heuristic",
    rank_fraction: float = 0.1,
    truncation_tolerance: float = 1e-6,
    binary_search_max_iter: int = 12,
) -> CD.Matrix2DDisentanglementResult:
    shape_4d = choose_tensor_shape(matrix_2d)
    tensor_4d = matrix_to_tensor4d(matrix_2d.to(dtype=torch.float64), shape_4d)
    inner = disentangle_tensor4d(
        tensor_4d,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        log_prefix=log_prefix,
        log_every=log_every,
        optimizer_type=optimizer_type,
        rank_selection=rank_selection,
        rank_fraction=rank_fraction,
        truncation_tolerance=truncation_tolerance,
        binary_search_max_iter=binary_search_max_iter,
    )
    return CD.Matrix2DDisentanglementResult(
        matrix_shape_2d=tuple(int(dim) for dim in matrix_2d.shape),
        tensor_shape_4d=shape_4d,
        inner=inner,
    )


def _local_weight_targets(
    checkpoint: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    architecture = checkpoint["config"]["architecture"]
    width = int(checkpoint["config"]["width"])
    targets: list[dict[str, Any]] = []
    for weight_key, value in state_dict.items():
        if not (weight_key.endswith("weight") and value.ndim == 2):
            continue
        if architecture == "attention" and weight_key.endswith("attention.in_proj_weight"):
            for role, start in (("q", 0), ("k", width), ("v", 2 * width)):
                end = start + width
                targets.append(
                    {
                        "logical_entity": weight_key.replace(
                            "in_proj_weight", f"{role}_proj.weight"
                        ),
                        "weight_key": weight_key,
                        "row_range": [start, end],
                        "matrix": value[start:end],
                    }
                )
            continue
        targets.append(
            {
                "logical_entity": weight_key,
                "weight_key": weight_key,
                "row_range": None,
                "matrix": value,
            }
        )
    return targets


def analyze_local_weight_bases(
    checkpoint: dict[str, Any],
    *,
    tensor_steps: int,
    tensor_learning_rate: float,
    tensor_optimizer_type: OptimizerType = "rcg",
    rank_selection: RankSelectionType = "fixed_heuristic",
    rank_fraction: float = 0.1,
    truncation_tolerance: float = 1e-6,
    binary_search_max_iter: int = 12,
    seed: int,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any], dict[str, Any]]:
    state_dict = {
        key: value.detach().clone()
        for key, value in checkpoint["model_state_dict"].items()
    }
    layer_report: dict[str, Any] = {
        key: {"status": "skipped", "operations": []} for key in state_dict
    }
    architecture = checkpoint["config"]["architecture"]
    transform_report: dict[str, Any] = {
        "architecture": architecture,
        "representation": "local_weight_basis",
        "rank_selection": rank_selection,
        "transforms": [],
    }
    transform_tensors: dict[str, Any] = {
        "architecture": architecture,
        "representation": "local_weight_basis",
        "transforms": [],
    }

    for weight_index, target in enumerate(_local_weight_targets(checkpoint, state_dict)):
        weight = target["matrix"].to(dtype=torch.float64)
        result = disentangle_matrix_via_reshape(
            weight,
            steps=tensor_steps,
            learning_rate=tensor_learning_rate,
            seed=seed + weight_index,
            device=device,
            log_prefix=target["logical_entity"],
            log_every=max(1, tensor_steps),
            optimizer_type=tensor_optimizer_type,
            rank_selection=rank_selection,
            rank_fraction=rank_fraction,
            truncation_tolerance=truncation_tolerance,
            binary_search_max_iter=binary_search_max_iter,
        )
        operation = {
            "logical_entity": target["logical_entity"],
            "procedure": "local_weight_rotation_basis",
            "weight_key": target["weight_key"],
            "row_range": target["row_range"],
            "optimizer_type": tensor_optimizer_type,
            "rank_selection": rank_selection,
            "keep_rank": getattr(result.inner, "keep_rank", None),
            "rank_search": getattr(result.inner, "rank_search", None),
            "best_cost": getattr(result.inner, "best_cost", None),
            "initial_entropy": result.inner.initial_stats.entropy,
            "best_entropy": result.inner.best_stats.entropy,
            "entropy_drop": result.inner.initial_stats.entropy
            - result.inner.best_stats.entropy,
            "initial_effective_rank": result.inner.initial_stats.effective_rank,
            "best_effective_rank": result.inner.best_stats.effective_rank,
            "matrix_shape_2d": list(result.matrix_shape_2d),
            "tensor_shape_4d": asdict(result.tensor_shape_4d),
            "history": result.inner.history,
        }
        transform_report["transforms"].append(operation)
        transform_tensors["transforms"].append(
            {
                "logical_entity": target["logical_entity"],
                "procedure": operation["procedure"],
                "weight_key": target["weight_key"],
                "row_range": target["row_range"],
                "optimizer_type": tensor_optimizer_type,
                "tensor_shape_4d": asdict(result.tensor_shape_4d),
                "best_q": result.inner.best_q.detach().cpu(),
            }
        )
        mark_layer(layer_report, target["weight_key"], "analyzed", operation)

    return state_dict, layer_report, transform_report, transform_tensors


def process_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path,
    tensor_steps: int,
    tensor_learning_rate: float,
    tensor_optimizer_type: OptimizerType = "rcg",
    rank_selection: RankSelectionType = "fixed_heuristic",
    rank_fraction: float = 0.1,
    truncation_tolerance: float = 1e-6,
    binary_search_max_iter: int = 12,
    eval_episodes: int,
    deterministic_eval: bool,
    seed: int,
    device: str,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint["config"]["architecture"]

    if architecture not in {"mlp", "attention"}:
        raise ValueError(f"Unsupported architecture '{architecture}'")
    modified_state_dict, layer_report, transform_report, transform_tensors = (
        analyze_local_weight_bases(
            checkpoint,
            tensor_steps=tensor_steps,
            tensor_learning_rate=tensor_learning_rate,
            tensor_optimizer_type=tensor_optimizer_type,
            rank_selection=rank_selection,
            rank_fraction=rank_fraction,
            truncation_tolerance=truncation_tolerance,
            binary_search_max_iter=binary_search_max_iter,
            seed=seed,
            device=device,
        )
    )

    exactness = max_model_difference(
        checkpoint, modified_state_dict, seed=seed, device=torch.device(device)
    )
    #TODO: check if ok without this?

    #evaluation = evaluate_state_dict(
    #    checkpoint,
    #    modified_state_dict,
    #    episodes=eval_episodes,
    #    seed=seed,
    #    deterministic=deterministic_eval,
    #    device=torch.device(device),
    #)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / checkpoint_output_name(checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "architecture": architecture,
        "representation": "local_weight_basis",
        "state_dict_modified": False,
        "inverse_applied_during_analysis": False,
        "tensor_optimizer_type": tensor_optimizer_type,
        "rank_selection": rank_selection,
        "rank_fraction": rank_fraction,
        "truncation_tolerance": truncation_tolerance,
        "exactness_check": exactness,
        #TODO: check if this is ok
        "evaluation": None,
        "transform_report": transform_report,
        "layer_report": layer_report,
    }

    #TODO: check if this is ok

    #visualize_permutation_trajectories(summary, checkpoint_dir / "trajectories.png")

    (checkpoint_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    collapsed_payload = {
        "checkpoint_path": str(checkpoint_path),
        "modified_state_dict": modified_state_dict,
        "summary": summary,
    }
    torch.save(collapsed_payload, checkpoint_dir / "disentangled_payload.pt")
    torch.save(
        {
            **collapsed_payload,
            "transform_tensors": transform_tensors,
        },
        checkpoint_dir / "disentangled_payload_extended.pt",
    )
    return summary
