"""
SOAP Optimizer: Improving and Stabilizing Shampoo Using Adam for Language Modeling.

Reference:
    Vyas, N., Kakade, S., & Barak, B. (2025).
    SOAP: Improving and Stabilizing Shampoo Using Adam for Language Modeling.
    International Conference on Learning Representations (ICLR 2025).

SOAP is a second-order optimizer that combines Shampoo-style preconditioning
with Adam-style diagonal moment estimation. The key idea is:

  1. Maintain running estimates of the left and right covariance matrices
     of the gradient (as in Shampoo).
  2. Periodically compute eigenbases of these covariance matrices.
  3. Project the gradient into the eigenbasis space.
  4. Run Adam (first + second moment tracking) in that projected space.
  5. Project the corrected update back to the original space.

This gives the benefits of Shampoo's second-order curvature information
while retaining the stability and per-coordinate adaptivity of Adam.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple
import math


def _merge_small_dims(shape: List[int], max_dim: int) -> List[int]:
    """Merge consecutive small dimensions so that each resulting dimension
    is as large as possible without exceeding ``max_dim``.

    This is used so that higher-dimensional tensors (e.g. Conv2d weights with
    shape [out, in, kH, kW]) can be treated as 2D matrices for Shampoo-style
    preconditioning. Small spatial dimensions are folded together.

    Args:
        shape: Original tensor shape as a list of ints.
        max_dim: Maximum allowed size for any merged dimension.

    Returns:
        A list of ints representing the merged shape.
    """
    merged: List[int] = []
    current = 1
    for dim in shape:
        if current * dim <= max_dim:
            current *= dim
        else:
            if current > 1:
                merged.append(current)
            current = dim
    if current > 1:
        merged.append(current)
    # If everything collapsed into a single dimension, keep it as-is.
    if len(merged) == 0:
        merged.append(1)
    return merged


class SOAP(Optimizer):
    """SOAP optimizer (Shampoo + Adam in the eigenbasis).

    For each parameter tensor W of shape (m, n) the optimizer maintains:
      - L (m x m) and R (n x n): running EMA estimates of the left / right
        Kronecker factors of the full-matrix Adagrad preconditioner, i.e.
        L_t = beta_shampoo * L_{t-1} + (1 - beta_shampoo) * G_t @ G_t^T
        R_t = beta_shampoo * R_{t-1} + (1 - beta_shampoo) * G_t^T @ G_t
      - Q_L, Q_R: eigenvectors of L, R (updated every ``precondition_frequency`` steps).
      - exp_avg, exp_avg_sq: Adam first / second moments, stored in the
        projected (eigenbasis) space.

    The per-step update is:
      1. Update L, R with the current gradient.
      2. Every ``precondition_frequency`` steps, recompute Q_L, Q_R.
      3. Project gradient: G_hat = Q_L^T @ G @ Q_R
      4. Adam update on G_hat (in eigenbasis space).
      5. Project back: Delta_W = Q_L @ G_hat_corrected @ Q_R^T
      6. Apply decoupled weight decay and learning rate.

    For 1-D parameters (biases, LayerNorm weights) or parameters whose
    every dimension exceeds ``max_precond_dim``, the optimizer falls back
    to standard AdamW.

    Args:
        params: Iterable of parameters or param-group dicts.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for first and second moment estimation
            (default: (0.9, 0.95)).
        shampoo_beta: EMA coefficient for covariance matrices. If negative,
            ``betas[1]`` is used (default: -1, meaning use beta2).
        eps: Term added to denominator for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        precondition_frequency: How often (in steps) to recompute eigenbases
            (default: 10).
        max_precond_dim: Maximum dimension size eligible for Shampoo
            preconditioning. Dimensions larger than this are not preconditioned
            (default: 10000).
        merge_dims: Whether to merge small consecutive dimensions for
            higher-order tensors so they can be treated as matrices
            (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = -1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_use_shampoo(
        shape: List[int], max_precond_dim: int
    ) -> bool:
        """Return True if the parameter is eligible for Shampoo preconditioning.

        We require the (possibly merged) tensor to be at least 2-D and every
        dimension to be <= ``max_precond_dim``.
        """
        if len(shape) < 2:
            return False
        return all(d <= max_precond_dim for d in shape)

    @staticmethod
    def _compute_eigenbasis(
        cov: torch.Tensor,
        fallback: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the eigenvectors of a symmetric PSD covariance matrix.

        Resilient against three failure modes we have observed in early
        steps of skipless ViT training:
          1. ``cov`` contains NaN/Inf (e.g. from a gradient overflow the
             optimizer didn't skip quickly enough).
          2. ``torch.linalg.eigh`` raises ``LinAlgError`` because the matrix
             is (numerically) ill-conditioned.
          3. ``eigh`` "succeeds" but returns non-finite eigenvectors when
             the input had very large dynamic range.

        In any of these cases we return ``fallback`` (typically the previous
        Q) if provided, otherwise an identity matrix. Returning a clean
        fallback is safe: projecting / back-projecting with the identity is
        equivalent to running plain Adam for one extra step, which is what
        SOAP does before the first eigendecomposition anyway.

        Args:
            cov: A symmetric positive semi-definite matrix.
            fallback: Matrix of the same shape as the expected output; if
                ``cov`` is unusable we return this instead of crashing.

        Returns:
            Q: Orthonormal eigenvector matrix (columns are eigenvectors).
        """
        n = cov.shape[0]
        if fallback is None:
            fallback = torch.eye(n, device=cov.device, dtype=cov.dtype)

        # 1) Reject pathological inputs outright.
        if not torch.isfinite(cov).all():
            return fallback

        # 2) Try eigh on the raw covariance.
        try:
            _, Q = torch.linalg.eigh(cov)
            if torch.isfinite(Q).all():
                return Q
        except Exception:
            pass

        # 3) One perturbed retry. Use a perturbation that's relative to the
        # trace to survive very-large-norm covariances in early steps.
        trace = float(cov.diagonal().abs().sum().item())
        eps = max(1e-6, trace * 1e-6 / max(n, 1))
        try:
            perturbed = cov + eps * torch.eye(n, device=cov.device, dtype=cov.dtype)
            _, Q = torch.linalg.eigh(perturbed)
            if torch.isfinite(Q).all():
                return Q
        except Exception:
            pass

        # 4) Give up this step; upstream will just keep the previous Q.
        return fallback

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that re-evaluates the model and returns the loss
                (not commonly used with SOAP).

        Returns:
            The loss value if ``closure`` is provided, else ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            if shampoo_beta < 0:
                shampoo_beta = beta2
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            precondition_frequency = group["precondition_frequency"]
            max_precond_dim = group["max_precond_dim"]
            merge_dims = group["merge_dims"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients.")

                # Always maintain SOAP state in fp32 so that Shampoo covariance
                # accumulation and eigendecomposition are numerically stable
                # regardless of the gradient dtype (fp16/bf16 under AMP).
                state = self.state[p]
                state_dtype = torch.float32

                # ----- State initialisation -----
                if len(state) == 0:
                    state["step"] = 0

                    original_shape = list(grad.shape)

                    # Determine the working shape (after optional dim merging).
                    if merge_dims and len(original_shape) > 2:
                        working_shape = _merge_small_dims(
                            original_shape, max_precond_dim
                        )
                    else:
                        working_shape = original_shape

                    state["original_shape"] = original_shape
                    state["working_shape"] = working_shape
                    state["use_shampoo"] = self._should_use_shampoo(
                        working_shape, max_precond_dim
                    )

                    if state["use_shampoo"]:
                        # We reshape the gradient to the working shape for
                        # covariance estimation.  For a 2-D working shape
                        # (m, n) we maintain L (m x m) and R (n x n).
                        m, n = working_shape[0], working_shape[1]

                        # Covariance accumulators (initialised to small identity
                        # so the first eigendecomposition is well-conditioned).
                        state["L"] = torch.zeros(
                            m, m, device=grad.device, dtype=state_dtype
                        )
                        state["R"] = torch.zeros(
                            n, n, device=grad.device, dtype=state_dtype
                        )

                        # Eigenbases - initialise to identity so that the first
                        # few steps (before the first eigendecomposition) behave
                        # like standard Adam.
                        state["Q_L"] = torch.eye(
                            m, device=grad.device, dtype=state_dtype
                        )
                        state["Q_R"] = torch.eye(
                            n, device=grad.device, dtype=state_dtype
                        )

                        # Adam moments are stored in the projected (eigenbasis)
                        # space with the working shape.
                        state["exp_avg"] = torch.zeros(
                            m, n, device=grad.device, dtype=state_dtype
                        )
                        state["exp_avg_sq"] = torch.zeros(
                            m, n, device=grad.device, dtype=state_dtype
                        )
                    else:
                        # Fallback to plain AdamW - moments in original space.
                        state["exp_avg"] = torch.zeros_like(
                            grad, dtype=state_dtype
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            grad, dtype=state_dtype
                        )

                # Skip entirely if the grad contains NaN/Inf - the caller
                # (e.g. GradScaler) is expected to lower the loss scale on the
                # next iteration. Touching the state with non-finite values
                # would poison it permanently.
                if not torch.isfinite(grad).all():
                    continue

                # Increment step counter.
                state["step"] += 1
                step = state["step"]

                # ----- Decoupled weight decay (applied before the update) -----
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # ============================================================
                # Path A: Shampoo-preconditioned Adam (2-D working shape)
                # ============================================================
                if state["use_shampoo"]:
                    original_shape = state["original_shape"]
                    working_shape = state["working_shape"]
                    m, n = working_shape[0], working_shape[1]

                    # Reshape gradient to working 2-D shape and promote to the
                    # state dtype (fp32) so AMP fp16 grads don't break eigh.
                    G = grad.reshape(m, n).to(state_dtype)

                    # --- 1. Update covariance running estimates ---
                    # L_t = shampoo_beta * L_{t-1} + (1 - shampoo_beta) * G @ G^T
                    # R_t = shampoo_beta * R_{t-1} + (1 - shampoo_beta) * G^T @ G
                    # Compute the outer products first so we can reject any
                    # non-finite contributions without poisoning L / R.
                    GGt = G @ G.t()
                    GtG = G.t() @ G
                    if torch.isfinite(GGt).all() and torch.isfinite(GtG).all():
                        state["L"].mul_(shampoo_beta).add_(
                            GGt, alpha=1.0 - shampoo_beta
                        )
                        state["R"].mul_(shampoo_beta).add_(
                            GtG, alpha=1.0 - shampoo_beta
                        )

                    # --- 2. Periodically recompute eigenbases ---
                    if step % precondition_frequency == 0:
                        state["Q_L"] = self._compute_eigenbasis(
                            state["L"], fallback=state["Q_L"]
                        )
                        state["Q_R"] = self._compute_eigenbasis(
                            state["R"], fallback=state["Q_R"]
                        )

                    Q_L = state["Q_L"]
                    Q_R = state["Q_R"]

                    # --- 3. Project gradient into eigenbasis ---
                    # G_hat = Q_L^T @ G @ Q_R
                    G_hat = Q_L.t() @ G @ Q_R

                    # --- 4. Adam update in projected space ---
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # First moment: m_t = beta1 * m_{t-1} + (1 - beta1) * G_hat
                    exp_avg.mul_(beta1).add_(G_hat, alpha=1.0 - beta1)
                    # Second moment: v_t = beta2 * v_{t-1} + (1 - beta2) * G_hat^2
                    exp_avg_sq.mul_(beta2).addcmul_(
                        G_hat, G_hat, value=1.0 - beta2
                    )

                    # Bias correction.
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    m_hat = exp_avg / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2

                    # Adam-style corrected update in projected space.
                    update_projected = m_hat / (v_hat.sqrt() + eps)

                    # --- 5. Project back to original space ---
                    # Delta_W = Q_L @ update_projected @ Q_R^T
                    update = Q_L @ update_projected @ Q_R.t()

                    # Defensive: never push NaN/Inf into the parameter tensor
                    # (would persist forever and also kill downstream layers).
                    if not torch.isfinite(update).all():
                        continue

                    # Reshape back to original parameter shape, cast to the
                    # parameter dtype and apply.
                    p.data.add_(
                        update.reshape(original_shape).to(p.dtype), alpha=-lr
                    )

                # ============================================================
                # Path B: Standard AdamW fallback (1-D or oversized params)
                # ============================================================
                else:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    g = grad.to(state_dtype)

                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        g, g, value=1.0 - beta2
                    )

                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    m_hat = exp_avg / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2

                    update = m_hat / (v_hat.sqrt() + eps)

                    if not torch.isfinite(update).all():
                        continue

                    p.data.add_(update.to(p.dtype), alpha=-lr)

        return loss
