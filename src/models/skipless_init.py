"""
Skipless initialization scheme from "Cutting the Skip: Training Residual-Free
Transformers".

Two key initialization strategies:
1. W_V, W_O initialization (Section 5.1): Makes W_V * W_O a scaled orthonormal
   matrix with condition number kappa = 1.
2. W_Q, W_K initialization (Section 5.2): Makes W_Q * W_K^T = alpha*Z + beta*I
   to encourage diagonal dominance in attention logits.
3. MLP initialization: Scaled-corrected orthogonal initialization.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _orthogonal_matrix(n: int, m: int, device: torch.device = None) -> torch.Tensor:
    """Generate a (semi-)orthogonal matrix of shape (n, m) via QR decomposition."""
    A = torch.randn(max(n, m), max(n, m), device=device)
    Q, _ = torch.linalg.qr(A)
    return Q[:n, :m]


def _init_wv_wo(attn_module: nn.Module, c: float = 3.0) -> None:
    """Initialize W_V and W_O so that W_V_h @ W_O_h = c^2 * I_{head_dim}.

    Storage layout:
        W_V.weight is (inner_dim, dim), so rows [h*hd:(h+1)*hd] are W_V_h
            (shape (hd, dim), maps dim -> head_dim).
        W_O.weight is (dim, inner_dim), so cols [h*hd:(h+1)*hd] are W_O_h
            (shape (dim, hd), maps head_dim -> dim).

    Construction (for each head h, independently):
        Let Q be a random orthogonal (dim x dim) matrix. Take its first
        ``head_dim`` rows, call that ``A`` with shape (hd, dim). Since Q
        is orthogonal its row-slice has orthonormal rows, so
        ``A @ A.T = I_hd``.

        Set W_V_h = c * A           (shape (hd, dim))
            W_O_h = c * A.T         (shape (dim, hd))

        Then  W_V_h @ W_O_h  =  c^2 * (A @ A.T)  =  c^2 * I_hd.

    This makes the per-head value-to-output projection a scaled identity,
    which gives W_V*W_O condition number = 1 exactly (verify_init confirms).
    """
    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim
    dim = attn_module.W_V.in_features
    device = attn_module.W_V.weight.device

    wv_data = attn_module.W_V.weight.data  # (inner_dim, dim)
    wo_data = attn_module.W_O.weight.data  # (dim, inner_dim)

    for h in range(num_heads):
        # Random orthogonal (dim, dim) via QR. Using QR here (not SVD) gives
        # us a single orthogonal matrix instead of the two separate U and V
        # factors of SVD, which is what we need for W_V*W_O = c^2 * I.
        rand = torch.randn(dim, dim, device=device)
        Q, R = torch.linalg.qr(rand)
        # Fix sign ambiguity so Q is uniquely distributed on O(dim).
        Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)

        A = Q[:head_dim, :]                  # (hd, dim), rows orthonormal

        # W_V_h  = c * A            shape (hd, dim)
        wv_data[h * head_dim : (h + 1) * head_dim, :] = c * A
        # W_O_h  = c * A.T          shape (dim, hd)
        wo_data[:, h * head_dim : (h + 1) * head_dim] = c * A.t()

    # Zero out biases
    if attn_module.W_V.bias is not None:
        nn.init.zeros_(attn_module.W_V.bias)
    if attn_module.W_O.bias is not None:
        nn.init.zeros_(attn_module.W_O.bias)


def _init_wq_wk(
    attn_module: nn.Module,
    alpha: float = 2.0,
    beta: float = 0.6,
) -> None:
    """Initialize W_Q and W_K so that W_Q_h * W_K_h^T = alpha*Z + beta*I.

    This encourages diagonal dominance in the attention logits, producing
    attention patterns closer to identity (each token attends mostly to itself).

    For each head h:
        Target M_h = alpha * Z_h + beta * I  where Z_h ~ N(0, 1/head_dim)
        Factor via SVD: M_h = U S V^T
        W_Q_h = U * sqrt(S),  W_K_h = V * sqrt(S)

    Storage: W_Q.weight is (inner_dim, dim), rows [h*hd:(h+1)*hd] for head h.
    The product in attention is: q_h k_h^T = x W_Q_h^T (x W_K_h^T)^T = x (W_Q_h^T W_K_h) x^T
    Wait -- in our model W_Q maps (dim -> inner_dim), so W_Q.weight is (inner_dim, dim).
    For head h, q_h = x @ W_Q.weight[h*hd:(h+1)*hd, :].T  (B, N, hd)
    The per-head product in logits: q @ k^T = x @ W_Q_h^T @ W_K_h @ x^T
    So the effective matrix is W_Q_h^T @ W_K_h (dim x dim).

    We want W_Q_h^T @ W_K_h = alpha*Z + beta*I (dim x dim).
    Set M = alpha*Z + beta*I, SVD -> U S V^T.
    W_Q_h^T = U sqrt(S) -> W_Q_h = (U sqrt(S))^T = sqrt(S) U^T
    W_K_h = sqrt(S) V^T -> stored as rows of W_K.weight

    Actually let's be more careful. W_Q_h has shape (head_dim, dim) in storage.
    The product W_Q_h^T @ W_K_h = (dim, head_dim) @ (head_dim, dim) = (dim, dim).
    This is rank-limited to head_dim. So we can't directly set it to a full-rank
    dim x dim matrix.

    The paper likely means the per-head logit matrix in the head_dim space:
    After projecting to head space, q_h = x W_Q_h^T has dim head_dim.
    The logit is q_h @ k_h^T = (x W_Q_h^T)(x W_K_h^T)^T, which depends on
    W_Q_h^T W_K_h (hd x hd? No... dim -> hd, so W_Q_h is (hd, dim)).

    Let's think again: W_Q_h is (hd, dim). q = x @ W_Q_h.T gives (B,N,hd).
    Logit = q @ k.T = x @ W_Q_h.T @ W_K_h @ x.T. Product is W_Q_h.T @ W_K_h
    which is (dim, dim). But this is rank hd.

    The paper's Eq 10-11 works in the head dimension. They consider the
    attention matrix A = softmax(X W_Q W_K^T X^T / sqrt(d_h)). The product
    W_Q W_K^T is in the projected space.

    For the initialization: we create target M_h = alpha*Z + beta*I where
    M_h is (head_dim x head_dim). Then factor it.
    But W_Q_h is (head_dim, dim) and W_K_h is (head_dim, dim).
    Product W_Q_h @ W_K_h.T is (head_dim, head_dim)... no.
    Actually in the attention: q = x W_Q_h^T (shape B,N,hd), k = x W_K_h^T (B,N,hd).
    Logits = q k^T = x W_Q_h^T W_K_h x^T. The "kernel" is W_Q_h^T W_K_h (dim, dim).

    I think the paper's approach is: initialize so that the effect in the
    input space W_Q_h^T W_K_h approximates alpha*Z + beta*I (dim x dim).
    Since this is rank hd, we do our best.

    Simpler interpretation: set W_Q_h and W_K_h so they're well-conditioned.
    Per the paper algorithm:
    - Generate Z ∈ R^{dim x dim}, Z_ij ~ N(0, 1/dim)
    - M = alpha * Z + beta * I
    - SVD: M = U S V^T
    - Take top head_dim singular values
    - W_Q_h = U[:, :hd] @ diag(sqrt(S[:hd]))  -> (dim, hd) -> transpose to (hd, dim)
    - W_K_h = V[:, :hd] @ diag(sqrt(S[:hd]))  -> (dim, hd) -> transpose to (hd, dim)
    """
    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim
    dim = attn_module.W_Q.in_features
    device = attn_module.W_Q.weight.device

    wq_data = attn_module.W_Q.weight.data  # (inner_dim, dim)
    wk_data = attn_module.W_K.weight.data  # (inner_dim, dim)

    for h in range(num_heads):
        # Target matrix: alpha * Z + beta * I (dim x dim)
        Z = torch.randn(dim, dim, device=device) / math.sqrt(dim)
        M = alpha * Z + beta * torch.eye(dim, device=device)

        # SVD
        U, S, Vt = torch.linalg.svd(M, full_matrices=True)

        # Take top head_dim components (truncated SVD)
        S_sqrt = torch.sqrt(S[:head_dim].clamp(min=1e-8))

        # W_Q_h: stored as (hd, dim) in rows of wq_data
        # = (diag(sqrt(S)) @ U[:, :hd].T) = (hd, dim)
        wq_data[h * head_dim : (h + 1) * head_dim, :] = (
            torch.diag(S_sqrt) @ U[:, :head_dim].T
        )

        # W_K_h: stored as (hd, dim) in rows of wk_data
        # = (diag(sqrt(S)) @ V[:, :hd].T) = (diag(sqrt(S)) @ Vt[:hd, :])
        wk_data[h * head_dim : (h + 1) * head_dim, :] = (
            torch.diag(S_sqrt) @ Vt[:head_dim, :]
        )

    # Zero out biases
    if attn_module.W_Q.bias is not None:
        nn.init.zeros_(attn_module.W_Q.bias)
    if attn_module.W_K.bias is not None:
        nn.init.zeros_(attn_module.W_K.bias)


def _init_mlp_orthogonal(mlp_module: nn.Module) -> None:
    """Scaled-corrected orthogonal initialization for MLP weights.

    Uses the approach from Martens et al. (2021): generate an orthogonal
    matrix and apply a scaling correction based on the fan-in/fan-out ratio.
    """
    for name, param in [("fc1", mlp_module.fc1), ("fc2", mlp_module.fc2)]:
        weight = param.weight  # (out_features, in_features)
        m, n = weight.shape

        # Generate orthogonal matrix
        Q = _orthogonal_matrix(m, n, device=weight.device)

        # Scaling correction: sqrt(max(m,n) / min(m,n)) for rectangular matrices
        # This ensures the expected squared norm of the output equals the input
        scale = math.sqrt(max(m, n) / min(m, n))

        weight.data.copy_(scale * Q)

        if param.bias is not None:
            nn.init.zeros_(param.bias)


def apply_skipless_init(
    model: nn.Module,
    alpha: float = 2.0,
    beta: float = 0.6,
    c: float = 3.0,
) -> None:
    """Apply the paper's initialization scheme to a skipless ViT model.

    Parameters
    ----------
    model : nn.Module
        A VisionTransformer instance (should have skipless=True).
    alpha : float
        Scaling factor for random component in W_Q*W_K^T initialization.
        Paper uses 2.0 for ViT-Base (supervised), 1.8 for ViT-Small (DINO).
    beta : float
        Scaling factor for identity component in W_Q*W_K^T initialization.
        Paper uses 0.6 for ViT-Base (supervised), 1.0 for ViT-Small (DINO).
    c : float
        Scaling constant for W_V*W_O initialization. Paper uses 3.0.
    """
    for block in model.blocks:
        attn = block.attn

        # 1. Initialize W_V, W_O for orthonormal product
        _init_wv_wo(attn, c=c)

        # 2. Initialize W_Q, W_K for diagonal-dominant attention
        _init_wq_wk(attn, alpha=alpha, beta=beta)

        # 3. Initialize MLP with scaled orthogonal
        _init_mlp_orthogonal(block.mlp)

    print(f"[skipless_init] Applied initialization: alpha={alpha}, beta={beta}, c={c}")


def verify_init(model: nn.Module, verbose: bool = True) -> dict:
    """Verify that initialization has the desired properties.

    Checks:
    - W_V * W_O product condition number (should be close to 1)
    - W_Q * W_K^T diagonal dominance (diagonal should dominate off-diagonal)
    - MLP weight orthogonality

    Returns a dict with summary statistics.
    """
    results = {
        "wv_wo_cond_numbers": [],
        "wqk_diag_dominance": [],
        "mlp_orthogonality": [],
    }

    for i, block in enumerate(model.blocks):
        attn = block.attn
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        dim = attn.W_V.in_features

        # Check W_V * W_O condition number per head
        wv = attn.W_V.weight.data  # (inner_dim, dim)
        wo = attn.W_O.weight.data  # (dim, inner_dim)

        for h in range(num_heads):
            wv_h = wv[h * head_dim : (h + 1) * head_dim, :]  # (hd, dim)
            wo_h = wo[:, h * head_dim : (h + 1) * head_dim]  # (dim, hd)
            # Product: wv_h @ wo_h -> (hd, hd)... or wo_h @ wv_h -> (dim, dim)?
            # The paper wants W_V W_O to be well-conditioned.
            # wv_h^T is (dim, hd), wo_h is (dim, hd), product wv_h @ wo_h is (hd, hd)
            product = wv_h @ wo_h  # (hd, dim) @ (dim, hd) = (hd, hd)
            s = torch.linalg.svdvals(product)
            cond = (s[0] / s[-1]).item() if s[-1] > 1e-10 else float("inf")
            results["wv_wo_cond_numbers"].append(cond)

        # Check W_Q * W_K^T diagonal dominance
        wq = attn.W_Q.weight.data
        wk = attn.W_K.weight.data

        for h in range(num_heads):
            wq_h = wq[h * head_dim : (h + 1) * head_dim, :]  # (hd, dim)
            wk_h = wk[h * head_dim : (h + 1) * head_dim, :]  # (hd, dim)
            # Product in input space: wq_h^T @ wk_h (dim, dim)
            product = wq_h.T @ wk_h  # (dim, dim)
            diag_mean = product.diag().abs().mean().item()
            offdiag = product - torch.diag(product.diag())
            offdiag_mean = offdiag.abs().mean().item()
            dominance = diag_mean / (offdiag_mean + 1e-10)
            results["wqk_diag_dominance"].append(dominance)

        # Check MLP orthogonality
        for fc in [block.mlp.fc1, block.mlp.fc2]:
            W = fc.weight.data
            m, n = W.shape
            if m <= n:
                WWT = W @ W.T
                identity = torch.eye(m, device=W.device)
                # Normalize by expected scale
                scale = W.norm() ** 2 / m
                deviation = (WWT / scale - identity).norm().item() / m
            else:
                WTW = W.T @ W
                identity = torch.eye(n, device=W.device)
                scale = W.norm() ** 2 / n
                deviation = (WTW / scale - identity).norm().item() / n
            results["mlp_orthogonality"].append(deviation)

    if verbose:
        import numpy as np
        conds = results["wv_wo_cond_numbers"]
        doms = results["wqk_diag_dominance"]
        orths = results["mlp_orthogonality"]
        print(f"[verify_init] W_V*W_O condition numbers: "
              f"mean={np.mean(conds):.3f}, max={np.max(conds):.3f}")
        print(f"[verify_init] W_Q*W_K^T diagonal dominance: "
              f"mean={np.mean(doms):.3f}, min={np.min(doms):.3f}")
        print(f"[verify_init] MLP orthogonality deviation: "
              f"mean={np.mean(orths):.6f}")

    return results
