"""
Vision Transformer (ViT) with optional skip-connection removal.

Implements the standard ViT architecture (Dosovitskiy et al., 2020) with the
modification from "Cutting the Skip: Training Residual-Free Transformers"
that removes residual (skip) connections when ``skipless=True``.

Standard ViT block:
    X_l  = X_{l-1} + SA(LN(X_{l-1}))
    X~_l = X_l     + MLP(LN(X_l))

Skipless ViT block:
    X_l  = SA(LN(X_{l-1}))
    X~_l = MLP(LN(X_l))

When skipless=True the drop-path rate is forced to 0.0 because stochastic
depth is not applicable to residual-free models.
"""

from __future__ import annotations

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Drop-path (stochastic depth) -- only used when skip connections are present
# ---------------------------------------------------------------------------

def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Drop paths (stochastic depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Work with arbitrary leading dims -- (batch, ..., channels)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop-path (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention with individually accessible projection
    weights (W_Q, W_K, W_V, W_O) for fine-grained initialization control."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        # Separate Q, K, V projections so skipless_init can target each one.
        self.W_Q = nn.Linear(dim, self.inner_dim, bias=True)
        self.W_K = nn.Linear(dim, self.inner_dim, bias=True)
        self.W_V = nn.Linear(dim, self.inner_dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection
        self.W_O = nn.Linear(self.inner_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        q = self.W_Q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_K(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_V(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)

        x = self.W_O(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Feed-Forward Network (MLP)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or 4 * in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer block with optional skip (residual) connections.

    Parameters
    ----------
    dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension per attention head.
    mlp_ratio : float
        Ratio of MLP hidden dim to embedding dim.
    skipless : bool
        If True, residual connections are removed.
    drop_path : float
        Drop-path rate (forced to 0.0 when skipless=True).
    attn_drop : float
        Dropout rate inside attention.
    proj_drop : float
        Dropout rate after projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        skipless: bool = False,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.skipless = skipless

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
        )

        # Drop-path is disabled for skipless models.
        dp_rate = 0.0 if skipless else drop_path
        self.drop_path = DropPath(dp_rate) if dp_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skipless:
            # Residual-free forward pass
            x = self.attn(self.norm1(x))
            x = self.mlp(self.norm2(x))
        else:
            # Standard ViT forward pass with residual connections
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convert an image into a sequence of patch embeddings.

    Parameters
    ----------
    img_size : int
        Input image size (assumed square).
    patch_size : int
        Patch size (assumed square).
    in_chans : int
        Number of input channels.
    embed_dim : int
        Embedding dimension.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Vision Transformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) with optional skip-connection removal.

    Parameters
    ----------
    img_size : int
        Input image size (assumed square).
    patch_size : int
        Patch size.
    in_chans : int
        Number of input image channels.
    num_classes : int
        Number of classification classes. Set to 0 for feature extraction
        (e.g. DINO) -- no classification head is created.
    embed_dim : int
        Token embedding dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension per attention head.
    mlp_ratio : float
        Ratio of MLP hidden dim to embedding dim.
    skipless : bool
        If True, residual connections are removed from all blocks.
    drop_path_rate : float
        Maximum drop-path rate (linearly increases across depth).
        Forced to 0.0 when skipless=True.
    attn_drop_rate : float
        Dropout rate inside attention.
    drop_rate : float
        Dropout rate after embeddings and projections.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        skipless: bool = False,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.skipless = skipless
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Drop-path is not applicable for skipless models.
        if skipless:
            drop_path_rate = 0.0

        # --- Patch embedding ---
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # --- CLS token and positional embedding ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches, embed_dim),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # --- Transformer blocks ---
        # Linearly increasing drop-path rate across depth.
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, depth)
        ]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                skipless=skipless,
                drop_path=dpr[i],
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
            )
            for i in range(depth)
        ])

        # --- Final layer norm ---
        self.norm = nn.LayerNorm(embed_dim)

        # --- Classification head ---
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self) -> None:
        """Default weight initialization (truncated normal + zeros).

        This mirrors the standard ViT initialization from timm. The
        skipless_init module may override these values afterwards.
        """
        # Positional embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # CLS token: truncated normal
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m: nn.Module) -> None:
        """Per-module initialization applied via ``self.apply(...)``."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # Fan-out initialization for patch embedding conv.
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def interpolate_pos_encoding(
        self,
        x: torch.Tensor,
        w: int,
        h: int,
    ) -> torch.Tensor:
        """Bicubically interpolate the positional embedding to match an
        input of arbitrary spatial size.

        This is required for DINO-style multi-crop training where global
        crops (e.g. 64x64) and local crops (e.g. 32x32) produce different
        sequence lengths, as well as for dense prediction tasks that feed
        in non-standard input resolutions.
        """
        N_tokens = x.shape[1] - 1  # exclude CLS
        N_pos = self.pos_embed.shape[1] - 1
        if N_tokens == N_pos and w == h:
            return self.pos_embed

        class_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        patch_size = self.patch_embed.patch_size
        w0 = w // patch_size
        h0 = h // patch_size
        # +0.1 nudge avoids floating-point edge cases on grid boundaries.
        w0, h0 = w0 + 0.1, h0 + 0.1

        # Reshape (1, N_pos, D) -> (1, D, sqrt(N_pos), sqrt(N_pos))
        grid = int(math.sqrt(N_pos))
        assert grid * grid == N_pos, (
            f"pos_embed grid is not square: N_pos={N_pos}"
        )
        patch_pos = patch_pos.reshape(1, grid, grid, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            scale_factor=(w0 / grid, h0 / grid),
            mode="bicubic",
            align_corners=False,
        )
        assert int(w0) == patch_pos.shape[-2] and int(h0) == patch_pos.shape[-1]
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([class_pos, patch_pos], dim=1)

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Embed patches, prepend CLS token, add positional embedding.

        Positional embedding is interpolated on the fly when the input
        resolution does not match the training resolution (needed for DINO
        multi-crop and for dense prediction at arbitrary sizes).
        """
        B, _, H, W = x.shape
        x = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N, D)

        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run through patch embed + transformer blocks + final norm.

        Returns the full sequence (CLS + patch tokens).
        """
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: features -> CLS token -> classification head."""
        x = self.forward_features(x)
        cls_out = x[:, 0]  # CLS token
        x = self.head(cls_out)
        return x

    # ------------------------------------------------------------------
    # DINO compatibility
    # ------------------------------------------------------------------

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 1,
    ) -> List[torch.Tensor]:
        """Return intermediate features from the last *n* transformer blocks.

        This is used by DINO for building features from multiple layers.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape ``(B, C, H, W)``.
        n : int
            Number of last blocks whose outputs to collect.

        Returns
        -------
        list of torch.Tensor
            Each tensor has shape ``(B, 1+num_patches, embed_dim)`` and
            includes the CLS token at position 0.  The list is ordered
            from the earliest requested block to the last block.
        """
        x = self.prepare_tokens(x)
        output: List[torch.Tensor] = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i >= len(self.blocks) - n:
                # Apply final norm to each collected output.
                output.append(self.norm(x))

        return output

    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return only patch tokens (without CLS) from the final layer.

        Useful for dense prediction tasks.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_patches, embed_dim)``.
        """
        x = self.forward_features(x)
        return x[:, 1:]  # strip CLS token

    # ------------------------------------------------------------------
    # Introspection helpers (used by skipless_init)
    # ------------------------------------------------------------------

    def get_num_layers(self) -> int:
        """Return the number of transformer blocks."""
        return len(self.blocks)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def vit_base(
    skipless: bool = False,
    num_classes: int = 1000,
    **kwargs,
) -> VisionTransformer:
    """ViT-Base: 12 layers, 12 heads, embed_dim=768, head_dim=64."""
    defaults = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        head_dim=64,
        mlp_ratio=4.0,
    )
    defaults.update(kwargs)
    return VisionTransformer(
        skipless=skipless,
        num_classes=num_classes,
        **defaults,
    )


def vit_small(
    skipless: bool = False,
    num_classes: int = 0,
    **kwargs,
) -> VisionTransformer:
    """ViT-Small: 12 layers, 6 heads, embed_dim=384, head_dim=64.

    Default ``num_classes=0`` for DINO / feature extraction usage.
    """
    defaults = dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        head_dim=64,
        mlp_ratio=4.0,
    )
    defaults.update(kwargs)
    return VisionTransformer(
        skipless=skipless,
        num_classes=num_classes,
        **defaults,
    )
