"""Classification and segmentation heads for ViT."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Simple linear classification head on CLS token."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SegmentationHead(nn.Module):
    """Linear head for dense semantic segmentation.

    Takes patch token features and produces per-pixel class predictions.
    """

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, patch_tokens: torch.Tensor, img_size: int = 224) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_tokens : (B, N, D) patch token features (no CLS).
        img_size : original image size.

        Returns
        -------
        (B, num_classes, H, W) logits at original resolution.
        """
        B, N, D = patch_tokens.shape
        h = w = int(N ** 0.5)
        x = patch_tokens.transpose(1, 2).reshape(B, D, h, w)
        x = self.linear(x)
        x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        return x
