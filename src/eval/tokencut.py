"""TokenCut object discovery evaluation (Section 6.2.2).

TokenCut (Wang et al., 2023) uses self-attention features from a ViT to
perform salient object detection via Normalized Cut.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage


class TokenCutEvaluator:
    """Evaluate object discovery using TokenCut on ViT features."""

    def __init__(
        self,
        backbone: nn.Module,
        block_idx: int = 11,
        device: str = "cuda",
        image_size: int = 224,
    ):
        self.backbone = backbone.to(device).eval()
        self.block_idx = block_idx
        self.device = device
        self.image_size = image_size
        self.patch_size = getattr(backbone, "patch_embed", None)
        if self.patch_size is not None:
            self.patch_size = self.patch_size.patch_size
        else:
            self.patch_size = 16

    @torch.no_grad()
    def get_key_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract key features from the specified block's attention.

        Returns (B, N, head_dim) features from the key projection.
        """
        images = images.to(self.device)
        x = self.backbone.prepare_tokens(images)

        # Run through blocks up to block_idx
        for i, blk in enumerate(self.backbone.blocks):
            if i == self.block_idx:
                # Extract keys before the full forward
                normed = blk.norm1(x)
                k = blk.attn.W_K(normed)
                B, N, D = k.shape
                num_heads = blk.attn.num_heads
                head_dim = blk.attn.head_dim
                # Use last head's keys as features
                k = k.reshape(B, N, num_heads, head_dim)
                # Average across heads
                k = k.mean(dim=2)  # (B, N, head_dim)
                return k[:, 1:]  # Remove CLS token
            x = blk(x)

        raise ValueError(f"block_idx {self.block_idx} out of range")

    def normalized_cut(self, features: torch.Tensor, tau: float = 0.15) -> torch.Tensor:
        """Apply Normalized Cut to segment foreground from background.

        Parameters
        ----------
        features : (N, D) patch features for a single image.
        tau : threshold for binarizing the affinity matrix.

        Returns
        -------
        mask : (N,) binary mask (1 = foreground).
        """
        features = features.cpu().numpy()
        N, D = features.shape

        # Compute cosine similarity affinity matrix
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features_normed = features / norms
        A = features_normed @ features_normed.T

        # Threshold
        A = (A > tau).astype(float)

        # Degree matrix
        d = A.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(d) + 1e-10))

        # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
        L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

        # Second smallest eigenvector (Fiedler vector)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]
        except np.linalg.LinAlgError:
            return np.ones(N)

        # Binarize: segment with more positive CLS-adjacent values is foreground
        mask = (fiedler > 0).astype(float)

        # Pick the segment that is more likely foreground (smaller area = object)
        if mask.sum() > N / 2:
            mask = 1.0 - mask

        return mask

    def mask_to_bbox(self, mask: np.ndarray, h: int, w: int) -> tuple:
        """Convert a binary mask to a bounding box (x1, y1, x2, y2)."""
        mask_2d = mask.reshape(h, w)
        # Find connected components and take largest
        labeled, num_features = ndimage.label(mask_2d)
        if num_features == 0:
            return (0, 0, w, h)

        # Find largest component
        best_size = 0
        best_label = 1
        for label_id in range(1, num_features + 1):
            size = (labeled == label_id).sum()
            if size > best_size:
                best_size = size
                best_label = label_id

        component = (labeled == best_label)
        rows = np.where(component.any(axis=1))[0]
        cols = np.where(component.any(axis=0))[0]

        if len(rows) == 0 or len(cols) == 0:
            return (0, 0, w, h)

        y1, y2 = rows[0], rows[-1]
        x1, x2 = cols[0], cols[-1]

        # Scale mask-grid coords to the actual evaluation image size.
        scale_h = float(self.image_size) / h
        scale_w = float(self.image_size) / w
        return (
            int(x1 * scale_w),
            int(y1 * scale_h),
            int((x2 + 1) * scale_w),
            int((y2 + 1) * scale_h),
        )

    def compute_iou(self, box1: tuple, box2: tuple) -> float:
        """Compute IoU between two boxes (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / (union + 1e-10)

    @torch.no_grad()
    def evaluate(self, dataset, iou_threshold: float = 0.5) -> float:
        """Evaluate CorLoc on a dataset.

        The dataset should yield (image, gt_bbox) tuples where gt_bbox is
        (x1, y1, x2, y2).

        Returns CorLoc (fraction of images with IoU > threshold).
        """
        correct = 0
        total = 0

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for images, gt_bboxes in loader:
            features = self.get_key_features(images)  # (1, N, D)
            features = features[0]  # (N, D)

            N = features.shape[0]
            h = w = int(N ** 0.5)

            mask = self.normalized_cut(features)
            pred_bbox = self.mask_to_bbox(mask, h, w)

            # gt_bboxes should be (1, 4) or similar
            if isinstance(gt_bboxes, torch.Tensor):
                gt = tuple(gt_bboxes[0].tolist())
            else:
                gt = gt_bboxes

            iou = self.compute_iou(pred_bbox, gt)
            if iou >= iou_threshold:
                correct += 1
            total += 1

        corloc = correct / max(total, 1)
        return corloc
