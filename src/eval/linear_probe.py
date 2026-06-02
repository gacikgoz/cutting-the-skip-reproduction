"""Dense linear probing for semantic segmentation evaluation (Section 6.2.1)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class LinearProbeSegmentation:
    """Train a linear classifier on frozen ViT features for segmentation.

    Extracts dense patch token features from one or more transformer blocks
    and trains a 1x1 conv classifier on top.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        feature_blocks: list = None,
        patch_size: int = 16,
        device: str = "cuda",
    ):
        self.backbone = backbone.to(device).eval()
        self.num_classes = num_classes
        self.feature_blocks = feature_blocks or [11]
        self.patch_size = patch_size
        self.device = device

        # Determine feature dimension
        embed_dim = backbone.embed_dim
        feat_dim = embed_dim * len(self.feature_blocks)

        self.head = nn.Conv2d(feat_dim, num_classes, kernel_size=1).to(device)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract dense features from specified blocks.

        Returns (B, feat_dim, h, w) feature maps.
        """
        images = images.to(self.device)
        n_blocks = max(self.feature_blocks) + 1
        features = self.backbone.get_intermediate_layers(images, n=n_blocks)

        # Select requested blocks and concatenate
        selected = []
        depth = self.backbone.depth
        for idx in self.feature_blocks:
            # get_intermediate_layers returns last n blocks
            # so index into the returned list accordingly
            offset = depth - n_blocks
            list_idx = idx - offset
            if 0 <= list_idx < len(features):
                feat = features[list_idx][:, 1:]  # remove CLS token
                selected.append(feat)

        # (B, N, D*num_blocks)
        combined = torch.cat(selected, dim=-1)
        B, N, D = combined.shape
        h = w = int(N ** 0.5)
        return combined.transpose(1, 2).reshape(B, D, h, w)

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> float:
        """Train the linear head and return best validation mIoU."""
        optimizer = torch.optim.Adam(
            self.head.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        best_miou = 0.0

        for epoch in range(epochs):
            self.head.train()
            total_loss = 0.0
            n_batches = 0

            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                features = self.extract_features(images)
                targets = targets.to(self.device)

                # Resize targets to feature map size
                h, w = features.shape[2], features.shape[3]
                targets_small = F.interpolate(
                    targets.unsqueeze(1).float(), size=(h, w), mode="nearest"
                ).squeeze(1).long()

                logits = self.head(features)
                loss = criterion(logits, targets_small)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            miou = self.evaluate(val_loader)
            best_miou = max(best_miou, miou)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, mIoU={miou:.4f}")

        return best_miou

    @torch.no_grad()
    def evaluate(self, val_loader) -> float:
        """Evaluate and return mIoU."""
        self.head.eval()
        intersection = np.zeros(self.num_classes)
        union = np.zeros(self.num_classes)

        for images, targets in val_loader:
            features = self.extract_features(images)
            targets = targets.to(self.device)

            h, w = features.shape[2], features.shape[3]
            targets_small = F.interpolate(
                targets.unsqueeze(1).float(), size=(h, w), mode="nearest"
            ).squeeze(1).long()

            logits = self.head(features)
            preds = logits.argmax(dim=1)

            for c in range(self.num_classes):
                pred_c = preds == c
                target_c = targets_small == c
                intersection[c] += (pred_c & target_c).sum().item()
                union[c] += (pred_c | target_c).sum().item()

        ious = []
        for c in range(self.num_classes):
            if union[c] > 0:
                ious.append(intersection[c] / union[c])
        return np.mean(ious) if ious else 0.0
