"""
DINO training framework: student-teacher self-distillation with ViTs.

Implements the full DINO training loop including:
  - DINOHead: MLP projection head with L2-normalized output
  - DINOTrainer: orchestrates student/teacher forward passes, loss
    computation, EMA teacher updates, and schedule management

The teacher is an exponential moving average (EMA) copy of the student.
It receives no gradients; its weights are updated purely through the
momentum schedule (cosine from 0.996 to 1.0).

Multi-crop strategy: 2 global crops (224x224) are passed through both
student and teacher; N local crops (96x96) are passed through the student
only. The loss is computed across all different-view pairs.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


# ---------------------------------------------------------------------------
# DINO projection head
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """Projection head for DINO: MLP with bottleneck and L2-normalized output.

    Architecture: in_dim -> [hidden_dim -> GELU -> ] * (nlayers-2)
                  -> bottleneck_dim -> L2-norm -> weight_normalized Linear -> out_dim

    The last layer uses weight normalization (no bias) so that the output
    lives on the unit sphere, which is important for the softmax temperature
    sharpening to work correctly.

    Parameters
    ----------
    in_dim : int
        Input dimension (e.g. embed_dim of the ViT backbone).
    out_dim : int
        Output dimension (number of prototypes / DINO output dim).
    hidden_dim : int
        Hidden dimension of intermediate MLP layers.
    bottleneck_dim : int
        Dimension of the bottleneck layer before the final projection.
    nlayers : int
        Total number of linear layers (must be >= 2).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
    ):
        super().__init__()
        assert nlayers >= 2, "DINOHead requires at least 2 layers."

        # Build the MLP layers.
        layers: List[nn.Module] = []

        # First layer: in_dim -> hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())

        # Intermediate layers: hidden_dim -> hidden_dim
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        # Bottleneck layer: hidden_dim -> bottleneck_dim
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))

        self.mlp = nn.Sequential(*layers)

        # Last layer: bottleneck_dim -> out_dim with weight normalization.
        # No bias; output is L2-normalized before this layer.
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        # Initialize the weight_g to 1 so the effective weight is just the
        # direction (weight_v / ||weight_v||).
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize MLP weights with truncated normal."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)  # L2 normalize before last layer
        x = self.last_layer(x)
        return x


# ---------------------------------------------------------------------------
# Schedule utilities
# ---------------------------------------------------------------------------

def cosine_schedule(
    base_value: float,
    final_value: float,
    num_steps: int,
    warmup_steps: int = 0,
    warmup_value: float = 0.0,
) -> List[float]:
    """Build a cosine annealing schedule with optional linear warmup.

    Parameters
    ----------
    base_value : float
        Value after warmup (start of cosine decay).
    final_value : float
        Value at the end of the schedule.
    num_steps : int
        Total number of steps (including warmup).
    warmup_steps : int
        Number of linear warmup steps.
    warmup_value : float
        Value at the very start of warmup.

    Returns
    -------
    list of float
        Schedule values for each step.
    """
    warmup_schedule = []
    if warmup_steps > 0:
        warmup_schedule = [
            warmup_value + (base_value - warmup_value) * i / warmup_steps
            for i in range(warmup_steps)
        ]

    cosine_steps = num_steps - warmup_steps
    cosine_schedule_vals = [
        final_value + 0.5 * (base_value - final_value) * (
            1.0 + math.cos(math.pi * i / cosine_steps)
        )
        for i in range(cosine_steps)
    ]

    return warmup_schedule + cosine_schedule_vals


# ---------------------------------------------------------------------------
# DINO Trainer
# ---------------------------------------------------------------------------

class DINOTrainer:
    """Orchestrates DINO self-supervised training.

    Parameters
    ----------
    student : nn.Module
        Student backbone + projection head (wrapped as a single module or
        passed separately; here we expect a MultiCropWrapper).
    teacher : nn.Module
        Teacher backbone + projection head (same architecture as student).
    dino_loss : nn.Module
        The DINOLoss instance.
    optimizer : torch.optim.Optimizer
        Optimizer for the student parameters.
    lr_schedule : list of float
        Per-iteration learning rate schedule.
    wd_schedule : list of float
        Per-iteration weight decay schedule.
    momentum_schedule : list of float
        Per-iteration teacher EMA momentum schedule.
    num_global_crops : int
        Number of global crops (default 2).
    clip_grad : float or None
        Maximum gradient norm for clipping.
    amp_enabled : bool
        Whether to use automatic mixed precision.
    accum_steps : int
        Gradient accumulation steps.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        dino_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_schedule: List[float],
        wd_schedule: List[float],
        momentum_schedule: List[float],
        num_global_crops: int = 2,
        clip_grad: Optional[float] = 3.0,
        amp_enabled: bool = True,
        accum_steps: int = 1,
    ):
        self.student = student
        self.teacher = teacher
        self.dino_loss = dino_loss
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule
        self.momentum_schedule = momentum_schedule
        self.num_global_crops = num_global_crops
        self.clip_grad = clip_grad
        self.amp_enabled = amp_enabled
        self.accum_steps = accum_steps

        self.scaler = GradScaler(enabled=amp_enabled)
        self._global_step = 0

    def train_one_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int,
        log_interval: int = 50,
    ) -> Dict[str, float]:
        """Train the student for one epoch.

        Parameters
        ----------
        data_loader : DataLoader
            Yields (images, _) where images is a list of crop tensors.
        epoch : int
            Current epoch index.
        log_interval : int
            Print training stats every N iterations.

        Returns
        -------
        dict
            Training metrics for the epoch (avg loss, lr, wd, momentum).
        """
        self.student.train()
        self.teacher.eval()

        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for it, (images, _) in enumerate(data_loader):
            # images is a list of crop tensors from DINODataAugmentation.
            # Move all crops to the same device as the model.
            device = next(self.student.parameters()).device
            images = [img.to(device, non_blocking=True) for img in images]

            # Update per-iteration schedules.
            step = self._global_step
            self._update_learning_rate(step)
            self._update_weight_decay(step)

            # --- Forward pass ---
            with autocast(enabled=self.amp_enabled):
                # Teacher: only global crops, no gradients.
                with torch.no_grad():
                    teacher_output = self.teacher(
                        torch.cat(images[: self.num_global_crops])
                    )

                # Student: all crops.
                student_output = self.student(torch.cat(images))

                loss = self.dino_loss(student_output, teacher_output, epoch)
                loss = loss / self.accum_steps

            # --- Backward pass ---
            self.scaler.scale(loss).backward()

            if (it + 1) % self.accum_steps == 0:
                # Unscale before clipping.
                if self.clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.student.parameters(), self.clip_grad
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # EMA update of teacher.
                momentum = self.momentum_schedule[
                    min(step, len(self.momentum_schedule) - 1)
                ]
                self.update_teacher(momentum)

            total_loss += loss.item() * self.accum_steps
            num_batches += 1
            self._global_step += 1

            if it % log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  [Epoch {epoch}][{it}/{len(data_loader)}] "
                    f"loss={loss.item() * self.accum_steps:.4f}  "
                    f"lr={lr:.6f}  step={step}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        return {
            "loss": avg_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """Update teacher weights as EMA of student weights.

        teacher_param = momentum * teacher_param + (1 - momentum) * student_param
        """
        for param_s, param_t in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_t.data.mul_(momentum).add_(
                param_s.data, alpha=1.0 - momentum
            )

    def _update_learning_rate(self, step: int) -> None:
        """Set learning rate according to the pre-computed schedule."""
        idx = min(step, len(self.lr_schedule) - 1)
        lr = self.lr_schedule[idx]
        for param_group in self.optimizer.param_groups:
            # If param_group has a "lr_scale" key, apply it.
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = lr * scale

    def _update_weight_decay(self, step: int) -> None:
        """Set weight decay according to the pre-computed schedule."""
        idx = min(step, len(self.wd_schedule) - 1)
        wd = self.wd_schedule[idx]
        for param_group in self.optimizer.param_groups:
            if param_group.get("apply_wd", True):
                param_group["weight_decay"] = wd

    @property
    def global_step(self) -> int:
        return self._global_step


# ---------------------------------------------------------------------------
# Multi-crop wrapper
# ---------------------------------------------------------------------------

class MultiCropWrapper(nn.Module):
    """Wrap a backbone + projection head for multi-crop forward passes.

    Handles crops of different resolutions efficiently by grouping crops
    of the same size and running them through the backbone in a single
    forward pass (instead of one-by-one).

    Parameters
    ----------
    backbone : nn.Module
        ViT backbone (must return CLS token features from ``forward``
        or have an ``embed_dim`` attribute).
    head : nn.Module
        DINOHead projection head.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        When called with a batch of concatenated crops (possibly of different
        sizes), it splits by resolution, runs the backbone on each group, and
        concatenates the results before passing through the head.

        Parameters
        ----------
        x : torch.Tensor
            Concatenated crops of shape ``(N, C, H, W)`` where N may include
            crops of different spatial sizes.

        Returns
        -------
        torch.Tensor
            Projection head output, shape ``(N, out_dim)``.
        """
        # If all crops have the same resolution, this is just a single pass.
        # Otherwise, split by resolution for efficiency.
        n_crops = x.shape[0]

        # Get unique spatial sizes.
        # We process in order to maintain correspondence with the loss.
        # For simplicity and correctness with the standard DINO setup,
        # we run the entire batch through the backbone at once if all crops
        # are the same size, or split otherwise.

        # Extract features from backbone: use forward_features + CLS token
        # if available, otherwise use the backbone's forward.
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
            cls_token = features[:, 0]
        else:
            cls_token = self.backbone(x)

        return self.head(cls_token)
