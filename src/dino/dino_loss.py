"""
DINO loss: self-distillation with no labels (Caron et al., 2021).

The teacher produces soft targets by centering and sharpening its logits.
The student is trained to match these targets via cross-entropy, computed
only across *different* crop views (i.e., student crop i vs teacher crop j
for i != j).

Key design choices:
  - Teacher output is centered (EMA of mean teacher logits) and sharpened
    with a low temperature (linearly warmed up from warmup_teacher_temp to
    teacher_temp over the first warmup_teacher_temp_epochs).
  - Student output is sharpened with a higher temperature (student_temp).
  - The center is updated as an exponential moving average of teacher
    outputs, which prevents mode collapse.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Cross-entropy loss between centered/sharpened teacher and student.

    Parameters
    ----------
    out_dim : int
        Dimensionality of the projection head output.
    num_crops : int
        Total number of crops (global + local). The first 2 are always
        the global crops used for the teacher.
    warmup_teacher_temp : float
        Initial teacher temperature (low value = sharper).
    teacher_temp : float
        Final teacher temperature after warmup.
    warmup_teacher_temp_epochs : int
        Number of epochs to linearly warm up teacher temperature.
    num_epochs : int
        Total number of training epochs (used to build temp schedule).
    student_temp : float
        Student temperature (fixed throughout training).
    center_momentum : float
        EMA momentum for the center update.
    """

    def __init__(
        self,
        out_dim: int,
        num_crops: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 30,
        num_epochs: int = 300,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_crops = num_crops
        self.num_global_crops = 2  # always 2 global crops for the teacher

        # Register the center as a buffer (not a parameter).
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Build the teacher temperature schedule: linear warmup then constant.
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.full(max(0, num_epochs - warmup_teacher_temp_epochs), teacher_temp),
        ])

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Compute the DINO cross-entropy loss.

        Parameters
        ----------
        student_output : torch.Tensor
            Concatenated student logits for all crops, shape
            ``(batch_size * num_crops, out_dim)``.
        teacher_output : torch.Tensor
            Concatenated teacher logits for the global crops only, shape
            ``(batch_size * num_global_crops, out_dim)``.
        epoch : int
            Current epoch index (used to look up teacher temperature).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        student_out = student_output / self.student_temp
        # Split student output into per-crop chunks.
        student_out = student_out.chunk(self.num_crops)

        # Teacher: center, sharpen, and stop gradients.
        teacher_temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(
            (teacher_output - self.center) / teacher_temp, dim=-1
        )
        teacher_out = teacher_out.detach().chunk(self.num_global_crops)

        # Cross-entropy: for each teacher global crop, compute loss against
        # every *different* student crop.
        total_loss = 0.0
        num_loss_terms = 0
        for t_idx, t in enumerate(teacher_out):
            for s_idx, s in enumerate(student_out):
                if s_idx == t_idx:
                    # Skip same-view pairs (both are global crop indices 0, 1).
                    continue
                loss = -torch.sum(t * F.log_softmax(s, dim=-1), dim=-1)
                total_loss += loss.mean()
                num_loss_terms += 1

        total_loss /= num_loss_terms

        # Update the center with EMA of teacher outputs.
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Update center with exponential moving average of teacher outputs.

        Parameters
        ----------
        teacher_output : torch.Tensor
            Raw (un-centered) teacher logits, shape
            ``(batch_size * num_global_crops, out_dim)``.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # In distributed training, all-reduce the batch center.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center)
            batch_center /= torch.distributed.get_world_size()

        # EMA update.
        self.center = self.center * self.center_momentum + batch_center * (
            1.0 - self.center_momentum
        )
