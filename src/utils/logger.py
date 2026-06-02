"""Simple training logger with TensorBoard support."""

import os
import sys
import logging
from datetime import datetime


class TrainingLogger:
    """Logger that writes to console and optionally to TensorBoard."""

    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Console + file logger
        self.logger = logging.getLogger("training")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)

        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                self.logger.warning("TensorBoard not available, logging to file only.")

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_value_dict: dict, step: int):
        for tag, value in tag_value_dict.items():
            self.log_scalar(tag, value, step)

    def info(self, msg: str):
        self.logger.info(msg)

    def close(self):
        if self.writer is not None:
            self.writer.close()
