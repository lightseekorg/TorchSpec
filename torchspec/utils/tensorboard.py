import logging
import os
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


_WRITER: SummaryWriter | None = None


def init_tensorboard(args) -> None:
    global _WRITER
    if not getattr(args, "use_tensorboard", False):
        _WRITER = None
        return
    if _WRITER is not None:
        return

    log_dir = getattr(args, "tensorboard_dir", None)
    if not log_dir:
        output_dir = getattr(args, "output_dir", None)
        log_dir = str(Path(output_dir) / "tensorboard") if output_dir else "./tensorboard"

    os.makedirs(log_dir, exist_ok=True)
    _WRITER = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs will be stored in: {log_dir}")


def log_metrics(metrics: dict[str, Any]) -> None:
    if _WRITER is None or not metrics:
        return

    step = metrics.get("train/step", metrics.get("eval/step"))
    if step is None:
        return

    for key, value in metrics.items():
        if key in {"train/step", "eval/step", "inference/step"}:
            continue
        if isinstance(value, bool):
            _WRITER.add_scalar(key, int(value), step)
        elif isinstance(value, (int, float)):
            _WRITER.add_scalar(key, value, step)


def finish_tensorboard() -> None:
    global _WRITER
    if _WRITER is None:
        return
    _WRITER.flush()
    _WRITER.close()
    _WRITER = None
