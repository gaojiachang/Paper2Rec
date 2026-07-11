"""Shared utilities for reproducibility, persistence, and AMP setup."""

from __future__ import annotations

import hashlib
import json
import random
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def stable_seed(*parts: int) -> int:
    """Create a process-independent 63-bit seed from integer components."""
    digest = hashlib.blake2b(
        ":".join(str(int(part)) for part in parts).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def choose_device(requested: str | None = None) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    index = torch.cuda.current_device() if device.index is None else device.index
    return f"cuda:{index} ({torch.cuda.get_device_name(index)})"


class AmpController:
    """CUDA bf16/fp16 autocast with a no-op CPU fallback."""

    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.device = device
        self.enabled = enabled and device.type == "cuda"
        self.dtype: torch.dtype | None = None
        self.scaler: torch.amp.GradScaler | None = None
        if self.enabled:
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)

    @property
    def description(self) -> str:
        if not self.enabled:
            return "disabled"
        return "bf16" if self.dtype == torch.bfloat16 else "fp16"

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True)

    def backward_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def file_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
