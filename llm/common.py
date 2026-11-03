"""Small shared helpers: device selection, seeding, memory, and data metadata."""

from __future__ import annotations

import json
import os
import random
import resource
import sys
from pathlib import Path

import numpy as np
import torch


def pick_device(requested: str = "auto") -> torch.device:
    """CUDA if present, else Apple MPS, else CPU. Override with an explicit device name."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tune_malloc() -> None:
    """On glibc, stop the allocator returning large temporaries to the OS after every step.

    Training allocates and frees (tokens x vocab) logits buffers each step; without this,
    the kernel spends a large share of CPU time on page faults (~1.6x slower on a 4-core box).
    No-op on macOS and other non-glibc systems.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 32 * 1024 * 1024)  # M_MMAP_THRESHOLD: keep blocks up to 32 MB on the heap
        libc.mallopt(-1, 1024 ** 3)         # M_TRIM_THRESHOLD: do not trim freed memory eagerly
        libc.mallopt(-2, 256 * 1024 ** 2)   # M_TOP_PAD: grow the heap in large steps
    except (OSError, AttributeError):
        pass


def sync(device: torch.device) -> None:
    """Block until queued device work finishes, so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def peak_rss_mb() -> float:
    """Peak resident memory of this process. ru_maxrss is bytes on macOS, KiB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def load_meta(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text())


def cpu_threads() -> int:
    return os.cpu_count() or 4
