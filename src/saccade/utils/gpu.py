"""CUDA discovery for CTranslate2, including NVIDIA's pip-installed libraries.

``pip install "saccade-video[gpu]"`` installs cuBLAS and cuDNN as wheels (``nvidia-*-cu12``).
Their DLLs/shared objects live inside site-packages, where the OS loader does not look,
so they are registered here before CTranslate2 needs them. No CUDA toolkit install is
required — only an NVIDIA driver.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import os
import sys
from pathlib import Path
from typing import Literal

from saccade.exceptions import ConfigError
from saccade.utils.logs import get_logger, log_event

logger = get_logger(__name__)

Device = Literal["cpu", "cuda", "auto"]

_LINUX_LIBS = ("libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9")


def _nvidia_roots() -> list[Path]:
    roots = []
    for entry in sys.path:
        candidate = Path(entry) / "nvidia"
        if candidate.is_dir():
            roots.append(candidate)
    return roots


@functools.cache
def register_cuda_libraries() -> int:
    """Make pip-installed CUDA libraries loadable; returns how many directories were added.

    Process-wide by nature (it configures the dynamic loader), so it runs once.
    """
    added = 0
    for root in _nvidia_roots():
        for package in sorted(p for p in root.iterdir() if p.is_dir()):
            if sys.platform == "win32":
                folder = package / "bin"
                if folder.is_dir():
                    os.add_dll_directory(str(folder))
                    os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
                    added += 1
            else:
                folder = package / "lib"
                for name in _LINUX_LIBS:
                    library = folder / name
                    if library.exists():
                        # Preloading by full path lets CTranslate2's later dlopen(soname) succeed.
                        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                        added += 1
    if added:
        log_event(logger, logging.DEBUG, "cuda.libraries", directories=added)
    return added


@functools.cache
def cuda_device_count() -> int:
    try:
        import ctranslate2
    except ImportError:  # pragma: no cover - hard dependency through faster-whisper
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def resolve_device(device: str) -> Literal["cpu", "cuda"]:
    """``"auto"`` → ``"cuda"`` when an NVIDIA GPU is usable, else ``"cpu"``."""
    device = device.lower()
    if device == "cpu":
        return "cpu"
    if device not in ("cuda", "auto"):
        raise ConfigError(f"Unknown device {device!r}. Use 'cpu', 'cuda' or 'auto'.")
    available = cuda_device_count() > 0
    if device == "auto":
        if available:
            register_cuda_libraries()
            return "cuda"
        return "cpu"
    if not available:
        raise ConfigError(
            "device='cuda' was requested but no CUDA-capable NVIDIA GPU is visible. "
            "Check the NVIDIA driver (nvidia-smi), or use device='auto' to fall back to the CPU."
        )
    register_cuda_libraries()
    return "cuda"


def missing_library_hint(error: BaseException) -> str | None:
    """An actionable message when CUDA libraries are missing at model load/run time."""
    text = str(error)
    if any(name in text for name in ("cublas", "cudnn", "cudart", ".dll", ".so")):
        return (
            f"{text}\n\nThe GPU needs NVIDIA's CUDA 12 libraries (cuBLAS, cuDNN 9). Install them with:\n"
            '    pip install "saccade-video[gpu]"\n'
            "or use device='cpu'."
        )
    return None
