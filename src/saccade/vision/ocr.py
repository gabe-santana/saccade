"""Optional local OCR using RapidOCR's bundled ONNX models."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from saccade.config import VisualConfig
from saccade.exceptions import ConfigError


def installed_version() -> str | None:
    try:
        return version("rapidocr-onnxruntime")
    except PackageNotFoundError:
        return None


def ocr_identity(config: VisualConfig) -> dict[str, str] | None:
    if config.ocr is False or config.strategy == "off":
        return None
    available = installed_version()
    if available is None:
        if config.ocr is True:
            raise ConfigError('Install OCR support: pip install "saccade-video[ocr]".')
        return None
    return {"engine": "rapidocr-onnxruntime", "version": available}


class FrameOCR:
    """Recognize text from one stored JPEG at a time, loading the engine on first use."""

    def __init__(self, score: float) -> None:
        self.score = score
        self._engine: Any = None

    def __call__(self, jpeg: bytes) -> str:
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise ConfigError('Install OCR support: pip install "saccade-video[ocr]".') from exc
            self._engine = RapidOCR(
                text_score=self.score,
                intra_op_num_threads=1,
                inter_op_num_threads=1,
                det_use_cuda=False,
                cls_use_cuda=False,
                rec_use_cuda=False,
            )
        rows, _ = self._engine(jpeg)
        if rows is None:
            return ""
        return "\n".join(
            str(text).strip()
            for _, text, score in rows
            if float(score) >= self.score and str(text).strip()
        )
