"""Structured logging helper.

Saccade logs through the standard :mod:`logging` module under the ``saccade`` logger and
never installs handlers itself. Each record carries its fields both in the message
(``event key=value ...``) and as ``record.saccade`` for JSON formatters.
"""

from __future__ import annotations

import logging
from typing import Any

logging.getLogger("saccade").addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    if not logger.isEnabledFor(level):
        return
    rendered = " ".join(f"{key}={_render(value)}" for key, value in fields.items())
    logger.log(level, f"{event} {rendered}".rstrip(), extra={"saccade": {"event": event, **fields}})


def _render(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    return f'"{text}"' if " " in text else text
