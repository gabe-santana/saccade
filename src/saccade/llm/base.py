"""LLM connection interface.

Saccade's core never talks to the network. An LLM is used only when you configure one
explicitly and call :meth:`saccade.Video.ask`. Anything with a ``complete(messages)``
method works — the bundled clients speak the OpenAI chat-completions format, which
Azure AI Foundry, OpenAI, Ollama, LM Studio, vLLM and most gateways accept.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from saccade.exceptions import SaccadeError

Message = dict[str, Any]
"""An OpenAI-style chat message; ``content`` is a string or a list of text/image parts."""


class LLMError(SaccadeError):
    """The LLM endpoint rejected the request or could not be reached."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


@runtime_checkable
class LLM(Protocol):
    """Anything that can answer a list of chat messages."""

    @property
    def supports_images(self) -> bool: ...

    def complete(self, messages: list[Message], *, max_tokens: int = 8000) -> LLMResponse: ...


_RETRY_STATUS = {429, 500, 502, 503, 504}


def post_chat(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout: float = 300.0,
    retries: int = 2,
) -> LLMResponse:
    """POST an OpenAI-format chat request (standard library only), with retry on 429/5xx."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    shown = url.split("?", maxsplit=1)[0]
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            if exc.code in _RETRY_STATUS and attempt < retries:
                wait = float(exc.headers.get("Retry-After") or 2 ** (attempt + 1))
                time.sleep(min(wait, 30.0))
                continue
            raise LLMError(
                f"{shown} returned HTTP {exc.code}:\n{detail}", status=exc.code, body=detail
            ) from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise LLMError(f"Could not reach {shown}: {exc.reason}") from exc
    try:
        choice = payload["choices"][0]
        text = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMError(f"Unexpected response from {shown}: {str(payload)[:500]}") from exc
    return LLMResponse(
        text=str(text),
        model=payload.get("model"),
        usage=payload.get("usage") or {},
        finish_reason=choice.get("finish_reason"),
    )


def chat_with_fallback(
    url: str, headers: dict[str, str], body: dict[str, Any], *, max_tokens: int, timeout: float
) -> LLMResponse:
    """Send with ``max_completion_tokens`` (required by reasoning models such as GPT-5 and
    o-series); fall back to the older ``max_tokens`` for servers that reject it."""
    try:
        response = post_chat(
            url, headers, {**body, "max_completion_tokens": max_tokens}, timeout=timeout
        )
    except LLMError as exc:
        if exc.status != 400 or "max_completion_tokens" not in exc.body:
            raise
        response = post_chat(url, headers, {**body, "max_tokens": max_tokens}, timeout=timeout)
    if not response.text.strip() and response.finish_reason == "length":
        raise LLMError(
            f"The model used its whole output budget ({max_tokens} tokens) before answering — "
            "reasoning models spend tokens thinking first. Pass a larger max_answer_tokens."
        )
    return response
