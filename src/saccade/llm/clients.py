"""Ready-made LLM connections: Azure AI Foundry and any OpenAI-compatible endpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from saccade.exceptions import ConfigError
from saccade.llm.base import LLMResponse, Message, chat_with_fallback

_AZURE_API_VERSION = "2024-10-21"
_AZURE_INFERENCE_API_VERSION = "2024-05-01-preview"


@dataclass(frozen=True)
class OpenAICompatible:
    """Any endpoint that implements ``POST {base_url}/chat/completions``.

    Examples: ``https://api.openai.com/v1``, Ollama ``http://localhost:11434/v1``,
    LM Studio ``http://localhost:1234/v1``, vLLM, LiteLLM, OpenRouter.
    """

    base_url: str
    model: str
    api_key: str | None = None
    images: bool = True
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 300.0

    @property
    def supports_images(self) -> bool:
        return self.images

    def complete(self, messages: list[Message], *, max_tokens: int = 8000) -> LLMResponse:
        headers = dict(self.headers)
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        return chat_with_fallback(url, headers, body, max_tokens=max_tokens, timeout=self.timeout)

    def __repr__(self) -> str:
        return f"OpenAICompatible(base_url={self.base_url!r}, model={self.model!r})"


@dataclass(frozen=True)
class AzureFoundry:
    """A model deployed in Azure AI Foundry (or Azure OpenAI).

    ``endpoint`` may be any URL the Foundry portal shows for the deployment:

    * ``https://<resource>.openai.azure.com`` or ``https://<resource>.services.ai.azure.com``
    * ``.../openai/v1`` or ``.../openai/v1/responses`` (the v1 API)
    * ``https://<resource>.services.ai.azure.com/models`` (model inference API)
    * the full *Target URI* ending in ``/chat/completions?api-version=...``
    """

    endpoint: str
    api_key: str
    deployment: str
    images: bool = True
    timeout: float = 300.0

    def __post_init__(self) -> None:
        if not self.endpoint or not self.api_key:
            raise ConfigError("AzureFoundry needs an endpoint and an api_key.")
        if not self.deployment and "/deployments/" not in self.endpoint:
            raise ConfigError(
                "AzureFoundry needs the deployment (model) name, e.g. deployment='gpt-4o'."
            )

    @property
    def supports_images(self) -> bool:
        return self.images

    def _url(self) -> tuple[str, bool]:
        """Chat-completions URL, and whether the body must name the model."""
        url = self.endpoint.strip()
        if "/chat/completions" in url:
            return url, "/deployments/" not in url
        base = url.rstrip("/")
        if base.endswith("/models"):
            return f"{base}/chat/completions?api-version={_AZURE_INFERENCE_API_VERSION}", True
        if "/openai/v1" in base:  # v1 API, including the /responses URL Foundry shows
            v1 = base[: base.index("/openai/v1") + len("/openai/v1")]
            return f"{v1}/chat/completions", True
        parts = urlsplit(base)
        root = f"{parts.scheme}://{parts.netloc}"  # drop e.g. /api/projects/<name>
        return (
            f"{root}/openai/deployments/{self.deployment}/chat/completions?api-version={_AZURE_API_VERSION}",
            False,
        )

    def complete(self, messages: list[Message], *, max_tokens: int = 8000) -> LLMResponse:
        url, needs_model = self._url()
        body: dict[str, Any] = {"messages": messages}
        if needs_model:
            body["model"] = self.deployment
        headers = {"api-key": self.api_key}
        return chat_with_fallback(url, headers, body, max_tokens=max_tokens, timeout=self.timeout)

    def __repr__(self) -> str:  # never print the key
        return f"AzureFoundry(endpoint={self.endpoint!r}, deployment={self.deployment!r})"


def azure(
    endpoint: str | None = None,
    api_key: str | None = None,
    deployment: str | None = None,
    *,
    images: bool = True,
) -> AzureFoundry:
    """Connect to an Azure AI Foundry deployment.

    Arguments default to the environment variables ``AZURE_AI_ENDPOINT``,
    ``AZURE_AI_API_KEY`` and ``AZURE_AI_DEPLOYMENT``.
    """
    endpoint = endpoint or os.environ.get("AZURE_AI_ENDPOINT", "")
    api_key = api_key or os.environ.get("AZURE_AI_API_KEY", "")
    deployment = deployment or os.environ.get("AZURE_AI_DEPLOYMENT", "")
    if not endpoint or not api_key:
        raise ConfigError(
            "Azure AI Foundry needs an endpoint and key: saccade.azure(endpoint=..., api_key=..., "
            "deployment=...) or set AZURE_AI_ENDPOINT / AZURE_AI_API_KEY / AZURE_AI_DEPLOYMENT."
        )
    return AzureFoundry(endpoint=endpoint, api_key=api_key, deployment=deployment, images=images)


def openai(
    model: str = "gpt-4o",
    api_key: str | None = None,
    *,
    base_url: str = "https://api.openai.com/v1",
) -> OpenAICompatible:
    """Connect to OpenAI (``OPENAI_API_KEY`` by default) or another compatible server."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    return OpenAICompatible(base_url=base_url, model=model, api_key=key)


def ollama(
    model: str, *, base_url: str = "http://localhost:11434/v1", images: bool = False
) -> OpenAICompatible:
    """A local model served by Ollama — keeps the whole workflow on your machine."""
    return OpenAICompatible(base_url=base_url, model=model, images=images)
