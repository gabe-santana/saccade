"""Optional LLM connections used by :meth:`saccade.Video.ask`."""

from saccade.llm.base import LLM, LLMError, LLMResponse, Message
from saccade.llm.clients import AzureFoundry, OpenAICompatible, azure, ollama, openai

__all__ = [
    "LLM",
    "AzureFoundry",
    "LLMError",
    "LLMResponse",
    "Message",
    "OpenAICompatible",
    "azure",
    "ollama",
    "openai",
]
