"""video.ask(): evidence selection, images, and the LLM HTTP clients (against a local server)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

import saccade
from saccade import ConfigError, LLMError
from saccade.llm import AzureFoundry, LLMResponse
from support import ToneASR, place, slides, tone, write_video

TEXTS = {
    440: "Tell me about your experience with distributed systems.",
    660: "I led the migration of our payment platform to Kubernetes.",
    880: "What are your salary expectations?",
}


class FakeLLM:
    def __init__(self, images: bool = True) -> None:
        self.images = images
        self.calls: list[list[dict]] = []

    @property
    def supports_images(self) -> bool:
        return self.images

    def complete(self, messages, *, max_tokens=1500, temperature=0.2):
        self.calls.append(messages)
        return LLMResponse(
            text="  The interview went well [0:09.0].  ", model="fake-1", usage={"total_tokens": 42}
        )


@pytest.fixture
def interview(make_video, tmp_path: Path):
    audio = place(40.0, [(2.0, tone(440, 3.0)), (9.0, tone(660, 4.0)), (25.0, tone(880, 2.0))])
    path = write_video(
        tmp_path / "entrevista ü.mp4", slides([(0, 15, 1), (15, 30, 2), (30, 40, 3)]), audio=audio
    )
    return lambda **kw: make_video(path, ToneASR(texts=TEXTS), **kw)


def test_ask_sends_full_transcript_and_frames(interview) -> None:
    llm = FakeLLM()
    video = interview(llm=llm)
    answer = video.ask("How was the interview?")

    assert str(answer) == "The interview went well [0:09.0]."
    assert (
        answer.mode == "full" and answer.model == "fake-1" and answer.usage == {"total_tokens": 42}
    )
    system, user = llm.calls[0]
    assert system["role"] == "system" and "ONLY the evidence" in system["content"]
    texts = "\n".join(p["text"] for p in user["content"] if p["type"] == "text")
    images = [p for p in user["content"] if p["type"] == "image_url"]
    for line in TEXTS.values():
        assert line in texts
    assert "QUESTION: How was the interview?" in texts
    assert len(images) == len(answer.frames) >= 2
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    for frame in answer.frames:
        assert f"| {frame.id}]" in texts
    kinds = {e.type for e in answer.evidence}
    assert kinds == {"transcript", "frame"}
    assert json.loads(answer.to_json())["mode"] == "full"


def test_ask_indexes_on_first_use_and_reuses_cache(interview) -> None:
    llm = FakeLLM()
    events = []
    interview(llm=llm).ask("How was it?", progress=events.append)
    assert any(e.stage is saccade.Stage.TRANSCRIBE for e in events)
    backend_video = interview(llm=llm)
    backend_video.ask("Another question")
    assert backend_video._custom_backend.calls == 0, "the second question never re-transcribes"


def test_ask_retrieves_when_transcript_is_too_long(make_video, tmp_path: Path) -> None:
    long_texts = {
        440: "We talked about the weather and the traffic on the way here. " * 40,
        660: "Then a long story about previous jobs and many different projects. " * 40,
        880: "What are your salary expectations?",
    }
    audio = place(40.0, [(2.0, tone(440, 3.0)), (9.0, tone(660, 4.0)), (25.0, tone(880, 2.0))])
    path = write_video(tmp_path / "long.mp4", slides([(0, 40, 1)]), audio=audio)
    llm = FakeLLM()
    video = make_video(path, ToneASR(texts=long_texts), llm=llm)
    answer = video.ask("salary expectations", evidence_tokens=400)
    assert answer.mode == "retrieved"
    texts = "\n".join(p["text"] for p in llm.calls[0][1]["content"] if p["type"] == "text")
    assert long_texts[880].strip() in texts
    assert "weather" not in texts


def test_ask_text_only_llm_gets_plain_string(interview) -> None:
    llm = FakeLLM(images=False)
    answer = interview(llm=llm).ask("How was the interview?")
    assert answer.frames == []
    assert isinstance(llm.calls[0][1]["content"], str)


def test_ask_without_llm_is_actionable(interview) -> None:
    with pytest.raises(ConfigError, match=r"saccade\.azure"):
        interview().ask("anything")


def test_llm_override_and_max_images(interview) -> None:
    default, override = FakeLLM(), FakeLLM()
    answer = interview(llm=default).ask("?", llm=override, max_images=1)
    assert not default.calls and override.calls
    assert len(answer.frames) == 1


# -- HTTP clients ---------------------------------------------------------------------------


class _Recorder(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        headers = {k.lower(): v for k, v in self.headers.items()}
        type(self).requests.append({"path": self.path, "headers": headers, "body": body})
        if type(self).status != 200:
            self.send_response(type(self).status)
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "invalid key"}}')
            return
        data = json.dumps(
            {
                "model": "gpt-4o-2024",
                "choices": [{"message": {"content": "olá ✓"}}],
                "usage": {"total_tokens": 7},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Recorder.requests = []
    _Recorder.status = 200
    httpd = HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", _Recorder
    httpd.shutdown()


MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    ("suffix", "path", "model_in_body"),
    [
        ("", "/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21", False),
        (
            "/api/projects/my-project",
            "/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
            False,
        ),
        ("/openai/v1/", "/openai/v1/chat/completions", True),
        ("/openai/v1/responses", "/openai/v1/chat/completions", True),
        ("/models", "/models/chat/completions?api-version=2024-05-01-preview", True),
        (
            "/openai/deployments/x/chat/completions?api-version=2025-01-01-preview",
            "/openai/deployments/x/chat/completions?api-version=2025-01-01-preview",
            False,
        ),
    ],
)
def test_azure_endpoint_shapes(server, suffix: str, path: str, model_in_body: bool) -> None:
    base, recorder = server
    llm = saccade.azure(endpoint=base + suffix, api_key="secret", deployment="gpt-4o")
    response = llm.complete(MESSAGES)
    request = recorder.requests[-1]
    assert request["path"] == path
    assert request["headers"]["api-key"] == "secret"
    assert ("model" in request["body"]) is model_in_body
    assert response.text == "olá ✓" and response.usage == {"total_tokens": 7}


def test_azure_reads_environment(monkeypatch, server) -> None:
    base, _ = server
    monkeypatch.setenv("AZURE_AI_ENDPOINT", base)
    monkeypatch.setenv("AZURE_AI_API_KEY", "k")
    monkeypatch.setenv("AZURE_AI_DEPLOYMENT", "dep")
    llm = saccade.azure()
    assert isinstance(llm, AzureFoundry) and llm.deployment == "dep"
    assert "k" not in repr(llm), "keys are never printed"
    monkeypatch.delenv("AZURE_AI_ENDPOINT")
    with pytest.raises(ConfigError, match="AZURE_AI_ENDPOINT"):
        saccade.azure()


def test_openai_compatible_and_errors(server) -> None:
    base, recorder = server
    llm = saccade.OpenAICompatible(base_url=base + "/v1", model="llama3", api_key="tok")
    assert llm.complete(MESSAGES).text == "olá ✓"
    assert recorder.requests[-1]["path"] == "/v1/chat/completions"
    assert recorder.requests[-1]["headers"]["authorization"] == "Bearer tok"
    assert recorder.requests[-1]["body"]["model"] == "llama3"
    recorder.status = 401
    with pytest.raises(LLMError, match="HTTP 401"):
        llm.complete(MESSAGES)


def test_unreachable_endpoint() -> None:
    llm = saccade.OpenAICompatible(base_url="http://127.0.0.1:9/v1", model="m", timeout=2)
    with pytest.raises(LLMError, match="Could not reach"):
        llm.complete(MESSAGES)


def test_reasoning_model_parameters(server) -> None:
    base, recorder = server
    saccade.azure(endpoint=base, api_key="k", deployment="gpt-5-mini").complete(
        MESSAGES, max_tokens=900
    )
    body = recorder.requests[-1]["body"]
    assert body["max_completion_tokens"] == 900
    assert "max_tokens" not in body and "temperature" not in body


def test_answer_token_usage(interview) -> None:
    answer = interview(llm=FakeLLM()).ask("How was it?")
    assert answer.total_tokens == 42
    usage = saccade.Answer(
        "q", "a", [], [], "full", usage={"prompt_tokens": 30, "completion_tokens": 12}
    )
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (30, 12, 42)


def test_clients_send_tools(server) -> None:
    from saccade.explore import TOOLS

    base, recorder = server
    llm = saccade.azure(endpoint=base, api_key="k", deployment="gpt-5-mini")
    assert llm.supports_tools
    llm.complete(MESSAGES, tools=TOOLS, tool_choice="none")
    body = recorder.requests[-1]["body"]
    assert body["tools"] == TOOLS and body["tool_choice"] == "none"
    llm.complete(MESSAGES)
    assert "tools" not in recorder.requests[-1]["body"]
    assert not saccade.ollama("llama3.1").supports_tools


def test_reasoning_effort_is_sent_only_when_set(server) -> None:
    base, recorder = server
    saccade.azure(
        endpoint=base, api_key="k", deployment="gpt-5-mini", reasoning_effort="low"
    ).complete(MESSAGES)
    assert recorder.requests[-1]["body"]["reasoning_effort"] == "low"
    saccade.azure(endpoint=base, api_key="k", deployment="gpt-4o").complete(MESSAGES)
    assert "reasoning_effort" not in recorder.requests[-1]["body"]
