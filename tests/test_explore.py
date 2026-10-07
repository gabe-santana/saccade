"""Exploration mode: the LLM navigates the video with tools."""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest

from saccade.explore import TOOLS
from saccade.llm.base import LLMResponse, ToolCall, _tool_calls
from saccade.media.frames import grab_frame
from support import ToneASR, pattern, place, slides, tone, write_video

TEXTS = {440: "Welcome everyone, let's begin.", 880: "What are your salary expectations?"}


def jpeg_pixels(data: bytes) -> np.ndarray:
    packet = av.Packet(data)
    codec = av.CodecContext.create("mjpeg", "r")
    frames = codec.decode(packet)
    return frames[0].to_ndarray(format="rgb24")


@pytest.fixture
def meeting(make_video, tmp_path: Path):
    audio = place(40.0, [(2.0, tone(440, 3.0)), (25.0, tone(880, 2.0))])
    deck = slides([(0, 10, 1), (10, 20, 2), (20, 40, 3)])
    path = write_video(tmp_path / "meeting.mp4", deck, audio=audio)
    return lambda **kw: make_video(path, ToneASR(texts=TEXTS), **kw)


def test_grab_frame_returns_the_frame_on_screen(meeting) -> None:
    video = meeting()
    early = grab_frame(video.path, 5.0)
    late = grab_frame(video.path, 15.0)
    assert early.time == pytest.approx(5.0, abs=0.11)
    assert late.time == pytest.approx(15.0, abs=0.11)
    for grabbed, seed in ((early, 1), (late, 2)):
        expected = pattern(seed).astype(float)
        got = jpeg_pixels(grabbed.jpeg).astype(float)
        assert np.abs(got - expected).mean() < 12, "decoded the slide shown at that time"


def test_grab_frame_zoom_crops_at_native_resolution(meeting) -> None:
    video = meeting()
    full = grab_frame(video.path, 12.34)
    assert 12.2 <= full.time <= 12.34
    zoom = grab_frame(video.path, 12.34, box=(0.5, 0.5, 0.5, 0.5))
    assert (zoom.width, zoom.height) == (80, 48)
    expected = pattern(2)[48:, 80:].astype(float)
    assert np.abs(jpeg_pixels(zoom.jpeg).astype(float) - expected).mean() < 12


class ScriptedLLM:
    """Replays tool calls, then answers; records what it was sent."""

    supports_images = True
    supports_tools = True

    def __init__(self, script: list[list[ToolCall]]) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def complete(self, messages, *, max_tokens=8000, tools=None, tool_choice=None):
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "tool_choice": tool_choice}
        )
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 60},
            "completion_tokens_details": {"reasoning_tokens": 4},
        }
        if tool_choice != "none" and self.script:
            calls = tuple(self.script.pop(0))
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in calls
                ],
            }
            return LLMResponse(
                text="", usage=usage, tool_calls=calls, message=message, finish_reason="tool_calls"
            )
        return LLMResponse(text="The candidate was calm [12:00.0].", model="scripted", usage=usage)


def test_exploration_loop(meeting) -> None:
    llm = ScriptedLLM(
        [
            [
                ToolCall("a", "view_frames", {"timestamps": [12.0, 30.0]}),
                ToolCall("b", "search_transcript", {"query": "salary"}),
            ],
            [
                ToolCall(
                    "c", "zoom", {"timestamp": 12.0, "x": 0, "y": 0, "width": 0.5, "height": 0.5}
                )
            ],
        ]
    )
    events = []
    video = meeting(llm=llm)
    answer = video.ask("How did the candidate look?", progress=events.append)

    assert str(answer) == "The candidate was calm [12:00.0]."
    assert answer.steps == [
        "Looking at 0:12.0, 0:30.0",
        "Searching the transcript for 'salary'",
        "Zooming into 0:12.0",
    ]
    views = [f for f in answer.frames if f.id.startswith("view_")]
    assert [f.reason for f in views] == ["view", "view", "zoom"]
    assert views[0].timestamp == pytest.approx(12.0, abs=0.11)
    assert all(Path(f.path).is_file() for f in views)
    assert {"view_001", "view_002", "view_003"} <= {e.id for e in answer.evidence}
    assert answer.total_tokens == 3 * 110, "usage is summed over every round"
    assert answer.cached_tokens == 3 * 60 and answer.reasoning_tokens == 3 * 4
    assert any(e.stage.value == "explore" and "Zooming" in e.message for e in events)

    first, second, third = llm.calls
    assert first["tools"] == TOOLS and first["tool_choice"] == "auto"
    sent = second["messages"]
    tool_messages = [m for m in sent if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"a", "b"}
    assert TEXTS[880] in next(m["content"] for m in tool_messages if m["tool_call_id"] == "b")
    images = [p for p in sent[-1]["content"] if p["type"] == "image_url"]
    assert len(images) == 2 and all(p["image_url"]["detail"] == "high" for p in images)
    assert "zoom" in str(third["messages"][-1]["content"][1])
    assert "LOOK before you answer" in sent[0]["content"]


def test_exploration_stops_at_step_limit(meeting) -> None:
    endless = [[ToolCall(str(i), "view_frames", {"timestamps": [1.0]})] for i in range(10)]
    llm = ScriptedLLM(endless)
    answer = meeting(llm=llm).ask("?", max_steps=2)
    assert len(answer.steps) == 2
    assert llm.calls[-1]["tool_choice"] == "none", "the last round must answer"
    assert answer.text


def test_bad_tool_arguments_are_reported_not_raised(meeting) -> None:
    llm = ScriptedLLM([[ToolCall("x", "zoom", {"timestamp": 3}), ToolCall("y", "dance", {})]])
    answer = meeting(llm=llm).ask("?")
    results = [m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert results[0].startswith("Invalid arguments for zoom")
    assert results[1] == "Unknown tool 'dance'."
    assert answer.text


def test_explore_can_be_disabled(meeting) -> None:
    llm = ScriptedLLM([[ToolCall("a", "view_frames", {"timestamps": [1.0]})]])
    answer = meeting(llm=llm).ask("?", explore=False)
    assert answer.steps == []
    assert llm.calls[0]["tools"] is None


def test_tool_call_parsing() -> None:
    message = {
        "tool_calls": [
            {"id": "1", "function": {"name": "zoom", "arguments": '{"timestamp": 3.5, "x": 0.1}'}},
            {"id": "2", "function": {"name": "view_frames", "arguments": "not json"}},
        ]
    }
    calls = _tool_calls(message)
    assert calls[0] == ToolCall("1", "zoom", {"timestamp": 3.5, "x": 0.1})
    assert calls[1].arguments == {}
