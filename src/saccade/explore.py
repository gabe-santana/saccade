"""Agentic question answering: the LLM navigates the video until it can answer.

The model starts from the same overview as a single-shot :func:`saccade.ask.ask` — the
timestamped transcript and small thumbnails of representative frames — and can then call
tools to gather more evidence:

* ``view_frames(timestamps)``: the exact frames at those moments, in high resolution;
* ``zoom(timestamp, x, y, width, height)``: a region of one frame at the video's native
  resolution (a participant's tile, a slide's small print, a terminal);
* ``search_transcript(query)``: when something was said;
* ``read_transcript(start, end)``: the verbatim transcript of a time range.

Frames are decoded on demand from the source file (seek + decode, well under a second), so
the model can look at any moment, not just the stored representative frames. Every image
it saw is saved next to the index and returned as evidence with its exact timestamp.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from saccade.ask import build_messages
from saccade.media.frames import Box, grab_frame
from saccade.models.answer import Answer
from saccade.models.context import Evidence
from saccade.models.frame import Frame
from saccade.progress import Reporter, Stage
from saccade.utils.time import format_span, format_timestamp

if TYPE_CHECKING:
    from saccade.llm.base import LLM, ToolCall
    from saccade.video import Video

EXPLORE_PROMPT = """You are a video analyst answering a question about ONE video. You start \
with its timestamped transcript and small thumbnails of representative frames. The thumbnails \
are too small for details: whenever the question involves anything visual (people, \
appearance, posture, clothing, slides, screens, text on screen), LOOK before you answer.

Tools:
- view_frames(timestamps): exact frames at those times, in high resolution (up to 4 per call).
- zoom(timestamp, x, y, width, height): one region of a frame at full resolution; the box is \
given as fractions of the frame (0–1, origin top-left). Use it on a person's tile, a slide or small text.
- search_transcript(query): find when something was said.
- read_transcript(start, end): the verbatim transcript between two times (seconds).

How to work:
- Use the transcript and search to find WHEN things happen, then look at those moments.
- For questions about people or the whole video, sample several moments across the timeline \
and zoom in on each person.
- Be efficient: request several timestamps per call and stop once the evidence is sufficient.

Answer rules:
- Base every statement on what you read or saw; never guess.
- Names said in the transcript or shown on screen (e.g. meeting tiles) may be used.
- Cite the moments you rely on with their timestamps, e.g. [12:22.1] or [12:22.1–12:49.8].
- Be direct and specific. Only mention limitations that actually matter for the question.
- Answer in the same language as the question."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "view_frames",
            "description": "Get high-resolution frames of the video at the given times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamps": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Times in seconds from the start of the video (max 4).",
                    }
                },
                "required": ["timestamps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom",
            "description": "Get one region of the frame at a time, at full resolution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "number", "description": "Time in seconds."},
                    "x": {
                        "type": "number",
                        "description": "Left edge, fraction of frame width (0-1).",
                    },
                    "y": {
                        "type": "number",
                        "description": "Top edge, fraction of frame height (0-1).",
                    },
                    "width": {
                        "type": "number",
                        "description": "Region width, fraction of frame width.",
                    },
                    "height": {
                        "type": "number",
                        "description": "Region height, fraction of frame height.",
                    },
                },
                "required": ["timestamp", "x", "y", "width", "height"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transcript",
            "description": "Find the passages where something was said; returns them with timestamps.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_transcript",
            "description": "Read the verbatim transcript between two times (seconds).",
            "parameters": {
                "type": "object",
                "properties": {"start": {"type": "number"}, "end": {"type": "number"}},
                "required": ["start", "end"],
            },
        },
    },
]

_MAX_FRAMES_PER_CALL = 4
_MAX_TRANSCRIPT_CHARS = 12_000


@dataclass
class _Session:
    video: Video
    reporter: Reporter
    max_images: int
    viewed: list[Frame] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def images_left(self) -> int:
        return self.max_images - len(self.viewed)

    def add_usage(self, usage: dict[str, Any]) -> None:
        _sum_into(self.usage, usage)


def _sum_into(total: dict[str, Any], usage: dict[str, Any]) -> None:
    """Add token counts, including nested details such as ``cached_tokens``."""
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, dict):
            _sum_into(total.setdefault(key, {}), value)


def explore(
    video: Video,
    question: str,
    llm: LLM,
    *,
    reporter: Reporter,
    evidence_tokens: int = 48_000,
    overview_images: int = 12,
    max_steps: int = 4,
    max_images: int = 16,
    image_width: int = 1024,
    zoom_width: int = 768,
    max_answer_tokens: int = 8000,
) -> Answer:
    """Let ``llm`` gather visual and textual evidence with tools, then answer ``question``."""
    messages, evidence, overview, mode = build_messages(
        video,
        question,
        evidence_tokens=evidence_tokens,
        max_images=overview_images,
        image_detail="low",
        images=True,
    )
    messages[0] = {"role": "system", "content": EXPLORE_PROMPT}
    session = _Session(video=video, reporter=reporter, max_images=max_images)
    complete: Callable[..., Any] = llm.complete
    response = None

    for step in range(max_steps + 1):
        last_turn = step == max_steps or session.images_left <= 0
        reporter(
            Stage.EXPLORE, "Writing the answer" if last_turn else f"Thinking (step {step + 1})"
        )
        response = complete(
            messages,
            max_tokens=max_answer_tokens,
            tools=TOOLS,
            tool_choice="none" if last_turn else "auto",
        )
        session.add_usage(response.usage)
        if not response.tool_calls or last_turn:
            break
        messages.append(response.message or {"role": "assistant", "content": response.text or None})
        attachments: list[dict[str, Any]] = []
        for call in response.tool_calls:
            text, images = _run_tool(session, call, image_width, zoom_width)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
            attachments += images
        if attachments:
            # Tool messages are text-only in the chat format; images follow as a user turn.
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Images returned by the tools:"},
                        *attachments,
                    ],
                }
            )

    assert response is not None
    evidence += [Evidence(id=f.id, type="frame", timestamp=f.timestamp) for f in session.viewed]
    return Answer(
        question=question,
        text=response.text.strip(),
        evidence=evidence,
        frames=[*overview, *session.viewed],
        mode=mode,
        model=response.model,
        usage=dict(session.usage),
        steps=list(session.steps),
    )


def _run_tool(
    session: _Session, call: ToolCall, image_width: int, zoom_width: int
) -> tuple[str, list[dict[str, Any]]]:
    args = call.arguments
    try:
        if call.name == "view_frames":
            times = [float(t) for t in (args.get("timestamps") or [])][:_MAX_FRAMES_PER_CALL]
            return _view(session, times, None, image_width)
        if call.name == "zoom":
            box: Box = (
                float(args["x"]),
                float(args["y"]),
                float(args["width"]),
                float(args["height"]),
            )
            return _view(session, [float(args["timestamp"])], box, zoom_width)
        if call.name == "search_transcript":
            return _search(session, str(args.get("query", ""))), []
        if call.name == "read_transcript":
            return _read(session, float(args.get("start", 0)), float(args.get("end", 0))), []
        return f"Unknown tool {call.name!r}.", []
    except (KeyError, TypeError, ValueError) as exc:
        return f"Invalid arguments for {call.name}: {exc}", []


def _view(
    session: _Session, times: list[float], box: Box | None, image_width: int
) -> tuple[str, list[dict[str, Any]]]:
    if not times:
        return "No timestamps given.", []
    times = times[: max(0, session.images_left)]
    if not times:
        return "Image budget exhausted; answer with the evidence you have.", []
    duration = session.video.media_info().duration
    folder = session.video.index_dir / "views"
    folder.mkdir(parents=True, exist_ok=True)
    labels: list[str] = []
    parts: list[dict[str, Any]] = []
    for requested in times:
        at = (
            min(max(0.0, requested), max(0.0, duration - 0.05)) if duration else max(0.0, requested)
        )
        try:
            grabbed = grab_frame(session.video.path, at, box=box, max_width=image_width)
        except Exception as exc:  # decoding problems are reported to the model, not raised
            labels.append(f"{format_timestamp(at)}: could not decode ({exc})")
            continue
        index = len(session.viewed) + 1
        suffix = "" if box is None else "_zoom"
        name = f"view_{int(time.time() * 1000)}_{index:03d}{suffix}.jpg"
        path = folder / name
        path.write_bytes(grabbed.jpeg)
        frame = Frame(
            id=f"view_{index:03d}",
            timestamp=grabbed.time,
            path=str(path),
            width=grabbed.width,
            height=grabbed.height,
            reason="zoom" if box else "view",
        )
        session.viewed.append(frame)
        label = f"[{format_timestamp(frame.timestamp)} | {frame.id}]"
        if box is not None:
            label += " zoom x={:.2f} y={:.2f} w={:.2f} h={:.2f}".format(*box)
        labels.append(label)
        parts.append({"type": "text", "text": label})
        parts.append(
            {"type": "image_url", "image_url": {"url": frame.data_url(), "detail": "high"}}
        )
    what = "Zooming into" if box else "Looking at"
    step = f"{what} {', '.join(format_timestamp(t) for t in times)}"
    session.steps.append(step)
    session.reporter(Stage.EXPLORE, step)
    return "Images attached in the next message: " + "; ".join(labels), parts


def _search(session: _Session, query: str) -> str:
    step = f"Searching the transcript for {query!r}"
    session.steps.append(step)
    session.reporter(Stage.EXPLORE, step)
    results = session.video.search(query, limit=8, expand=5)
    if not results:
        return "No matches."
    lines = [f"[{format_span(r.start, r.end)}] {r.text}" for r in results]
    return "\n".join(lines)[:_MAX_TRANSCRIPT_CHARS]


def _read(session: _Session, start: float, end: float) -> str:
    if end <= start:
        return "end must be after start."
    step = f"Reading the transcript {format_span(start, end)}"
    session.steps.append(step)
    session.reporter(Stage.EXPLORE, step)
    segments = session.video.transcript().segments
    lines = [
        f"[{format_span(s.start, s.end)}] {s.text}"
        for s in segments
        if s.end >= start and s.start <= end
    ]
    return ("\n".join(lines) or "Nothing was said in that range.")[:_MAX_TRANSCRIPT_CHARS]
