"""Question answering over a video: Saccade's evidence + your LLM.

Strategy:

* if the whole timestamped transcript fits ``evidence_tokens``, send all of it — broad
  questions ("How was the interview?") need the full conversation, not a few keyword hits;
* otherwise send the passages :meth:`Video.context` retrieves for the question;
* attach representative frames as images (spread over the video, or near the retrieved
  passages), each labelled with its source timestamp, when the LLM accepts images.

The LLM is instructed to use only that evidence and to cite timestamps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from saccade.models.answer import Answer
from saccade.models.context import Evidence
from saccade.models.frame import Frame
from saccade.retrieval.context import estimate_tokens
from saccade.utils.time import format_span, format_timestamp

if TYPE_CHECKING:
    from saccade.llm.base import LLM, Message
    from saccade.video import Video

SYSTEM_PROMPT = """You answer questions about a video using ONLY the evidence provided: \
a transcript with source timestamps and, when available, frames (images) taken from the video, \
each labelled with its timestamp.

Rules:
- Base every statement on the evidence. Never invent names, facts, numbers or events.
- Cite the moments you rely on with their timestamps exactly as given, e.g. [12:22.1–12:49.8] or [13:32.5].
- Use the frames for what can be seen (people, setting, slides, screens, body language), \
the transcript for what was said.
- If the evidence does not answer the question, say so plainly.
- Answer in the same language as the question."""


def build_messages(
    video: Video,
    question: str,
    *,
    evidence_tokens: int,
    max_images: int,
    image_detail: str = "low",
    images: bool,
) -> tuple[list[Message], list[Evidence], list[Frame], Literal["full", "retrieved"]]:
    transcript = video.transcript()
    header = _header(video)
    chunks = transcript.chunks
    full_text = "\n\n".join(f"[{format_span(c.start, c.end)} | {c.id}]\n{c.text}" for c in chunks)
    mode: Literal["full", "retrieved"]

    if not transcript.segments:
        mode = "full"
        body = "TRANSCRIPT: no speech was found in this video."
        evidence: list[Evidence] = []
        nearby: list[Frame] = []
    elif estimate_tokens(header + full_text) <= evidence_tokens:
        mode = "full"
        body = "FULL TRANSCRIPT (verbatim; timestamps refer to the source video):\n\n" + full_text
        evidence = [
            Evidence(id=s.id, type="transcript", start=s.start, end=s.end, text=s.text)
            for s in transcript.segments
        ]
        nearby = []
    else:
        mode = "retrieved"
        context = video.context(question, max_tokens=evidence_tokens)
        body = context.text
        evidence = [e for e in context.evidence if e.type == "transcript"]
        nearby = context.frames

    frames: list[Frame] = []
    if images and max_images > 0:
        frames = _pick_frames(video.frames(), nearby, max_images)
    evidence += [Evidence(id=f.id, type="frame", timestamp=f.timestamp) for f in frames]

    parts: list[dict[str, Any]] = [{"type": "text", "text": f"{header}\n\n{body}"}]
    if frames:
        parts.append(
            {"type": "text", "text": f"\nFRAMES FROM THE VIDEO ({len(frames)}, in timeline order):"}
        )
        for frame in frames:
            parts.append(
                {"type": "text", "text": f"[{format_timestamp(frame.timestamp)} | {frame.id}]"}
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": frame.data_url(), "detail": image_detail},
                }
            )
    parts.append({"type": "text", "text": f"\nQUESTION: {question}"})

    content: Any = (
        parts if frames else "".join(p["text"] + "\n" for p in parts if p["type"] == "text")
    )
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return messages, evidence, frames, mode


def ask(
    video: Video,
    question: str,
    llm: LLM,
    *,
    evidence_tokens: int = 48_000,
    max_images: int = 12,
    image_detail: str = "low",
    max_answer_tokens: int = 8000,
) -> Answer:
    messages, evidence, frames, mode = build_messages(
        video,
        question,
        evidence_tokens=evidence_tokens,
        max_images=max_images,
        image_detail=image_detail,
        images=llm.supports_images,
    )
    response = llm.complete(messages, max_tokens=max_answer_tokens)
    return Answer(
        question=question,
        text=response.text.strip(),
        evidence=evidence,
        frames=frames,
        mode=mode,
        model=response.model,
        usage=response.usage,
    )


def _header(video: Video) -> str:
    info = video.media_info()
    lines = [f"VIDEO: {video.name}"]
    if info.duration:
        lines.append(f"DURATION: {format_timestamp(info.duration, precision=0)}")
    return "\n".join(lines)


def _pick_frames(all_frames: list[Frame], preferred: list[Frame], limit: int) -> list[Frame]:
    """Frames near the evidence first, then evenly spread over the video, in time order."""
    chosen: dict[str, Frame] = {f.id: f for f in preferred[:limit]}
    remaining = [f for f in all_frames if f.id not in chosen]
    slots = limit - len(chosen)
    if slots > 0 and remaining:
        if len(remaining) <= slots:
            picks = remaining
        else:
            step = (len(remaining) - 1) / (slots - 1) if slots > 1 else 0
            picks = [remaining[round(i * step)] for i in range(slots)]
        chosen.update((f.id, f) for f in picks)
    return sorted(chosen.values(), key=lambda f: f.timestamp)
