# Frames

Alongside the transcript, `index()` stores a small set of **representative frames**: still
images that show what was on screen. They are what `context()` attaches as visual evidence
and what `ask()` shows the LLM as an overview. During exploration, the LLM can also request
*any* other frame on demand (see [Asking questions](asking-questions.md)).

```python
for frame in video.frames():                 # or video.frames(start=600, end=900)
    print(frame.timestamp, frame.id, frame.reason, frame.path)

frame.data_url()       # "data:image/jpeg;base64,..." for multimodal LLM APIs
frame.read_bytes()     # the JPEG
```

Each `Frame` has these fields:

| Field | Meaning |
|---|---|
| `id` | `frame_00031` |
| `timestamp` | Presentation time of the decoded frame, in seconds |
| `path` | Absolute path of the JPEG |
| `width`, `height` | Stored image size |
| `reason` | Why it was kept: `start`, `change`, `interval` or `end` |

## How frames are chosen

No vision model is involved, so frame selection is cheap.

1. **Sample.** About one decoded frame per second is reduced to a 64×36 grayscale thumbnail.
   Non-reference frames are skipped at decode time.
2. **Detect changes.** A frame is kept when it differs clearly from the last kept one. On
   screen recordings and slides, Saccade waits until a transition has settled, so
   half-rendered pages and animations are skipped.
3. **Drop duplicates.** A 256-bit perceptual hash plus the thumbnail difference are compared
   with recently kept frames. Returning to an earlier slide does not store it again.
4. **Store.** Kept frames are saved as JPEGs, at most 1280 px wide by default, next to the
   index.

## Strategies

| `visual_strategy` | Best for | Behaviour |
|---|---|---|
| `auto` (default) | Anything | Detects static content: there it keeps every settled change; on camera footage, clear changes plus one frame per minute while the picture moves |
| `screen` | Screen recordings, slides | Sensitive to UI changes, waits for transitions to settle |
| `scenes` | Edited footage | Hard cuts only |
| `interval` | Even coverage | One frame every `interval_s` seconds, duplicates skipped |
| `off` | Audio-only workflows | No frames |

```python
video.index(visual_strategy="screen")
```

## `VisualConfig`

```python
from saccade import VisualConfig, Video

Video("talk.mp4", visual=VisualConfig(
    strategy="auto",
    sample_fps=1.0,         # frames examined per second
    interval_s=60.0,        # spacing for "interval" (and auto's periodic frames)
    max_width=1280,         # stored JPEG width
    jpeg_quality=3,         # FFmpeg qscale: 2 (best) .. 31
    max_frames=1000,        # hard cap per video
    keyframes_only=False,   # True: decode keyframes only (about 5x faster, coarser)
))
```

### `keyframes_only`

Decoding every frame is the slowest part of indexing for long or high-resolution videos.
A 26-minute 3440×1440 recording takes about 30–40 s. With `keyframes_only=True`, only
keyframes are decoded, typically one every 2–10 s, which is about 5× faster. You get fewer,
coarser frames, which is usually fine because `ask()` can fetch any exact frame on demand
anyway.

## Videos without video (or without audio)

- **Audio-only files** (`.wav`, `.mp3`) simply have no frames.
- **Silent screen recordings** still get frames. Their transcript status is `empty`.
