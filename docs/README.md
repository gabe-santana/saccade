# Saccade documentation

Saccade turns a video into evidence an LLM can use. It works out what was **said**, using
local speech recognition, and what was **shown**, using representative frames, and stamps
everything with source timestamps. You can then search that evidence, build prompt context
from it, or ask an LLM questions about the video in one call.

```python
import saccade

video = saccade.Video("meeting.mp4", llm=saccade.azure())
print(video.ask("What did they decide about the release date?"))
```

## Start here

| If you want to… | Read |
|---|---|
| Install Saccade and get a first answer in five minutes | [Getting started](getting-started.md) |
| Understand what happens to your video, and what gets cached | [Concepts](concepts.md) |

## Guides

| Topic | What it covers |
|---|---|
| [Asking questions with an LLM](asking-questions.md) | `video.ask()`, LLM connections (Azure, OpenAI, Ollama, your own), exploration mode, token costs |
| [Search and RAG context](search-and-context.md) | `search()`, `context()`, `chunks()`, plugging Saccade into your own RAG stack |
| [Transcription](transcription.md) | Profiles, languages, streaming, async, background indexing, progress, exports |
| [Frames](frames.md) | Representative frames, visual strategies, `frames()` |
| [Performance: CPU and GPU](performance.md) | `device="cuda"`, threads, keyframe sampling, measured numbers |
| [Command line](cli.md) | The `saccade` command |

## Reference

| Page | What it covers |
|---|---|
| [API reference](api-reference.md) | Every public class, method, parameter and return type |
| [Extending Saccade](extending.md) | Custom ASR backends, VAD, and LLM clients |
| [Architecture and limitations](architecture.md) | How the pipeline works inside, design decisions, source layout, known limits |
| [Troubleshooting](troubleshooting.md) | Common errors and how to fix them |

## Requirements at a glance

- Python 3.11 or newer, on Windows, macOS or Linux.
- No GPU, Docker or database server. FFmpeg is bundled through PyAV.
- An LLM only if you call `ask()`; everything else runs offline.
- An NVIDIA GPU is optional and makes indexing much faster (see [Performance](performance.md)).
