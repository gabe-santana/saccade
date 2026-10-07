# Getting started

This page takes you from installation to a first answer about your own video.

## 1. Install

```bash
pip install saccade-video                # CPU
pip install "saccade-video[gpu]"         # optional: NVIDIA GPU support (see performance.md)
```

The package is called `saccade-video` on PyPI and is imported as `saccade`.

To work on Saccade itself, install it from source:

```bash
git clone https://github.com/gabe-santana/saccade
cd saccade
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

This installs:

- **faster-whisper**, for speech recognition;
- **PyAV**, which includes FFmpeg, so you don't install FFmpeg separately;
- **onnxruntime**, for voice activity detection;
- **numpy**.

PyTorch is not needed.

## 2. Index a video

```python
from saccade import Video

video = Video("meeting.mp4")
summary = video.index(progress=print)

print(summary.segments, "segments,", summary.frames, "frames, language:", summary.language)
```

The first call does the real work, entirely on your machine:

1. **Speech.** Saccade transcribes the speech. The Whisper model, about 480 MB for `small`,
   downloads once on first use.
2. **Frames.** It picks representative frames from the video.
3. **Index.** It stores everything in a local SQLite index.

Calling `index()` again on the same file, even from a new `Video` object or a later Python
session, returns in milliseconds.

## 3. Search and build context (no LLM needed)

```python
for hit in video.search("deployment failed"):
    print(hit.start, hit.end, hit.text)

context = video.context("Why did the deployment fail?", max_tokens=4000)
print(context.text)   # timestamped evidence, ready to paste into any LLM prompt
```

## 4. Ask an LLM

Configure an LLM once, then ask questions. With Azure AI Foundry, set the three environment
variables from your deployment's page in the Foundry portal:

```bash
export AZURE_AI_ENDPOINT="https://<resource>.services.ai.azure.com/openai/v1/responses"
export AZURE_AI_API_KEY="<key>"
export AZURE_AI_DEPLOYMENT="gpt-5-mini"
```

In PowerShell:

```powershell
$env:AZURE_AI_ENDPOINT = "https://<resource>.services.ai.azure.com/openai/v1/responses"
$env:AZURE_AI_API_KEY = "<key>"
$env:AZURE_AI_DEPLOYMENT = "gpt-5-mini"
```

Then:

```python
import saccade

video = saccade.Video("meeting.mp4", llm=saccade.azure())
answer = video.ask("Summarize the meeting and list the decisions.", progress=print)

print(answer)                                         # the LLM's answer, citing [mm:ss] timestamps
print(answer.total_tokens, "tokens")
```

`ask()` indexes the video first if needed. The LLM receives only text and images that
Saccade extracted (transcript and frames); the video file itself never leaves your machine.
See [Asking questions](asking-questions.md) for OpenAI, local Ollama and custom clients.

## 5. From the command line

```bash
saccade index meeting.mp4
saccade search meeting.mp4 "deployment failed"
saccade context meeting.mp4 "Why did the deployment fail?"
saccade ask meeting.mp4 "Summarize the meeting"      # uses the AZURE_AI_* variables
```

See [Command line](cli.md) for every option.

## Where things are stored

Indexes, frames and models live in a cache directory:

| Platform | Default location |
|---|---|
| Windows | `%LOCALAPPDATA%\saccade` |
| macOS | `~/Library/Caches/saccade` |
| Linux | `$XDG_CACHE_HOME/saccade` (default `~/.cache/saccade`) |

To change it, pass `Video(..., cache_dir=...)` or set `SACCADE_CACHE_DIR`.
`video.info().index_dir` shows the folder for a specific video.

## Next steps

- [Concepts](concepts.md) explains what Saccade stores and when it reprocesses a video.
- [Performance](performance.md) shows how to index in seconds on an NVIDIA GPU.
