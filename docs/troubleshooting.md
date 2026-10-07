# Troubleshooting

To see what Saccade is doing, run the CLI with `-v` (or `-vv` for debug output). In Python,
enable logging for the `saccade` logger:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Installation and models

**`ModelNotFoundError: Whisper model 'small' is not in ... and downloads are disabled`**
You are offline (`SACCADE_OFFLINE` / `HF_HUB_OFFLINE` is set, or `offline=True`) and the
model hasn't been downloaded yet. Run `saccade models download small` once while online.
Alternatively, point `ASRConfig(model=...)` at a local model directory.

**`ModelNotFoundError: Unknown Whisper model 'smal'`**
That name isn't a known model. `saccade models list` shows the valid names.

**The first run "hangs" for a while**
The first use of a model downloads it (about 480 MB for `small`), and a progress event says
so. Pass `progress=print` to see it.

## GPU

**`Library cublas64_12.dll is not found` / `libcublas.so.12: cannot open shared object file`**
The CUDA libraries aren't installed. Run `pip install -e ".[gpu]"` (or
`pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"`).

**`ConfigError: device='cuda' was requested but no CUDA-capable NVIDIA GPU is visible`**
Check that `nvidia-smi` works. Use `device="auto"` to fall back to the CPU automatically.

**The GPU isn't much faster**
Usually frame extraction is now the bottleneck: decoding every frame of a high-resolution
video. Add `visual=VisualConfig(keyframes_only=True)`. The first GPU run on a new GPU
architecture is also slower while CUDA compiles kernels.

## Asking an LLM

**`ConfigError: No LLM configured`**
Pass one: `Video(path, llm=saccade.azure())` or `video.ask(question, llm=...)`.

**`ConfigError: Azure AI Foundry needs an endpoint and key`**
Set `AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY` and `AZURE_AI_DEPLOYMENT`, or pass them to
`saccade.azure()`. Environment variables set in one terminal don't carry over to another.

**`LLMError: ... returned HTTP 401`**
The API key is wrong or belongs to another resource.

**`LLMError: ... returned HTTP 404`**
The deployment name is wrong. Use the *deployment* name from the Foundry portal, which is not
always the model name. Also check that the endpoint belongs to the same resource.

**`LLMError: ... returned HTTP 400`**
The endpoint's message says which parameter it rejected. Saccade handles reasoning models
(`max_completion_tokens`, no `temperature`). If you passed `reasoning_effort` to a
non-reasoning model such as gpt-4o, remove it.

**`LLMError: ... returned HTTP 429`**
You hit a rate limit. Saccade retries twice, honouring `Retry-After`. If it persists, raise
the deployment's quota or lower `max_views`.

**`LLMError: The model used its whole output budget ... before answering`**
A reasoning model spent its budget thinking. Raise `max_answer_tokens`, or use
`saccade.azure(reasoning_effort="low")`.

**Too many tokens**
See [Controlling token costs](asking-questions.md#controlling-token-costs). In short:

- `reasoning_effort="low"`;
- `max_steps=2, max_views=6`;
- `explore=False` for questions about what was said.

**The answer misses visual details**
Make sure exploration is on: it is by default with Azure and OpenAI, and `answer.steps`
should list "Looking at …" or "Zooming into …". For local models, use one that accepts images
(`images=True`) and supports tools (`function_calling=True`).

## Media

**`MediaNotFoundError`**
The path doesn't exist. Relative paths are resolved from the current working directory.

**`UnsupportedFormatError: "x.mp4" is not a media file FFmpeg can read`**
The file is corrupted, incomplete, or not a media file at all.

**`AudioStreamNotFoundError`**
The file has no audio track. `transcribe()` raises this. `index()` instead records an empty
transcript and still extracts frames.

## Indexing

**`NotIndexedError`**
Call `video.index()` before `search()`, `context()`, `transcript()` or `chunks()`. `ask()`
and the CLI index automatically.

**`IndexLockedError`**
Another process is indexing the same video. Wait for it; searching the partial index works
in the meantime.

**The video is reprocessed even though I indexed it already**
A setting that changes the output is different: model, profile, language, device, VAD or
visual settings. See [what gets cached](concepts.md#what-gets-cached-and-when-saccade-reprocesses).
`video.info().runs` lists the stored runs.

**Search finds nothing**
Search is lexical: it matches the words used, in the language spoken. Try the words a
speaker would actually say, or use `ask()`, which lets the LLM search and read the transcript
itself.

## Output

**Garbled accents or `UnicodeEncodeError` on the Windows console**
The CLI forces UTF-8. In your own scripts, add
`sys.stdout.reconfigure(encoding="utf-8")` or set `PYTHONIOENCODING=utf-8`.

## Disk usage

Each video's index and frames live in `video.index_dir`, and models live in
`<cache>/models`. Deleting a video's folder under `<cache>/videos/` removes its index; it is
rebuilt on the next `index()`.
