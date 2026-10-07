# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Copilot, Cursor and others) working in
this repository. Human contributors: see [CONTRIBUTING.md](CONTRIBUTING.md). It says the
same things at more length.

## What this project is

Saccade is a Python library and CLI (`pip install saccade-video`, `import saccade`). It turns a
video into timestamped evidence for LLMs: local Whisper transcription with Silero VAD,
representative frames, SQLite FTS5 search, token-budgeted context, and `Video.ask()`, which
lets an LLM explore the video through tools. It runs on CPU and, optionally, on NVIDIA GPUs.

## Commands

```bash
pip install -e ".[dev]"                                # setup (Python 3.11+)
pytest                                                 # full default suite, no models, under a minute
pytest tests/test_pipeline.py -k seam                  # a focused subset
ruff check src tests examples benchmarks               # lint
ruff format src tests examples benchmarks              # format (100 columns)
mypy                                                   # strict type check of src/saccade
SACCADE_INTEGRATION=1 pytest -m integration            # real models; Windows only (SAPI TTS)
```

Run `ruff check`, `ruff format --check`, `mypy` and `pytest` before you consider a change
done. All four must pass.

## Layout

```text
src/saccade/
  video.py      Video facade: index/transcribe/search/context/frames/ask; cache-key material
  pipeline.py   decode → VAD → pack → ASR → store (resumable, one transaction per pack)
  ask.py        single-shot ask();  explore.py: tool-calling exploration loop
  config.py     ASRConfig, VadConfig, ChunkConfig, VisualConfig, PROFILES
  media/        probe, fingerprint, streaming audio decode, frame sampling and grab
  vad/ asr/     Silero streaming VAD + segmenter; faster-whisper backend (CPU, batched GPU)
  vision/       frame selection (change/scene/settle), dHash dedup, extraction
  index/        SQLite schema, FTS normalisation, chunking
  retrieval/    search, ranking, context assembly
  llm/          LLM protocol and stdlib-only HTTP clients (Azure, OpenAI-compatible)
  models/       public dataclasses (segments, frames, context, answers)
tests/          pytest; support.py has the deterministic test doubles
docs/           user documentation (one page per topic)
examples/       tiny runnable scripts (keep them ~15 lines)
benchmarks/     media generator and benchmark runner
```

## Rules

- **No new runtime dependencies** without discussion. Heavy optional features go in an
  extra, with imports inside the function that needs them (lint rule PLC0415 is disabled
  for this reason). LLM clients must stay standard-library only (`urllib`).
- **Never compute timestamps from frame numbers.** Use decoded PTS relative to the
  container start. Keep `Pack.map_span` seam logic intact; tests cover it.
- **Bump cache versions when stored output changes.** Use `PROCESSING_VERSION` in
  `_version.py` for transcripts and chunks, `_VISUAL_VERSION` in `video.py` for frames,
  and `SCHEMA_VERSION` in `index/schema.py` for the database layout.
- **Memory must not grow with video length.** Stream; don't collect whole tracks into
  arrays.
- **The public API is what `saccade/__init__.py` exports.** Changes to it must be typed,
  documented in `docs/api-reference.md` and noted in `CHANGELOG.md`.
- **Tests use stand-ins, never real models or APIs.** Use `make_video`, `ToneASR`,
  `EnergyModel` and `write_media`/`write_video` from `tests/`. Tests must not touch the
  network.
- **Don't commit media or secrets.** Media files (`*.mp4` and anything under
  `benchmarks/media/`) are gitignored. API keys come from environment variables
  (`AZURE_AI_*`, `OPENAI_API_KEY`) and never appear in code, tests, docs or examples.
- **Keep examples minimal.** An example is a ~15-line script: `Video(path, llm=...)` and
  then `ask()`. Put explanations in `docs/`, not in examples.
- **Style:** Python 3.11+ syntax, full type annotations, `from __future__ import
  annotations` where the module already uses it, docstrings on public functions, comments
  that explain *why*. Multilingual text and en dashes in strings are intentional.

## Where things are documented

- User docs: `docs/README.md` (index), and `docs/api-reference.md` for every public symbol.
- Internals and trade-offs: `docs/architecture.md`.
- Releasing: `RELEASING.md`. Releases are tag-triggered (`vX.Y.Z`) and only maintainers cut them.
