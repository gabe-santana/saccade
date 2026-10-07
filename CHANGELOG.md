# Changelog

All notable changes to Saccade are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). While the version is 0.x, minor releases may
change the API.

## [Unreleased]

### Added

- Timestamped Markdown transcript export via `Transcript.to_markdown()` and `saccade transcribe -f md`.
- Project logo, README banner and animated demo (`assets/`).
- `CONTRIBUTING.md`, `AGENTS.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and PR
  templates, Dependabot and pre-commit configuration.
- `docs/architecture.md`, covering pipeline internals, design notes and limitations.

### Changed

- README rewritten as a landing page; reference material moved into `docs/`.
- Relative README links are rewritten to absolute GitHub URLs on PyPI.

## [0.1.0] - 2026-10-07

First public release.

### Added

- `Video` API: `index()`, `transcribe()`, `atranscribe()`, `transcript()`, `chunks()`,
  `search()`, `context()`, `frames()`, `info()`, `ask()` and async variants.
- Streaming audio decode with PyAV, aligned to the media timeline.
- Streaming Silero VAD (ONNX) with hysteresis, padding and merging. Speech is packed into
  ~30 s windows with exact timestamp restoration across seams.
- faster-whisper ASR: int8 on CPU, batched float16 on NVIDIA GPUs (`device="cuda"`, the
  `gpu` extra installs the CUDA libraries from pip).
- Resumable, incremental SQLite index (WAL) with content fingerprints and
  configuration-keyed runs.
- Multilingual FTS5 search: diacritic and case folding, stop words, light stemming, CJK
  bigrams, phrase bonus, temporal grouping.
- Token-budgeted, timestamped RAG context with transcript and frame evidence.
- Representative frame extraction (`auto`, `screen`, `scenes`, `interval` strategies) with
  perceptual deduplication and keyframe-only sampling.
- `ask()` with Azure AI Foundry, OpenAI, Ollama and any OpenAI-compatible client, plus an
  exploration mode where the model views frames, zooms and reads transcript ranges through
  tools. Token usage includes cached and reasoning tokens.
- `saccade` CLI: `index`, `transcribe`, `search`, `context`, `frames`, `ask`, `info`,
  `models`.
- Documentation in `docs/`, examples, benchmarks, and CI on Linux, Windows and macOS.

[Unreleased]: https://github.com/gabe-santana/saccade/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gabe-santana/saccade/releases/tag/v0.1.0
