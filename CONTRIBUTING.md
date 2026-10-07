# Contributing to Saccade

Thanks for your interest in Saccade! Bug reports, docs fixes, benchmarks on your hardware,
new backends: every contribution helps. This guide gets you from clone to merged PR.

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Running the checks](#running-the-checks)
- [Project principles](#project-principles)
- [Making a change](#making-a-change)
- [Pull requests](#pull-requests)

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug.** Open a [bug report](https://github.com/gabe-santana/saccade/issues/new?template=bug_report.yml).
  The most useful reports include the output of `saccade info <video> --json`, your OS,
  Python version and CPU/GPU, and the smallest snippet that reproduces the problem.
- **Suggest a feature.** Open a [feature request](https://github.com/gabe-santana/saccade/issues/new?template=feature_request.yml)
  and describe the problem first, then the solution you have in mind.
- **Pick up an issue.** Issues labelled
  [`good first issue`](https://github.com/gabe-santana/saccade/labels/good%20first%20issue)
  are scoped to be approachable without knowing the whole codebase.
  [`help wanted`](https://github.com/gabe-santana/saccade/labels/help%20wanted) issues are
  bigger. Comment on an issue before starting, so nobody duplicates work.
- **Share benchmarks.** Results from `benchmarks/bench.py` on other CPUs, GPUs and real
  recordings are valuable. Post them in a discussion or an issue.
- **Improve the docs.** If something confused you, it will confuse the next person. Fix it.

Security problems should **not** be reported in public issues; see [SECURITY.md](SECURITY.md).

## Development setup

You need Python 3.11+ and git. FFmpeg is bundled through PyAV, so there is nothing else to
install.

```bash
git clone https://github.com/gabe-santana/saccade
cd saccade
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"              # add ,gpu for NVIDIA GPU work: ".[dev,gpu]"
```

Optional but recommended: install the [pre-commit](https://pre-commit.com) hooks so lint and
formatting run on every commit.

```bash
pip install pre-commit
pre-commit install
```

## Running the checks

CI runs exactly these. Run them before opening a PR:

```bash
ruff check src tests examples benchmarks
ruff format --check src tests examples benchmarks     # `ruff format ...` fixes formatting
mypy                                                   # strict mode, configured in pyproject.toml
pytest                                                 # ~180 tests, under a minute
```

### How the tests work

The default suite **downloads no models and needs no GPU**. It runs the real pipeline
(decode, VAD segmentation, packing, timestamp mapping, SQLite, search, context, frames,
`ask()`) with deterministic stand-ins from [`tests/support.py`](tests/support.py):

- `EnergyModel` replaces Silero's neural network with an energy detector.
- `ToneASR` "transcribes" tone bursts by naming their pitch. Test media places tones at known
  times, so timestamp restoration is exactly checkable.
- `write_media` / `write_video` synthesise test files in any container with PyAV encoders.
- LLM tests use scripted fake clients; nothing calls a real API.

Fixtures such as `make_video` live in [`tests/conftest.py`](tests/conftest.py). Use them in
new tests instead of real models.

### Integration tests

```bash
SACCADE_INTEGRATION=1 pytest -m integration
```

These run the real faster-whisper + Silero stack on synthesised speech. They download the
`base` model on first run, and the speech synthesis currently uses Windows SAPI voices, so
they only run on Windows. Making them cross-platform is an
[open issue](https://github.com/gabe-santana/saccade/issues).

## Project principles

These are the decisions that keep Saccade fast, correct and small. PRs that conflict with
them need a strong reason, so please discuss in an issue first.

1. **Local first.** Media never leaves the machine. Network access happens only for model
   downloads (which `SACCADE_OFFLINE=1` forbids) and for calls to the LLM the user configured.
2. **Few, light dependencies.** Runtime deps are faster-whisper, PyAV, onnxruntime and numpy.
   No PyTorch, OpenCV or Pillow. LLM clients use only the standard library. New features with
   heavy deps belong in an optional extra, imported lazily.
3. **Timestamps are the truth.** Every time comes from decoded presentation timestamps on the
   source timeline, never from `frame_index / fps`. Any change to decoding, VAD or packing
   must keep timestamps exact; there are tests for this.
4. **Evidence, not conclusions.** Saccade retrieves and labels source material. It never
   generates summaries or claims of its own; only the user's LLM does, and the answer comes
   back together with the evidence it was given.
5. **Cache correctness.** If a change alters stored output (segments, chunks, frames),
   bump `PROCESSING_VERSION` in `src/saccade/_version.py` (or `_VISUAL_VERSION` in
   `video.py` for frames), so old indexes are not reused with new semantics. Schema
   changes bump `SCHEMA_VERSION` in `index/schema.py`.
6. **Bounded memory, streaming results.** Nothing may scale with video length in memory.
   Results are committed and visible as they are produced.
7. **A small, typed public API.** Everything public is exported from `saccade/__init__.py`,
   fully typed (mypy strict) and documented in `docs/api-reference.md`.

## Making a change

- **Keep PRs focused.** One fix or feature per PR is much faster to review.
- **Add tests.** Bug fixes come with a test that fails without the fix. Features come with
  tests for the main path and the edge cases.
- **Update the docs.** User-visible changes update the relevant page in [`docs/`](docs/)
  and, if it's a headline feature, the README.
- **Add a changelog entry** under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
- **Match the surrounding code.** The formatter targets 100 columns, every function is
  type-annotated, and comments explain *why*, not *what*.

### Commit messages

Write the subject in the imperative mood, under ~72 characters, for example *"Skip
non-reference frames when sampling"*. Use the body to explain why the change is needed.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your change, with tests and docs.
3. Run the checks above.
4. Open a PR and fill in the template. Link the issue it closes (`Closes #123`).

A maintainer will review your PR, usually within a few days. CI must pass on Linux, Windows
and macOS before merge.

Releases are cut by maintainers; see [RELEASING.md](RELEASING.md).

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
