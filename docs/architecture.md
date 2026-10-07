# Architecture and limitations

This page is for people who want to know how Saccade works inside: contributors, and anyone
deciding whether its trade-offs fit their use case.

## The pipeline

```text
Video ─► probe (PyAV) ─► fingerprint ─► SQLite run lookup ──► cached? ─► done
                                              │
      ┌───────────── background thread ───────┴─────────────┐
      │ decode audio (PyAV, 16 kHz mono, streamed in 5 s     │
      │ blocks; timeline gaps/overlaps corrected)            │
      │   ─► Silero VAD (ONNX, streaming, exact)             │
      │   ─► speech regions (padded, merged, ≤ 28 s)         │
      │   ─► packs of ~30 s speech + timestamp map           │
      └──────────────────────┬───────────────────────────────┘
                     bounded queue (3)
                             ▼
      faster-whisper (1 model, N threads; batched float16 on GPU) on speech only
         ─► map segment times back to the source timeline
         ─► SQLite transaction: segments + FTS5 + chunks + resume point
         ─► yield segment (already searchable)

      frames (in parallel on GPU, after ASR on CPU):
      decode ~1 fps ─► 64×36 gray thumbnail ─► change / scene / settle detection
         ─► 256-bit dHash dedup ─► JPEG (FFmpeg mjpeg) ─► frames table

      search():  FTS5 BM25 ─► temporal grouping ─► coverage / phrase scoring
      context(): hits ─► ±expansion ─► merge ─► token budget ─► evidence text
      ask():     transcript + thumbnails ─► LLM ⇄ tools (view_frames, zoom,
                 search_transcript, read_transcript) ─► answer + evidence
```

## Design notes

- **Silence never reaches Whisper.** Speech regions are concatenated into packs of about one
  Whisper window (30 s; the first pack is 10 s so the first results arrive quickly). Each
  pack keeps a map back to the original timeline.
- **Timestamps come from decoded PTS, never from guesses.** Audio is aligned to the media
  timeline (`pts − start_time`, gaps filled with silence, overlaps dropped). A pack only
  joins regions less than 3 s apart, because Whisper does not respect seams. A segment that
  merely grazes a neighbouring region across a seam is trimmed to the region it really
  covers.
- **Streaming everywhere.** Audio is decoded in 5-second blocks, VAD keeps its LSTM state
  between calls, and only the current pack is held. Memory stays flat however long the
  video is.
- **Every pack is one transaction.** Segments, FTS rows, chunks and the resume point commit
  together, so an interrupted run resumes exactly where it stopped and readers never see
  half a pack.
- **Cache keys cover everything that changes the output.** Backend identity (model,
  quantization, device, batching), language, VAD, chunking and the pipeline version. A new
  configuration creates a new run; results are never mixed.
- **Content fingerprints, not paths.** Size, first and last MiB, eight sampled 64 KiB
  windows, duration and codecs. Copying or renaming a file reuses its index.
- **Frames without a vision model.** Representative frames are chosen from tiny grayscale
  thumbnails, so selection costs almost nothing next to decoding. Screen recordings wait for
  transitions to settle before a frame is kept.
- **The LLM navigates; Saccade fetches.** In exploration mode the model receives an
  overview and requests exact frames, zooms (decoded at native resolution) and transcript
  ranges through tools. Saccade never writes conclusions itself.
- **Few dependencies.** faster-whisper, PyAV, onnxruntime and numpy. LLM clients use only
  the standard library. There is no PyTorch, OpenCV or Pillow.
- **Replaceable stages.** ASR backends implement `ASRBackend`, VAD implements
  `VoiceActivityDetector`, and LLM clients implement the `LLM` protocol. See
  [Extending Saccade](extending.md).

## Source layout

```text
src/saccade/
  video.py          public facade (Video, IndexSummary, IndexJob)
  pipeline.py       decode → VAD → pack → ASR → store, resumable
  ask.py            single-shot ask(): evidence selection and prompt
  explore.py        exploration agent: tools, loop, token accounting
  config.py         ASRConfig, VadConfig, ChunkConfig, VisualConfig, profiles
  media/            probe, fingerprint, streaming audio decode, frame grab
  vad/              streaming Silero + segmenter, fixed-window fallback
  asr/              backend protocol, faster-whisper backend (CPU and GPU)
  vision/           frame sampling, change detection, perceptual hashing
  index/            SQLite schema & access, FTS normalisation, chunking
  retrieval/        search, ranking, context assembly
  llm/              LLM protocol, Azure / OpenAI-compatible clients (stdlib only)
  models/           typed results: segments, chunks, frames, context, answers
  utils/            cache dirs & file lock, threading, GPU libraries, time
  cli.py            the `saccade` command
```

## Limitations

- **Frames are not "understood" locally.** Saccade selects and stores frames but runs no
  vision model; there is no OCR or captioning. What is *in* a frame is interpreted only by
  your LLM, through `ask()` or your own prompt. Search covers speech only.
- **Frame extraction decodes the video stream.** That is cheap next to ASR at typical
  resolutions, but not free for long 4K files. Use `VisualConfig(keyframes_only=True)` or a
  lower `sample_fps`.
- **Lexical search only.** There is no synonym or cross-language matching; the light
  stemmer is deliberately conservative. Semantic search is on the roadmap.
- **Language** is detected once per video, from its first speech. Videos that switch
  languages are transcribed in the first language detected; set `language=` if that is
  wrong.
- **No speaker labels.** Segments are not attributed to speakers (no diarization yet).
- **Thai, Lao, Khmer and other unspaced scripts** besides Chinese and Japanese are indexed
  as whole runs, so substring search inside them is limited.
- **Token budgets** are estimated (≈3.5 characters, or 1 CJK character, per token) unless
  you pass `token_counter=`.
- **Benchmarks** use synthetic TTS speech plus one real recording. Real recordings with
  noise, crosstalk and accents will have different accuracy and speech density.
