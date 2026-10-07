# Saccade

Fast, local video context for LLMs.

Saccade turns hours of video into multilingual transcripts, representative frames,
timestamped evidence and searchable, RAG-ready context — locally on your CPU, or in seconds
on an NVIDIA GPU. No GPU required, no Docker, no database server.

```python
import saccade

video = saccade.Video("interview.mp4", llm=saccade.azure(endpoint=..., api_key=..., deployment="gpt-4o"))

print(video.ask("How was the interview?"))
```

`ask()` processes the video locally: it transcribes what was said and picks representative
frames of what was shown, and caches both. It then sends that evidence, with source
timestamps, to *your* LLM. Without an LLM, everything stays on your machine:

```python
from saccade import Video

video = Video("meeting.mp4")
video.index()

context = video.context("What did the team decide about authentication?")

print(context.text)
```

```text
VIDEO: meeting.mp4
DURATION: 1:02:13 · LANGUAGE: en · TRANSCRIPT: complete
QUERY: What did the team decide about authentication?

Relevant evidence (verbatim transcript; times refer to the source video):

[12:22.1–12:49.8 | seg_00042–seg_00045]
We are switching authentication to managed identity because the client secret expired...

[31:04.0–31:30.2 | seg_00118–seg_00120]
The App Service configuration still uses the old connection string...

Relevant visual evidence (frames from the video at these times):

[12:25.0 | frame_00031]
C:\Users\me\AppData\Local\saccade\videos\s1-…\frames\run1\frame_00031.jpg
```

Paste `context.text` into any LLM prompt. Every line is source material, labelled with the
source timestamps and segment or frame ids it came from. Saccade retrieves evidence; it
never writes conclusions of its own. The only generated text is your LLM's answer from
`ask()`, which comes back together with the exact evidence it was given.

**Documentation:** see [docs/](https://github.com/gabe-santana/saccade/blob/main/docs/README.md) for getting started, guides, the full API
reference, extension points and troubleshooting.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Asking an LLM](#asking-an-llm)
- [Performance profiles](#performance-profiles)
- [Transcription](#transcription)
- [Representative frames](#representative-frames)
- [Search](#search)
- [RAG context](#rag-context)
- [Multilingual support](#multilingual-support)
- [CPU tuning](#cpu-tuning)
- [GPU](#gpu)
- [Caching and incremental indexing](#caching-and-incremental-indexing)
- [Command line](#command-line)
- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [Limitations](#limitations)
- [Development](#development)

## Installation

```bash
pip install saccade-video        # the package is imported as `saccade`
```

Python 3.11+. Saccade has four runtime dependencies:

- **faster-whisper**, for ASR on CTranslate2;
- **PyAV**, FFmpeg bindings used for all decoding (wheels bundle FFmpeg, so you don't
  install it separately);
- **onnxruntime**, which runs Silero VAD;
- **numpy**.

PyTorch is not required.

The Whisper model (about 480 MB for `small`) is downloaded once, on first use, into
Saccade's cache directory. After that, nothing touches the network. To download ahead of
time and then run fully offline:

```bash
saccade models download small base
export SACCADE_OFFLINE=1        # forbid any download; missing models raise ModelNotFoundError
```

Frame extraction also uses PyAV, with JPEG encoding through FFmpeg, so it needs no OpenCV
or Pillow. LLM connections use only the standard library. Upcoming optional extras are
`saccade[embeddings]` (semantic search) and `saccade[whispercpp]`; see
[Roadmap](#roadmap).

## Quick start

```python
from saccade import Video

video = Video("tutorial.mp4")          # cheap: no work happens yet
summary = video.index()                # transcribe + index (cached afterwards)
print(summary.segments, summary.language, f"RTF {summary.real_time_factor:.3f}")

for hit in video.search("Azure App Service", limit=3):
    print(hit.start, hit.end, hit.score, hit.text)

context = video.context("How is Azure App Service configured?", max_tokens=4000)
prompt = f"Answer using only this evidence:\n\n{context.text}\n\nQuestion: {context.query}"
```

A second `Video("tutorial.mp4").index()` returns in milliseconds, and asking a different
question never re-transcribes.

## Asking an LLM

```python
import saccade

llm = saccade.azure()     # reads AZURE_AI_ENDPOINT, AZURE_AI_API_KEY, AZURE_AI_DEPLOYMENT
video = saccade.Video("interview.mp4", llm=llm)

answer = video.ask("How was the interview?", progress=print)
print(answer)                   # the LLM's answer, citing [mm:ss] timestamps
answer.evidence                 # transcript segments and frames that were sent, with source times
answer.frames                   # the images that were sent
```

Supported LLM connections:

| Connection | How to create it |
|---|---|
| Azure AI Foundry / Azure OpenAI | `saccade.azure(endpoint, api_key, deployment)` |
| OpenAI | `saccade.openai("gpt-4o")` (uses `OPENAI_API_KEY`) |
| Local Ollama (fully offline) | `saccade.ollama("llama3.1")` |
| Any OpenAI-compatible server | `saccade.OpenAICompatible(base_url, model, api_key)` |
| Your own client | any object with `complete(messages, max_tokens=...)` and `supports_images` (add `supports_tools` to enable exploration) |

For Azure, `endpoint` can be any URL the Foundry portal shows. That includes the resource
URL, `.../openai/v1`, `.../openai/v1/responses`, `.../models`, and the full target URI.
Reasoning models such as GPT-5 and o-series are supported: requests use
`max_completion_tokens`, and no `temperature` is sent.

### The model explores the video

Thumbnails are too small for questions like "describe everyone's appearance". So when the
LLM supports tool calling (Azure and OpenAI do), `ask()` lets the model **navigate the
video**. It starts from an overview and then asks for what it needs to see:

| Tool | What the model gets back |
|---|---|
| `view_frames(timestamps)` | The exact frames at those moments, in high resolution |
| `zoom(timestamp, x, y, width, height)` | One region of a frame (a participant's tile, small text on a slide) at the video's native resolution |
| `search_transcript(query)` | When something was said |
| `read_transcript(start, end)` | The verbatim transcript of a time range |

The overview is the timestamped transcript plus small thumbnails of the representative
frames. Frames are decoded on demand from the source file, about 0.1–0.25 s each for a
3440×1440 recording, so the model can look at *any* moment, not only the stored frames.

The model typically finds *when* things happen in the transcript, looks at those moments,
zooms in where details matter, then answers. Exploration is bounded by `max_steps` (4 rounds)
and `max_views` (16 images). Every image the model saw is saved next to the index and returned
as evidence with its exact timestamp:

```python
answer = video.ask("Describe everyone's appearance in the meeting", progress=print)
answer.steps     # ["Searching the transcript for 'introduce'", "Looking at 1:30.0, 6:15.0", "Zooming into 6:15.0", ...]
answer.frames    # overview thumbnails + every frame and zoom the model looked at
```

`explore=False` turns this off and sends one fixed set of evidence instead. That is cheaper,
and it is what LLMs without tool support get automatically. This fixed evidence is:

- **Transcript.** The full timestamped transcript when it fits `evidence_tokens` (default
  48k). Longer videos get only the passages retrieved for the question.
- **Frames.** Up to `max_images` frames (default 12) at `image_detail="low"`, spread over the
  video and near the retrieved passages.
- **Instructions.** Use only that evidence, cite timestamps, and say when the evidence is
  insufficient.

### Token costs

Chat APIs are stateless, so every exploration round resends the conversation so far: the
transcript (about 5k tokens for a 26-minute meeting), the thumbnails and every image fetched
so far. Saccade keeps this small:

- **Small images.** Full frames are sent at 1024 px (about 425 tokens) and zooms at 768 px
  (about 425 tokens). The model zooms only where it needs detail.
- **Prompt caching.** Each round only appends to the conversation, so providers that cache
  prompt prefixes bill the repeated part at a large discount. Azure OpenAI does this
  automatically; for GPT-5 models cached input costs about 90% less. `answer.cached_tokens`
  shows how much was cached.
- **Your controls:**
  - `saccade.azure(reasoning_effort="low")` reduces the hidden thinking tokens of reasoning
    models. They are reported in `answer.reasoning_tokens`.
  - `max_steps` and `max_views` cap how much the model explores.
  - `explore=False` makes a single call.

```python
print(answer.input_tokens, answer.cached_tokens, answer.output_tokens, answer.reasoning_tokens)
```

Saccade contacts an LLM only when you configure one and call `ask()`. `video.aask()` is the
async version. From the CLI:

```bash
saccade ask interview.mp4 "How was the interview?"   # with AZURE_AI_* set
```

## Performance profiles

| profile | model | quantization | beam |
|---|---|---|---|
| `fast` | `base` | int8 | 1 |
| `balanced` (default) | `small` | int8 | 1 |
| `accurate` | `medium` | int8 | 5 |

```python
video.index(profile="fast")
Video("talk.mkv", profile="accurate")
```

All profiles are multilingual Whisper models. English-only `*.en` models are never used by
default. For full control:

```python
from saccade import ASRConfig, VadConfig, Video

video = Video(
    "meeting.mp4",
    asr=ASRConfig(model="large-v3-turbo", compute_type="int8", beam_size=1, language=None),
    vad=VadConfig(min_speech_ms=250, min_silence_ms=500, padding_ms=200, merge_gap_ms=350),
)
```

`ASRConfig.model` also accepts a local CTranslate2 model directory.

## Transcription

`transcribe()` streams segments in timeline order. Each segment is yielded only once it has
been committed to the index, so it is already searchable.

```python
for segment in video.transcribe():
    print(f"[{segment.start:7.1f} → {segment.end:7.1f}] {segment.text}")
```

```python
async for segment in video.atranscribe():   # decoding and inference run off the event loop
    print(segment.text)
```

- **Language** is detected automatically from the first speech, or set explicitly:
  `Video("aula.mp4", language="pt")`.
- **Interrupting** a transcription (break out of the loop, Ctrl-C, cancel the task) keeps
  everything done so far. The next call resumes from that point.
- **Word timestamps** are off by default because they cost extra decoding time. Turn them on
  with `ASRConfig(word_timestamps=True)`.
- **Exports:** `video.transcript()` returns the stored transcript, with `.to_srt()`,
  `.to_vtt()`, `.to_json()` and `.text`.
- **Confidence:** `segment.confidence` is `exp(avg_logprob)` as reported by Whisper. It is
  useful for spotting doubtful passages, but it is *not* a calibrated probability of
  correctness.

Long-running jobs can index in the background while the rest of your program queries the
partial index:

```python
job = video.index(background=True, progress=print)
...
video.search("rollback")          # sees everything committed so far
summary = job.wait()
```

### Progress

```python
video.index(progress=lambda event: print(event))
```

```text
Inspecting media: meeting.mp4
Detecting spoken language (0% of timeline)
Loading Whisper model 'small' (int8)
Detected language: en (p=1.00) (0% of timeline)
Transcribing 00:00:00–00:00:08 (1% of timeline)
Transcribing 00:00:09–00:00:33 (6% of timeline)
...
Completed: 62 segments, language en
```

Events are typed (`ProgressEvent` with `stage`, `position`, `duration` and `fraction`).
`fraction` is the share of the *timeline* processed so far. It is never an estimated
percentage of remaining work.

## Representative frames

`index()` also stores a small set of frames that show what is on screen. Frames are chosen
cheaply, without a vision model and without extracting every frame:

1. About one decoded frame per second is reduced to a 64×36 grayscale thumbnail. Non-reference
   frames are skipped at decode time.
2. A frame is kept when it differs clearly from the last kept one. On screen recordings and
   slides, Saccade waits for the transition to settle, so half-drawn pages and animations are
   skipped.
3. Duplicates are dropped by comparing a 256-bit difference hash plus the thumbnail difference
   against recently kept frames. Going back to an earlier slide does not store it again.
4. Timestamps are the decoded presentation times on the video timeline (correct for
   variable-frame-rate video). They are never a frame number multiplied by a nominal fps.

```python
for frame in video.frames():            # or video.frames(start=600, end=900)
    print(frame.timestamp, frame.id, frame.path, frame.reason)
frame.data_url()                        # "data:image/jpeg;base64,..." for multimodal APIs
```

| `visual_strategy` | Use for | Behaviour |
|---|---|---|
| `auto` (default) | anything | Detects static content: there it keeps every settled change; on camera footage, clear changes plus one frame per minute while the picture moves |
| `screen` | screen recordings, slides | Sensitive to UI changes, waits for transitions to settle |
| `scenes` | edited footage | Hard cuts only |
| `interval` | sparse coverage | One frame every `interval_s` (60 s), duplicates skipped |
| `off` | audio-only use | No frames |

```python
video.index(visual_strategy="screen")
saccade.Video("talk.mp4", visual=saccade.VisualConfig(strategy="interval", interval_s=30, max_width=1024))
```

Frames are JPEGs, at most 1280 px wide by default, stored next to the index. A 26-minute
interview produced 52 frames.

## Search

```python
results = video.search("OAuth authentication error", limit=5)
results = video.search("deployment error", start=600, end=1200)   # 10:00–20:00 only
results = video.search("rollback", expand=10)                      # ±10 s of surrounding text
```

```python
SearchResult(start=742.2, end=769.8, text="...", score=1.0, source="transcript",
             segment_ids=("seg_00042", "seg_00043"), matched_terms=("oauth", "authentication"))
```

Search uses SQLite FTS5 with BM25 and needs no embeddings. On top of BM25:

- **Stop words are dropped.** Natural questions such as *"Why did the deployment fail?"* turn
  into an OR query over their content words (stop-word lists for en/pt/es/fr/de).
- **Light stemming.** Each term also matches a stemmed prefix, so `deployment` matches
  `deploy`, `deployed` and `deploying`.
- **Phrase bonus.** Passages containing the query as a verbatim phrase rank higher.
- **Temporal grouping.** Hits within 8 s of each other become one passage, so several
  query terms said close together outrank an isolated mention.

`score` is relative to the best result for that query, in (0, 1]. It ranks results; it is
not a probability.

## RAG context

```python
context = video.context(
    "Why did deployment fail?",
    max_tokens=4000,          # budget for context.text
    before=12, after=15,      # seconds of surrounding transcript around each hit
    segment_timestamps=False, # True: timestamp every segment inside a passage
    token_counter=None,       # e.g. lambda s: len(tokenizer.encode(s))
)

context.text        # the prompt-ready string
context.segments    # passages (TranscriptChunk), each with segment_ids
context.evidence    # one Evidence(id, type, start, end, text) per source segment
context.metadata    # video, duration, language, transcript status, token estimate...
context.to_json()
```

How `context()` works:

1. Each hit is widened by `before`/`after` seconds of transcript and snapped to segment
   boundaries, so the LLM sees what was said around a mention rather than an isolated
   sentence.
2. Overlapping windows are merged.
3. Passages are added best-first until the budget is spent, then printed in timeline order.
4. While a video is still being indexed, the header says `TRANSCRIPT: partial, indexed up
   to 23:10`, so the LLM knows the evidence may be incomplete.

For your own vector store, `video.chunks()` returns the stored retrieval chunks. These are
groups of 20–60 s of consecutive segments that close at sentence boundaries or long pauses.

## Multilingual support

- **Transcription:** any language multilingual Whisper supports, detected automatically.
- **Search:** text is NFKC-normalised and case-folded, and Latin diacritics are folded.
  `configuracao` finds *configuração*, and `strasse` finds *Straße*. Combining marks stay
  part of the word, so Devanagari, Arabic and Cyrillic work. Chinese and Japanese are
  indexed per character and matched by bigrams, so `身份验证` finds 部署失败是因为**身份验证**配置错误.
- **Unicode** is preserved end to end: ASR → SQLite → search → CLI → JSON. The CLI forces
  UTF-8 output even on legacy Windows code pages, and JSON is written with
  `ensure_ascii=False`.

Search is lexical. A query in one language does not find passages spoken in another; that
needs the optional embeddings planned for Phase 3.

## CPU tuning

```python
Video("meeting.mp4", threads=8, workers=1)
```

- `threads` (default: number of physical cores, capped at 8) is CTranslate2's intra-op thread
  pool. More than about 8 rarely helps Whisper, and leaving cores free keeps the machine
  responsive.
- `workers` (default 1) is the number of concurrent ASR calls. One worker with internal
  threading is the efficient configuration; raise it only on machines with many idle cores.
  Each worker gets `threads // workers` threads.

Concurrency layout: one background thread decodes audio and runs VAD (single-threaded
onnxruntime) into a bounded queue of at most 3 packs. The caller's thread runs one Whisper
model. Memory stays bounded however long the video is. There is no multiprocessing, and
never one Whisper per core.

## GPU

```bash
pip install "saccade-video[gpu]"      # NVIDIA's CUDA 12 libraries as pip wheels; no CUDA toolkit needed
```

```python
video = saccade.Video(
    "interview.mp4",
    device="cuda",                                     # or "auto": GPU when available, else CPU
    visual=saccade.VisualConfig(keyframes_only=True),  # frame extraction that keeps up with the GPU
)
```

What changes on the GPU:

- **Batched transcription.** Whisper runs in float16 and transcribes 16 speech chunks per
  batch (`ASRConfig(batch_size=...)`).
- **Overlapped work.** Frames are extracted while transcription runs, since the CPU is mostly
  idle.

Your existing CPU indexes stay valid. GPU results are cached separately, because batched
decoding segments text slightly differently.

Measured on a 26-minute 3440×1440 interview recording, from a cold index cache with the model
already downloaded (i7-14650HX, RTX 5050 Laptop GPU):

| Setup | Full indexing | Speech recognition |
|---|---|---|
| CPU (`small`, int8) | 175 s | 122 s |
| GPU (`small`, float16) | 41 s, limited by decoding every 3440×1440 frame | 7 s |
| GPU + `keyframes_only=True` | **9 s** (first transcript lines after 2.3 s) | 7 s |

On the GPU, the larger models become practical. For example,
`Video(..., device="cuda", profile="accurate")` uses `medium` with beam search.

The first GPU run on a brand-new GPU architecture can take a few extra seconds while CUDA
compiles its kernels; the driver caches the result.

## Caching and incremental indexing

All results live in SQLite under the cache directory (default `%LOCALAPPDATA%\saccade`,
`~/Library/Caches/saccade` or `~/.cache/saccade`; override with `cache_dir=` or
`SACCADE_CACHE_DIR`). There is one database per video, at
`videos/<fingerprint>/index.db`. `video.info().index_dir` tells you where.

- **Fingerprint.** Saccade identifies media by content, without hashing whole files: size,
  the first and last MiB, eight evenly spaced 64 KiB samples, duration and codecs. It reads
  about 2.5 MiB. Modification time is deliberately ignored, so copying or moving a file
  reuses its index. `Video(..., fingerprint="strict")` hashes every byte with SHA-256.
- **Cache key.** Every transcript belongs to a *run* keyed by everything that changes the
  output: ASR backend, model, quantization, beam size, language setting, word timestamps,
  VAD settings, chunking and pipeline version. Changing any of them creates a new run and
  never mixes results. `index(force=True)` redoes the current configuration.
- **Incremental indexing.** Each ~30 s block of speech is committed in one transaction,
  together with the resume point. Searches from any thread or process see new segments
  immediately (WAL mode), and an interrupted run continues where it stopped. A file lock
  prevents two writers indexing the same video at once.

## Command line

```bash
saccade index meeting.mp4 [more.mkv ...] [--profile fast] [--language pt]
saccade transcribe meeting.mp4                  # streams lines as they are transcribed
saccade transcribe meeting.mp4 -f srt -o meeting.srt
saccade search meeting.mp4 "authentication" [--start 10:00 --end 20:00] [--json]
saccade context meeting.mp4 "Why did deployment fail?" [--max-tokens 4000] [--json]
saccade frames meeting.mp4 [--start 10:00] [--visual screen] [--json]
saccade ask meeting.mp4 "How was the interview?"   # Azure via AZURE_AI_*, or --base-url/--llm-model
saccade info meeting.mp4 [--json]
saccade models download small
saccade models list
```

`search` and `context` index the video first if needed. Progress goes to stderr and results
to stdout. `-q` silences progress, and `-v`/`-vv` enable structured logs.

## Architecture

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
      faster-whisper (1 model, N threads) on speech only
         ─► map segment times back to the source timeline
         ─► SQLite transaction: segments + FTS5 + chunks + resume point
         ─► yield segment (already searchable)
                             ▼
      search(): FTS5 BM25 ─► temporal grouping ─► coverage/phrase scoring
      context(): hits ─► ±expansion ─► merge ─► token budget ─► evidence text
```

Design notes:

- **Silence never reaches Whisper.** Speech regions are concatenated into packs of about one
  Whisper window. Each pack keeps a map back to the original timeline.
- **Timestamps come from decoded PTS, never from guesses.** A pack only joins regions less
  than 3 s apart, because Whisper does not respect seams. A segment that merely grazes a
  neighbouring region across a seam is trimmed to the region it really covers.
- **Replaceable stages.** ASR backends implement `ASRBackend`
  (`transcribe(audio, language=...)` and `detect_language(audio)`). VAD implements
  `VoiceActivityDetector`, and the probability model behind Silero is pluggable too. Pass
  them with `Video(..., asr_backend=..., vad_factory=...)`.

```text
src/saccade/
  video.py          public facade (Video, IndexSummary, IndexJob)
  pipeline.py       decode → VAD → pack → ASR → store, resumable
  config.py         ASRConfig, VadConfig, ChunkConfig, profiles
  media/            probe, fingerprint, streaming audio decode
  vad/              streaming Silero + segmenter, fixed-window fallback
  asr/              backend protocol, faster-whisper backend
  index/            SQLite schema & access, FTS normalisation, chunking
  retrieval/        search, ranking, context assembly
  models/           typed results: segments, chunks, search results, context
  utils/            cache dirs & file lock, threading helpers, time formatting
  cli.py            the `saccade` command
```

## Benchmarks

Measured with `benchmarks/bench.py` on synthetic meetings: Windows SAPI speech laid out with
realistic pauses, from `benchmarks/make_media.py`. Every run starts from a cold index cache;
the model is already downloaded. These numbers come from one machine, so treat them as
indicative and run the suite on your own hardware.

**Machine:** Intel Core i7-14650HX (16 cores / 24 threads), 32 GB RAM, Windows 11,
Python 3.12. Saccade 0.1.0, `threads=8`, `workers=1`.

| Video | Lang | Speech | Profile | RTF | First segment | Index time | Peak RSS | Warm cache | Search p50 |
|---|---|---|---|---|---|---|---|---|---|
| 10 min | en | 50% | balanced (`small`) | 0.045 | 3.1 s | 27 s | 594 MB | 19 ms | 0.3 ms |
| 10 min | pt | 59% | balanced (`small`) | 0.046 | 2.8 s | 27 s | 746 MB | 19 ms | 0.5 ms |
| 30 min | en | 58% | balanced (`small`) | 0.053 | 3.0 s | 1 min 36 s | 696 MB | 18 ms | 0.6 ms |
| 2 h | en | 57% | balanced (`small`) | 0.074 | 3.0 s | 8 min 54 s | 1135 MB | 36 ms | 2.3 ms |
| 10 min | en | 50% | fast (`base`) | 0.027 | 7.6 s¹ | 16 s | 381 MB | 19 ms | 0.4 ms |
| 10 min | pt | 59% | fast (`base`) | 0.024 | 1.1 s | 14 s | 416 MB | 21 ms | 0.9 ms |

Language was detected correctly in every run.

¹ Includes the one-time download of the `base` model.

These runs cover transcription and indexing only; they were measured before frame
extraction was added. Frame extraction decodes the video stream on top of this. Decode,
VAD and packing alone process the 2-hour file in 14 s at a flat ~95 MB. Whisper inference
dominates both time and memory.

Columns:

- **RTF:** wall-clock indexing time ÷ media duration, including decode, VAD, ASR and
  indexing.
- **First segment:** time from the start of indexing until the first segment is committed,
  including model load.
- **Warm cache:** a fresh `Video(...).index()` on an indexed file.
- **Search:** median over 35 queries.
- **Speech:** the share of the timeline that VAD sent to ASR.

```bash
python benchmarks/make_media.py --minutes 30 --language en benchmarks/media/meeting-30m-en.mp4
python benchmarks/bench.py benchmarks/media/*.mp4 --profile balanced
```

## Limitations

- **Frames are not "understood" locally.** Saccade selects and stores frames but runs no
  vision model; there is no OCR or captioning. What is *in* a frame is interpreted only by
  your LLM, through `ask()` or your own prompt. Search covers speech only.
- **Frame extraction decodes the video stream.** That is cheap next to ASR at typical
  resolutions, but it is not free for long 4K files. Lower `VisualConfig.sample_fps` to
  reduce it.
- **Lexical search only.** There is no synonym or cross-language matching; the light
  stemmer is deliberately conservative.
- **Language** is detected once per video, from its first speech. Videos that switch
  languages are transcribed in the first language detected; set `language=` if that is
  wrong.
- **Thai, Lao, Khmer and other unspaced scripts** besides Chinese and Japanese are indexed
  as whole runs, so substring search inside them is limited.
- **Token budgets** are estimated (≈3.5 characters, or 1 CJK character, per token) unless
  you pass `token_counter=`.
- **Benchmarks** use synthetic TTS speech. Real recordings with noise, crosstalk and accents
  will have different accuracy and speech density.

## Roadmap

- **Done:** Phase 1 (transcription, indexing, search and context) and Phase 2
  (representative frames, change and scene detection, perceptual deduplication, frames in
  context), plus `ask()` with pluggable LLM connections.
- **Next (Phase 3):** optional multilingual embeddings and hybrid lexical/semantic/temporal
  ranking, a whisper.cpp backend, cache management commands, and extended benchmarks
  across CPU classes.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
pytest                                  # fast, deterministic suite (no models needed)
SACCADE_INTEGRATION=1 pytest -m integration   # real faster-whisper + Silero (Windows TTS voices)
ruff check src tests && ruff format --check src tests && mypy
```

The default test suite exercises the whole pipeline with deterministic stand-ins. An
energy-based speech model replaces Silero's network, and a fake ASR names the pitch of
tone bursts placed at known times. That makes timestamp restoration, resume, caching and
ranking exactly testable without downloading models.
