# API reference

Everything below is importable from the top-level package (`import saccade`) unless a module
path is shown.

- [`Video`](#video)
- [Configuration](#configuration): `ASRConfig`, `VadConfig`, `ChunkConfig`, `VisualConfig`, `PROFILES`
- [LLM connections](#llm-connections): `azure`, `openai`, `ollama`, `AzureFoundry`, `OpenAICompatible`, `LLM`
- [Results](#results): `IndexSummary`, `IndexJob`, `VideoInfo`, `Transcript`, `TranscriptSegment`, `TranscriptWord`, `TranscriptChunk`, `SearchResult`, `VideoContext`, `Evidence`, `ContextMetadata`, `Frame`, `Answer`
- [Progress](#progress): `ProgressEvent`, `Stage`
- [Exceptions](#exceptions)

---

## `Video`

```python
Video(
    path,
    *,
    language=None, profile=None, device=None,
    asr=None, vad=None, chunking=None, visual=None,
    llm=None,
    threads=None, workers=1,
    cache_dir=None, offline=None, fingerprint="sampled",
    asr_backend=None, vad_factory=None,
)
```

Creating a `Video` is cheap and does no I/O; work happens when you call a method.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `path` | `str \| PathLike` | | Any media file FFmpeg reads (MP4, MKV, MOV, WEBM, WAV, MP3, ...) |
| `language` | `str \| None` | `None` | Spoken language (`"en"`, `"pt"`, ...); `None` detects it |
| `profile` | `"fast" \| "balanced" \| "accurate"` | `None` (balanced) | Model preset |
| `device` | `"cpu" \| "cuda" \| "auto"` | `None` (cpu) | Where Whisper runs |
| `asr` | `ASRConfig` | `ASRConfig()` | Full speech recognition settings |
| `vad` | `VadConfig` | `VadConfig()` | Voice activity detection settings |
| `chunking` | `ChunkConfig` | `ChunkConfig()` | Retrieval chunk sizes |
| `visual` | `VisualConfig` | `VisualConfig()` | Representative frame settings |
| `llm` | `LLM` | `None` | The LLM `ask()` uses |
| `threads` | `int` | Physical cores, max 8 | Inference threads (CPU) |
| `workers` | `int` | 1 | Concurrent Whisper calls |
| `cache_dir` | `str \| PathLike` | Platform cache dir | Where indexes, frames and models live |
| `offline` | `bool` | From `SACCADE_OFFLINE` / `HF_HUB_OFFLINE` | Never download models |
| `fingerprint` | `"sampled" \| "strict"` | `"sampled"` | Media identity: sampled bytes (instant) or full SHA-256 |
| `asr_backend` | `ASRBackend` | faster-whisper | Custom speech recognizer ([Extending](extending.md)) |
| `vad_factory` | `Callable[[], VoiceActivityDetector]` | Silero | Custom speech detector |

`language` and `profile` override the corresponding parts of `asr`.

### Methods

| Method | Returns | Description |
|---|---|---|
| `index(*, profile=None, force=False, progress=None, background=False, visual_strategy=None)` | `IndexSummary` (or `IndexJob` if `background=True`) | Transcribe and extract frames; cached afterwards |
| `transcribe(*, profile=None, force=False, progress=None)` | `Iterator[TranscriptSegment]` | Stream segments, transcribing only what's missing |
| `transcript()` | `Transcript` | The stored transcript; never transcribes |
| `chunks()` | `list[TranscriptChunk]` | 20–60 s retrieval chunks |
| `search(query, *, limit=5, start=None, end=None, expand=0.0)` | `list[SearchResult]` | Full-text search over the transcript |
| `context(query, *, max_tokens=4000, before=12.0, after=15.0, segment_timestamps=False, token_counter=None)` | `VideoContext` | Token-budgeted, timestamped evidence |
| `frames(*, start=None, end=None)` | `list[Frame]` | Stored representative frames |
| `ask(question, *, llm=None, progress=None, explore=None, max_steps=4, max_views=16, max_images=12, image_detail="low", evidence_tokens=48000, max_answer_tokens=8000)` | `Answer` | Answer a question with your LLM ([guide](asking-questions.md)) |
| `info()` | `VideoInfo` | Metadata and index status |
| `media_info()` | `MediaInfo` | Container and stream metadata |

Async versions:

| Async method | Equivalent of |
|---|---|
| `await aindex(*, profile=None, force=False, progress=None)` | `index()` |
| `async for s in atranscribe(...)` | `transcribe()` |
| `await asearch(query, **kw)` | `search()` |
| `await acontext(query, **kw)` | `context()` |
| `await aask(question, **kw)` | `ask()` |

### Properties

| Property | Meaning |
|---|---|
| `name` | File name |
| `path` | Absolute path |
| `fingerprint` | Content identity, e.g. `s1-a6aa13d5…` |
| `index_dir` | Folder holding this video's index and frames |
| `asr_config` | The active `ASRConfig` |

---

## Configuration

All configuration objects are immutable dataclasses. Derive variants with
`dataclasses.replace(config, field=value)`.

### `ASRConfig`

| Field | Default | Meaning |
|---|---|---|
| `model` | `"small"` | faster-whisper model name, Hugging Face repo id, or local CTranslate2 directory |
| `compute_type` | `"auto"` | `int8` on CPU, `float16` on GPU; or any CTranslate2 type |
| `device` | `"cpu"` | `"cpu"`, `"cuda"` or `"auto"` |
| `beam_size` | 1 | 1 = greedy (fastest), 5 = Whisper's accuracy default |
| `language` | `None` | ISO-639-1 code or `None` to detect |
| `word_timestamps` | `False` | Store per-word timings |
| `carry_context` | `True` | Prompt each window with the previous text (CPU, single worker) |
| `batch_size` | 16 | Speech chunks per batch on GPU |

`ASRConfig().with_profile("accurate")` returns a copy with that profile's model, quantization
and beam size.

### `PROFILES`

| Profile | Settings |
|---|---|
| `"fast"` | `base` model, beam 1 |
| `"balanced"` | `small` model, beam 1 |
| `"accurate"` | `medium` model, beam 5 |

All profiles use `compute_type="auto"`.

### `VadConfig`

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `True` | Skip non-speech before ASR |
| `threshold` | 0.5 | Speech probability threshold |
| `min_speech_ms` | 250 | Drop shorter detections |
| `min_silence_ms` | 500 | Silence needed to close a region |
| `padding_ms` | 200 | Audio kept around speech |
| `merge_gap_ms` | 350 | Merge padded regions closer than this |
| `max_region_s` | 28.0 | Split longer regions at a pause |

### `ChunkConfig`

| Field | Default | Meaning |
|---|---|---|
| `min_seconds` | 20.0 | Close a chunk at the first sentence end after this |
| `max_seconds` | 60.0 | Never exceed this |
| `break_on_pause_s` | 8.0 | A pause this long ends a chunk early |

### `VisualConfig`

| Field | Default | Meaning |
|---|---|---|
| `strategy` | `"auto"` | `"auto"`, `"screen"`, `"scenes"`, `"interval"` or `"off"` |
| `sample_fps` | 1.0 | Frames examined per second |
| `interval_s` | 60.0 | Spacing for `interval` and auto's periodic frames |
| `max_width` | 1280 | Stored JPEG width |
| `jpeg_quality` | 3 | FFmpeg qscale, 2 (best) .. 31 |
| `max_frames` | 1000 | Cap per video |
| `keyframes_only` | `False` | Decode keyframes only (about 5× faster, coarser) |

---

## LLM connections

| Function | Returns | Notes |
|---|---|---|
| `azure(endpoint=None, api_key=None, deployment=None, *, images=True, reasoning_effort=None)` | `AzureFoundry` | Arguments default to `AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY`, `AZURE_AI_DEPLOYMENT` |
| `openai(model="gpt-4o", api_key=None, *, base_url="https://api.openai.com/v1")` | `OpenAICompatible` | Key defaults to `OPENAI_API_KEY` |
| `ollama(model, *, base_url="http://localhost:11434/v1", images=False)` | `OpenAICompatible` | Local; tool calling off |

`OpenAICompatible(base_url, model, api_key=None, images=True, function_calling=True, reasoning_effort=None, headers={}, timeout=300.0)`
works with any server implementing `POST {base_url}/chat/completions`.

`AzureFoundry(endpoint, api_key, deployment, images=True, function_calling=True, reasoning_effort=None, timeout=300.0)`
is the class behind `azure()`. Its `repr()` never shows the key.

`LLM` is the protocol both satisfy. See [Extending](extending.md#llm-clients).

Both clients:

- use only the Python standard library;
- retry 429 and 5xx responses twice, honouring `Retry-After`;
- raise `LLMError` with the endpoint's response body otherwise.

---

## Results

### `IndexSummary`

| Fields | Meaning |
|---|---|
| `video`, `status`, `duration`, `language`, `language_probability` | What was indexed |
| `segments`, `chunks`, `frames`, `speech_seconds`, `indexed_until`, `model` | What the index holds |
| `cached` | True when nothing new was processed |
| `elapsed_seconds`, `transcription_seconds` | Timing |

The `real_time_factor` property gives ASR seconds per second of media. `to_dict()` serialises
the summary.

### `IndexJob`

Returned by `index(background=True)`:

| Member | Meaning |
|---|---|
| `wait(timeout=None)` | Returns the `IndexSummary`, re-raising any error |
| `done` | True when indexing has finished |
| `error` | The exception, if indexing failed |
| `cancel()` | Stops after the current block; progress so far is kept |

### `VideoInfo`

| Fields | Meaning |
|---|---|
| `path`, `name`, `fingerprint`, `media`, `index_dir` | The file and where its index lives |
| `transcript_status`, `language`, `segments`, `chunks`, `frames`, `indexed_until`, `model` | State of the active index |
| `runs` | Every stored processing run |

### `Transcript`

| Fields | Meaning |
|---|---|
| `segments`, `chunks` | The content |
| `language`, `language_probability`, `status`, `indexed_until`, `duration`, `model` | About the transcript |

Members: `text`, `to_srt()`, `to_vtt()`, `to_json()`, `to_dict()`, `len()` and iteration.

### `TranscriptSegment` and `TranscriptWord`

| Type | Fields |
|---|---|
| `TranscriptSegment` | `id`, `start`, `end`, `text`, `language`, `confidence`, `no_speech_prob`, `words`; property `duration` |
| `TranscriptWord` | `text`, `start`, `end`, `probability` |

### `TranscriptChunk`

Fields: `id`, `start`, `end`, `text`, `segment_ids`.

### `SearchResult`

Fields: `start`, `end`, `text`, `score` (relative, in (0, 1]), `source`, `segment_ids`,
`matched_terms`. Property `id` gives `seg_a–seg_b`.

### `VideoContext`

| Fields | Meaning |
|---|---|
| `query`, `text` | The question and the prompt-ready evidence |
| `segments` | Passages, as `TranscriptChunk`s |
| `frames` | Attached `Frame`s |
| `evidence` | `Evidence` items |
| `metadata` | A `ContextMetadata` |

Members: `found`, `to_json()`.

### `Evidence`

Fields: `id`, `type` (`"transcript"` or `"frame"`), `start`, `end`, `timestamp`, `text`.

### `ContextMetadata`

Fields: `video`, `duration`, `language`, `transcript_status`, `indexed_until`, `max_tokens`,
`estimated_tokens`, `passages`, `hits`, `model`.

### `Frame`

Fields: `id`, `timestamp`, `path`, `width`, `height`, `reason`. Methods: `read_bytes()`,
`data_url()`, `to_dict()`.

### `Answer`

| Fields | Meaning |
|---|---|
| `question`, `text` | The question and the LLM's answer |
| `evidence`, `frames`, `steps`, `mode` | What the LLM was given and did |
| `model`, `usage` | As reported by the endpoint |

Properties: `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens`,
`reasoning_tokens`. Methods: `to_dict()`, `to_json()`. `str(answer)` returns `text`.

---

## Progress

`ProgressEvent(stage, message, position, duration, start, end)`. The `fraction` property
gives `position / duration`, or `None` when unknown. `str(event)` gives a readable line.

`Stage` values: `inspect`, `cached`, `resume`, `download_model`, `load_model`,
`detect_language`, `detect_speech`, `transcribe`, `index`, `frames`, `explore`, `complete`,
`warning`.

---

## Exceptions

All errors derive from `SaccadeError`.

| Exception | Raised when |
|---|---|
| `ConfigError` (also a `ValueError`) | An invalid setting, e.g. `device="cuda"` without a GPU, or `ask()` without an LLM |
| `MediaNotFoundError` (also a `FileNotFoundError`) | The file does not exist |
| `UnsupportedFormatError` | Not a media file FFmpeg can read |
| `MediaDecodeError` | FFmpeg failed to decode; the message includes FFmpeg's error |
| `AudioStreamNotFoundError` | `transcribe()` on a file without audio |
| `ModelNotFoundError` | Unknown model, model missing while offline, or GPU libraries missing |
| `TranscriptionError` | The ASR backend failed |
| `NotIndexedError` | `search()`, `context()`, `transcript()` or `chunks()` before `index()` |
| `IndexLockedError` | Another process is indexing the same video |
| `DatabaseError` | The SQLite index could not be read or written |
| `LLMError` | The LLM endpoint rejected the request or was unreachable; carries `status` and `body` |
