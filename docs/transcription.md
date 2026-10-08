# Transcription

Saccade transcribes speech with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(Whisper on CTranslate2). Silence is removed first by Silero VAD.

## Profiles and models

| Profile | Model | Beam | Use it for |
|---|---|---|---|
| `fast` | `base` | 1 | Quick previews, clean audio |
| `balanced` (default) | `small` | 1 | Most videos |
| `accurate` | `medium` | 5 | Hard audio, accents, names. Practical on a GPU |

```python
Video("talk.mp4", profile="fast")
video.index(profile="accurate")      # also becomes this object's active profile
```

All profiles use multilingual models. Quantization is automatic: int8 on CPU, float16 on GPU.
For full control, use `ASRConfig`:

```python
from saccade import ASRConfig, Video

Video("talk.mp4", asr=ASRConfig(
    model="large-v3-turbo",   # any faster-whisper model name, HF repo id or local directory
    compute_type="auto",      # or "int8", "int8_float16", "float16", ...
    beam_size=1,
    language=None,            # None = detect automatically
    word_timestamps=False,    # per-word timings (slower)
    carry_context=True,       # prompt each window with the previous text
))
```

Models download once into `<cache>/models`. To use another location, set
`SACCADE_MODELS_DIR`. To download in advance, for example before going offline:

```bash
saccade models download small medium
saccade models list
```

## Language

By default the language is detected from the first speech in the video:
`summary.language` and `summary.language_probability`. To set it explicitly, use
`Video("aula.mp4", language="pt")`.

The language is detected once per video. Videos that switch languages are transcribed in the
first language detected; set `language=` if that is wrong.

## Streaming segments

`transcribe()` yields segments in timeline order. Each one is already committed to the index,
and therefore searchable, when you receive it.

```python
for segment in video.transcribe(progress=print):
    print(f"[{segment.start:7.1f} → {segment.end:7.1f}] {segment.text}")
```

- **Already transcribed:** segments come straight from the cache.
- **Stopping early:** breaking out of the loop, or Ctrl-C, keeps everything done so far, and
  the next call resumes from there.
- **No audio:** raises `AudioStreamNotFoundError`. `index()` handles such videos gracefully
  and still extracts their frames.

Each segment is a `TranscriptSegment` with these fields:

| Field | Meaning |
|---|---|
| `id` | `seg_00042` |
| `start`, `end` | Seconds on the video timeline |
| `text` | The spoken text |
| `language` | Language code |
| `confidence` | `exp(avg_logprob)` as reported by Whisper. Useful to flag doubtful lines; it is **not** a calibrated probability |
| `no_speech_prob` | Whisper's probability that the window held no speech |
| `words` | Per-word timings, only with `word_timestamps=True` |

## `index()`

```python
summary = video.index(
    profile=None,            # "fast" | "balanced" | "accurate"
    force=False,             # redo the transcript and frames for this configuration
    progress=print,          # progress callback
    background=False,        # True: return an IndexJob immediately
    visual_strategy=None,    # "auto" | "screen" | "scenes" | "interval" | "off"
)
```

`IndexSummary` reports:

- **Counts:** `segments`, `chunks`, `frames`.
- **Speech:** `language`, `language_probability`, `speech_seconds`.
- **Status:** `status`, `indexed_until`, and `cached` (True when nothing new was processed).
- **Timing:** `elapsed_seconds` for this call, `transcription_seconds` of ASR time, and
  `real_time_factor`.

### In the background

```python
job = video.index(background=True, progress=print)

# The partial index is already searchable while the job runs:
video.search("rollback")

summary = job.wait()       # or job.done / job.error / job.cancel()
```

### Async

```python
summary = await video.aindex()
async for segment in video.atranscribe():
    ...
```

Decoding and inference run in worker threads, never on the event loop. Cancelling the task
stops after the current block and keeps progress.

## Progress events

Progress callbacks receive a `ProgressEvent`. `str(event)` gives a readable line.

| Field | Meaning |
|---|---|
| `stage` | A `Stage`: `inspect`, `cached`, `resume`, `download_model`, `load_model`, `detect_language`, `detect_speech`, `transcribe`, `frames`, `explore`, `complete`, `warning` |
| `message` | Human-readable text |
| `position`, `duration` | Seconds into the video, and the video length |
| `fraction` | `position / duration`: the share of the *timeline* processed, not a time estimate |

```python
def on_progress(event):
    if event.fraction is not None:
        bar.update(event.fraction)
```

## Voice activity detection

The defaults suit most videos. Tune them with `VadConfig`:

```python
from saccade import VadConfig

Video("lecture.mp4", vad=VadConfig(
    threshold=0.5,        # speech probability threshold
    min_speech_ms=250,    # drop shorter blips
    min_silence_ms=500,   # silence needed to end a region
    padding_ms=200,       # audio kept around speech
    merge_gap_ms=350,     # merge regions closer than this
    max_region_s=28.0,
))
```

`VadConfig(enabled=False)` sends every second of audio to Whisper. That is slower, and
Whisper tends to invent text during silence.

## Exports

```python
t = video.transcript()
Path("talk.srt").write_text(t.to_srt(), encoding="utf-8")
Path("talk.vtt").write_text(t.to_vtt(), encoding="utf-8")
Path("talk.md").write_text(t.to_markdown(title="talk.mp4"), encoding="utf-8")
Path("talk.json").write_text(t.to_json(), encoding="utf-8")
```

Markdown output uses a heading followed by one timestamped paragraph per segment:

```markdown
# talk.mp4

**[0:12.0]** We are switching authentication to managed identity...
```

Or from the CLI: `saccade transcribe talk.mp4 -f md -o talk.md`.
