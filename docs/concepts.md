# Concepts

A short tour of how Saccade works. Read it once and the rest of the API will feel
predictable.

## The pipeline

```text
video file
 ├─ audio ─► decode (16 kHz mono) ─► voice activity detection ─► speech only ─► Whisper ─► segments
 └─ video ─► sample ~1 frame/s ─► detect visual changes ─► drop duplicates ─► representative frames
                                                   │
                                     SQLite index (segments, chunks, frames, full-text search)
                                                   │
             search() · context() · transcript() · frames() · ask() ─► your LLM
```

- **Speech only reaches Whisper.** Silence is detected and skipped before transcription.
  That makes transcription faster and avoids text Whisper sometimes invents for silence.
- **Everything is streamed.** Audio and video are decoded in small blocks, so memory use does
  not grow with video length.
- **Results become searchable immediately.** Each transcribed block is committed to the index
  as soon as it is done. You can search a 2-hour video while it is still being indexed.

## Timestamps are the source of truth

Every segment and frame carries times in seconds from the start of the video, measured from
the media itself. Frame times come from the decoded frames' timestamps (correct for
variable-frame-rate video). Times are never estimated from frame numbers.

When an LLM answers with `ask()`, its citations point back to these timestamps.
`answer.evidence` lists exactly which segments and frames it was given.

## Identifiers

| Thing | Example id | Where you see it |
|---|---|---|
| Transcript segment (a sentence or two) | `seg_00042` | `transcript()`, search results, evidence |
| Retrieval chunk (20–60 s of segments) | `chunk_00007` | `chunks()` |
| Representative frame | `frame_00031` | `frames()`, context, evidence |
| Frame the LLM requested while exploring | `view_003` | `answer.frames`, `answer.evidence` |

## What gets cached, and when Saccade reprocesses

Each video gets its own index, identified by a **content fingerprint**. Saccade hashes the
file size, about 2.5 MB of sampled bytes and basic stream info. The rest of the file is not
read, so fingerprinting is instant even for huge files.

| You change… | Effect |
|---|---|
| The question you ask | Nothing is reprocessed |
| The file's name or location (copy or move) | Nothing is reprocessed; the fingerprint is the same |
| The file's content | New fingerprint, new index |
| Model, profile, language, VAD settings or `device` | A new transcript **run** is made; the old one is kept |
| Visual settings (strategy, `keyframes_only`, ...) | A new frames run is made |

`index(force=True)` redoes the current configuration.

Interrupted runs resume. If indexing stops through Ctrl-C, a crash or `job.cancel()`, the
next `index()` or `transcribe()` continues from the last committed block.

## Statuses

`transcript().status`, `IndexSummary.status` and `VideoInfo.transcript_status` take these
values:

| Status | Meaning |
|---|---|
| `complete` | The whole video was transcribed |
| `partial` | Indexing stopped or is still running; `indexed_until` says how far it got |
| `running` | Another process or thread is indexing right now |
| `empty` | The video has no audio, or no speech was found |
| `failed` | The last run failed (the error is logged; re-run to resume) |

## Offline by design

Saccade only uses the network in two situations, and both are explicit:

1. **Model downloads.** A Whisper model is downloaded the first time you use it. After
   that, models load from disk without any network check. Set `SACCADE_OFFLINE=1` or pass
   `Video(..., offline=True)` to forbid downloads entirely.
2. **LLM calls.** A request goes to your LLM only when you configure one and call `ask()`.

## Concurrency

- **Reading is always safe.** `search()`, `context()` and `frames()` can run from any thread
  or process at any time, including while indexing is in progress.
- **One writer per video.** If a second process tries to index the same video at the same
  time, it gets `IndexLockedError`.
