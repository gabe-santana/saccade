# Search and RAG context

These methods run entirely locally and need no LLM. Use them to build your own retrieval or
chat pipeline on top of Saccade.

All of them read the index, so call `video.index()` first. If you don't, they raise
`NotIndexedError`. They also work on a partial index while indexing is still running.

## `search()`: find where something was said

```python
results = video.search("authentication error", limit=5)

for r in results:
    print(f"[{r.start:.1f}–{r.end:.1f}] score={r.score:.2f} {r.text}")
```

| Parameter | Default | Meaning |
|---|---|---|
| `query` | | Free text in any language |
| `limit` | 5 | Maximum number of results |
| `start`, `end` | `None` | Restrict to part of the timeline (seconds) |
| `expand` | 0 | Seconds of surrounding transcript to add to each result |

Each `SearchResult` has these fields:

| Field | Meaning |
|---|---|
| `start`, `end` | Source timestamps of the passage |
| `text` | The passage's transcript text |
| `score` | Relative to the best result, in (0, 1]. It ranks results; it is not a probability |
| `segment_ids` | The segments the passage is made of |
| `matched_terms` | Which query terms were found |

How matching works:

- **Algorithm.** SQLite FTS5 with BM25 ranking.
- **Question-style queries.** Stop words are dropped (en, pt, es, fr, de), so *"Why did the
  deployment fail?"* works.
- **Word variants.** Words also match by a stemmed prefix: `deployment` finds *deploy*,
  *deployed* and *deploying*.
- **Ranking.** A verbatim phrase match ranks higher, and hits said close together are
  grouped into one passage.
- **Normalisation.** Accents and case are folded, so `configuracao` finds *configuração*.
  Chinese and Japanese are matched by character bigrams.

Search is lexical. It finds the words you use, not synonyms or translations.

## `context()`: prompt-ready evidence for a question

```python
context = video.context(
    "Why did the deployment fail?",
    max_tokens=4000,        # budget for context.text
    before=12, after=15,    # seconds of transcript around each hit
)
print(context.text)
```

```text
VIDEO: meeting.mp4
DURATION: 1:02:13 · LANGUAGE: en · TRANSCRIPT: complete
QUERY: Why did the deployment fail?

Relevant evidence (verbatim transcript; times refer to the source video):

[12:22.1–12:49.8 | seg_00042–seg_00045]
The release failed on Friday because the authentication secret expired...

Relevant visual evidence (frames from the video at these times):

[12:25.0 | frame_00031]
/home/me/.cache/saccade/videos/s1-…/frames/run1/frame_00031.jpg
```

How passages are chosen:

1. **Expand.** Each hit is widened by `before` and `after` seconds of transcript, so the LLM
   sees what was said around a mention, not an isolated sentence.
2. **Merge.** Overlapping windows are merged.
3. **Budget.** Passages are added best-first until `max_tokens` is used, then printed in
   timeline order.
4. **Frames.** Up to two frames from inside each passage are attached.

Token counts are estimated, at roughly 3.5 characters per token. For exact budgets, pass your
tokenizer:

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
context = video.context("...", max_tokens=8000, token_counter=lambda s: len(enc.encode(s)))
```

`segment_timestamps=True` prefixes every sentence inside a passage with its own time. That
helps an LLM cite precise moments.

The returned `VideoContext` has these fields:

| Field | Meaning |
|---|---|
| `text` | The prompt-ready string |
| `segments` | The passages, each with `segment_ids` |
| `frames` | Frames attached to the passages |
| `evidence` | One `Evidence(id, type, start/end or timestamp, text)` per source item |
| `metadata` | Video, duration, language, transcript status, estimated tokens, hit count |
| `found` | `False` when nothing matched |

`to_json()` serialises everything.

### Using it with any LLM

```python
context = video.context(question)
prompt = (
    "Answer using only the evidence below and cite timestamps.\n\n"
    f"{context.text}\n\nQuestion: {question}"
)
reply = my_llm(prompt)
```

`context.frames[i].data_url()` returns a `data:image/jpeg;base64,...` string. Use it for
multimodal APIs.

## `chunks()`: units for your own vector store

```python
for chunk in video.chunks():
    my_vector_store.add(
        id=f"{video.fingerprint}:{chunk.id}",
        text=chunk.text,
        metadata={"start": chunk.start, "end": chunk.end, "video": video.name},
    )
```

Each chunk covers 20–60 s of consecutive speech. It closes at a sentence boundary or a long
pause, never overlaps another chunk, and lists its `segment_ids`, so every embedding stays
traceable to exact timestamps. Tune the grouping with
`Video(..., chunking=ChunkConfig(min_seconds=20, max_seconds=60, break_on_pause_s=8))`.

## `transcript()`: the full transcript

```python
transcript = video.transcript()
transcript.status            # "complete", "partial", ...
transcript.language          # detected or configured language
transcript.text              # plain text, one segment per line
transcript.to_srt()          # subtitles
transcript.to_vtt()
transcript.to_json()
for segment in transcript:   # TranscriptSegment(id, start, end, text, language, confidence, ...)
    ...
```

## Async variants

`await video.asearch(query, ...)` and `await video.acontext(query, ...)` run in a worker
thread and don't block your event loop.
