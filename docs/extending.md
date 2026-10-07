# Extending Saccade

Saccade's main stages are replaceable through small protocols. You don't subclass anything;
any object with the right methods works.

## LLM clients

To use an LLM SDK or gateway that the bundled clients don't cover, implement `complete()`:

```python
from saccade.llm import LLMResponse, ToolCall

class MyLLM:
    supports_images = True     # False: ask() sends text only
    supports_tools = True      # True: enables exploration mode (optional; default False)

    def complete(self, messages, *, max_tokens=8000, tools=None, tool_choice=None):
        # ``messages`` and ``tools`` use the OpenAI chat-completions format.
        reply = my_sdk.chat(messages=messages, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens)
        return LLMResponse(
            text=reply.content or "",
            model=reply.model,
            usage={"prompt_tokens": reply.input_tokens, "completion_tokens": reply.output_tokens},
            tool_calls=tuple(ToolCall(c.id, c.name, c.arguments_dict) for c in reply.tool_calls),
            message=reply.raw_assistant_message,   # appended to the conversation as-is
        )

video = saccade.Video("talk.mp4", llm=MyLLM())
```

If you don't set `supports_tools`, `complete()` only needs `(messages, *, max_tokens)`; the
`tools` and `tool_choice` arguments are passed only to clients with `supports_tools = True`.

In the messages Saccade sends:

- `content` is a string or a list of `{"type": "text"}` and
  `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": ...}}`
  parts;
- tool results use `{"role": "tool", "tool_call_id": ..., "content": str}`;
- images requested by tools follow in a separate user message.

## ASR backends

A backend receives 16 kHz mono float32 NumPy audio and returns segments whose times are
**relative to that audio**. Saccade maps them back to the video timeline.

```python
from saccade.asr import ASRSegment, LanguageDetection

class MyASR:
    @property
    def identity(self):
        # Everything that changes the output. It becomes part of the cache key.
        return {"backend": "my-asr", "model": "v2"}

    def detect_language(self, audio):
        return LanguageDetection(language="en", probability=0.99)

    def transcribe(self, audio, *, language=None, word_timestamps=False, prompt=None):
        for start, end, text in my_engine.run(audio, sample_rate=16_000, language=language):
            yield ASRSegment(start=start, end=end, text=text)

video = saccade.Video("talk.mp4", asr_backend=MyASR())
```

To transcribe several speech chunks per call (for example on a GPU), also provide
`batch_size` (an int greater than 1) and
`transcribe_batch(audios, *, language, word_timestamps) -> list[list[ASRSegment]]`. Each
chunk is at most 30 s long, and each returned list is relative to its own chunk.

## Voice activity detection

The simplest extension is a different speech probability model inside the streaming
segmenter:

```python
from saccade import VadConfig
from saccade.vad import SileroVAD

class MyProbabilities:
    window_samples = 512                 # samples per decision at 16 kHz

    def reset(self): ...

    def __call__(self, windows):         # (n, window_samples) float32 -> (n,) probabilities
        return my_model(windows)

video = saccade.Video("talk.mp4", vad_factory=lambda: SileroVAD(VadConfig(), model=MyProbabilities()))
```

For full control, implement `VoiceActivityDetector` yourself:

| Member | Contract |
|---|---|
| `feed(block)` | Takes an `AudioBlock(start, samples)` and returns finished `SpeechRegion(start, end)`s |
| `flush()` | Returns the remaining regions at the end of the stream |
| `retain_from` | Seconds; the earliest time a future region could start, so older audio can be freed |

## Lower-level building blocks

These modules are usable on their own:

| Module | What it offers |
|---|---|
| `saccade.media` | `probe(path)`, `decode_audio(path, start=...)` (streamed 16 kHz blocks), `fingerprint(path, info)` |
| `saccade.media.frames` | `sample_frames(path, sample_fps=...)` and `grab_frame(path, time, box=...)`: exact frame at any time, optionally cropped, as JPEG |
| `saccade.index` | The SQLite `Database`, `build_match(query)` (FTS5 query builder) and `Chunker` |
| `saccade.retrieval` | `search_transcript(db, run_id, query)` and `build_context(...)` |
