# Asking questions with an LLM

`video.ask(question)` answers a question from what was said **and** shown in the video, using
an LLM you configure. Saccade does the video work locally and sends your LLM only extracted
evidence (transcript text and still images), never the video file.

```python
import saccade

video = saccade.Video("interview.mp4", llm=saccade.azure())
answer = video.ask("How did the candidate describe their last project?", progress=print)

print(answer)            # same as answer.text
```

## Connecting an LLM

Pass the connection when you create the `Video`, or per call with `ask(..., llm=...)`.

### Azure AI Foundry / Azure OpenAI

```python
llm = saccade.azure()     # reads AZURE_AI_ENDPOINT, AZURE_AI_API_KEY, AZURE_AI_DEPLOYMENT

llm = saccade.azure(
    endpoint="https://<resource>.services.ai.azure.com/openai/v1/responses",
    api_key="...",                 # prefer the environment variable; never commit keys
    deployment="gpt-5-mini",       # your deployment name
    reasoning_effort="low",        # optional, for reasoning models (GPT-5, o-series)
)
```

`endpoint` accepts any URL the Foundry portal shows for a deployment:

- the resource URL (`https://<resource>.openai.azure.com` or `https://<resource>.services.ai.azure.com`);
- the v1 API (`.../openai/v1` or `.../openai/v1/responses`);
- the model inference API (`.../models`);
- the full target URI ending in `/chat/completions?api-version=...`.

Reasoning models are handled automatically. Requests use `max_completion_tokens`, and no
`temperature` is sent.

### OpenAI

```python
llm = saccade.openai("gpt-4o")          # uses OPENAI_API_KEY
```

### Local models (Ollama, LM Studio, vLLM)

```python
llm = saccade.ollama("llama3.1")        # text-only by default; fully offline
llm = saccade.ollama("llava", images=True)

llm = saccade.OpenAICompatible(
    base_url="http://localhost:1234/v1",
    model="qwen2.5-vl-7b",
    images=True,              # the model accepts images
    function_calling=True,    # the server supports tool calls (enables exploration)
)
```

### Your own client

Any object with `supports_images`, a `complete(messages, *, max_tokens)` method and,
optionally, `supports_tools` works. See [Extending](extending.md#llm-clients).

## Two ways of answering

### Exploration (default for Azure and OpenAI)

When the LLM supports tool calling and images, `ask()` lets the model **navigate the
video**:

1. **Overview.** It starts from the timestamped transcript and small thumbnails of the
   representative frames.
2. **Tools.** It then calls tools to gather what it needs:

   | Tool | What the model gets |
   |---|---|
   | `view_frames(timestamps)` | The exact frames at those times, in high resolution (up to 4 per call) |
   | `zoom(timestamp, x, y, width, height)` | One region of a frame at the video's native resolution, with the box given as fractions of the frame |
   | `search_transcript(query)` | The passages where something was said, with timestamps |
   | `read_transcript(start, end)` | The verbatim transcript of a time range |

3. **Answer.** It answers once it has enough evidence, citing timestamps.

Frames are decoded on demand, so **any moment of the video** is reachable, not only the
stored representative frames. Each look takes about 0.1–0.25 s on a 3440×1440 recording.

```python
answer = video.ask("Describe what each participant was wearing", progress=print)

answer.steps    # ['Looking at 1:30.0, 6:15.0, 14:35.0', 'Zooming into 6:15.0', ...]
answer.frames   # overview thumbnails + every frame and zoom the model looked at (saved as JPEGs)
```

With `progress=print`, each step prints live ("Looking at …", "Zooming into …").

### Single shot (`explore=False`)

One request with a fixed set of evidence: the full transcript when it fits
`evidence_tokens` (otherwise the passages relevant to the question), plus up to `max_images`
frames. It is cheaper and faster, but the model can't look closer. LLMs without tool
support always use this mode.

```python
answer = video.ask("Summarize the meeting", explore=False)
```

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `llm` | the `Video`'s `llm` | Override the connection for this call |
| `progress` | `None` | Callback for indexing and exploration progress (`print` works) |
| `explore` | auto | `True`/`False` to force exploration on or off |
| `max_steps` | 4 | Upper bound on exploration rounds |
| `max_views` | 16 | Upper bound on frames and zooms the model may request |
| `max_images` | 12 | Overview thumbnails (or images in single-shot mode) |
| `image_detail` | `"low"` | Detail level for single-shot images (`"high"` for small on-screen text) |
| `evidence_tokens` | 48 000 | Above this, only the passages relevant to the question are sent |
| `max_answer_tokens` | 8 000 | Output budget, including reasoning models' hidden thinking |

## The `Answer` object

| Attribute | Meaning |
|---|---|
| `text` | The LLM's answer (`str(answer)` gives the same) |
| `evidence` | Every transcript segment and frame that was sent, with source timestamps |
| `frames` | All images sent (`Frame` objects with a JPEG `path`) |
| `steps` | What the model did while exploring |
| `mode` | `"full"` (whole transcript sent) or `"retrieved"` (relevant passages only) |
| `model`, `usage` | As reported by the endpoint (usage is summed over all rounds) |
| `input_tokens`, `output_tokens`, `total_tokens` | Token counts |
| `cached_tokens` | Input tokens served from the provider's prompt cache (billed at a discount) |
| `reasoning_tokens` | Hidden thinking tokens of reasoning models (included in `output_tokens`) |

`answer.to_json()` serialises everything, including the evidence.

## Controlling token costs

Chat APIs are stateless, so each exploration round resends the conversation so far: the
transcript, the thumbnails and the images fetched so far. To keep costs down:

- **Use `reasoning_effort="low"`** on reasoning models: `saccade.azure(reasoning_effort="low")`.
- **Cap exploration:** `ask(..., max_steps=2, max_views=6)`.
- **Use `explore=False`** for questions about what was *said*. The transcript alone answers
  them.
- **Rely on prompt caching.** Each round only appends to the conversation, so providers that
  cache prompt prefixes bill the repeated part at a large discount. Azure OpenAI does this
  automatically; `answer.cached_tokens` shows how much was cached.

Images are already sized for cost. Full frames are sent at 1024 px and zooms at 768 px,
about 425 tokens each.

```python
print(f"{answer.input_tokens} in ({answer.cached_tokens} cached) + {answer.output_tokens} out")
```

## Async

```python
answer = await video.aask("What was decided?")
```

## Privacy

- **What the LLM receives:** the question, transcript text and JPEG stills.
- **What it never receives:** the video file and its audio.
- **Where viewed images are kept:** frames requested while exploring are saved under
  `video.index_dir / "views"` for traceability.
