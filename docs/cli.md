# Command line

Installing Saccade adds a `saccade` command. You can also run it as `python -m saccade`.

```text
saccade [--cache-dir DIR] [-q] [-v|-vv] COMMAND ...
```

| Global option | Meaning |
|---|---|
| `--cache-dir DIR` | Index and model cache (default: the platform cache directory) |
| `-q`, `--quiet` | No progress output |
| `-v`, `-vv` | Structured logs on stderr (info, debug) |

Progress goes to stderr and results to stdout, so you can pipe the output. Output is always
UTF-8.

## Commands

### `index`

Transcribes and indexes one or more videos.

```bash
saccade index meeting.mp4 talk.mkv --profile fast --device auto
saccade index meeting.mp4 --json          # machine-readable summary
saccade index meeting.mp4 --force         # redo this configuration
```

### `transcribe`

Prints the transcript, streaming lines as they are transcribed.

```bash
saccade transcribe meeting.mp4
saccade transcribe meeting.mp4 -f srt -o meeting.srt     # formats: text, srt, vtt, md, json
saccade transcribe meeting.mp4 -f md -o meeting.md
```

### `search`

Searches what was said. It indexes the video first if needed.

```bash
saccade search meeting.mp4 "authentication" -n 10 --start 10:00 --end 20:00 --expand 5
saccade search meeting.mp4 "authentication" --json
```

`--start` and `--end` accept seconds, `mm:ss` or `hh:mm:ss`.

### `context`

Builds prompt-ready evidence for a question.

```bash
saccade context meeting.mp4 "Why did the deployment fail?" --max-tokens 4000 --before 12 --after 15
saccade context meeting.mp4 "..." --segment-timestamps --json
```

### `frames`

Lists the representative frames, indexing the video first if needed.

```bash
saccade frames meeting.mp4 --start 5:00 --visual screen
```

### `ask`

Asks an LLM about the video.

```bash
# Azure AI Foundry, using the AZURE_AI_ENDPOINT / AZURE_AI_API_KEY / AZURE_AI_DEPLOYMENT variables
saccade ask meeting.mp4 "What was decided?"

# Any OpenAI-compatible server (OPENAI_API_KEY is used if set)
saccade ask meeting.mp4 "What was decided?" --base-url http://localhost:11434/v1 --llm-model llama3.1

saccade ask meeting.mp4 "..." --max-images 6 --json
```

### `info`

Shows media metadata and the state of the index.

```bash
saccade info meeting.mp4
saccade info meeting.mp4 --json
```

### `models`

Manages local Whisper models.

```bash
saccade models download small medium      # pre-download for offline use
saccade models list                       # known models and which are installed
```

## Processing options

Every command that may index a video accepts these options:

| Option | Meaning |
|---|---|
| `--profile {fast,balanced,accurate}` | Model preset |
| `--model NAME_OR_PATH` | Whisper model (overrides `--profile`) |
| `--language CODE` | Spoken language, e.g. `pt` (default: detect) |
| `--device {cpu,cuda,auto}` | Run Whisper on the CPU or an NVIDIA GPU |
| `--threads N`, `--workers N` | CPU tuning |
| `--word-timestamps` | Store per-word timings |
| `--no-vad` | Transcribe everything, including silence |
| `--offline` | Never download models |
| `--visual {auto,screen,scenes,interval,off}` | Frame strategy |
| `--keyframes-only` | Faster, coarser frame extraction |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | A Saccade error: the message names the file and the cause |
| 2 | Invalid command-line usage |
| 130 | Interrupted. Progress is saved; run the same command again to resume |
