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
| `speaker` | Optional speaker ID or assigned name; `None` when disabled or insufficient speech |

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

## Speaker labeling

Install `pip install "saccade-video[speakers]"` and supply a local ONNX speaker embedding
model supported by sherpa-onnx. The extra uses ONNX Runtime on CPU, not PyTorch. Nothing
is downloaded automatically for speaker labeling, and it adds no inference work when off.

```python
from saccade import DiarizationConfig, Video

video = Video("meeting.mp4", diarization=DiarizationConfig(model="nemo_en_titanet_small.onnx"))
video.index()
video.rename_speaker("SPEAKER_1", "Ana")
hits = video.search("deployment", speaker="Ana")
print(video.transcript().to_srt())
```

Labels also appear in JSON, VTT, `context()` and the full transcript sent to `ask()` and
exploration tools. Canonical IDs such as `SPEAKER_1` still work in search after renaming.
Names are user assignments, not verified identities. Reviewing the frames may help you
or an LLM associate a voice with a visible meeting tile, but that association can be wrong.

The existing VAD/ASR packs are reused; only a central window of at most `window_s` seconds
is embedded per segment. Online cosine clustering retains at most `max_speakers` centroids,
so memory does not grow with recording length. Centroids are checkpointed atomically with
each pack. Configuration and model contents have separate cache identities from ordinary
transcripts. Renaming does not alter the embeddings or require reprocessing.

**Limitations:** one label per ASR segment; no separate speaker-change detector, overlapping
speech separation or per-word attribution. Short turns can stay unlabeled. Mixed-speaker
segments can contaminate centroids; similar voices can merge, and one voice can split into
several IDs. At the configured cluster limit, new voices are assigned to the closest existing
cluster. Treat labels as provisional and choose thresholds using representative development
audio, not a test-set score. The default threshold is not calibrated to every model.

### Reproducing the AMI measurement

The first 300 seconds of **AMI ES2002a, Mix-Headset** produced **37.38% DER**:
27.096 s missed speech, 22.508 s false alarms, and 18.078 s speaker confusion over
181.050 s reference speaker time. Seven speaker clusters were predicted across 18 ASR
segments. This is a single-recording smoke benchmark, not a corpus-wide accuracy estimate.

Scoring uses `pyannote.metrics` 4.1 with optimal speaker mapping, **zero collar, overlap
included**, and the complete [0, 300] s interval. Neither reference speech boundaries nor
the reference speaker count is supplied to inference. Silero VAD and faster-whisper
`tiny.en` (CPU int8, greedy) feed TitaNet-S embeddings with the default configuration.
The measured inference/transcription wall time was 11.497 s on Windows/Python 3.12.10,
four inference threads. This includes initialization; it is not a hardware-normalized speed
claim. Full hashes, versions and score components are in
[`benchmarks/results/diarization-ami-es2002a.json`](../benchmarks/results/diarization-ami-es2002a.json).

Download these files to the ignored `benchmarks/media/` directory:

- [AMI ES2002a audio](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav).
  AMI Meeting Corpus, Carletta et al. (2006), [CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml).
  The benchmark converts its first five minutes to mono 16 kHz PCM; no audio is redistributed.
- [Word-only RTTM reference](https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/67c2d539286e89f68952d5dcf83912bd9f01dfae/only_words/rttms/train/ES2002a.rttm).
  BUT Speech@FIT setup, Landini et al., *Bayesian HMM clustering of x-vector sequences (VBx)*;
  pinned pyannote fork, Apache-2.0 setup code, derived from the CC BY 4.0 AMI annotations.
- [sherpa-onnx TitaNet-S export](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx).
  NVIDIA [TitaNet-S model card](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small),
  licensed under the [NeMo Apache-2.0 license](https://github.com/NVIDIA/NeMo/blob/main/LICENSE).

Install the benchmark-only scorer with `pip install "pyannote.metrics>=4,<5"`, then run:

```bash
python benchmarks/diarization.py benchmarks/media/ES2002a.Mix-Headset.wav benchmarks/media/ES2002a.rttm --recording ES2002a --speaker-model benchmarks/media/nemo_en_titanet_small.onnx --seconds 300 --output benchmarks/results/diarization-ami-es2002a.json
```

The benchmark does not download audio, annotations or speaker weights. Whisper/VAD follow
the normal model-cache policy. Default tests use deterministic stand-ins and never download.

## Exports

```python
t = video.transcript()
Path("talk.srt").write_text(t.to_srt(), encoding="utf-8")
Path("talk.vtt").write_text(t.to_vtt(), encoding="utf-8")
Path("talk.json").write_text(t.to_json(), encoding="utf-8")
```

Or from the CLI: `saccade transcribe talk.mp4 -f srt -o talk.srt`.
