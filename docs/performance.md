# Performance: CPU and GPU

## Measured numbers

These were measured on one laptop: Intel i7-14650HX (16 cores), RTX 5050 Laptop GPU and
Windows 11, with the model already downloaded and a cold index cache. Run
`benchmarks/bench.py` on your own hardware for your own numbers.

**A 26-minute meeting screen recording (3440×1440, 30 fps):**

| Setup | Full `index()` | Speech recognition |
|---|---|---|
| CPU (`small`, int8) | 175 s | 122 s |
| GPU (`small`, float16) | 41 s, limited by decoding every frame | 7 s |
| GPU + `keyframes_only=True` | **9 s**, first transcript lines after 2.3 s | 7 s |

**Synthetic meetings, CPU only (`balanced`, transcription only):**

| Video | Indexing | Real-time factor | First segment | Peak memory |
|---|---|---|---|---|
| 10 min | 27 s | 0.045 | 3 s | ~600–750 MB |
| 30 min | 1 min 36 s | 0.053 | 3 s | ~700 MB |
| 2 h | 8 min 54 s | 0.074 | 3 s | ~1.1 GB |

Repeated `index()` calls on an indexed video take 20–40 ms. `search()` takes under 3 ms.

## Using an NVIDIA GPU

```bash
pip install "saccade-video[gpu]"     # installs NVIDIA's CUDA 12 libraries (cuBLAS, cuDNN 9) from pip
```

You need an NVIDIA driver (check with `nvidia-smi`). There is no CUDA toolkit to install:
Saccade finds the pip-installed libraries automatically.

```python
import saccade

video = saccade.Video(
    "meeting.mp4",
    device="cuda",                                     # or "auto": GPU if available, else CPU
    visual=saccade.VisualConfig(keyframes_only=True),  # frame extraction that keeps up with the GPU
)
```

What changes on the GPU:

- **Float16 inference.** Whisper runs in float16 and transcribes 16 speech chunks per batch
  (`ASRConfig(batch_size=...)`).
- **Overlap.** Frame extraction runs at the same time as transcription.
- **Larger models.** Bigger models become affordable: `profile="accurate"` or
  `ASRConfig(model="large-v3-turbo")`.

GPU and CPU results are cached separately, because batched decoding segments text slightly
differently. Switching device therefore reprocesses the video once.

The first run on a brand-new GPU architecture can take a few extra seconds while CUDA
compiles kernels; the driver caches the result.

## CPU tuning

```python
Video("meeting.mp4", threads=8, workers=1)
```

| Option | Default | Meaning |
|---|---|---|
| `threads` | Physical cores, capped at 8 | Whisper's inference threads. More than ~8 rarely helps, and leaving cores free keeps the machine responsive |
| `workers` | 1 | Concurrent Whisper calls. One worker with internal threading is the efficient choice on almost every machine |

Other levers:

- **Smaller model:** `profile="fast"` (the `base` model) roughly halves the time.
- **Faster frames:** `VisualConfig(keyframes_only=True)` speeds up frame extraction on
  high-resolution video.
- **No frames:** `visual_strategy="off"` skips frames entirely.

## Memory

Audio is streamed in 5-second blocks and only the current speech pack is held. Decoding and
VAD alone stay flat at about 95 MB, even for a 2-hour file. Most memory is the Whisper model
itself: about 450 MB for `small` int8 on CPU.

## Time to first result

A speech pack is sent to Whisper as soon as it is complete, so the first segments of a long
video are searchable within seconds. Use `transcribe()` or `index(background=True)` to start
working with them before indexing finishes.
