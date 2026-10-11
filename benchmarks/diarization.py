"""Evaluate an audio prefix against RTTM references; requires pyannote.metrics.

See docs/transcription.md for the AMI sample, licenses and reproducible invocation.
No downloads are performed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import tempfile
import time
import wave
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np

from saccade import ASRConfig, DiarizationConfig, Video
from saccade.media.audio import SAMPLE_RATE, decode_audio


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--speaker-model", type=Path, required=True)
    parser.add_argument("--asr-model", default="tiny.en")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600:
        parser.error("--seconds must be positive and at most 3600")

    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.database.util import load_rttm
    from pyannote.metrics.diarization import DiarizationErrorRate

    reference = load_rttm(args.reference)[args.recording]
    config = DiarizationConfig(model=str(args.speaker_model.resolve()), threshold=args.threshold)
    hypothesis = Annotation(uri=args.recording)
    with tempfile.TemporaryDirectory(prefix="saccade-speakers-") as directory:
        excerpt = Path(directory) / "excerpt.wav"
        sample_count = 0
        with wave.open(str(excerpt), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            with closing(decode_audio(args.audio)) as blocks:
                for block in blocks:
                    remaining = round(args.seconds * SAMPLE_RATE) - sample_count
                    if remaining <= 0:
                        break
                    samples = block.samples[:remaining]
                    output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
                    sample_count += len(samples)
        duration = sample_count / SAMPLE_RATE
        video = Video(
            excerpt,
            asr=ASRConfig(model=args.asr_model, language="en"),
            diarization=config,
            cache_dir=Path(directory) / "cache",
            threads=4,
        )
        started = time.perf_counter()
        segments = 0
        unlabeled = 0
        try:
            for segment in video.transcribe():
                segments += 1
                if segment.speaker and segment.end > segment.start:
                    hypothesis[Segment(segment.start, segment.end), segment.id] = segment.speaker
                else:
                    unlabeled += 1
        finally:
            video._db().close()
        elapsed = time.perf_counter() - started

    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    scores = metric(reference, hypothesis, uem=Timeline([Segment(0, duration)]), detailed=True)
    result = {
        "recording": args.recording,
        "audio": args.audio.name,
        "audio_sha256": digest(args.audio),
        "reference_sha256": digest(args.reference),
        "speaker_model_sha256": digest(args.speaker_model),
        "diarization": {**asdict(config), "model": args.speaker_model.name},
        "asr_model": args.asr_model,
        "interval_seconds": [0, duration],
        "collar_seconds": 0,
        "overlap_scored": True,
        "oracle_speech_or_speakers": False,
        "scores": {key: float(value) for key, value in scores.items()},
        "predicted_speakers": len(hypothesis.labels()),
        "segments": segments,
        "unlabeled_segments": unlabeled,
        "elapsed_seconds": round(elapsed, 3),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["sherpa-onnx", "faster-whisper", "pyannote.metrics"]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
