"""Measure Saccade end to end on real files.

Reports, per file: total indexing time, real-time factor, time to first segment, ASR
time, peak memory, warm-cache time, search latency, database size — together with the
machine, model, quantization, language and speech density, so numbers can be compared.

    python benchmarks/bench.py benchmarks/media/*.mp4 --profile balanced --threads 8

Each run uses a fresh index cache (models are reused) so results are cold-cache numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from saccade import Video, __version__
from saccade.utils.cache import default_cache_dir, models_dir

QUERIES = [
    "Why did the deployment fail?",
    "authentication secret",
    "database migration duration",
    "monitoring dashboard errors",
    "cache configuration",
    "falhou a implantação",
    "latência da busca",
]


def cpu_model() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


class PeakMemory:
    """Samples resident memory of this process every 100 ms."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> PeakMemory:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def bench_file(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    cache = Path(tempfile.mkdtemp(prefix="saccade-bench-"))
    os.environ.setdefault("SACCADE_MODELS_DIR", str(models_dir(default_cache_dir())))
    try:
        video = Video(
            path,
            profile=args.profile,
            threads=args.threads,
            cache_dir=cache,
            language=args.language,
        )
        baseline_rss = psutil.Process().memory_info().rss
        began = time.perf_counter()
        first_segment = None
        segments = 0
        with PeakMemory() as memory:
            for _ in video.transcribe():
                if first_segment is None:
                    first_segment = time.perf_counter() - began
                segments += 1
        total = time.perf_counter() - began
        summary = video.index()  # cached: just collects the summary

        warm_began = time.perf_counter()
        warm = Video(
            path,
            profile=args.profile,
            threads=args.threads,
            cache_dir=cache,
            language=args.language,
        )
        warm.index()
        warm_seconds = time.perf_counter() - warm_began

        warm.search(QUERIES[0])  # open connections, page cache
        latencies = []
        for _ in range(5):
            for query in QUERIES:
                t = time.perf_counter()
                warm.search(query, limit=5)
                latencies.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        context = warm.context(QUERIES[0], max_tokens=4000)
        context_ms = (time.perf_counter() - t) * 1000

        db_path = warm.index_dir / "index.db"
        db_bytes = sum(p.stat().st_size for p in db_path.parent.glob("index.db*"))
        meta_file = path.with_suffix(".json")
        source = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        duration = summary.duration or 0.0
        return {
            "file": path.name,
            "duration_s": round(duration, 1),
            "source_language": source.get("language"),
            "detected_language": summary.language,
            "speech_density_vad": round(summary.speech_seconds / duration, 3) if duration else None,
            "speech_density_source": source.get("speech_density"),
            "model": video.asr_config.model,
            "compute_type": video.asr_config.compute_type,
            "beam_size": video.asr_config.beam_size,
            "threads": video.threads,
            "segments": segments,
            "chunks": summary.chunks,
            "time_to_first_segment_s": round(first_segment or 0, 2),
            "index_seconds": round(total, 1),
            "real_time_factor": round(total / duration, 4) if duration else None,
            "asr_seconds": round(summary.transcription_seconds, 1),
            "peak_rss_mb": round(memory.peak / 2**20),
            "rss_growth_mb": round((memory.peak - baseline_rss) / 2**20),
            "warm_cache_seconds": round(warm_seconds, 3),
            "search_ms_median": round(statistics.median(latencies), 2),
            "search_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2),
            "context_ms": round(context_ms, 1),
            "context_tokens_est": context.metadata.estimated_tokens,
            "db_kb": round(db_bytes / 1024),
            "frames": None,  # visual pipeline arrives in Phase 2
        }
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--profile", default="balanced", choices=["fast", "balanced", "accurate"])
    parser.add_argument("--threads", type=int)
    parser.add_argument("--language")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    machine = {
        "cpu": cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "saccade": __version__,
    }
    print(json.dumps(machine, indent=2))
    results = []
    for path in args.files:
        print(f"\n== {path.name} ({args.profile})", flush=True)
        result = bench_file(path, args)
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.output / f"bench-{args.profile}-{stamp}.json"
    out.write_text(
        json.dumps({"machine": machine, "results": results}, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
