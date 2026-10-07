"""Command-line interface: ``saccade index | transcribe | search | context | info | models``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from saccade._version import __version__
from saccade.config import PROFILES, ASRConfig, VadConfig, VisualConfig
from saccade.exceptions import ConfigError, NotIndexedError, SaccadeError
from saccade.models.segment import TranscriptSegment
from saccade.models.transcript import Transcript
from saccade.progress import ProgressEvent, Stage
from saccade.utils.time import format_span, format_timestamp
from saccade.video import Video

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


# -- output helpers ---------------------------------------------------------------------


def _utf8(stream: TextIO) -> None:
    """Make sure non-ASCII transcripts never crash on legacy Windows code pages."""
    reconfigure = getattr(stream, "reconfigure", None)
    encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
    if reconfigure is not None and encoding != "utf8":
        reconfigure(encoding="utf-8", errors="replace")


def _dump_json(data: Any) -> None:
    sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


class ProgressPrinter:
    """Human progress on stderr: an updating line on terminals, plain lines otherwise."""

    def __init__(self, stream: TextIO, *, quiet: bool) -> None:
        self.stream = stream
        self.quiet = quiet
        self.tty = stream.isatty()
        self._width = 0

    def __call__(self, event: ProgressEvent) -> None:
        if self.quiet:
            return
        line = str(event)
        transient = event.stage in (Stage.TRANSCRIBE, Stage.DETECT_SPEECH)
        if self.tty and transient:
            pad = max(0, self._width - len(line))
            self.stream.write("\r" + line + " " * pad)
            self._width = len(line)
        else:
            self.clear()
            self.stream.write(line + "\n")
        self.stream.flush()

    def clear(self) -> None:
        if self.tty and self._width:
            self.stream.write("\r" + " " * self._width + "\r")
            self._width = 0
            self.stream.flush()


def parse_time(value: str) -> float:
    """Accept ``90``, ``1:30``, ``01:01:30`` or ``1:30.5``."""
    try:
        parts = [float(p) for p in value.strip().split(":")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid time {value!r}; use seconds or [hh:]mm:ss"
        ) from None
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts):
        raise argparse.ArgumentTypeError(f"invalid time {value!r}; use seconds or [hh:]mm:ss")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


# -- argument parsing -------------------------------------------------------------------


def _add_video_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("transcription options")
    group.add_argument("--profile", choices=list(PROFILES), help="model preset (default: balanced)")
    group.add_argument(
        "--model", help="Whisper model name or local model directory (overrides --profile)"
    )
    group.add_argument("--language", help="spoken language code, e.g. pt (default: auto-detect)")
    group.add_argument(
        "--threads", type=int, help="inference threads (default: physical cores, max 8)"
    )
    group.add_argument("--workers", type=int, default=1, help="concurrent ASR calls (default: 1)")
    group.add_argument("--word-timestamps", action="store_true", help="store per-word timings")
    group.add_argument(
        "--no-vad", action="store_true", help="transcribe everything, including silence"
    )
    group.add_argument("--offline", action="store_true", help="never download models")
    group.add_argument(
        "--device", choices=["cpu", "cuda", "auto"], help="run Whisper on CPU or NVIDIA GPU"
    )
    group.add_argument(
        "--keyframes-only", action="store_true", help="faster, coarser frame extraction"
    )
    group.add_argument(
        "--visual",
        choices=["auto", "screen", "scenes", "interval", "off"],
        default="auto",
        help="how representative frames are chosen (default: auto)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saccade",
        description="Fast, local video context for LLMs: transcribe, index, search and build RAG context.",
    )
    parser.add_argument("--version", action="version", version=f"saccade {__version__}")
    parser.add_argument(
        "--cache-dir", type=Path, help="index and model cache (default: platform cache dir)"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="log details to stderr (-vv: debug)"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("index", help="transcribe and index one or more videos")
    p.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    p.add_argument("--force", action="store_true", help="redo the transcript even if cached")
    p.add_argument("--json", action="store_true", help="print a JSON summary")
    _add_video_options(p)

    p = sub.add_parser("transcribe", help="print the transcript (streams while transcribing)")
    p.add_argument("path", type=Path)
    p.add_argument("-f", "--format", choices=["text", "srt", "vtt", "md", "json"], default="text")
    p.add_argument("-o", "--output", type=Path, help="write to this file instead of stdout")
    p.add_argument("--force", action="store_true", help="redo the transcript even if cached")
    _add_video_options(p)

    p = sub.add_parser("search", help="search the transcript")
    p.add_argument("path", type=Path)
    p.add_argument("query")
    p.add_argument("-n", "--limit", type=int, default=5)
    p.add_argument("--start", type=parse_time, help="only after this time (seconds or mm:ss)")
    p.add_argument("--end", type=parse_time, help="only before this time (seconds or mm:ss)")
    p.add_argument(
        "--expand", type=float, default=0.0, help="seconds of surrounding transcript to include"
    )
    p.add_argument("--json", action="store_true")
    _add_video_options(p)

    p = sub.add_parser("context", help="build LLM-ready context for a question")
    p.add_argument("path", type=Path)
    p.add_argument("query")
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--before", type=float, default=12.0, help="seconds of context before each hit")
    p.add_argument("--after", type=float, default=15.0, help="seconds of context after each hit")
    p.add_argument("--segment-timestamps", action="store_true", help="timestamp every segment")
    p.add_argument("--json", action="store_true")
    _add_video_options(p)

    p = sub.add_parser("frames", help="list the representative frames (indexes first if needed)")
    p.add_argument("path", type=Path)
    p.add_argument("--start", type=parse_time)
    p.add_argument("--end", type=parse_time)
    p.add_argument("--json", action="store_true")
    _add_video_options(p)

    p = sub.add_parser(
        "ask",
        help="ask an LLM about the video (Azure AI Foundry via AZURE_AI_* env vars, or --base-url)",
    )
    p.add_argument("path", type=Path)
    p.add_argument("question")
    p.add_argument(
        "--base-url",
        help="OpenAI-compatible endpoint instead of Azure, e.g. http://localhost:11434/v1",
    )
    p.add_argument("--llm-model", help="model name for --base-url")
    p.add_argument("--max-images", type=int, default=12)
    p.add_argument("--json", action="store_true")
    _add_video_options(p)

    p = sub.add_parser("info", help="show media metadata and index status")
    p.add_argument("path", type=Path)
    p.add_argument("--json", action="store_true")
    _add_video_options(p)

    p = sub.add_parser("models", help="manage local Whisper models")
    models = p.add_subparsers(dest="models_command", required=True, metavar="ACTION")
    d = models.add_parser("download", help="download models for offline use")
    d.add_argument("names", nargs="+", metavar="MODEL")
    models.add_parser("list", help="list known model names and local availability")
    return parser


def _video(args: argparse.Namespace, path: Path) -> Video:
    asr = ASRConfig()
    if args.profile:
        asr = asr.with_profile(args.profile)
    if args.model:
        asr = replace(asr, model=args.model)
    if args.word_timestamps:
        asr = replace(asr, word_timestamps=True)
    return Video(
        path,
        asr=asr,
        language=args.language,
        vad=VadConfig(enabled=not args.no_vad),
        threads=args.threads,
        workers=args.workers,
        cache_dir=args.cache_dir,
        offline=True if args.offline else None,
        visual=VisualConfig(strategy=args.visual, keyframes_only=args.keyframes_only),
        device=args.device,
    )


# -- commands ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace, progress: ProgressPrinter) -> int:
    summaries = []
    for path in args.paths:
        video = _video(args, path)
        summary = video.index(force=args.force, progress=progress)
        progress.clear()
        summaries.append(summary.to_dict())
        if not args.json:
            rtf = (
                f", RTF {summary.real_time_factor:.3f}"
                if summary.real_time_factor and not summary.cached
                else ""
            )
            lang = f"{summary.language}" if summary.language else "n/a"
            print(
                f"{summary.video}: {summary.status} · {summary.segments} segments · "
                f"{summary.chunks} chunks · language {lang} · "
                f"{'cached' if summary.cached else f'{summary.elapsed_seconds:.1f}s'}{rtf}"
            )
    if args.json:
        _dump_json(summaries if len(summaries) > 1 else summaries[0])
    return EXIT_OK


def cmd_transcribe(args: argparse.Namespace, progress: ProgressPrinter) -> int:
    video = _video(args, args.path)
    streaming = args.format == "text" and args.output is None
    segments: list[TranscriptSegment] = []
    for segment in video.transcribe(force=args.force, progress=progress):
        segments.append(segment)
        if streaming:
            progress.clear()
            sys.stdout.write(f"[{format_span(segment.start, segment.end)}] {segment.text}\n")
            sys.stdout.flush()
    progress.clear()
    if streaming:
        return EXIT_OK
    transcript = video.transcript()
    renderers: dict[str, Callable[[Transcript], str]] = {
        "text": lambda t: str(t) + "\n",
        "srt": Transcript.to_srt,
        "vtt": Transcript.to_vtt,
        "md": lambda t: t.to_markdown(title=video.name),
        "json": lambda t: t.to_json() + "\n",
    }
    rendered = renderers[args.format](transcript)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {len(transcript)} segments to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
    return EXIT_OK


def _ensure_indexed(video: Video, progress: ProgressPrinter) -> None:
    try:
        video.transcript()
    except NotIndexedError:
        progress(ProgressEvent(Stage.INSPECT, f"{video.name} is not indexed yet; indexing now"))
        video.index(progress=progress)
        progress.clear()


def cmd_search(args: argparse.Namespace, progress: ProgressPrinter) -> int:
    video = _video(args, args.path)
    _ensure_indexed(video, progress)
    began = time.perf_counter()
    results = video.search(
        args.query, limit=args.limit, start=args.start, end=args.end, expand=args.expand
    )
    elapsed_ms = (time.perf_counter() - began) * 1000
    if args.json:
        _dump_json({"query": args.query, "results": [r.to_dict() for r in results]})
        return EXIT_OK
    if not results:
        print("No matches.")
        return EXIT_OK
    for i, result in enumerate(results, start=1):
        print(
            f"{i}. [{format_span(result.start, result.end)}]  score {result.score:.2f}  ({result.id})"
        )
        print(f"   {result.text}\n")
    print(f"{len(results)} result(s) in {elapsed_ms:.1f} ms", file=sys.stderr)
    return EXIT_OK


def cmd_context(args: argparse.Namespace, progress: ProgressPrinter) -> int:
    video = _video(args, args.path)
    _ensure_indexed(video, progress)
    context = video.context(
        args.query,
        max_tokens=args.max_tokens,
        before=args.before,
        after=args.after,
        segment_timestamps=args.segment_timestamps,
    )
    if args.json:
        sys.stdout.write(context.to_json() + "\n")
    else:
        sys.stdout.write(context.text + "\n")
    return EXIT_OK


def cmd_frames(args: argparse.Namespace, progress: ProgressPrinter) -> int:
    video = _video(args, args.path)
    video.index(progress=progress)
    progress.clear()
    frames = video.frames(start=args.start, end=args.end)
    if args.json:
        _dump_json([f.to_dict() for f in frames])
        return EXIT_OK
    for frame in frames:
        size = f"{frame.width}x{frame.height}"
        print(
            f"[{format_timestamp(frame.timestamp)}] {frame.id}  {size}  {frame.reason:<8} {frame.path}"
        )
    print(f"{len(frames)} frame(s)", file=sys.stderr)
    return EXIT_OK


def cmd_ask(args: argparse.Namespace, progress: ProgressPrinter) -> int:
    from saccade.llm import LLM, OpenAICompatible, azure

    llm: LLM
    if args.base_url:
        if not args.llm_model:
            raise ConfigError("--base-url needs --llm-model.")
        llm = OpenAICompatible(
            base_url=args.base_url, model=args.llm_model, api_key=os.environ.get("OPENAI_API_KEY")
        )
    else:
        llm = azure()
    video = _video(args, args.path)
    answer = video.ask(args.question, llm=llm, progress=progress, max_images=args.max_images)
    progress.clear()
    if args.json:
        sys.stdout.write(answer.to_json() + "\n")
    else:
        sys.stdout.write(answer.text + "\n")
        kinds = [e.type for e in answer.evidence]
        print(
            f"\n(based on {kinds.count('transcript')} transcript segments and {kinds.count('frame')} frames)",
            file=sys.stderr,
        )
    return EXIT_OK


def cmd_info(args: argparse.Namespace, progress: ProgressPrinter) -> int:
    info = _video(args, args.path).info()
    if args.json:
        _dump_json(info.to_dict())
        return EXIT_OK
    media = info.media
    print(f"File:        {info.path}")
    print(f"Format:      {media.format} · {media.size_bytes / 1e6:.1f} MB")
    print(
        f"Duration:    {format_timestamp(media.duration, precision=1) if media.duration else 'unknown'}"
    )
    for stream in media.video_streams:
        fps = f" @ {stream.average_fps:.2f} fps" if stream.average_fps else ""
        print(f"Video:       #{stream.index} {stream.codec} {stream.width}x{stream.height}{fps}")
    for audio in media.audio_streams:
        lang = f" [{audio.language}]" if audio.language else ""
        print(
            f"Audio:       #{audio.index} {audio.codec} {audio.sample_rate} Hz · {audio.channels} ch{lang}"
        )
    if not media.audio_streams:
        print("Audio:       none")
    print(f"Fingerprint: {info.fingerprint}")
    print(f"Index:       {info.index_dir}")
    if info.transcript_status is None:
        print("Transcript:  not indexed")
    else:
        until = (
            f" up to {format_timestamp(info.indexed_until, precision=0)}"
            if info.indexed_until
            else ""
        )
        print(
            f"Transcript:  {info.transcript_status}{until} · {info.segments} segments · "
            f"{info.chunks} chunks · language {info.language or 'n/a'} · model {info.model}"
        )
    print(f"Frames:      {info.frames}")
    return EXIT_OK


def cmd_models(args: argparse.Namespace, progress: ProgressPrinter) -> int:
    from faster_whisper.utils import available_models

    from saccade.asr.faster_whisper import resolve_model
    from saccade.utils.cache import default_cache_dir, models_dir

    target = models_dir(args.cache_dir or default_cache_dir())
    if args.models_command == "list":
        for name in available_models():
            try:
                resolve_model(name, target, allow_download=False)
                state = "installed"
            except SaccadeError:
                state = "-"
            print(f"{name:<22} {state}")
        print(f"\nModels directory: {target}", file=sys.stderr)
        return EXIT_OK
    for name in args.names:
        path = resolve_model(name, target, allow_download=True)
        print(f"{name}: {path}")
    return EXIT_OK


COMMANDS = {
    "index": cmd_index,
    "transcribe": cmd_transcribe,
    "search": cmd_search,
    "context": cmd_context,
    "info": cmd_info,
    "frames": cmd_frames,
    "ask": cmd_ask,
    "models": cmd_models,
}


def main(argv: Sequence[str] | None = None) -> int:
    _utf8(sys.stdout)
    _utf8(sys.stderr)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose > 1 else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
        )
    progress = ProgressPrinter(sys.stderr, quiet=args.quiet)
    try:
        return COMMANDS[args.command](args, progress)
    except KeyboardInterrupt:
        progress.clear()
        print(
            "Interrupted. Progress is saved; run the same command again to resume.", file=sys.stderr
        )
        return EXIT_INTERRUPTED
    except SaccadeError as exc:
        progress.clear()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
