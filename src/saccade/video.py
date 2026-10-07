"""The :class:`Video` facade — the whole public workflow in one object."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from saccade._version import PROCESSING_VERSION, __version__
from saccade.asr.base import ASRBackend
from saccade.asr.faster_whisper import FasterWhisperBackend
from saccade.config import (
    ASRConfig,
    ChunkConfig,
    FingerprintMode,
    Profile,
    VadConfig,
    VisualConfig,
    VisualStrategy,
    validate_profile,
)
from saccade.exceptions import AudioStreamNotFoundError, ConfigError, NotIndexedError
from saccade.index.database import Database, RunRecord, utcnow
from saccade.llm.base import LLM
from saccade.media.fingerprint import fingerprint as compute_fingerprint
from saccade.media.probe import MediaInfo, probe
from saccade.models.answer import Answer
from saccade.models.context import VideoContext
from saccade.models.frame import Frame
from saccade.models.result import SearchResult
from saccade.models.segment import TranscriptSegment
from saccade.models.transcript import Transcript, TranscriptChunk, TranscriptStatus
from saccade.pipeline import (
    FIRST_PACK_SECONDS,
    MAX_JOIN_GAP_S,
    MAX_PACK_SPAN_S,
    PACK_SECONDS,
    PipelineContext,
    run_transcription,
)
from saccade.progress import ProgressCallback, Reporter, Stage
from saccade.retrieval.context import ContextOptions, TokenCounter, build_context
from saccade.retrieval.search import search_transcript
from saccade.utils.cache import FileLock, default_cache_dir, models_dir, video_cache_dir
from saccade.utils.concurrency import default_threads, iterate_in_thread
from saccade.vad import create_vad
from saccade.vad.base import VoiceActivityDetector
from saccade.vision.extract import extract_frames, frame_id

_TRANSCRIPT = "transcript"
_FRAMES = "frames"
_VISUAL_VERSION = 1
_OFFLINE_ENV = ("SACCADE_OFFLINE", "HF_HUB_OFFLINE")


@dataclass(frozen=True, slots=True)
class IndexSummary:
    """Outcome of :meth:`Video.index`."""

    video: str
    status: TranscriptStatus
    duration: float | None
    language: str | None
    language_probability: float | None
    segments: int
    chunks: int
    speech_seconds: float
    indexed_until: float | None
    model: str | None
    cached: bool
    elapsed_seconds: float
    transcription_seconds: float
    frames: int = 0

    @property
    def real_time_factor(self) -> float | None:
        """ASR seconds per second of media (lower is faster); ``None`` if not measurable."""
        if not self.duration or not self.transcription_seconds:
            return None
        return self.transcription_seconds / self.duration

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["real_time_factor"] = self.real_time_factor
        return data


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Media metadata plus the state of the local index."""

    path: str
    name: str
    fingerprint: str
    media: MediaInfo
    index_dir: str
    transcript_status: TranscriptStatus | None
    language: str | None
    segments: int
    chunks: int
    indexed_until: float | None
    model: str | None
    frames: int = 0
    runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["media"] = self.media.to_dict()
        return data


class IndexJob:
    """Handle for :meth:`Video.index` running in the background.

    Search and context work while the job runs; they see everything committed so far.
    """

    def __init__(self, target: Callable[[threading.Event], IndexSummary]) -> None:
        self._cancel = threading.Event()
        self._result: IndexSummary | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, args=(target,), name="saccade-index", daemon=True
        )
        self._thread.start()

    def _run(self, target: Callable[[threading.Event], IndexSummary]) -> None:
        try:
            self._result = target(self._cancel)
        except BaseException as exc:
            self._error = exc

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def cancel(self) -> None:
        """Stop after the block currently being transcribed; progress so far is kept."""
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> IndexSummary:
        """Block until indexing finishes and return its summary (re-raising its error)."""
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("Indexing is still running.")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class Video:
    """A local video file and its Saccade index.

    Construction is cheap and does no I/O; work happens when you call a method. Results are
    cached under ``cache_dir`` keyed by the file's content fingerprint and the processing
    configuration, so asking new questions never re-transcribes.

    Args:
        path: The media file (any container FFmpeg reads: MP4, MKV, MOV, WEBM, audio files...).
        language: Spoken language (``"pt"``, ``"en"``...) or ``None`` to detect automatically.
        profile: ``"fast"``, ``"balanced"`` (default model) or ``"accurate"``.
        asr: Full ASR configuration; ``language`` and ``profile`` override parts of it.
        vad: Voice activity detection configuration.
        chunking: How segments are grouped into retrieval chunks.
        threads: Inference threads. Defaults to the number of physical cores, capped at 8.
        workers: Concurrent ASR calls. Keep 1 unless you have many cores to spare.
        cache_dir: Where indexes and models live (default: platform cache dir, or
            ``SACCADE_CACHE_DIR``).
        offline: Never download models. Defaults to true when ``SACCADE_OFFLINE`` or
            ``HF_HUB_OFFLINE`` is set.
        fingerprint: ``"sampled"`` (fast, default) or ``"strict"`` (full SHA-256).
        asr_backend: Use this backend instead of faster-whisper (advanced).
        vad_factory: Build the speech detector for each run (advanced).
        visual: How representative frames are chosen (``VisualConfig(strategy="off")`` disables).
        llm: The LLM :meth:`ask` sends questions to, e.g. ``saccade.azure(...)``. Saccade only
            contacts it when you call :meth:`ask`.
        device: ``"cuda"`` runs Whisper on an NVIDIA GPU (``pip install "saccade-video[gpu]"``),
            ``"auto"`` uses one when available, ``"cpu"`` is the default.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        language: str | None = None,
        profile: Profile | None = None,
        asr: ASRConfig | None = None,
        vad: VadConfig | None = None,
        chunking: ChunkConfig | None = None,
        threads: int | None = None,
        workers: int = 1,
        cache_dir: str | os.PathLike[str] | None = None,
        offline: bool | None = None,
        fingerprint: FingerprintMode = "sampled",
        asr_backend: ASRBackend | None = None,
        vad_factory: Callable[[], VoiceActivityDetector] | None = None,
        visual: VisualConfig | None = None,
        llm: LLM | None = None,
        device: Literal["cpu", "cuda", "auto"] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().absolute()
        asr = asr or ASRConfig()
        if profile is not None:
            asr = asr.with_profile(validate_profile(profile) or "balanced")
        if language is not None:
            asr = replace(asr, language=language)
        if device is not None:
            asr = replace(asr, device=device)
        if threads is not None and threads < 1:
            raise ConfigError("threads must be >= 1.")
        if workers < 1:
            raise ConfigError("workers must be >= 1.")
        self._asr = asr
        self.vad_config = vad or VadConfig()
        self.chunk_config = chunking or ChunkConfig()
        self.visual_config = visual or VisualConfig()
        self.llm = llm
        self.threads = threads or default_threads()
        self.workers = workers
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
        self.offline = (
            offline
            if offline is not None
            else any(_truthy(os.environ.get(v)) for v in _OFFLINE_ENV)
        )
        self.fingerprint_mode = fingerprint
        self._custom_backend = asr_backend
        self._vad_factory = vad_factory
        self._backend: ASRBackend | None = None
        self._media: MediaInfo | None = None
        self._fingerprint: str | None = None
        self._database: Database | None = None
        self._lock = threading.Lock()

    # -- identity & storage ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def asr_config(self) -> ASRConfig:
        return self._asr

    def media_info(self) -> MediaInfo:
        """Container and stream metadata (probed once, then cached on this object)."""
        with self._lock:
            if self._media is None:
                self._media = probe(self.path)
            return self._media

    @property
    def fingerprint(self) -> str:
        info = self.media_info()
        with self._lock:
            if self._fingerprint is None:
                self._fingerprint = compute_fingerprint(self.path, info, self.fingerprint_mode)
            return self._fingerprint

    @property
    def index_dir(self) -> Path:
        """Directory holding this video's index database and its frames."""
        return video_cache_dir(self.cache_dir, self.fingerprint)

    def _db(self) -> Database:
        fp = self.fingerprint
        info = self.media_info()
        with self._lock:
            if self._database is None:
                database = Database(video_cache_dir(self.cache_dir, fp) / "index.db")
                database.upsert_video(
                    fingerprint=fp, path=self.path, probe=info.to_dict(), duration=info.duration
                )
                self._database = database
            return self._database

    def _backend_for(self, reporter: Reporter | None = None) -> ASRBackend:
        if self._custom_backend is not None:
            return self._custom_backend
        backend = self._backend
        if not isinstance(backend, FasterWhisperBackend) or backend.config != self._asr:
            backend = FasterWhisperBackend(
                self._asr,
                models_dir=models_dir(self.cache_dir),
                threads=max(1, self.threads // self.workers),
                workers=self.workers,
                allow_download=not self.offline,
            )
            self._backend = backend
        backend.reporter = reporter
        return backend

    def _cache_material(self, backend: ASRBackend) -> dict[str, Any]:
        return {
            "processing_version": PROCESSING_VERSION,
            "asr": backend.identity,
            "language": self._asr.language,
            "word_timestamps": self._asr.word_timestamps,
            "carry_context": self._asr.carry_context and self.workers == 1,
            "vad": asdict(self.vad_config),
            "chunking": asdict(self.chunk_config),
            "packing": [PACK_SECONDS, FIRST_PACK_SECONDS, MAX_PACK_SPAN_S, MAX_JOIN_GAP_S],
        }

    def _cache_key(self, backend: ASRBackend) -> tuple[str, dict[str, Any]]:
        material = self._cache_material(backend)
        blob = json.dumps(material, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:24], material

    def _read_run(self) -> RunRecord | None:
        key, _ = self._cache_key(self._backend_for())
        return self._db().best_run(_TRANSCRIPT, key)

    def _require_run(self) -> RunRecord:
        run = self._read_run()
        if run is None:
            raise NotIndexedError(
                f'"{self.name}" has not been indexed yet. Call video.index() first '
                f'(or `saccade index "{self.path}"`).'
            )
        return run

    def _apply_profile(self, profile: Profile | str | None) -> None:
        if profile is not None:
            self._asr = self._asr.with_profile(profile)

    # -- processing -------------------------------------------------------------------

    def transcribe(
        self,
        *,
        profile: Profile | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Iterator[TranscriptSegment]:
        """Stream transcript segments in timeline order, transcribing only what is missing.

        Already-indexed segments are yielded straight from the cache; new ones are yielded
        as soon as they are transcribed *and* committed, so they are immediately searchable.
        Stopping the iteration early keeps everything transcribed so far; the next call
        resumes from there.

        Raises:
            AudioStreamNotFoundError: the media has no audio.
            IndexLockedError: another process is indexing this video right now.
        """
        self._apply_profile(profile)
        return self._transcribe(force=force, progress=progress)

    def _transcribe(
        self, *, force: bool, progress: ProgressCallback | None
    ) -> Generator[TranscriptSegment, None, None]:
        info = self.media_info()
        reporter = Reporter(progress, info.duration)
        reporter(Stage.INSPECT, f"Inspecting media: {self.name}")
        if not info.has_audio:
            raise AudioStreamNotFoundError(
                f'"{self.name}" has no audio stream, so there is nothing to transcribe.'
            )

        db = self._db()
        backend = self._backend_for(reporter)
        key, material = self._cache_key(backend)
        run = db.get_run(_TRANSCRIPT, key)
        if run is not None and run.status == "complete" and not force:
            reporter(Stage.CACHED, "Using cached transcript", position=info.duration)
            for _, segment in db.segments(run.id, with_words=self._asr.word_timestamps):
                yield segment
            reporter(Stage.COMPLETE, "Completed (cached)", position=info.duration)
            return

        with FileLock(self.index_dir / "index.lock"):
            run = db.get_run(_TRANSCRIPT, key)
            if run is not None and force:
                db.delete_run(run.id)
                run = None
            if run is None:
                run = db.create_run(
                    kind=_TRANSCRIPT, config_key=key, config=material, library_version=__version__
                )
            elif run.status == "complete":
                for _, segment in db.segments(run.id, with_words=self._asr.word_timestamps):
                    yield segment
                return
            for _, segment in db.segments(run.id, with_words=self._asr.word_timestamps):
                yield segment

            context = PipelineContext(
                path=self.path,
                db=db,
                backend=backend,
                asr=self._asr,
                chunking=self.chunk_config,
                vad_factory=self._vad_factory or (lambda: create_vad(self.vad_config)),
                reporter=reporter,
                duration=info.duration,
                workers=self.workers,
                stream_index=info.primary_audio.index if info.primary_audio else None,
            )
            yield from run_transcription(context, run)
            final = db.get_run_by_id(run.id)
            count = db.stats(run.id).segments
            reporter(
                Stage.COMPLETE,
                f"Completed: {count} segments, language {final.language if final else '?'}",
                position=info.duration,
            )

    def index(
        self,
        *,
        profile: Profile | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
        background: bool = False,
        visual_strategy: VisualStrategy | None = None,
    ) -> Any:
        """Transcribe the audio and extract representative frames (cached afterwards).

        Args:
            profile: ``"fast"``, ``"balanced"`` or ``"accurate"``; also becomes this object's
                active profile for later ``search``/``context`` calls.
            force: Discard the cached transcript and frames for this configuration and redo them.
            progress: Called with a :class:`~saccade.progress.ProgressEvent` per step.
            background: Return an :class:`IndexJob` immediately and index in a thread.
            visual_strategy: ``"auto"`` (default), ``"screen"``, ``"scenes"``, ``"interval"`` or
                ``"off"``; see :class:`~saccade.config.VisualConfig`.

        Returns:
            :class:`IndexSummary`, or :class:`IndexJob` when ``background=True``.
        """
        self._apply_profile(profile)
        if visual_strategy is not None:
            self.visual_config = replace(self.visual_config, strategy=visual_strategy)
        if background:
            return IndexJob(
                lambda cancel: self._index(force=force, progress=progress, cancel=cancel)
            )
        return self._index(force=force, progress=progress, cancel=None)

    def _index(
        self, *, force: bool, progress: ProgressCallback | None, cancel: threading.Event | None
    ) -> IndexSummary:
        began = time.perf_counter()
        if not self.media_info().has_audio:
            self._record_silent_video(progress)
            frames_cached = self._index_frames(force=force, progress=progress)
            return self._summary(began, cached=frames_cached)
        cached = self._is_cached() and not force
        # With Whisper on the GPU the CPU is mostly idle, so frames are extracted meanwhile.
        frames_job = self._frames_in_background(force, progress) if self._on_gpu() else None
        stream = self._transcribe(force=force, progress=progress)
        try:
            for _ in stream:
                if cancel is not None and cancel.is_set():
                    break
        finally:
            stream.close()
        if frames_job is not None:
            cached = frames_job.result() and cached
        elif cancel is None or not cancel.is_set():
            cached = self._index_frames(force=force, progress=progress) and cached
        return self._summary(began, cached=cached)

    def _on_gpu(self) -> bool:
        return getattr(self._backend_for(), "device", "cpu") == "cuda"

    def _frames_in_background(self, force: bool, progress: ProgressCallback | None) -> Future[bool]:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="saccade-frames")
        future = executor.submit(self._index_frames, force=force, progress=progress)
        executor.shutdown(wait=False)
        return future

    # -- frames -----------------------------------------------------------------------

    def _visual_key(self) -> tuple[str, dict[str, Any]]:
        material = {"visual_version": _VISUAL_VERSION, "visual": asdict(self.visual_config)}
        blob = json.dumps(material, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:24], material

    def _frames_dir(self, run_id: int) -> Path:
        return self.index_dir / "frames" / f"run{run_id}"

    def _index_frames(self, *, force: bool, progress: ProgressCallback | None) -> bool:
        """Extract representative frames; returns True when they were already cached."""
        info = self.media_info()
        if not info.has_video or self.visual_config.strategy == "off":
            return True
        db = self._db()
        key, material = self._visual_key()
        run = db.get_run(_FRAMES, key)
        if run is not None and run.status == "complete" and not force:
            return True
        reporter = Reporter(progress, info.duration)
        with FileLock(self.index_dir / "frames.lock"):
            run = db.get_run(_FRAMES, key)
            if run is not None and (force or run.status != "complete"):
                shutil.rmtree(self._frames_dir(run.id), ignore_errors=True)
                db.delete_run(run.id)
                run = None
            if run is not None:
                return True
            run = db.create_run(
                kind=_FRAMES, config_key=key, config=material, library_version=__version__
            )
            try:
                stats = extract_frames(
                    self.path,
                    db=db,
                    run=run,
                    out_dir=self._frames_dir(run.id),
                    config=self.visual_config,
                    reporter=reporter,
                )
            except BaseException:
                db.update_run(run.id, status="failed")
                raise
        reporter(
            Stage.COMPLETE,
            f"Completed: {stats.kept} representative frames from {stats.sampled} sampled",
            position=info.duration,
        )
        return False

    def _frames_run(self) -> RunRecord | None:
        if self.visual_config.strategy == "off":
            return None
        key, _ = self._visual_key()
        run = self._db().best_run(_FRAMES, key)
        return run if run is not None and run.status == "complete" else None

    def frames(self, *, start: float | None = None, end: float | None = None) -> list[Frame]:
        """Representative frames stored by :meth:`index`, in timeline order.

        Each frame is a JPEG on disk (``frame.path``) with its presentation timestamp;
        ``frame.data_url()`` gives a base64 data URL for multimodal LLM APIs.

        Args:
            start: Only frames at or after this time (seconds).
            end: Only frames at or before this time (seconds).
        """
        run = self._frames_run()
        if run is None:
            return []
        folder = self._frames_dir(run.id)
        return [
            Frame(
                id=frame_id(row.idx),
                timestamp=row.timestamp,
                path=str(folder / row.path),
                width=row.width,
                height=row.height,
                reason=row.reason,
            )
            for row in self._db().frames(run.id, start=start, end=end)
        ]

    def _frame_count(self) -> int:
        run = self._frames_run()
        return self._db().frame_count(run.id) if run else 0

    def _is_cached(self) -> bool:
        key, _ = self._cache_key(self._backend_for())
        run = self._db().get_run(_TRANSCRIPT, key)
        return run is not None and run.status == "complete"

    def _record_silent_video(self, progress: ProgressCallback | None) -> None:
        reporter = Reporter(progress, self.media_info().duration)
        reporter(Stage.WARNING, f'"{self.name}" has no audio stream; nothing to transcribe.')
        db = self._db()
        key, material = self._cache_key(self._backend_for())
        run = db.get_run(_TRANSCRIPT, key) or db.create_run(
            kind=_TRANSCRIPT, config_key=key, config=material, library_version=__version__
        )
        if run.status != "complete":
            db.update_run(
                run.id,
                status="complete",
                processed_until=self.media_info().duration or 0.0,
                completed_at=utcnow(),
            )
        reporter(Stage.COMPLETE, "Completed (no audio)")

    def _summary(self, began: float, *, cached: bool) -> IndexSummary:
        run = self._read_run()
        stats = self._db().stats(run.id) if run else None
        return IndexSummary(
            video=self.name,
            status=_status(run, stats.segments if stats else 0),
            duration=self.media_info().duration,
            language=run.language if run else None,
            language_probability=run.language_probability if run else None,
            segments=stats.segments if stats else 0,
            chunks=stats.chunks if stats else 0,
            speech_seconds=run.speech_seconds if run else 0.0,
            indexed_until=run.processed_until if run else None,
            model=_model(run),
            cached=cached,
            elapsed_seconds=time.perf_counter() - began,
            transcription_seconds=run.elapsed_seconds if run else 0.0,
            frames=self._frame_count(),
        )

    # -- queries ----------------------------------------------------------------------

    def transcript(self) -> Transcript:
        """The stored transcript (complete or, while indexing, partial). Never transcribes."""
        run = self._require_run()
        db = self._db()
        segments = tuple(s for _, s in db.segments(run.id, with_words=True))
        return Transcript(
            segments=segments,
            language=run.language,
            status=_status(run, len(segments)),
            indexed_until=run.processed_until,
            duration=self.media_info().duration,
            model=_model(run),
            language_probability=run.language_probability,
            chunks=tuple(db.chunks(run.id)),
        )

    def chunks(self) -> list[TranscriptChunk]:
        """Retrieval chunks (20–60 s of consecutive segments), ready for your own RAG store."""
        run = self._require_run()
        return self._db().chunks(run.id)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        start: float | None = None,
        end: float | None = None,
        expand: float = 0.0,
    ) -> list[SearchResult]:
        """Find the passages that best match ``query`` (BM25 over the transcript).

        Works on partial indexes, so it can be used while :meth:`index` is still running.

        Args:
            query: Free text in any language; stop words are ignored.
            limit: Maximum number of results.
            start: Only consider transcript after this time (seconds).
            end: Only consider transcript before this time (seconds).
            expand: Seconds of surrounding transcript to add to each result.
        """
        run = self._require_run()
        return search_transcript(
            self._db(), run.id, query, limit=limit, start=start, end=end, expand=expand
        )

    def context(
        self,
        query: str,
        *,
        max_tokens: int = 4000,
        before: float = 12.0,
        after: float = 15.0,
        segment_timestamps: bool = False,
        token_counter: TokenCounter | None = None,
    ) -> VideoContext:
        """Build LLM-ready, timestamped evidence for ``query`` within a token budget.

        Args:
            query: The question you will ask your LLM.
            max_tokens: Budget for ``context.text`` (estimated unless ``token_counter`` is given).
            before: Seconds of transcript to include before each hit.
            after: Seconds of transcript to include after each hit.
            segment_timestamps: Timestamp every segment inside each passage.
            token_counter: ``str -> int`` using your model's tokenizer.
        """
        run = self._require_run()
        db = self._db()
        segments = db.stats(run.id).segments
        return build_context(
            db,
            run,
            query=query,
            video_name=self.name,
            duration=self.media_info().duration,
            status=_status(run, segments),
            max_tokens=max_tokens,
            options=ContextOptions(
                before=before, after=after, segment_timestamps=segment_timestamps
            ),
            token_counter=token_counter,
            frames=lambda a, b: self.frames(start=a, end=b),
        )

    def ask(
        self,
        question: str,
        *,
        llm: LLM | None = None,
        progress: ProgressCallback | None = None,
        max_images: int = 12,
        image_detail: Literal["low", "high", "auto"] = "low",
        evidence_tokens: int = 48_000,
        max_answer_tokens: int = 8000,
        explore: bool | None = None,
        max_steps: int = 4,
        max_views: int = 16,
    ) -> Answer:
        """Ask your LLM a question about the video, answered from what was said and shown.

        With an LLM that supports tool calling (Azure, OpenAI), the model *explores* the
        video: starting from the transcript and frame thumbnails, it requests exact
        high-resolution frames and zooms into regions at any timestamp until it has enough
        evidence (``explore=False`` sends one fixed set of evidence instead).

        Indexes the video first if needed (cached afterwards). The LLM receives the
        timestamped transcript (all of it when it fits ``evidence_tokens``, otherwise the
        passages relevant to the question) plus up to ``max_images`` representative frames,
        and is told to rely only on that evidence and cite timestamps.

        Args:
            question: Anything, in any language.
            llm: Overrides the ``llm`` given to :class:`Video`.
            progress: Progress callback for the indexing step.
            max_images: Frames sent as images (0 for text only).
            image_detail: ``"low"`` (cheap, default) or ``"high"`` when small on-screen text matters.
            evidence_tokens: Budget for transcript evidence.
            max_answer_tokens: Output budget, including any hidden reasoning of reasoning models.
            explore: Let the model look around the video with tools. Defaults to on when the LLM
                supports tools and images.
            max_steps: Upper bound on exploration rounds (each round may request several images).
            max_views: Upper bound on frames/zooms the model may look at while exploring.

        Returns:
            :class:`~saccade.models.answer.Answer` — ``str(answer)`` is the text; ``answer.evidence``
            and ``answer.frames`` are the source material that was sent.
        """
        from saccade.ask import ask as ask_video

        model = llm or self.llm
        if model is None:
            raise ConfigError(
                "No LLM configured. Pass one to the video, e.g. "
                "Video(path, llm=saccade.azure(endpoint=..., api_key=..., deployment=...)), "
                "or call video.ask(question, llm=...)."
            )
        self._index(force=False, progress=progress, cancel=None)
        use_tools = explore
        if use_tools is None:
            use_tools = bool(getattr(model, "supports_tools", False)) and model.supports_images
        if use_tools:
            from saccade.explore import explore as explore_video

            return explore_video(
                self,
                question,
                model,
                reporter=Reporter(progress, self.media_info().duration),
                evidence_tokens=evidence_tokens,
                overview_images=max_images,
                max_steps=max_steps,
                max_images=max_views,
                max_answer_tokens=max_answer_tokens,
            )
        return ask_video(
            self,
            question,
            model,
            evidence_tokens=evidence_tokens,
            max_images=max_images,
            image_detail=image_detail,
            max_answer_tokens=max_answer_tokens,
        )

    async def aask(self, question: str, **kwargs: Any) -> Answer:
        """Async version of :meth:`ask` (runs in a worker thread)."""
        return await asyncio.to_thread(self.ask, question, **kwargs)

    def info(self) -> VideoInfo:
        """Media metadata and index status. Does not transcribe."""
        media = self.media_info()
        db = self._db()
        run = self._read_run()
        stats = db.stats(run.id) if run else None
        runs = [
            {
                "id": r.id,
                "status": r.status,
                "model": (r.config.get("asr") or {}).get("model"),
                "language": r.language,
                "processed_until": r.processed_until,
                "updated_at": r.updated_at,
            }
            for r in db.runs(_TRANSCRIPT)
        ]
        return VideoInfo(
            path=str(self.path),
            name=self.name,
            fingerprint=self.fingerprint,
            media=media,
            index_dir=str(self.index_dir),
            transcript_status=_status(run, stats.segments) if run and stats else None,
            language=run.language if run else None,
            segments=stats.segments if stats else 0,
            chunks=stats.chunks if stats else 0,
            indexed_until=run.processed_until if run else None,
            model=_model(run),
            frames=self._frame_count(),
            runs=runs,
        )

    # -- async ------------------------------------------------------------------------

    async def atranscribe(
        self,
        *,
        profile: Profile | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> AsyncIterator[TranscriptSegment]:
        """Async version of :meth:`transcribe`; decoding and inference run in a worker thread."""
        self._apply_profile(profile)
        async for segment in iterate_in_thread(
            lambda: self._transcribe(force=force, progress=progress)
        ):
            yield segment

    async def aindex(
        self,
        *,
        profile: Profile | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IndexSummary:
        """Async version of :meth:`index`. Cancelling the task stops after the current block."""
        self._apply_profile(profile)
        began = time.perf_counter()
        if not await asyncio.to_thread(lambda: self.media_info().has_audio):
            await asyncio.to_thread(self._record_silent_video, progress)
            cached = await asyncio.to_thread(self._index_frames, force=force, progress=progress)
            return await asyncio.to_thread(self._summary, began, cached=cached)
        cached = await asyncio.to_thread(self._is_cached) and not force
        async for _ in self.atranscribe(force=force, progress=progress):
            pass
        frames_cached = await asyncio.to_thread(self._index_frames, force=force, progress=progress)
        return await asyncio.to_thread(self._summary, began, cached=cached and frames_cached)

    async def asearch(self, query: str, **kwargs: Any) -> list[SearchResult]:
        """Async version of :meth:`search`."""
        return await asyncio.to_thread(self.search, query, **kwargs)

    async def acontext(self, query: str, **kwargs: Any) -> VideoContext:
        """Async version of :meth:`context`."""
        return await asyncio.to_thread(self.context, query, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Video({str(self.path)!r}, model={self._asr.model!r}, language={self._asr.language!r})"
        )


def _status(run: RunRecord | None, segments: int) -> TranscriptStatus:
    if run is None:
        return "empty"
    if run.status == "complete":
        return "complete" if segments else "empty"
    if run.status == "failed":
        return "failed"
    return "running" if run.status == "running" else "partial"


def _model(run: RunRecord | None) -> str | None:
    if run is None:
        return None
    model = (run.config.get("asr") or {}).get("model")
    return str(model) if model is not None else None


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


__all__ = ["IndexJob", "IndexSummary", "Video", "VideoInfo"]
