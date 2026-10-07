"""Turns per-window speech probabilities into padded, merged speech regions — streaming.

The raw detection logic is Silero's reference algorithm (hysteresis thresholds, minimum
silence before closing, splitting over-long regions at the last real pause), rewritten
to work incrementally. Post-processing then pads each region, drops short false
positives, and merges regions whose padded gap is below ``merge_gap_ms`` so sentences
are not cut at short breaths. All positions are integer sample offsets.
"""

from __future__ import annotations

from collections.abc import Iterable

from saccade.config import VadConfig

Span = tuple[int, int]


class SpeechSegmenter:
    def __init__(
        self, config: VadConfig, *, sample_rate: int = 16_000, window_samples: int = 512
    ) -> None:
        sr = sample_rate
        self.window = window_samples
        self.threshold = config.threshold
        self.neg_threshold = max(config.threshold - 0.15, 0.01)
        self.min_speech = sr * config.min_speech_ms / 1000
        self.min_silence = sr * config.min_silence_ms / 1000
        self.min_silence_at_max = sr * 98 / 1000
        self.pad = int(sr * config.padding_ms / 1000)
        self.merge_gap = int(sr * config.merge_gap_ms / 1000)
        self.max_region = int(sr * config.max_region_s)
        self.max_speech = self.max_region - self.window - 2 * self.pad

        self._index = 0
        self._triggered = False
        self._start = 0
        self._temp_end = 0
        self._prev_end = 0
        self._next_start = 0

        self._pending: Span | None = None
        self._pending_raw_end = 0
        self._emitted_end = 0
        self._out: list[Span] = []

    # -- public API -----------------------------------------------------------------

    @property
    def position(self) -> int:
        """Samples consumed so far."""
        return self._index * self.window

    def push(self, probabilities: Iterable[float]) -> list[Span]:
        for probability in probabilities:
            self._step(float(probability))
            self._index += 1
        self._emit_if_settled()
        return self._drain()

    def finish(self, total_samples: int) -> list[Span]:
        if self._triggered and total_samples - self._start > self.min_speech:
            self._on_raw(self._start, total_samples)
        self._triggered = False
        if self._pending is not None:
            self._emit()
        clipped = [(s, min(e, total_samples)) for s, e in self._drain()]
        return [(s, e) for s, e in clipped if e > s]

    @property
    def retain_from(self) -> int:
        """Earliest sample a not-yet-emitted region may still include."""
        candidates = [self.position - self.pad]
        if self._pending is not None:
            candidates.append(self._pending[0])
        if self._triggered:
            candidates.append(self._start - self.pad)
        return max(self._emitted_end, min(candidates), 0)

    # -- raw detection (Silero reference algorithm, streaming) ------------------------

    def _step(self, p: float) -> None:
        now = self._index * self.window
        if p >= self.threshold and self._temp_end:
            self._temp_end = 0
            if self._next_start < self._prev_end:
                self._next_start = now

        if p >= self.threshold and not self._triggered:
            self._triggered = True
            self._start = now
            return

        if self._triggered and now - self._start > self.max_speech:
            if self._prev_end:
                self._on_raw(self._start, self._prev_end)
                if self._next_start < self._prev_end:
                    self._triggered = False
                else:
                    self._start = self._next_start
                self._prev_end = self._next_start = self._temp_end = 0
            else:
                self._on_raw(self._start, now)
                self._prev_end = self._next_start = self._temp_end = 0
                self._triggered = False
                return

        if p < self.neg_threshold and self._triggered:
            if not self._temp_end:
                self._temp_end = now
            if now - self._temp_end > self.min_silence_at_max:
                self._prev_end = self._temp_end
            if now - self._temp_end < self.min_silence:
                return
            if self._temp_end - self._start > self.min_speech:
                self._on_raw(self._start, self._temp_end)
            self._prev_end = self._next_start = self._temp_end = 0
            self._triggered = False

    # -- padding / merging -----------------------------------------------------------

    def _on_raw(self, start: int, end: int) -> None:
        s = max(start - self.pad, self._emitted_end, 0)
        e = end + self.pad
        if self._pending is not None:
            ps, pe = self._pending
            if s - pe <= self.merge_gap and e - ps <= self.max_region:
                self._pending = (ps, max(pe, e))
                self._pending_raw_end = end
                return
            if s < pe:
                # Padding overlaps but merging would exceed max_region: split the gap evenly.
                middle = max(ps + 1, (self._pending_raw_end + start) // 2)
                self._pending = (ps, middle)
                s = middle
            self._emit()
            s = max(s, self._emitted_end)
        self._pending = (s, e)
        self._pending_raw_end = end

    def _emit_if_settled(self) -> None:
        if self._pending is None:
            return
        _, pending_end = self._pending
        if self.position < pending_end:
            return  # the padded tail has not been decoded yet
        earliest_next = (self._start if self._triggered else self.position) - self.pad
        if earliest_next - pending_end > self.merge_gap:
            self._emit()

    def _emit(self) -> None:
        assert self._pending is not None
        self._out.append(self._pending)
        self._emitted_end = self._pending[1]
        self._pending = None

    def _drain(self) -> list[Span]:
        out, self._out = self._out, []
        return out
