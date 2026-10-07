"""Choosing representative frames from a stream of cheap thumbnails.

The selector sees one grayscale thumbnail (64x36) per sampled frame and decides, in
O(1) per sample, whether that frame is worth storing:

* a *change* is a large difference from the last kept frame;
* on ``screen``-like content the selector waits until a transition has settled
  (two consecutive samples nearly identical) and keeps the settled view, so animations
  and half-rendered pages are skipped;
* a candidate that looks like one of the recently kept frames (slide A → B → A, a
  dialog opening and closing) is a duplicate and is not stored again;
* ``interval``/``auto`` add periodic frames while the picture is actually changing, so
  camera footage gets sparse coverage and static screens get none.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from saccade.config import VisualConfig
from saccade.vision.hashing import Thumb, hamming, mean_abs_diff

Reason = Literal["start", "change", "interval", "end"]

_RECENT = 8
_DUP_HAMMING = 10  # of 256 bits
_DUP_DIFF = 4.0
_SETTLED_DIFF = 1.5
_SETTLE_TIMEOUT_S = 4.0
_STATIC_DIFF = 0.6  # median consecutive difference below this = screen-like content
_ADAPT_WINDOW = 30


@dataclass(frozen=True, slots=True)
class Params:
    change: float
    settle: bool
    periodic: float | None
    periodic_min_diff: float
    min_gap: float


def params_for(config: VisualConfig, *, static: bool) -> Params:
    strategy = config.strategy
    if strategy == "screen" or (strategy == "auto" and static):
        return Params(change=3.0, settle=True, periodic=None, periodic_min_diff=0.0, min_gap=1.5)
    if strategy == "scenes":
        return Params(change=25.0, settle=False, periodic=None, periodic_min_diff=0.0, min_gap=1.0)
    if strategy == "interval":
        return Params(
            change=float("inf"),
            settle=False,
            periodic=config.interval_s,
            periodic_min_diff=0.0,
            min_gap=0.0,
        )
    # auto on camera-like content
    return Params(
        change=12.0, settle=True, periodic=config.interval_s, periodic_min_diff=1.5, min_gap=4.0
    )


@dataclass(slots=True)
class Candidate:
    time: float
    thumb: Thumb
    hash: int
    token: object  # opaque handle to the decoded frame (kept by the caller)


@dataclass(frozen=True, slots=True)
class Decision:
    candidate: Candidate
    reason: Reason


class FrameSelector:
    def __init__(self, config: VisualConfig) -> None:
        self.config = config
        self._static_votes: deque[float] = deque(maxlen=_ADAPT_WINDOW)
        self._previous: Candidate | None = None
        self._last_kept: Candidate | None = None
        self._reference: Thumb | None = None  # what "no change" is measured against
        self._recent: deque[Candidate] = deque(maxlen=_RECENT)
        self._pending: Candidate | None = None
        self.kept = 0

    @property
    def params(self) -> Params:
        static = (
            len(self._static_votes) >= 5
            and float(np.median(np.asarray(self._static_votes))) < _STATIC_DIFF
        )
        return params_for(self.config, static=static)

    def offer(self, candidate: Candidate) -> list[Decision]:
        out: list[Decision] = []
        if self._previous is not None:
            self._static_votes.append(mean_abs_diff(candidate.thumb, self._previous.thumb))
        previous, self._previous = self._previous, candidate

        if self._last_kept is None:
            self._keep(candidate, "start", out)
            return out
        p = self.params
        since_kept = candidate.time - self._last_kept.time

        if self._pending is not None:
            settled = (
                previous is not None
                and mean_abs_diff(candidate.thumb, previous.thumb) <= _SETTLED_DIFF
            )
            if settled or candidate.time - self._pending.time >= _SETTLE_TIMEOUT_S:
                self._pending = None
                self._keep(candidate, "change", out)
            return out

        assert self._reference is not None
        diff = mean_abs_diff(candidate.thumb, self._reference)
        if diff >= p.change and since_kept >= p.min_gap:
            if p.settle:
                self._pending = candidate
            else:
                self._keep(candidate, "change", out)
        elif p.periodic is not None and since_kept >= p.periodic and diff >= p.periodic_min_diff:
            self._keep(candidate, "interval", out)
        return out

    def finish(self) -> list[Decision]:
        out: list[Decision] = []
        if self._pending is not None and self._previous is not None:
            self._keep(self._previous, "end", out)
            self._pending = None
        return out

    def _keep(self, candidate: Candidate, reason: Reason, out: list[Decision]) -> None:
        # Whatever happens, this view becomes the new reference, so a duplicate does not
        # re-trigger on every following sample.
        self._reference = candidate.thumb
        if reason != "start" and self._is_duplicate(candidate):
            return
        if self.kept >= self.config.max_frames:
            return
        light = Candidate(
            candidate.time, candidate.thumb, candidate.hash, None
        )  # drop the decoded frame
        self._last_kept = light
        self._recent.append(light)
        self.kept += 1
        out.append(Decision(candidate, reason))

    def _is_duplicate(self, candidate: Candidate) -> bool:
        return any(
            hamming(candidate.hash, seen.hash) <= _DUP_HAMMING
            and mean_abs_diff(candidate.thumb, seen.thumb) <= _DUP_DIFF
            for seen in self._recent
        )
