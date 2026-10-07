from __future__ import annotations

import itertools

import numpy as np
import pytest

from saccade.config import VadConfig
from saccade.media.audio import AudioBlock
from saccade.vad import FixedWindowVAD, SileroModel, SileroVAD
from saccade.vad.segmenter import SpeechSegmenter
from support import EnergyModel, place, tone

SR = 16_000
W = 512  # samples per window = 32 ms


def probs_for(spans: list[tuple[float, float]], total: float) -> np.ndarray:
    """Window probabilities that are 1.0 inside ``spans`` (seconds) and 0.0 elsewhere."""
    n = int(total * SR / W)
    t = np.arange(n) * W / SR
    p = np.zeros(n, dtype=np.float32)
    for a, b in spans:
        p[(t >= a) & (t < b)] = 1.0
    return p


def segment(
    probs: np.ndarray, config: VadConfig, chunk: int | None = None
) -> list[tuple[float, float]]:
    seg = SpeechSegmenter(config)
    out = []
    step = chunk or len(probs)
    for i in range(0, len(probs), step):
        out += seg.push(probs[i : i + step])
    out += seg.finish(len(probs) * W)
    return [(s / SR, e / SR) for s, e in out]


def test_padding_and_absolute_spans() -> None:
    regions = segment(probs_for([(10.0, 12.0)], 20.0), VadConfig(padding_ms=200))
    assert len(regions) == 1
    start, end = regions[0]
    assert start == pytest.approx(9.8, abs=0.04)
    assert end == pytest.approx(12.2, abs=0.04)


def test_short_false_positives_are_dropped() -> None:
    regions = segment(probs_for([(5.0, 5.1), (10.0, 12.0)], 20.0), VadConfig(min_speech_ms=250))
    assert len(regions) == 1
    assert regions[0][0] > 9


def test_short_pauses_do_not_split_speech() -> None:
    # 300 ms pause < min_silence (500 ms): one region.
    regions = segment(probs_for([(1.0, 3.0), (3.3, 6.0)], 10.0), VadConfig())
    assert len(regions) == 1


def test_gaps_within_merge_gap_are_merged() -> None:
    # 0.9 s raw gap -> padded gap 0.5 s; merged with merge_gap 600 ms, separate with 350 ms.
    spans = [(1.0, 3.0), (3.9, 6.0)]
    assert len(segment(probs_for(spans, 10.0), VadConfig(merge_gap_ms=600))) == 1
    assert len(segment(probs_for(spans, 10.0), VadConfig(merge_gap_ms=350))) == 2


def test_long_speech_is_split_below_max_region() -> None:
    config = VadConfig(max_region_s=10.0)
    regions = segment(probs_for([(0.0, 35.0)], 40.0), config)
    assert len(regions) >= 4
    assert all(e - s <= 10.0 + 1e-6 for s, e in regions)
    assert regions[0][0] == pytest.approx(0.0, abs=0.04)
    assert regions[-1][1] == pytest.approx(35.2, abs=0.1)


def test_long_speech_prefers_splitting_at_pauses() -> None:
    config = VadConfig(max_region_s=10.0, merge_gap_ms=0)
    # 150 ms pause at 6 s: a split there is preferred over a hard cut at the limit.
    regions = segment(probs_for([(0.0, 6.0), (6.15, 14.0)], 16.0), config)
    assert any(abs(e - 6.2) < 0.15 for _, e in regions)


def test_regions_never_overlap_and_are_ordered() -> None:
    rng = np.random.default_rng(1)
    probs = (rng.random(3000) > 0.6).astype(np.float32)
    regions = segment(probs, VadConfig(min_silence_ms=100, merge_gap_ms=0))
    assert regions
    for (s1, e1), (s2, e2) in itertools.pairwise(regions):
        assert s1 < e1 <= s2 < e2


@pytest.mark.parametrize("chunk", [1, 7, 64, 1000])
def test_streaming_is_independent_of_block_size(chunk: int) -> None:
    probs = probs_for([(1.0, 2.0), (2.6, 4.0), (9.0, 30.0), (31.0, 31.5), (50.0, 53.0)], 60.0)
    config = VadConfig(max_region_s=12.0)
    assert segment(probs, config, chunk) == segment(probs, config)


def test_regions_are_emitted_before_end_of_stream() -> None:
    seg = SpeechSegmenter(VadConfig())
    early = seg.push(probs_for([(1.0, 3.0)], 10.0))
    assert len(early) == 1, "a region followed by long silence must not wait for the stream to end"


def test_retain_from_never_passes_unemitted_speech() -> None:
    probs = probs_for([(1.0, 3.0), (3.9, 5.0)], 12.0)
    seg = SpeechSegmenter(VadConfig())
    retained: list[int] = []
    emitted: list[tuple[int, int]] = []  # (step at which it was emitted, region start)
    for i in range(len(probs)):
        emitted += [(i, s) for s, _ in seg.push(probs[i : i + 1])]
        retained.append(seg.retain_from)
    emitted += [(len(probs), s) for s, _ in seg.finish(len(probs) * W)]
    assert len(emitted) == 2
    assert retained == sorted(retained), "retain_from must be monotonic"
    for step, start in emitted:
        # Until a region is emitted, the audio it starts with must not be released.
        assert all(r <= start for r in retained[:step])


def _blocks(audio: np.ndarray, start: float, size: int) -> list[AudioBlock]:
    return [AudioBlock(start + i / SR, audio[i : i + size]) for i in range(0, len(audio), size)]


def test_silero_vad_preserves_absolute_offset() -> None:
    audio = place(12.0, [(4.0, tone(300, 2.0))])
    vad = SileroVAD(VadConfig(), model=EnergyModel())
    regions = []
    for block in _blocks(audio, start=100.0, size=SR):
        regions += vad.feed(block)
    regions += vad.flush()
    assert len(regions) == 1
    assert regions[0].start == pytest.approx(103.8, abs=0.05)
    assert regions[0].end == pytest.approx(106.2, abs=0.05)


def test_silero_model_streaming_equals_batch() -> None:
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(W * 400) * 0.1).astype(np.float32)
    windows = audio.reshape(-1, W)
    batch = SileroModel()(windows)
    model = SileroModel()
    streamed = np.concatenate([model(windows[a:b]) for a, b in [(0, 13), (13, 200), (200, 400)]])
    np.testing.assert_allclose(batch, streamed, atol=1e-6)


def test_silero_finds_no_speech_in_silence_or_hum() -> None:
    audio = np.concatenate([np.zeros(SR * 5, np.float32), tone(50, 5.0, amp=0.05)])
    vad = SileroVAD(VadConfig())
    regions = []
    for block in _blocks(audio, 0.0, SR * 2):
        regions += vad.feed(block)
    regions += vad.flush()
    assert regions == []


def test_fixed_window_vad_covers_timeline() -> None:
    vad = FixedWindowVAD(window_s=10.0)
    audio = np.zeros(SR * 25, np.float32)
    regions = []
    for block in _blocks(audio, 0.0, SR * 3):
        regions += vad.feed(block)
    regions += vad.flush()
    assert [(r.start, r.end) for r in regions] == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]
