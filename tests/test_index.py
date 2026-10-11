"""Packing/timestamp maps, chunking, FTS normalisation and the SQLite layer."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from saccade.config import ChunkConfig
from saccade.index.chunking import Chunker, ends_sentence, join_texts
from saccade.index.database import Database, NewChunk
from saccade.index.fts import build_match, light_stem, matched_terms, normalize_text, tokenize
from saccade.models.segment import TranscriptSegment, segment_id
from saccade.pipeline import Pack, Packer
from saccade.retrieval.search import search_transcript
from saccade.retrieval.semantic import search_semantic
from saccade.vad.base import SpeechRegion

SR = 16_000


def seg(i: int, start: float, end: float, text: str) -> tuple[int, TranscriptSegment]:
    return i, TranscriptSegment(id=segment_id(i), start=start, end=end, text=text, language="en")


# -- packing and timestamp restoration -------------------------------------------------------


def make_pack(regions: list[tuple[float, float]]) -> Pack:
    packer = Packer(max_seconds=1e9, first_seconds=1e9, max_join_gap=1e9)
    for start, end in regions:
        packer.add(SpeechRegion(start, end), np.zeros(round((end - start) * SR), np.float32))
    pack = packer.flush()
    assert pack is not None
    return pack


def test_pack_maps_concatenated_time_back_to_source() -> None:
    pack = make_pack([(10.0, 15.0), (100.0, 104.0), (500.0, 510.0)])
    assert pack.speech_seconds == pytest.approx(19.0)
    assert pack.to_media_time(0.0) == pytest.approx(10.0)
    assert pack.to_media_time(2.5) == pytest.approx(12.5)
    assert pack.to_media_time(6.0) == pytest.approx(101.0)
    assert pack.to_media_time(9.0) == pytest.approx(500.0)
    assert pack.to_media_time(5.0) == pytest.approx(100.0), (
        "a start on the seam belongs to the next region"
    )
    assert pack.to_media_time(5.0, is_end=True) == pytest.approx(15.0), (
        "an end on the seam belongs to the previous"
    )
    assert pack.to_media_time(99.0, is_end=True) == pytest.approx(510.0), (
        "clamped to the last region"
    )


def test_segment_grazing_a_seam_is_trimmed() -> None:
    pack = make_pack([(10.0, 20.0), (80.0, 90.0)])
    # Starts 0.4 s before the seam (inside region 1's padding), mostly in region 2.
    assert pack.map_span(9.6, 16.0) == pytest.approx((80.0, 86.0))
    # Ends 0.3 s into region 2.
    assert pack.map_span(4.0, 10.3) == pytest.approx((14.0, 20.0))
    # Genuinely spans both: kept as is.
    assert pack.map_span(6.0, 14.0) == pytest.approx((16.0, 84.0))


def test_packer_limits_and_first_pack() -> None:
    packer = Packer(max_seconds=30, first_seconds=10, max_join_gap=5)
    one_s = np.zeros(SR, np.float32)
    packs = []
    for i in range(70):
        pack = packer.add(SpeechRegion(i * 2.0, i * 2.0 + 1.0), one_s)
        if pack:
            packs.append(pack)
    last = packer.flush()
    if last is not None:
        packs.append(last)
    sizes = [round(p.speech_seconds) for p in packs]
    assert sizes[0] == 10
    assert all(s <= 30 for s in sizes)
    assert sum(sizes) == 70


def test_packer_never_joins_across_long_gaps() -> None:
    packer = Packer(max_seconds=30, first_seconds=30, max_join_gap=3.0)
    one_s = np.zeros(SR, np.float32)
    assert packer.add(SpeechRegion(0.0, 1.0), one_s) is None
    assert packer.add(SpeechRegion(3.5, 4.5), one_s) is None  # 2.5 s gap: joined
    pack = packer.add(SpeechRegion(50.0, 51.0), one_s)  # 45.5 s gap: previous pack closed first
    assert pack is not None and [r.start for r in pack.regions] == [0.0, 3.5]


def test_packer_flushes_sparse_speech_by_span() -> None:
    packer = Packer(max_seconds=30, first_seconds=30, max_span=120)
    packer.add(SpeechRegion(0.0, 1.0), np.zeros(SR, np.float32))
    packer.add(SpeechRegion(2.0, 3.0), np.zeros(SR, np.float32))
    assert packer.due(60.0, next_region_from=3.5) is None
    pack = packer.due(121.0, next_region_from=3.5)
    assert pack is not None and pack.speech_seconds == pytest.approx(2.0)


def test_packer_flushes_when_nothing_can_join() -> None:
    packer = Packer(max_seconds=30, first_seconds=30, max_join_gap=3.0)
    packer.add(SpeechRegion(10.0, 12.0), np.zeros(2 * SR, np.float32))
    assert (
        packer.due(14.0, next_region_from=13.5) is None
    )  # a region starting soon could still join
    pack = packer.due(16.0, next_region_from=15.5)  # VAD guarantees nothing starts before 15.5
    assert pack is not None and pack.end == 12.0


# -- chunking --------------------------------------------------------------------------------


def chunk_all(segments: list[tuple[int, TranscriptSegment]], config: ChunkConfig | None = None):
    chunker = Chunker(config)
    out = []
    for idx, s in segments:
        out += chunker.add(idx, s)
    return out + chunker.flush()


def test_chunks_cover_each_segment_once_and_respect_bounds() -> None:
    rng = np.random.default_rng(7)
    segments, t = [], 0.0
    for i in range(400):
        dur = float(rng.uniform(1, 9))
        text = "word " * int(dur * 2) + rng.choice([".", ",", "", "?"])
        segments.append(seg(i, t, t + dur, text))
        t += dur + float(rng.choice([0.1, 0.3, 1.0, 12.0], p=[0.5, 0.3, 0.15, 0.05]))
    chunks = chunk_all(segments)
    covered = [i for c in chunks for i in range(c.first_segment, c.last_segment + 1)]
    assert covered == list(range(400))
    assert all(c.end - c.start <= 60.0 + 1e-6 for c in chunks)
    assert all(a.end <= b.start for a, b in itertools.pairwise(chunks))
    long_enough = [c for c in chunks[:-1] if c.end - c.start >= 20.0]
    assert len(long_enough) / len(chunks) > 0.6


def test_chunks_prefer_sentence_boundaries() -> None:
    segments = [
        seg(i, i * 5.0, i * 5.0 + 4.8, "a sentence." if i % 5 == 4 else "and more")
        for i in range(30)
    ]
    chunks = chunk_all(segments)
    assert all(ends_sentence(c.text) for c in chunks[:-1])


def test_long_pause_starts_new_chunk() -> None:
    segments = [seg(0, 0, 6, "one"), seg(1, 6, 12, "two"), seg(2, 40, 45, "three")]
    chunks = chunk_all(segments)
    assert [(c.first_segment, c.last_segment) for c in chunks] == [(0, 1), (2, 2)]


def test_incremental_chunking_with_pending_restore_matches_single_pass() -> None:
    segments = [seg(i, i * 4.0, i * 4.0 + 3.9, "x." if i % 7 == 6 else "y") for i in range(50)]
    whole = chunk_all(segments)
    first = Chunker()
    out = []
    for idx, s in segments[:23]:
        out += first.add(idx, s)
    done = {i for c in out for i in range(c.first_segment, c.last_segment + 1)}
    resumed = Chunker(next_index=len(out), pending=[x for x in segments[:23] if x[0] not in done])
    for idx, s in segments[23:]:
        out += resumed.add(idx, s)
    out += resumed.flush()
    assert out == whole


def test_join_texts_handles_cjk() -> None:
    assert join_texts(["部署", "失败了"]) == "部署失败了"
    assert join_texts(["Hello", " world "]) == "Hello world"


# -- text normalisation & query building ---------------------------------------------------------


def test_normalisation_casefolds_and_spaces_cjk() -> None:
    assert normalize_text("Straße") == "strasse"
    assert normalize_text("部署失败") == " 部  署  失  败 "
    assert [t.text for t in tokenize("Configuração da Autenticação!")] == [
        "configuracao",
        "da",
        "autenticacao",
    ]
    assert [t.text for t in tokenize("नमस्ते दुनिया")] == ["नमस्ते", "दुनिया"], (
        "combining marks stay in words"
    )


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("deployment", "deploy"),
        ("failed", "fail"),
        ("authentication", "authentic"),
        ("configuracao", "configur"),
        ("bereitstellung", "bereitstell"),
        ("errors", "error"),
        ("process", None),
        ("api", None),
    ],
)
def test_light_stem(word: str, stem: str | None) -> None:
    assert light_stem(word) == stem


def test_build_match_drops_stopwords_and_adds_phrase() -> None:
    match = build_match("Why did the deployment fail?")
    assert match is not None
    assert [t.text for t in match.terms] == ["deployment", "fail"]
    assert '"deploy" *' in match.expression
    assert match.phrase == "why did the deployment fail"


def test_build_match_keeps_stopword_only_queries() -> None:
    match = build_match("what is this")
    assert match is not None and len(match.terms) == 3


def test_build_match_cjk_bigrams_and_empty() -> None:
    match = build_match("部署失败")
    assert match is not None
    assert [t.text for t in match.terms] == ["部署", "署失", "失败"]
    assert build_match("?!  ...") is None


def test_matched_terms_uses_stems() -> None:
    match = build_match("deployment failures")
    assert match is not None
    assert matched_terms(match, "The deploy failed twice") == ("deployment", "failures")


# -- database ------------------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "index.db")


def _store(db: Database, texts: list[str], step: float = 5.0) -> int:
    run = db.create_run(kind="transcript", config_key="k", config={}, library_version="test")
    rows = [seg(i, i * step, i * step + step - 0.5, t) for i, t in enumerate(texts)]
    conn = db.connect()
    db.commit_batch(
        conn,
        run.id,
        segments=rows,
        regions=[],
        chunks=[],
        processed_until=len(texts) * step,
        speech_seconds=0,
        elapsed_seconds=0,
    )
    conn.close()
    return run.id


MULTILINGUAL = [
    "We configured OAuth authentication for the API.",
    "A configuração da autenticação falhou no App Service.",
    "La configuración de autenticación falló ayer.",
    "La configuration de l'authentification a échoué.",
    "Die Bereitstellung ist wegen der Straße fehlgeschlagen.",
    "Развертывание завершилось ошибкой аутентификации.",
    "部署失败是因为身份验证配置错误。",
    "デプロイが失敗しました。",
    "तैनाती विफल रही क्योंकि प्रमाणीकरण गलत था।",
    "فشل النشر بسبب المصادقة",
]


def test_unicode_round_trip(db: Database) -> None:
    run_id = _store(db, MULTILINGUAL)
    stored = [s.text for _, s in db.segments(run_id)]
    assert stored == MULTILINGUAL


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("OAuth authentication", 0),
        ("configuracao autenticacao", 1),  # diacritics folded
        ("configuración", 2),
        ("authentification échoué", 3),
        ("strasse", 4),  # ß casefolds to ss
        ("bereitstellen", 4),  # stem prefix
        ("аутентификации", 5),
        ("身份验证", 6),
        ("失败", 6),
        ("デプロイ", 7),
        ("प्रमाणीकरण", 8),
        ("المصادقة", 9),
    ],
)
def test_multilingual_fts(db: Database, query: str, expected: int) -> None:
    run_id = _store(db, MULTILINGUAL)
    match = build_match(query)
    assert match is not None
    hits = db.search(run_id, match.expression, start=None, end=None, limit=5)
    assert hits, f"no hits for {query!r}"
    assert hits[0].idx == expected


def test_runs_are_isolated_and_deletable(db: Database) -> None:
    a = _store(db, ["alpha beta"])
    b_run = db.create_run(kind="transcript", config_key="other", config={}, library_version="test")
    assert db.segments(b_run.id) == []
    match = build_match("alpha")
    assert match is not None
    assert db.search(b_run.id, match.expression, start=None, end=None, limit=5) == []
    db.delete_run(a)
    assert db.get_run_by_id(a) is None
    assert db.search(a, match.expression, start=None, end=None, limit=5) == []


def test_best_run_prefers_matching_key_then_complete(db: Database) -> None:
    partial = db.create_run(kind="transcript", config_key="partial", config={}, library_version="t")
    done = db.create_run(kind="transcript", config_key="done", config={}, library_version="t")
    db.update_run(done.id, status="complete")
    assert db.best_run("transcript", "partial").id == partial.id  # type: ignore[union-attr]
    assert db.best_run("transcript", "unknown").id == done.id  # type: ignore[union-attr]


def test_reopening_existing_database_is_safe(tmp_path: Path) -> None:
    path = tmp_path / "again.db"
    run_id = _store(Database(path), ["persisted text"])
    reopened = Database(path)
    assert [s.text for _, s in reopened.segments(run_id)] == ["persisted text"]


class BilingualEncoder:
    key = "bilingual-test-v1"

    def __init__(self):
        self.passages = 0

    def encode(self, texts, *, query=False):
        if not query:
            self.passages += len(texts)
        return np.asarray(
            [
                [1.0, 0.0] if "sign-in" in text or "autenticacao" in text else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_semantic_cross_language_cache_and_model_isolation(db):
    texts = ["A autenticacao falhou.", "O tempo esta bom."]
    run_id = _store(db, texts, step=40)
    with db.session() as conn:
        db.commit_chunks(
            conn,
            run_id,
            [
                NewChunk(idx, idx * 40, idx * 40 + 39.5, text, idx, idx)
                for idx, text in enumerate(texts)
            ],
        )
    encoder = BilingualEncoder()
    assert search_transcript(db, run_id, "sign-in") == []
    results = search_semantic(db, run_id, "sign-in", encoder)
    assert results[0].text == texts[0]
    assert results[0].segment_ids == (segment_id(0),)
    assert encoder.passages == 2
    assert search_semantic(db, run_id, "sign-in", encoder) == results
    assert encoder.passages == 2
    encoder.key = "bilingual-test-v2"
    search_semantic(db, run_id, "sign-in", encoder)
    assert encoder.passages == 4
    assert search_semantic(db, run_id, "sign-in", encoder, start=40)[0].text == texts[1]
    assert search_semantic(db, run_id, "sign-in", encoder, end=1)[0].text == texts[0]
    assert search_semantic(db, run_id, "", encoder) == []
    db.delete_run(run_id)
    with db.session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] == 0


def test_semantic_video_and_context(make_video, tone_video, monkeypatch):
    video = make_video(tone_video)
    video.index()
    video._encoder = BilingualEncoder()
    assert video.search("sign-in", mode="lexical") == []
    results = video.search("sign-in", mode="semantic")
    assert results and results[0].segment_ids
    assert video.context("sign-in", mode="semantic").segments
    assert video.context("sign-in", mode="lexical").segments == []
    monkeypatch.setattr("saccade.video.available", lambda: True)
    assert video.search("sign-in") == video.search("sign-in", mode="hybrid")


def test_semantic_requires_extra(tmp_path, monkeypatch):
    from saccade.exceptions import ConfigError
    from saccade.retrieval.semantic import FastEmbedEncoder

    monkeypatch.setattr("saccade.retrieval.semantic.available", lambda: False)
    with pytest.raises(ConfigError, match="embeddings"):
        FastEmbedEncoder("model", tmp_path, offline=True, threads=1)


def test_hybrid_fuses_overlapping_passages():
    from saccade.models.result import SearchResult
    from saccade.retrieval.ranking import reciprocal_rank_fusion

    lexical = SearchResult(0, 5, "login", 1.0, segment_ids=("s0",))
    semantic = SearchResult(0, 10, "sign-in", 1.0, segment_ids=("s0", "s1"))
    other = SearchResult(20, 25, "unrelated", 0.5, segment_ids=("s2",))
    results = reciprocal_rank_fusion([[lexical], [other, semantic]], limit=2)
    assert results[0] == lexical
    assert len(results) == 2
    assert results[1].score < 1.0


@pytest.mark.parametrize("vector", [[[0.0, 0.0]], [[float("nan"), 1.0]], [1.0, 0.0]])
def test_semantic_rejects_invalid_vectors(db, vector):
    from saccade.exceptions import ConfigError

    class InvalidEncoder:
        key = "invalid"

        def encode(self, texts, *, query=False):
            return np.asarray(vector, dtype=np.float32)

    run_id = _store(db, ["text"])
    with pytest.raises(ConfigError, match="vector"):
        search_semantic(db, run_id, "query", InvalidEncoder())


def test_embedding_backend_forwards_offline_and_loads_lazily(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from saccade.retrieval.semantic import FastEmbedEncoder

    calls = []

    class FakeModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def query_embed(self, texts):
            return [np.array([1.0, 0.0]) for text in texts]

        passage_embed = query_embed

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeModel))
    monkeypatch.setattr("saccade.retrieval.semantic.available", lambda: True)
    monkeypatch.setattr("saccade.retrieval.semantic.version", lambda package: "test")
    encoder = FastEmbedEncoder("test-model", tmp_path, offline=True, threads=2)
    assert calls == []
    assert encoder.encode(["query"], query=True).shape == (1, 2)
    encoder.encode(["passage"])
    assert calls == [
        {
            "model_name": "test-model",
            "cache_dir": str(tmp_path),
            "local_files_only": True,
            "threads": 2,
        }
    ]
    assert "test-model" in encoder.key
