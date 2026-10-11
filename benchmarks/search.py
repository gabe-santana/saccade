"""Measure optional multilingual search over two hours of synthetic transcript chunks."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from saccade.index.database import Database, NewChunk
from saccade.models.segment import TranscriptSegment, segment_id
from saccade.retrieval.semantic import DEFAULT_MODEL, FastEmbedEncoder, search_semantic
from saccade.utils.cache import default_cache_dir, models_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    encoder = FastEmbedEncoder(
        DEFAULT_MODEL,
        models_dir(default_cache_dir()) / "embeddings",
        offline=args.offline,
        threads=args.threads,
    )
    with TemporaryDirectory() as folder:
        database = Database(Path(folder) / "index.db")
        try:
            run = database.create_run(
                kind="transcript", config_key="benchmark", config={}, library_version="benchmark"
            )
            texts = ["Hoje vamos discutir a previsao do tempo e a chuva."] * 240
            texts[120] = "O usuario nao conseguiu entrar porque a senha estava incorreta."
            segments = [
                (
                    idx,
                    TranscriptSegment(
                        id=segment_id(idx), start=idx * 30.0, end=idx * 30.0 + 29.0, text=text
                    ),
                )
                for idx, text in enumerate(texts)
            ]
            chunks = [
                NewChunk(idx, idx * 30.0, idx * 30.0 + 29.0, text, idx, idx)
                for idx, text in enumerate(texts)
            ]
            with database.session() as connection:
                database.commit_batch(
                    connection,
                    run.id,
                    segments=segments,
                    regions=[],
                    chunks=chunks,
                    processed_until=7200.0,
                    speech_seconds=0.0,
                    elapsed_seconds=0.0,
                )
            query = "Why could the user not sign in?"
            result = search_semantic(database, run.id, query, encoder)
            if result[0].start != 3600.0:
                raise RuntimeError(f"Cross-language retrieval failed: {result[0]}")
            times = []
            for _ in range(10):
                started = perf_counter()
                search_semantic(database, run.id, query, encoder)
                times.append((perf_counter() - started) * 1000)
            print(f"Cross-language top hit: {result[0].text}")
            print(
                f"240 chunks; warm median {median(times):.2f} ms; "
                f"min {min(times):.2f} ms; max {max(times):.2f} ms"
            )
        finally:
            database.close()


if __name__ == "__main__":
    main()
