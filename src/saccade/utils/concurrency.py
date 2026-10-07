"""Thread-count defaults and bounded producer/consumer helpers.

Saccade deliberately uses very little concurrency: one background thread that decodes
audio and runs VAD, feeding a small bounded queue consumed by a single ASR worker whose
heavy lifting happens inside CTranslate2's own thread pool.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")

_MAX_DEFAULT_THREADS = 8


def physical_cores() -> int:
    """Best-effort physical core count (falls back to logical / 2 when SMT is likely)."""
    try:
        import psutil

        count = psutil.cpu_count(logical=False)
        if count:
            return int(count)
    except ImportError:
        pass
    logical = os.cpu_count() or 1
    return max(1, logical // 2) if logical >= 4 else logical


def default_threads() -> int:
    """Inference threads used when the caller does not specify ``threads``.

    Whisper on CTranslate2 scales with physical cores but gains little beyond ~8, and
    saturating every core makes the machine unresponsive, so the default is capped.
    """
    return max(1, min(_MAX_DEFAULT_THREADS, physical_cores()))


class _Done:
    pass


class _Failed:
    def __init__(self, error: BaseException) -> None:
        self.error = error


_DONE = _Done()


class BoundedProducer(Generic[T]):
    """Run an iterator in a background thread, handing items over through a bounded queue.

    The queue bound provides back-pressure: the producer never runs more than ``maxsize``
    items ahead of the consumer, so memory stays bounded on arbitrarily long inputs.
    Exceptions raised by the producer are re-raised in the consumer. Closing the producer
    (or abandoning iteration) stops the background thread promptly.
    """

    def __init__(
        self,
        factory: Callable[[], Iterator[T]],
        *,
        maxsize: int = 2,
        name: str = "saccade-producer",
    ) -> None:
        self._factory = factory
        self._queue: queue.Queue[T | _Done | _Failed] = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False

    def _put(self, item: T | _Done | _Failed) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        iterator = self._factory()
        try:
            for item in iterator:
                if not self._put(item):
                    return
            self._put(_DONE)
        except BaseException as exc:  # forwarded to the consumer
            self._put(_Failed(exc))
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def __iter__(self) -> Iterator[T]:
        if not self._started:
            self._started = True
            self._thread.start()
        while True:
            item = self._queue.get()
            if isinstance(item, _Done):
                return
            if isinstance(item, _Failed):
                raise item.error
            yield item

    def close(self) -> None:
        self._stop.set()
        # Drain so a producer blocked on put() can observe the stop flag.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._started:
            self._thread.join(timeout=30)


async def iterate_in_thread(
    factory: Callable[[], Iterator[T]], *, maxsize: int = 8
) -> AsyncIterator[T]:
    """Consume a blocking iterator from asyncio without blocking the event loop.

    The iterator runs in a worker thread; items cross over through a bounded queue.
    Cancelling the async iteration stops the worker at the next item boundary.
    """
    loop = asyncio.get_running_loop()
    items: asyncio.Queue[T | _Done | _Failed] = asyncio.Queue(maxsize=maxsize)
    stop = threading.Event()

    def deliver(item: T | _Done | _Failed) -> None:
        future = asyncio.run_coroutine_threadsafe(items.put(item), loop)
        while not stop.is_set():
            try:
                future.result(timeout=0.1)
                return
            except TimeoutError:
                continue
        future.cancel()

    def worker() -> None:
        iterator = factory()
        try:
            for item in iterator:
                if stop.is_set():
                    break
                deliver(item)
            else:
                deliver(_DONE)
        except BaseException as exc:
            deliver(_Failed(exc))
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    task = loop.run_in_executor(None, worker)
    try:
        while True:
            item = await items.get()
            if isinstance(item, _Done):
                break
            if isinstance(item, _Failed):
                raise item.error
            yield item
    finally:
        stop.set()
        while not items.empty():
            items.get_nowait()
        await asyncio.shield(task)
