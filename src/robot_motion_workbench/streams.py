"""Bounded best-effort display consumption, independent of robot scheduling."""
from collections import deque
from dataclasses import dataclass
from threading import Condition, Thread


@dataclass(frozen=True)
class DisplayStats:
    accepted: int
    consumed: int
    dropped_full: int
    discarded_close: int
    rejected_closed: int
    errors: int
    pending: int
    closed: bool
    alive: bool
    last_error: str | None
    coalesced: int = 0


class VisualizationWorker:
    """FIFO, drop-newest when full; consumer work never holds the queue lock.

    publish never waits for capacity or rendering. A small mutex protects only
    queue bookkeeping; this is not a hard-real-time or lock-free guarantee.
    Drain on close by default; timeout returns False if a consumer is still
    running. Python cannot kill a hung callback. Re-close after it returns.
    Consumer exceptions are counted and consumption continues. close_callback
    runs once on the display thread (e.g. renderer flush/disconnect).
    With batch=True the consumer receives all pending events as one tuple;
    capacity and statistics still count individual events, not tuples.
    """
    def __init__(self, consume, *, capacity=64, close_callback=None, coalesce_feedback=False, batch=False):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self._consume, self._finish = consume, close_callback
        self._capacity = capacity
        self._queue = deque()
        self._condition = Condition()
        self._closed = False
        self._accepted = self._consumed = self._dropped = self._rejected = self._errors = 0
        self._last_error = None
        self._discarded = 0
        self._coalesce_feedback = coalesce_feedback
        self._batch = batch
        self._coalesced = 0
        self._thread = Thread(target=self._run, name='rmw-visualization', daemon=True)
        self._thread.start()

    def publish(self, event):
        with self._condition:
            if self._closed:
                self._rejected += 1
                return False
            if (self._coalesce_feedback and getattr(event, 'stage', None) in ('feedback', 'planned')
                    and event.available):
                # Remove older pending display samples, then append the latest
                # at its proper chronological position. Never coalesce control
                # lifecycle, sent evidence, absence or provenance transitions.
                remove = set()
                for index in range(len(self._queue) - 1, -1, -1):
                    prior = self._queue[index]
                    if (prior.stage != event.stage or
                            getattr(prior, 'group', None) != getattr(event, 'group', None)):
                        continue
                    if not prior.available or prior.origin != event.origin:
                        break  # Preserve every availability/origin boundary.
                    remove.add(index)
                if event.stage == 'feedback' and remove:
                    # Let the oldest pending pose reach the consumer. Moving
                    # every pose to the tail can starve animation indefinitely
                    # while fresh submission events keep arriving.
                    remove.discard(min(remove))
                retained = deque(e for index, e in enumerate(self._queue) if index not in remove)
                self._coalesced += len(self._queue) - len(retained)
                self._queue = retained
            if len(self._queue) >= self._capacity:
                self._dropped += 1
                return False
            self._queue.append(event)
            self._accepted += 1
            self._condition.notify()
            return True

    def publish_many(self, events):
        """Enqueue one display frame atomically; capacity/stats count events.

        Reject the whole frame when full, so a pose cannot be separated from
        its associated reference changes. This path preserves every event.
        """
        events = tuple(events)
        with self._condition:
            if self._closed:
                self._rejected += len(events)
                return False
            if len(self._queue) + len(events) > self._capacity:
                self._dropped += len(events)
                return False
            self._queue.extend(events)
            self._accepted += len(events)
            self._condition.notify()
            return True

    @property
    def stats(self):
        with self._condition:
            return DisplayStats(self._accepted, self._consumed, self._dropped, self._discarded, self._rejected,
                                self._errors, len(self._queue), self._closed,
                                self._thread.is_alive(), self._last_error, self._coalesced)

    def _error(self, exc):
        with self._condition:
            self._errors += 1
            self._last_error = f'{type(exc).__name__}: {exc}'

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._queue or self._closed)
                    if not self._queue:
                        break
                    if self._batch:
                        event = tuple(self._queue)
                        self._queue.clear()
                        count = len(event)
                    else:
                        event = self._queue.popleft()
                        count = 1
                try:
                    self._consume(event)
                except Exception as exc:
                    self._error(exc)
                finally:
                    with self._condition:
                        self._consumed += count
        finally:
            if self._finish:
                try:
                    self._finish()
                except Exception as exc:
                    self._error(exc)

    def close(self, *, drain=True, timeout=5.0):
        with self._condition:
            self._closed = True
            if not drain:
                self._discarded += len(self._queue)
                self._queue.clear()
            self._condition.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
