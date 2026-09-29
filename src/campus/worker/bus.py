"""Transport between camera workers and the identity plane.

Redis Streams, not pub/sub. Streams give a consumer group with acknowledgements
and replay, which means a restarted identity service can reprocess events it
missed instead of losing attendance records. Pub/sub is fire-and-forget and
would silently drop commits during any deploy.

Backpressure matters more than throughput here. If the identity service stalls,
the stream grows and memory is the limit — so streams are capped with
``MAXLEN``, and the cap is set high enough to absorb a normal deploy but low
enough that a multi-day outage cannot exhaust the host.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator
from typing import Any

log = logging.getLogger("campus.bus")

OBSERVATION_STREAM = "campus:observations"
PRESENCE_STREAM = "campus:presence"
DEAD_LETTER_STREAM = "campus:deadletter"

# ~1.5 days at 400 cameras. Enough to survive a weekend deploy; small enough
# that a stuck consumer cannot exhaust the host's memory.
STREAM_MAXLEN = 400_000


class ObservationBus:
    """Publishes identity observations to Redis.

    Falls back to an in-memory list when Redis is unreachable, and says so
    loudly. Silently dropping commits would produce a system that looks healthy
    and under-reports attendance — the worst possible failure mode for this
    application, because nothing errors and the numbers are just wrong.
    """

    def __init__(
        self,
        redis_url: str,
        worker_id: str,
        maxlen: int = STREAM_MAXLEN,
        batch_size: int = 200,
        flush_interval_s: float = 1.0,
    ) -> None:
        self.redis_url = redis_url
        self.worker_id = worker_id
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self._client: Any = None
        self._buffer: list[dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self._degraded = False
        self._dropped = 0

    def connect(self) -> bool:
        try:
            import redis  # noqa: PLC0415

            client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_keepalive=True,
                health_check_interval=30,
            )
            client.ping()
            self._client = client
            self._degraded = False
            log.info("observation bus connected: %s", self.redis_url)
            return True
        except Exception as exc:  # noqa: BLE001
            self._client = None
            self._degraded = True
            log.error("observation bus unavailable (%s); buffering in memory", exc)
            return False

    def publish_observation(self, event: dict[str, Any]) -> None:
        self._buffer.append(event)
        if len(self._buffer) >= self.batch_size:
            self.flush()
            return
        now = time.monotonic()
        if now - self._last_flush >= self.flush_interval_s:
            self.flush()

    def flush(self) -> int:
        if not self._buffer:
            self._last_flush = time.monotonic()
            return 0
        pending, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()

        if self._client is None:
            self._dropped += len(pending)
            log.error("dropping %d observations: bus disconnected", len(pending))
            return 0

        try:
            self._client.xadd(
                OBSERVATION_STREAM,
                {"data": json.dumps(pending)},
                maxlen=self.maxlen,
                approximate=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._dropped += len(pending)
            log.error("failed to publish %d observations: %s", len(pending), exc)
            return 0
        return len(pending)

    def read_observations(
        self, group: str, consumer: str, count: int = 100, block_ms: int = 1000
    ) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        """Read and ack a batch. Yields ``(message_id, events)``."""
        if self._client is None:
            if not self.connect():
                return
        try:
            self._client.xgroup_create(OBSERVATION_STREAM, group, id="0", mkstream=True)
        except Exception:  # noqa: BLE001 - group already exists
            pass

        try:
            messages = self._client.xreadgroup(
                group, consumer, {OBSERVATION_STREAM: ">"}, count=count, block=block_ms
            )
        except Exception as exc:  # noqa: BLE001
            log.error("observation read failed: %s", exc)
            return

        for _stream, entries in messages or []:
            for msg_id, payload in entries:
                try:
                    events = json.loads(payload.get("data", "[]"))
                except json.JSONDecodeError:
                    self._client.xadd(
                        DEAD_LETTER_STREAM, {"raw": payload.get("data", ""), "id": msg_id}
                    )
                    continue
                yield msg_id, events

    def ack(self, message_id: str) -> None:
        if self._client is None:
            return
        try:
            self._client.xack(OBSERVATION_STREAM, self._group, message_id)
        except Exception:  # noqa: BLE001
            log.debug("ack failed for %s", message_id)

    _group: str = "identity"

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def degraded(self) -> bool:
        return self._degraded

    def close(self) -> None:
        self.flush()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


class InMemoryBus:
    """Drop-in replacement for tests and single-camera development."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish_observation(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def flush(self) -> int:
        return len(self.events)

    def read_observations(
        self, group: str, consumer: str, count: int = 100, block_ms: int = 1000
    ) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        batch, self.events = self.events[:count], self.events[count:]
        if batch:
            yield "1", batch

    def ack(self, message_id: str) -> None:
        return None

    @property
    def dropped(self) -> int:
        return 0

    @property
    def degraded(self) -> bool:
        return False

    def close(self) -> None:
        return None


def encode_events(events: Iterable[dict[str, Any]]) -> str:
    return json.dumps(list(events))
