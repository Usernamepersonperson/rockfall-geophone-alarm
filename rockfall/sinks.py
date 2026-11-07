"""Alarm sinks.  A sink failing must never break the alarm chain: errors are logged, not raised.

* ``StdoutSink``   human-readable line.
* ``JsonlSink``    append-only audit log; each line carries ``seq``, ``prev`` and ``hash``
                   (SHA-256 over the previous hash and the record) so edits are detectable.
* ``MqttSink``     publishes JSON to ``hazard/<site>/alarm`` (retained, QoS 1); needs ``paho-mqtt``.
* ``WebhookSink``  HTTP POST of the JSON record.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("rockfall.sinks")


class Sink(Protocol):
    def emit(self, record: dict[str, Any]) -> None: ...


class StdoutSink:
    def __init__(self, stream: Any = None) -> None:
        self.stream = stream or sys.stdout

    def emit(self, record: dict[str, Any]) -> None:
        print(
            f"[{record['t']:8.1f}s] {record['previous']:>8} -> {record['state']:<8} "
            f"light={record['light']:<15} road_closed={record['road_closed']}  {record['reason']}",
            file=self.stream,
        )


def _digest(prev: str, record: dict[str, Any]) -> str:
    body = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev + body).encode()).hexdigest()


class JsonlSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq, self._prev = 0, "0" * 64
        if self.path.exists():  # continue the chain of an existing log
            lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                last = json.loads(lines[-1])
                self._seq, self._prev = last["seq"], last["hash"]

    def emit(self, record: dict[str, Any]) -> None:
        self._seq += 1
        entry = {"seq": self._seq, "prev": self._prev, **record}
        entry["hash"] = _digest(self._prev, entry)
        self._prev = entry["hash"]
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")


def verify_audit_log(path: str | Path) -> bool:
    """Recompute the hash chain; False if any line was edited, removed or reordered."""
    prev = "0" * 64
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        entry = json.loads(line)
        claimed = entry.pop("hash")
        if entry["prev"] != prev or entry["seq"] != n or _digest(prev, entry) != claimed:
            return False
        prev = claimed
    return True


class MqttSink:
    """``client_factory`` is injectable for tests; by default paho-mqtt is imported lazily."""

    def __init__(self, host: str, port: int = 1883, site: str = "demo-gully",
                 client_factory: Callable[[], Any] | None = None) -> None:
        self.topic = f"hazard/{site}/alarm"
        if client_factory is None:
            import paho.mqtt.client as mqtt  # optional dependency

            def client_factory() -> Any:
                version = getattr(mqtt, "CallbackAPIVersion", None)
                return mqtt.Client(version.VERSION2) if version else mqtt.Client()

        self.client = client_factory()
        self.client.connect(host, port)
        self.client.loop_start()

    def emit(self, record: dict[str, Any]) -> None:
        try:
            self.client.publish(self.topic, json.dumps(record), qos=1, retain=True)
        except Exception:  # noqa: BLE001 - never break the alarm chain
            log.exception("MQTT publish failed")


class WebhookSink:
    def __init__(self, url: str, timeout_s: float = 3.0) -> None:
        self.url, self.timeout_s = url, timeout_s

    def emit(self, record: dict[str, Any]) -> None:
        req = urllib.request.Request(
            self.url, data=json.dumps(record).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout_s).close()  # noqa: S310 - user supplied URL
        except Exception:  # noqa: BLE001
            log.exception("webhook POST failed")
