import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from rockfall.sinks import JsonlSink, MqttSink, StdoutSink, WebhookSink, verify_audit_log

REC = {"t": 12.5, "site": "demo-gully", "state": "RED", "previous": "YELLOW", "light": "red", "road_closed": True,
       "reason": "event 1: rockfall", "event": None}


def test_stdout_line() -> None:
    buf = io.StringIO()
    StdoutSink(buf).emit(REC)
    out = buf.getvalue()
    assert "YELLOW" in out and "RED" in out and "road_closed=True" in out


def test_jsonl_hash_chain_and_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlSink(path)
    for i in range(3):
        sink.emit({**REC, "t": float(i)})
    assert verify_audit_log(path)
    lines = path.read_text().splitlines()
    assert json.loads(lines[1])["prev"] == json.loads(lines[0])["hash"]
    tampered = json.loads(lines[1])
    tampered["road_closed"] = False
    path.write_text("\n".join([lines[0], json.dumps(tampered), lines[2]]) + "\n")
    assert not verify_audit_log(path)


def test_jsonl_continues_existing_chain(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    JsonlSink(path).emit(REC)
    JsonlSink(path).emit(REC)
    assert verify_audit_log(path) and len(path.read_text().splitlines()) == 2


def test_deleted_line_detected(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    s = JsonlSink(path)
    for _ in range(3):
        s.emit(REC)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert not verify_audit_log(path)


class FakeClient:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, int, bool]] = []
        self.connected: tuple[str, int] | None = None

    def connect(self, host: str, port: int) -> None:
        self.connected = (host, port)

    def loop_start(self) -> None: ...

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> None:
        self.published.append((topic, payload, qos, retain))


def test_mqtt_topic_and_payload() -> None:
    fake = FakeClient()
    sink = MqttSink("broker", 1884, site="demo-gully", client_factory=lambda: fake)
    sink.emit(REC)
    assert fake.connected == ("broker", 1884)
    topic, payload, qos, retain = fake.published[0]
    assert topic == "hazard/demo-gully/alarm" and json.loads(payload)["state"] == "RED" and qos == 1 and retain


def test_mqtt_failure_does_not_raise() -> None:
    class Broken(FakeClient):
        def publish(self, *a: Any, **k: Any) -> None:
            raise OSError("broker down")

    MqttSink("x", client_factory=Broken).emit(REC)


def test_webhook_posts_json_and_survives_dead_endpoint() -> None:
    received: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a: Any) -> None: ...

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/hook"
    WebhookSink(url).emit(REC)
    srv.server_close()
    assert received and received[0]["light"] == "red"
    WebhookSink(url, timeout_s=0.5).emit(REC)  # nothing listening any more: must not raise
