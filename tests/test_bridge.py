"""The addon bridge: loopback only, token-gated, plan-only unless told."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from xander_agent.bridge import BridgeService, make_server


class RecordingInvoke:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, mode: str, **kwargs) -> dict:
        self.calls.append((mode, kwargs))
        return {
            "ok": True,
            "status": "completed",
            "task_id": "t-bridge",
            "task": {
                "plan": {"summary": "a small plan"},
                "evidence": [{"kind": "answer", "text": "flexbox for one dimension, grid for two"}],
            },
            "handoff": {},
        }


class FakeMemory:
    def __init__(self) -> None:
        self.preferences: list[tuple[str, str]] = []

    def add_preference(self, text: str, *, kind: str, source: str) -> bool:
        self.preferences.append((text, kind))
        return True


def _service(tmp_path: Path, invoke: RecordingInvoke, memory: FakeMemory | None = None) -> BridgeService:
    return BridgeService(tmp_path, variant="default", invoke=invoke, memory=memory or FakeMemory())


# -- service logic --------------------------------------------------------------
def test_order_is_plan_only_unless_apply_is_requested(tmp_path: Path) -> None:
    invoke = RecordingInvoke()
    service = _service(tmp_path, invoke)

    service.order("make the background green")
    mode, kwargs = invoke.calls[-1]
    assert mode == "plan"
    assert kwargs["autonomy"] == "proposal-only"

    service.order("make the background green", apply_changes=True)
    mode, kwargs = invoke.calls[-1]
    assert mode == "implement"
    assert kwargs["autonomy"] is None


def test_ask_extracts_the_answer_evidence(tmp_path: Path) -> None:
    service = _service(tmp_path, RecordingInvoke())
    result = service.ask("flexbox or grid?")
    assert result["answer"] == "flexbox for one dimension, grid for two"


def test_note_records_a_preference(tmp_path: Path) -> None:
    memory = FakeMemory()
    service = _service(tmp_path, RecordingInvoke(), memory)
    service.note("keep summaries short", kind="like")
    assert memory.preferences == [("keep summaries short", "like")]


def test_empty_payloads_are_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path, RecordingInvoke())
    for call in (lambda: service.ask("  "), lambda: service.note(""), lambda: service.order("")):
        with pytest.raises(ValueError):
            call()


# -- HTTP surface ---------------------------------------------------------------
def test_the_bridge_refuses_to_bind_a_routable_address(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        make_server(_service(tmp_path, RecordingInvoke()), token="t", host="0.0.0.0")


def _request(port: int, path: str, token: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_surface_gates_on_the_token_and_serves_json(tmp_path: Path) -> None:
    import threading

    invoke = RecordingInvoke()
    server = make_server(_service(tmp_path, invoke), token="secret-token", port=0, origin="moz-extension://abc")
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _request(port, "/health", "wrong-token")
        assert status == 401 and body["ok"] is False

        status, body = _request(port, "/health", "secret-token")
        assert status == 200 and body["ok"] is True
        assert body["workspace"] == str(tmp_path.resolve())

        status, body = _request(port, "/order", "secret-token", {"goal": "tidy the css"})
        assert status == 200
        assert body["applied"] is False and body["summary"] == "a small plan"

        status, body = _request(port, "/ask", "secret-token", {"question": ""})
        assert status == 400, "an empty question is a client error, not a crash"

        status, body = _request(port, "/nope", "secret-token")
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()
