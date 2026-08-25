"""A part of Xander that lives outside the terminal.

``xander serve`` opens a small loopback HTTP surface so another program —
a Firefox addon, a scratch script, a hotkey — can reach the same engine
the TUI drives. Stdlib only, no new dependencies, and off unless asked
for.

Safety, because this is a door into a coding agent:

- binds ``127.0.0.1`` only, never a routable address;
- every request needs the bearer token from ``config_dir()/bridge-token``
  (generated 0600 on first serve, printed once at startup);
- ``/ask`` and ``/stats`` are read-only; ``/order`` runs the engine and is
  **plan-only by default** — applying changes takes ``"apply": true``;
- CORS is granted to the single origin passed with ``--origin`` (a
  ``moz-extension://…`` id), not ``*``.

Endpoints (all JSON):

    GET  /health   -> {ok, workspace, variant}
    GET  /stats    -> the scoreboard payload
    POST /ask      {"question": "..."}            -> {answer}
    POST /note     {"text": "...", "kind": "..."} -> {recorded}
    POST /order    {"goal": "...", "apply": bool} -> {ok, status, task_id, summary}
"""

from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_MAX_BODY = 64 * 1024


def token_path() -> Path:
    from .paths import config_dir

    return config_dir() / "bridge-token"


def ensure_token() -> str:
    """Read the bridge token, minting a private one on first use."""

    path = token_path()
    if path.exists():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    path.chmod(0o600)
    return token


class BridgeService:
    """Engine-facing logic, kept free of HTTP so it can be tested directly."""

    def __init__(
        self,
        workspace: Path,
        *,
        variant: str = "default",
        invoke: Any = None,
        memory: Any = None,
    ) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.variant = variant
        self._invoke = invoke
        self._memory = memory

    def _invoker(self) -> Any:
        if self._invoke is None:
            from .cli import invoke_engine

            self._invoke = invoke_engine
        return self._invoke

    def _memory_store(self) -> Any:
        if self._memory is None:
            from .memory import MemoryStore

            self._memory = MemoryStore(namespace=self.variant)
        return self._memory

    def health(self) -> dict[str, Any]:
        return {"ok": True, "workspace": str(self.workspace), "variant": self.variant}

    def stats(self) -> dict[str, Any]:
        from .stats import stats_payload

        return stats_payload()

    def ask(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("ask needs a question")
        result = self._invoker()(
            "answer",
            workspace=self.workspace,
            goal=question,
            variant=self.variant,
            autonomy="proposal-only",
        )
        answer = ""
        task = result.get("task") if isinstance(result.get("task"), dict) else {}
        for item in reversed(task.get("evidence") or []):
            if isinstance(item, dict) and item.get("kind") == "answer":
                answer = str(item.get("text", ""))
                break
        return {"ok": bool(result.get("ok")), "answer": answer, "task_id": result.get("task_id", "")}

    def note(self, text: str, kind: str = "style") -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("note needs text")
        recorded = self._memory_store().add_preference(text, kind=kind, source="explicit")
        return {"ok": True, "recorded": recorded, "text": text, "kind": kind}

    def order(self, goal: str, apply_changes: bool = False) -> dict[str, Any]:
        goal = goal.strip()
        if not goal:
            raise ValueError("order needs a goal")
        result = self._invoker()(
            "implement" if apply_changes else "plan",
            workspace=self.workspace,
            goal=goal,
            variant=self.variant,
            autonomy=None if apply_changes else "proposal-only",
        )
        task = result.get("task") if isinstance(result.get("task"), dict) else {}
        plan = task.get("plan") if isinstance(task.get("plan"), dict) else {}
        return {
            "ok": bool(result.get("ok")),
            "status": str(result.get("status", "")),
            "task_id": str(result.get("task_id", "")),
            "applied": apply_changes,
            "summary": str(plan.get("summary", "")),
        }


def build_handler(service: BridgeService, token: str, origin: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "xander-bridge/1.0"

        # -- plumbing ---------------------------------------------------------
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # the narrator owns Xander's voice; keep stderr quiet

        def _cors(self) -> None:
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            header = self.headers.get("Authorization", "")
            presented = header[7:].strip() if header.lower().startswith("bearer ") else ""
            return bool(presented) and secrets.compare_digest(presented, token)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > _MAX_BODY:
                raise ValueError("request body too large")
            raw = self.rfile.read(length)
            parsed = json.loads(raw.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else {}

        # -- routes -----------------------------------------------------------
        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                self._reply(401, {"ok": False, "error": "bridge token required"})
                return
            try:
                if self.path == "/health":
                    self._reply(200, service.health())
                elif self.path == "/stats":
                    self._reply(200, service.stats())
                else:
                    self._reply(404, {"ok": False, "error": "unknown endpoint"})
            except Exception as exc:
                self._reply(500, {"ok": False, "error": str(exc)[:500]})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                self._reply(401, {"ok": False, "error": "bridge token required"})
                return
            try:
                payload = self._body()
                if self.path == "/ask":
                    self._reply(200, service.ask(str(payload.get("question", ""))))
                elif self.path == "/note":
                    self._reply(
                        200,
                        service.note(str(payload.get("text", "")), str(payload.get("kind", "style"))),
                    )
                elif self.path == "/order":
                    self._reply(
                        200,
                        service.order(str(payload.get("goal", "")), bool(payload.get("apply", False))),
                    )
                else:
                    self._reply(404, {"ok": False, "error": "unknown endpoint"})
            except ValueError as exc:
                self._reply(400, {"ok": False, "error": str(exc)[:500]})
            except Exception as exc:
                self._reply(500, {"ok": False, "error": str(exc)[:500]})

    return Handler


def make_server(
    service: BridgeService,
    *,
    token: str,
    host: str = "127.0.0.1",
    port: int = 8787,
    origin: str = "",
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the bridge only binds loopback; Xander is not a public service")
    server = ThreadingHTTPServer((host, port), build_handler(service, token, origin))
    server.daemon_threads = True
    return server


def serve(
    workspace: Path,
    *,
    variant: str = "default",
    host: str = "127.0.0.1",
    port: int = 8787,
    origin: str = "",
) -> None:
    token = ensure_token()
    service = BridgeService(workspace, variant=variant)
    server = make_server(service, token=token, host=host, port=port, origin=origin)
    print(f"xander bridge listening on http://{host}:{port}")
    print(f"token: {token}")
    print(f"token file: {token_path()}")
    if origin:
        print(f"allowed origin: {origin}")
    else:
        print("no --origin given: browser requests will be blocked by CORS")
    thread = threading.Thread(target=server.serve_forever, name="xander-bridge", daemon=True)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
