"""Pluggable LLM backend for Xander.

Two backends share one interface so the rest of Xander never cares which is
live:

  * OllamaBackend — the local default, reusing the same battle-tested transport
    Monica runs on (streaming, keep-alive, think-stripping, model fallback).
  * OrnithBackend — the operator's preferred backend, spoken to as an
    OpenAI-compatible chat endpoint. If Ornith's real API differs, this is the
    single class to adjust.

`get_backend()` picks per config and transparently falls back to Ollama if the
preferred backend is unreachable, so Xander always has a brain.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

from config import (
    CONNECT_TIMEOUT,
    FALLBACK_MODELS,
    GEN_TIMEOUT,
    KEEP_ALIVE,
    OLLAMA_URL,
    ORNITH_CHAT_PATH,
    ORNITH_KEY,
    ORNITH_MODEL,
    ORNITH_URL,
    UNCENSORED,
    model_for,
)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _http_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class Backend:
    """Common interface. Subclasses implement `_generate` and `available`."""

    name = "backend"

    def __init__(self, log=None, uncensored: bool = True):
        self.log = log
        self.uncensored = uncensored
        self.last_stats: dict = {}

    def available(self) -> bool:  # pragma: no cover - trivial
        raise NotImplementedError

    def warm(self) -> None:
        pass

    def ask(
        self,
        prompt: str,
        *,
        system: str = "",
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        timeout: int | None = None,
    ) -> str:
        sys_msg = ""
        if self.uncensored:
            sys_msg += UNCENSORED
        if system:
            sys_msg += ("\n" + system if sys_msg else system)
        messages = []
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": prompt})
        return self._generate(
            messages, stream=stream, on_token=on_token,
            temperature=temperature, num_predict=num_predict, timeout=timeout,
        )

    def _generate(self, messages, **kw) -> str:  # pragma: no cover
        raise NotImplementedError


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, log=None, uncensored: bool = True, tier: str = "coder"):
        super().__init__(log, uncensored)
        self.default_model = model_for(tier)

    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{OLLAMA_URL}/api/version")
            with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT):
                return True
        except Exception:
            return False

    def warm(self) -> None:
        def _load():
            try:
                _http_json(
                    f"{OLLAMA_URL}/api/chat",
                    {"model": self.default_model, "messages": [], "keep_alive": KEEP_ALIVE},
                    {"Content-Type": "application/json"}, 120,
                )
            except Exception:
                pass
        threading.Thread(target=_load, daemon=True).start()

    def _generate(self, messages, *, stream=False, on_token=None,
                  temperature=None, num_predict=None, timeout=None) -> str:
        chain = [self.default_model] + [m for m in FALLBACK_MODELS if m != self.default_model]
        last_err = ""
        for idx, mdl in enumerate(chain):
            try:
                out = self._call(mdl, messages, stream, on_token, temperature, num_predict, timeout)
                if out.strip():
                    if idx and self.log:
                        self.log.warn("used fallback model", model=mdl)
                    return strip_think(out)
                last_err = "empty response"
            except Exception as exc:
                last_err = str(exc)
                if self.log:
                    self.log.warn("ollama call failed, trying fallback", model=mdl, err=last_err[:120])
        if self.log:
            self.log.error("all ollama models failed", err=last_err[:200])
        return ""

    def _call(self, model, messages, stream, on_token, temperature, num_predict, timeout) -> str:
        streaming = stream or on_token is not None
        payload: dict = {"model": model, "messages": messages,
                         "stream": streaming, "keep_alive": KEEP_ALIVE}
        opts = {}
        if temperature is not None:
            opts["temperature"] = temperature
        if num_predict is not None:
            opts["num_predict"] = num_predict
        if opts:
            payload["options"] = opts
        to = timeout or GEN_TIMEOUT
        started = time.monotonic()
        self.last_stats = {"backend": self.name, "model": model}
        if not streaming:
            body = _http_json(f"{OLLAMA_URL}/api/chat", payload,
                              {"Content-Type": "application/json"}, to)
            self._stats(body.get("eval_count"), body.get("eval_duration"), started)
            return (body.get("message") or {}).get("content", "").strip()
        chunks: list[str] = []
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=to) as resp:
            for raw in resp:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw.decode("utf-8"))
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    chunks.append(piece)
                    if on_token:
                        on_token(piece)
                    else:
                        print(piece, end="", flush=True)
                if obj.get("done"):
                    self._stats(obj.get("eval_count"), obj.get("eval_duration"), started)
                    break
        if on_token is None:
            print()
        return "".join(chunks).strip()

    def _stats(self, tokens, dur_ns, started):
        try:
            tokens = int(tokens or 0)
            dur_ns = int(dur_ns or 0)
            self.last_stats.update(
                tokens=tokens, seconds=round(time.monotonic() - started, 1),
                tps=round(tokens / (dur_ns / 1e9), 1) if dur_ns else 0.0,
            )
        except Exception:
            pass


class OrnithBackend(Backend):
    """Ornith as an OpenAI-compatible chat backend.

    Adjust ORNITH_* in config.py (or the env) to point at the real service.
    Streaming uses SSE (`data: {...}` lines), the OpenAI convention.
    """

    name = "ornith"

    def available(self) -> bool:
        # Probe a couple of common health endpoints; any 2xx means alive.
        for path in ("/v1/models", "/health", "/"):
            try:
                req = urllib.request.Request(f"{ORNITH_URL}{path}")
                if ORNITH_KEY:
                    req.add_header("Authorization", f"Bearer {ORNITH_KEY}")
                with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as r:
                    if 200 <= r.status < 300:
                        return True
            except Exception:
                continue
        return False

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if ORNITH_KEY:
            h["Authorization"] = f"Bearer {ORNITH_KEY}"
        return h

    def _generate(self, messages, *, stream=False, on_token=None,
                  temperature=None, num_predict=None, timeout=None) -> str:
        url = f"{ORNITH_URL}{ORNITH_CHAT_PATH}"
        streaming = stream or on_token is not None
        payload: dict = {"model": ORNITH_MODEL, "messages": messages, "stream": streaming}
        if temperature is not None:
            payload["temperature"] = temperature
        if num_predict is not None:
            payload["max_tokens"] = num_predict
        to = timeout or GEN_TIMEOUT
        started = time.monotonic()
        self.last_stats = {"backend": self.name, "model": ORNITH_MODEL}
        try:
            if not streaming:
                body = _http_json(url, payload, self._headers(), to)
                choice = (body.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content", "")
                usage = body.get("usage") or {}
                self._stats(usage.get("completion_tokens"), started)
                return strip_think(text.strip())
            # SSE streaming
            chunks: list[str] = []
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=self._headers())
            with urllib.request.urlopen(req, timeout=to) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    delta = ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content", "")
                    if delta:
                        chunks.append(delta)
                        if on_token:
                            on_token(delta)
                        else:
                            print(delta, end="", flush=True)
            if on_token is None:
                print()
            self._stats(None, started, count=len(" ".join(chunks).split()))
            return strip_think("".join(chunks).strip())
        except Exception as exc:
            if self.log:
                self.log.error("ornith call failed", err=str(exc)[:200])
            raise

    def _stats(self, tokens, started, count=None):
        try:
            t = int(tokens) if tokens else (count or 0)
            secs = round(time.monotonic() - started, 1)
            self.last_stats.update(tokens=t, seconds=secs,
                                   tps=round(t / secs, 1) if secs else 0.0)
        except Exception:
            pass


class ResilientBackend(Backend):
    """Primary backend with an automatic fallback to a secondary one."""

    def __init__(self, primary: Backend, fallback: Backend, log=None):
        super().__init__(log)
        self.primary = primary
        self.fallback = fallback
        self._active = primary
        self.name = primary.name

    def available(self) -> bool:
        return self.primary.available() or self.fallback.available()

    def warm(self) -> None:
        self._active.warm()

    def _generate(self, messages, **kw) -> str:
        try:
            out = self._active._generate(messages, **kw)
            if out.strip():
                self.last_stats = self._active.last_stats
                return out
        except Exception as exc:
            if self.log:
                self.log.warn(f"{self._active.name} failed, switching backend", err=str(exc)[:120])
        # switch to the other backend once and retry
        other = self.fallback if self._active is self.primary else self.primary
        if other.available():
            self._active = other
            self.name = other.name
            if self.log:
                self.log.warn("backend switched", to=other.name)
            out = other._generate(messages, **kw)
            self.last_stats = other.last_stats
            return out
        return ""


def get_backend(log=None, prefer: str | None = None, tier: str = "coder") -> Backend:
    """Build the configured backend with Ollama as the safety net."""
    from config import BACKEND

    choice = (prefer or BACKEND).lower()
    ollama = OllamaBackend(log=log, tier=tier)
    if choice == "ollama":
        return ollama
    if choice == "ornith":
        ornith = OrnithBackend(log=log)
        if ornith.available():
            return ResilientBackend(ornith, ollama, log=log)
        if log:
            log.warn("ornith unreachable, using local ollama", url=ORNITH_URL)
        return ollama
    # unknown backend name → local default
    if log:
        log.warn("unknown backend, using ollama", backend=choice)
    return ollama
