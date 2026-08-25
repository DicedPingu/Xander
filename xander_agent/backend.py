from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import BaseModel

from .policy import neutral_intent_contract


DEFAULT_MODELS = {
    "coder": "huihui_ai/qwen2.5-coder-abliterate:7b",
    "planner": "huihui_ai/qwen3-abliterated:8b",
    "classifier": "huihui_ai/qwen2.5-vl-abliterated:3b",
    "critic": "huihui_ai/qwen3-abliterated:8b",
}

_GENERATION_LOCK = threading.Lock()


class Backend(Protocol):
    def available(self) -> bool: ...

    def generate(
        self,
        prompt: str,
        *,
        role: str = "coder",
        system: str = "",
        schema: type[BaseModel] | dict[str, Any] | None = None,
        think: bool | str | None = None,
        on_token: Callable[[str], None] | None = None,
        timeout: int | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class ModelResponse:
    content: str
    thinking: str
    model: str
    stats: dict[str, Any]


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        models: dict[str, str] | None = None,
        keep_alive: str = "45s",
        connect_timeout: int = 3,
        min_available_memory_mb: int = 1800,
        max_temperature_c: int = 88,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")).rstrip("/")
        self.models = {**DEFAULT_MODELS, **(models or {})}
        self.keep_alive = keep_alive
        self.connect_timeout = connect_timeout
        self.min_available_memory_mb = min_available_memory_mb
        self.max_temperature_c = max_temperature_c
        self.last_stats: dict[str, Any] = {}
        self.last_thinking = ""
        self._resident_model: str | None = None

    def available(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/api/version")
            with urllib.request.urlopen(request, timeout=self.connect_timeout) as response:
                return response.status == 200
        except Exception:
            return False

    def installed_models(self) -> list[str]:
        try:
            request = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(request, timeout=self.connect_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return sorted(item.get("name", "") for item in payload.get("models", []) if item.get("name"))
        except Exception:
            return []

    def doctor(self) -> dict[str, Any]:
        installed = self.installed_models()
        return {
            "available": bool(installed) or self.available(),
            "base_url": self.base_url,
            "installed_models": installed,
            "routing": self.models,
            "missing_routed_models": sorted({model for model in self.models.values() if model not in installed}),
            "resource_policy": {
                "parallel_generations": 1,
                "keep_alive": self.keep_alive,
                "context_tokens": 4096,
                "minimum_available_memory_mb": self.min_available_memory_mb,
                "maximum_temperature_c": self.max_temperature_c,
            },
            "resources": self.resource_status(),
        }

    def resource_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {"memory_available_mb": self._memory_available_mb()}
        temperatures = self._temperatures()
        if temperatures:
            status["temperatures_c"] = temperatures
            status["maximum_temperature_c"] = max(temperatures)
        try:
            run = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if run.returncode == 0 and run.stdout.strip():
                values = [int(float(value.strip())) for value in run.stdout.splitlines()[0].split(",")]
                status["nvidia"] = {
                    "temperature_c": values[0],
                    "memory_used_mb": values[1],
                    "memory_total_mb": values[2],
                    "utilization_percent": values[3],
                }
        except Exception:
            pass
        return status

    @staticmethod
    def _memory_available_mb() -> int:
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
        except Exception:
            pass
        return 0

    @staticmethod
    def _temperatures() -> list[int]:
        temperatures: list[int] = []
        for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                raw = int(path.read_text(encoding="utf-8").strip())
                value = raw // 1000 if raw > 1000 else raw
                if 0 < value < 150:
                    temperatures.append(value)
            except Exception:
                continue
        return temperatures

    def _resource_gate(self) -> None:
        memory = self._memory_available_mb()
        if memory and memory < self.min_available_memory_mb:
            self.unload()
            raise RuntimeError(f"resource gate: only {memory} MiB memory available")
        temperatures = self._temperatures()
        if temperatures and max(temperatures) >= self.max_temperature_c:
            self.unload()
            raise RuntimeError(f"resource gate: temperature reached {max(temperatures)} C")

    def unload(self, model: str | None = None) -> None:
        target = model or self._resident_model
        if not target:
            return
        try:
            payload = json.dumps({"model": target, "keep_alive": 0}).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30):
                pass
        except Exception:
            pass
        if target == self._resident_model:
            self._resident_model = None

    def _prepare_model(self, model: str) -> None:
        self._resource_gate()
        if self._resident_model and self._resident_model != model:
            self.unload(self._resident_model)
        self._resident_model = model

    def _models_for(self, role: str) -> list[str]:
        preferred = self.models.get(role, self.models["coder"])
        ordered = [preferred]
        for fallback_role in ("coder", "planner", "critic"):
            model = self.models[fallback_role]
            if model not in ordered:
                ordered.append(model)
        installed = set(self.installed_models())
        present = [model for model in ordered if model in installed]
        return present or ordered

    def generate(
        self,
        prompt: str,
        *,
        role: str = "coder",
        system: str = "",
        schema: type[BaseModel] | dict[str, Any] | None = None,
        think: bool | str | None = None,
        on_token: Callable[[str], None] | None = None,
        timeout: int | None = None,
    ) -> str:
        schema_value = schema.model_json_schema() if isinstance(schema, type) and issubclass(schema, BaseModel) else schema
        if think is None:
            think = role in {"planner", "critic"} and schema is None
        system_message = neutral_intent_contract()
        if system.strip():
            system_message += "\n" + system.strip()
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        errors: list[str] = []
        with _GENERATION_LOCK:
            for model in self._models_for(role):
                try:
                    self._prepare_model(model)
                    response = self._call(
                        model,
                        messages,
                        schema=schema_value,
                        think=think,
                        on_token=on_token,
                        timeout=timeout or 900,
                    )
                    if not response.content and response.thinking and think:
                        response = self._call(
                            model,
                            messages,
                            schema=schema_value,
                            think=False,
                            on_token=on_token,
                            timeout=timeout or 900,
                        )
                    if response.content.strip():
                        self.last_stats = response.stats
                        self.last_thinking = response.thinking
                        return response.content.strip()
                    errors.append(f"{model}: empty final content")
                except Exception as exc:
                    errors.append(f"{model}: {exc}")
        raise RuntimeError("all local models failed: " + "; ".join(errors))

    def generate_model(
        self,
        prompt: str,
        model_type: type[BaseModel],
        *,
        role: str,
        system: str = "",
        timeout: int | None = None,
    ) -> BaseModel:
        raw = self.generate(
            prompt,
            role=role,
            system=system,
            schema=model_type,
            think=False,
            timeout=timeout,
        )
        return model_type.model_validate_json(raw)

    def ask(self, prompt: str, **kwargs: Any) -> str:
        role = kwargs.pop("role", "coder")
        kwargs.pop("temperature", None)
        kwargs.pop("num_predict", None)
        return self.generate(prompt, role=role, **kwargs)

    def _call(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None,
        think: bool | str,
        on_token: Callable[[str], None] | None,
        timeout: int,
    ) -> ModelResponse:
        streaming = on_token is not None
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": streaming,
            "keep_alive": self.keep_alive,
            "think": think,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
                "num_batch": 128,
                "num_predict": 2048 if schema is not None else 3072,
            },
        }
        if schema is not None:
            payload["format"] = schema
        started = time.monotonic()
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not streaming:
                body = json.loads(response.read().decode("utf-8"))
                message = body.get("message") or {}
                return ModelResponse(
                    content=message.get("content", ""),
                    thinking=message.get("thinking", ""),
                    model=model,
                    stats=self._stats(body, model, started),
                )
            content: list[str] = []
            thinking: list[str] = []
            final: dict[str, Any] = {}
            for line in response:
                if not line.strip():
                    continue
                body = json.loads(line.decode("utf-8"))
                message = body.get("message") or {}
                if message.get("thinking"):
                    thinking.append(message["thinking"])
                if message.get("content"):
                    content.append(message["content"])
                    if on_token:
                        on_token(message["content"])
                if body.get("done"):
                    final = body
            return ModelResponse(
                content="".join(content),
                thinking="".join(thinking),
                model=model,
                stats=self._stats(final, model, started),
            )

    @staticmethod
    def _stats(body: dict[str, Any], model: str, started: float) -> dict[str, Any]:
        tokens = int(body.get("eval_count") or 0)
        duration = int(body.get("eval_duration") or 0)
        return {
            "backend": "ollama",
            "model": model,
            "tokens": tokens,
            "seconds": round(time.monotonic() - started, 3),
            "tokens_per_second": round(tokens / (duration / 1e9), 2) if duration else 0.0,
            "done_reason": body.get("done_reason", ""),
        }


class UnavailableBackend:
    name = "unavailable"

    def available(self) -> bool:
        return False

    def doctor(self) -> dict[str, Any]:
        return {"available": False, "reason": "no local backend configured"}

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("no local LLM backend is reachable")


CLOUD_PREFIX = "anthropic/"


class AnthropicBackend:
    """Optional Claude API brain behind the same generate() surface.

    Routing values look like ``anthropic/claude-opus-5``; the prefix is
    stripped before the call. The ``anthropic`` package is an optional
    extra (``xander-agent[cloud]``) — without it, or without credentials,
    this backend simply reports unavailable and Xander stays local.
    """

    name = "anthropic"

    def __init__(self, *, models: dict[str, str] | None = None, default_model: str = "claude-opus-5") -> None:
        self.models = {role: model.removeprefix(CLOUD_PREFIX) for role, model in (models or {}).items()}
        self.default_model = default_model
        self.last_stats: dict[str, Any] = {}
        self.last_thinking = ""
        self._client: Any = None

    @staticmethod
    def _has_credentials() -> bool:
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return True
        return (Path.home() / ".config" / "anthropic").exists()

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # optional dependency: xander-agent[cloud]

            self._client = anthropic.Anthropic()
        return self._client

    def available(self) -> bool:
        if not self._has_credentials():
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def doctor(self) -> dict[str, Any]:
        return {
            "available": self.available(),
            "backend": self.name,
            "routing": self.models,
            "default_model": self.default_model,
            "credentials": self._has_credentials(),
        }

    def generate(
        self,
        prompt: str,
        *,
        role: str = "coder",
        system: str = "",
        schema: type[BaseModel] | dict[str, Any] | None = None,
        think: bool | str | None = None,
        on_token: Callable[[str], None] | None = None,
        timeout: int | None = None,
    ) -> str:
        client = self._get_client()
        model = self.models.get(role, self.default_model)
        system_message = neutral_intent_contract()
        if system.strip():
            system_message += "\n" + system.strip()
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": 16_000,
            "system": system_message,
            "messages": [{"role": "user", "content": prompt}],
        }
        if schema is not None:
            schema_value = schema.model_json_schema() if isinstance(schema, type) and issubclass(schema, BaseModel) else schema
            request["output_config"] = {"format": {"type": "json_schema", "schema": schema_value}}
        started = time.monotonic()
        response = client.with_options(timeout=float(timeout or 600)).messages.create(**request)
        if getattr(response, "stop_reason", "") == "refusal":
            raise RuntimeError("the cloud model declined this request")
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "output_tokens", 0) or 0)
        seconds = round(time.monotonic() - started, 3)
        self.last_stats = {
            "backend": self.name,
            "model": model,
            "tokens": tokens,
            "seconds": seconds,
            "tokens_per_second": round(tokens / seconds, 2) if seconds else 0.0,
            "done_reason": str(getattr(response, "stop_reason", "")),
        }
        if on_token and text:
            on_token(text)
        return text.strip()


class HybridBackend:
    """Per-role dispatch: ``anthropic/``-prefixed roles go to the cloud,
    the rest to local Ollama. Cloud failures fall back to local so a lost
    network never strands a mission."""

    name = "hybrid"

    def __init__(
        self,
        models: dict[str, str],
        *,
        local: Any = None,
        cloud: Any = None,
    ) -> None:
        cloud_models = {role: model for role, model in models.items() if model.startswith(CLOUD_PREFIX)}
        local_models = {role: model for role, model in models.items() if not model.startswith(CLOUD_PREFIX)}
        self.cloud_roles = set(cloud_models)
        self.local = local if local is not None else OllamaBackend(models=local_models or None)
        self.cloud = cloud if cloud is not None else AnthropicBackend(models=cloud_models)
        self.last_stats: dict[str, Any] = {}
        self.last_thinking = ""

    def available(self) -> bool:
        return self.local.available() or (bool(self.cloud_roles) and self.cloud.available())

    def doctor(self) -> dict[str, Any]:
        local_doctor = getattr(self.local, "doctor", lambda: {"available": self.local.available()})()
        cloud_doctor = getattr(self.cloud, "doctor", lambda: {"available": self.cloud.available()})()
        return {
            "available": self.available(),
            "backend": self.name,
            "cloud_roles": sorted(self.cloud_roles),
            "local": local_doctor,
            "cloud": cloud_doctor,
        }

    def generate(self, prompt: str, *, role: str = "coder", **kwargs: Any) -> str:
        use_cloud = role in self.cloud_roles and self.cloud.available()
        target = self.cloud if use_cloud else self.local
        try:
            result = target.generate(prompt, role=role, **kwargs)
        except Exception:
            if use_cloud and self.local.available():
                target = self.local
                result = target.generate(prompt, role=role, **kwargs)
            else:
                raise
        self.last_stats = dict(getattr(target, "last_stats", {}) or {})
        self.last_thinking = str(getattr(target, "last_thinking", "") or "")
        return result


def backend_for(models: dict[str, str] | None = None) -> Backend:
    """Pick the backend shape a routing table implies."""

    models = models or {}
    if any(value.startswith(CLOUD_PREFIX) for value in models.values()):
        return HybridBackend(models)
    return OllamaBackend(models=models or None)


def get_backend() -> OllamaBackend:
    return OllamaBackend()
