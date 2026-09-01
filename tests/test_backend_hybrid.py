"""Hybrid routing: cloud roles reach Claude, everything else stays local."""

from __future__ import annotations

import pytest

from xander_agent.backend import AnthropicBackend, HybridBackend, backend_for, OllamaBackend
from xander_agent.variants import VariantProfile


class FakeLocal:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls: list[str] = []
        self.last_stats = {"backend": "ollama", "tokens": 10}

    def available(self) -> bool:
        return self._available

    def generate(self, prompt, *, role="coder", **kwargs) -> str:
        self.calls.append(role)
        return f"local:{role}"


class FakeCloud:
    def __init__(self, available: bool = True, explode: bool = False) -> None:
        self._available = available
        self.explode = explode
        self.calls: list[str] = []
        self.last_stats = {"backend": "anthropic", "tokens": 99}

    def available(self) -> bool:
        return self._available

    def generate(self, prompt, *, role="coder", **kwargs) -> str:
        self.calls.append(role)
        if self.explode:
            raise RuntimeError("cloud is down")
        return f"cloud:{role}"


def _hybrid(**kwargs) -> tuple[HybridBackend, FakeLocal, FakeCloud]:
    local, cloud = FakeLocal(), FakeCloud(**kwargs)
    backend = HybridBackend(
        {"coder": "abliterated-coder:7b", "critic": "anthropic/claude-opus-5"},
        local=local,
        cloud=cloud,
    )
    return backend, local, cloud


def test_cloud_roles_route_to_the_cloud_and_the_rest_stay_local() -> None:
    backend, local, cloud = _hybrid()
    assert backend.generate("p", role="critic") == "cloud:critic"
    assert backend.generate("p", role="coder") == "local:coder"
    assert cloud.calls == ["critic"] and local.calls == ["coder"]


def test_a_cloud_failure_falls_back_to_local() -> None:
    backend, local, cloud = _hybrid(explode=True)
    assert backend.generate("p", role="critic") == "local:critic"
    assert backend.last_stats["backend"] == "ollama", "stats follow whoever actually answered"


def test_stats_track_the_backend_that_answered() -> None:
    backend, _, _ = _hybrid()
    backend.generate("p", role="critic")
    assert backend.last_stats["backend"] == "anthropic"
    backend.generate("p", role="coder")
    assert backend.last_stats["backend"] == "ollama"


def test_backend_for_picks_hybrid_only_when_a_cloud_route_exists() -> None:
    assert isinstance(backend_for({"coder": "huihui_ai/qwen-abliterated:7b"}), OllamaBackend)
    assert isinstance(backend_for({"critic": "anthropic/claude-opus-5"}), HybridBackend)


def test_anthropic_backend_is_unavailable_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(AnthropicBackend, "_has_credentials", staticmethod(lambda: False))
    assert AnthropicBackend().available() is False


def test_ollama_availability_accepts_a_tags_only_compatible_server(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OllamaBackend()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("version unavailable")))
    monkeypatch.setattr(backend, "installed_models", lambda: ["huihui_ai/qwen3-abliterated:8b"])

    assert backend.available() is True


def test_the_cloud_prefix_is_stripped_from_routed_model_names() -> None:
    backend = AnthropicBackend(models={"critic": "anthropic/claude-opus-5"})
    assert backend.models["critic"] == "claude-opus-5"


def test_variant_profiles_accept_cloud_routes_but_still_reject_raw_local_models() -> None:
    profile = VariantProfile(
        name="hybrid",
        model_routing={
            "coder": "huihui_ai/qwen2.5-coder-abliterate:7b",
            "critic": "anthropic/claude-opus-5",
        },
    )
    assert profile.model_routing["critic"] == "anthropic/claude-opus-5"

    with pytest.raises(ValueError, match="abliterated"):
        VariantProfile(name="leaky", model_routing={"coder": "llama3:8b"})
