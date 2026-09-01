from xander_agent.calibers import (
    DEFAULT_MODELS,
    EMBEDDER,
    alternatives,
    best_installed,
    is_sibling,
    ordered_models,
    parameter_size,
    precision_bits,
    role_for_task,
    upgrades,
)

INSTALLED = [
    "huihui_ai/qwen2.5-coder-abliterate:7b",
    "huihui_ai/qwen2.5-vl-abliterated:3b",
    "huihui_ai/qwen2.5-vl-abliterated:3b-instruct-q8_0",
    "huihui_ai/qwen3-abliterated:8b",
    "huihui_ai/qwen3-abliterated:8b-v2",
    "qwen3-embedding:0.6b",
]


def test_precision_ranking_reads_quant_suffixes() -> None:
    assert precision_bits("huihui_ai/qwen2.5-vl-abliterated:3b-instruct-q8_0") == 8
    assert precision_bits("family:7b-fp16") == 16
    # A bare tag is q4_K_M, which must outrank an explicit q4_0 build.
    assert precision_bits("family:8b") > precision_bits("family:8b-q4_0")


def test_parameter_size_ignores_quant_digits() -> None:
    assert parameter_size("huihui_ai/qwen2.5-vl-abliterated:3b-instruct-q8_0") == "3b"
    assert parameter_size("huihui_ai/qwen3-abliterated:8b-v2") == "8b"
    assert parameter_size("qwen3-embedding:0.6b") == "0.6b"


def test_siblings_never_cross_family_or_parameter_size() -> None:
    assert is_sibling("huihui_ai/qwen3-abliterated:8b", "huihui_ai/qwen3-abliterated:8b-v2")
    assert not is_sibling("huihui_ai/qwen3-abliterated:8b", "huihui_ai/qwen2.5-vl-abliterated:3b")
    # An 8b must never be "upgraded" to a smaller 3b build.
    assert not is_sibling("huihui_ai/qwen3-abliterated:8b", "huihui_ai/qwen3-abliterated:3b-q8_0")


def test_a_local_upgrade_never_leaves_the_abliterated_set() -> None:
    installed = [*INSTALLED, "huihui_ai/qwen3-abliterated-safe:8b-q8_0", "qwen3:8b-q8_0"]
    assert best_installed("huihui_ai/qwen3-abliterated:8b", installed) == "huihui_ai/qwen3-abliterated:8b"


def test_higher_precision_build_on_disk_wins() -> None:
    assert (
        best_installed("huihui_ai/qwen2.5-vl-abliterated:3b", INSTALLED)
        == "huihui_ai/qwen2.5-vl-abliterated:3b-instruct-q8_0"
    )


def test_a_retag_of_identical_weights_is_an_alternative_not_an_upgrade() -> None:
    model = "huihui_ai/qwen3-abliterated:8b"
    assert best_installed(model, INSTALLED) == model
    assert alternatives(model, INSTALLED) == ["huihui_ai/qwen3-abliterated:8b-v2"]


def test_cloud_routes_are_left_alone() -> None:
    assert best_installed("anthropic/claude-opus-5", INSTALLED) == "anthropic/claude-opus-5"


def test_upgrades_reports_stale_routing() -> None:
    stale = {"classifier": "huihui_ai/qwen2.5-vl-abliterated:3b"}
    report = upgrades(stale, INSTALLED)
    assert report["classifier"]["better"] == "huihui_ai/qwen2.5-vl-abliterated:3b-instruct-q8_0"
    assert "8-bit" in report["classifier"]["reason"]


def test_shipped_routing_is_already_the_best_installed_build() -> None:
    assert upgrades(DEFAULT_MODELS, INSTALLED) == {}


def test_role_order_falls_back_when_the_preferred_model_is_absent() -> None:
    without_classifier = [m for m in INSTALLED if "vl-abliterated" not in m]
    order = ordered_models("classifier", without_classifier)
    assert order[0] == "huihui_ai/qwen2.5-coder-abliterate:7b"
    assert all(model in without_classifier for model in order)


def test_nothing_installed_still_yields_a_model_to_attempt() -> None:
    assert ordered_models("planner", []) == list(dict.fromkeys(
        [DEFAULT_MODELS["planner"], DEFAULT_MODELS["coder"], DEFAULT_MODELS["critic"]]
    ))


def test_the_embedder_stays_out_of_chat_routing() -> None:
    # It is not an abliterated build, so it would be rejected by variant policy.
    assert EMBEDDER not in DEFAULT_MODELS.values()
    assert "abliterat" not in EMBEDDER


def test_task_shape_selects_the_model_role_automatically() -> None:
    assert role_for_task("implement", "add the parser", complexity=2) == "coder"
    assert role_for_task("implement", "diagnose the flaky parser", complexity=2) == "planner"
    assert role_for_task("research", "compare maintained options") == "critic"
    assert role_for_task("test-triage", "explain the failure") == "planner"
    assert role_for_task("implement", "add a parser", complexity=2, retry=True) == "planner"
