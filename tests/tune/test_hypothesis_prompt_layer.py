from pathlib import Path

from onboard.domain.models import ApplyMode, DirectiveValueType, PriorityTier
from tune.application.hypothesis_prompt_layer import (
    format_candidate_line_for_llm,
    format_hybrid_hypothesis_prompt,
    format_preflight_digest_lines,
    format_runtime_config_snippet,
    format_service_yaml_reference_snippet,
    hypothesis_prompt_layer_preamble,
)
from tune.application.rule_based_triage import RuleBasedTriage, TriageRulesLoader
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import (
    CandidateAvailability,
    CandidateParameter,
    CandidateSource,
    TunePhase,
)
from tune.domain.tuning_layer import TuningLayer

from tests.tune.test_candidate_catalog_builder import FakeExecutor, build_tune_context
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder


def test_preamble_mentions_curated_and_snippets() -> None:
    lines = hypothesis_prompt_layer_preamble()
    assert any("selected raw snippets" in line for line in lines)
    assert any("no deterministic autofix was applied" in line for line in lines)


def test_preflight_digest_includes_ring_and_storage_facts() -> None:
    ctx = build_tune_context()
    lines = format_preflight_digest_lines(ctx.preflight)
    joined = "\n".join(lines)
    assert "rings_rx=512/4096" in joined
    assert "rings_tx=512/4096" in joined
    assert "storage=" in joined


def test_candidate_line_truncates_long_hint() -> None:
    long_hint = "x" * 500
    candidate = CandidateParameter(
        parameter_key="k",
        domain="d",
        tuning_layer=TuningLayer.RUNTIME,
        parameter_name="p",
        source=CandidateSource.SERVICE_DIRECTIVE,
        value_type=DirectiveValueType.INTEGER,
        apply_mode=ApplyMode.RELOAD,
        priority_tier=PriorityTier.HIGH,
        allowed_values=(),
        forbidden_values=(),
        min_value=1,
        max_value=10,
        rationale_hint=long_hint,
        current_value="5",
    )
    line = format_candidate_line_for_llm(candidate)
    assert "truncated" in line
    assert len(line) < len(long_hint) + 200


def test_runtime_config_snippet_prefers_interesting_lines() -> None:
    snippet = format_runtime_config_snippet(
        "\n".join(
            (
                "events {",
                "worker_connections 1024;",
                "}",
                "http {",
                "access_log off;",
                "}",
            )
        )
    )
    assert "worker_connections 1024;" in snippet
    assert "access_log off;" in snippet


def test_service_yaml_reference_snippet_mentions_tunable_surface() -> None:
    snippet = format_service_yaml_reference_snippet(build_tune_context())
    assert "tunable_surface.allowed_directives" in snippet
    assert "benchmark_hints.primary_metric" in snippet


def test_hybrid_prompt_includes_triage_section() -> None:
    tune_context = build_tune_context()
    built = CandidateCatalogBuilder().build(tune_context, FakeExecutor())
    context = HypothesisContext(
        tune_context=tune_context,
        phase=TunePhase.WIDE_SWEEP,
        iteration_number=1,
        candidates=tuple(c for c in built if c.availability is CandidateAvailability.ACTIVE),
        deferred_candidates=tuple(
            c for c in built if c.availability is CandidateAvailability.DEFERRED
        ),
        history=(),
        active_parameter_keys=(),
        best_parameter_values=(),
    )
    triage = RuleBasedTriage(TriageRulesLoader().load(Path("triage-rules.yaml"))).evaluate(context)
    prompt = format_hybrid_hypothesis_prompt(context, triage)
    assert "Rule-based triage result:" in prompt
    assert "autofix_action=" in prompt
    assert "Selected runtime config snippet:" in prompt
    assert "Selected service YAML reference snippet:" in prompt
    assert "triage autofix is already resolved before this prompt" in prompt
    assert "do not invent unsupported knobs mentioned only in signal text" in prompt
    assert "rollback_plan" in prompt
