from pathlib import Path

from onboard.domain.models import ApplyMode, DirectiveValueType, PriorityTier
from preflight.domain.runtime_artifacts import RuntimeArtifacts
from tune.application.hypothesis_prompt_layer import (
    format_blocked_prior_pairs,
    format_candidate_line_for_llm,
    format_compact_history_lines,
    format_hybrid_hypothesis_prompt,
    format_prior_run_memory,
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
    HypothesisRecord,
    HypothesisStatus,
    TuningHypothesis,
    TunePhase,
)
from tune.domain.tuning_layer import TuningLayer

from tests.tune.test_candidate_catalog_builder import FakeExecutor, build_tune_context
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from preflight.infrastructure.knowledge_base import KnowledgeBase


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


def test_compact_history_lines_summarize_older_iterations() -> None:
    tune_context = build_tune_context()
    history = tuple(
        HypothesisRecord(
            iteration_number=index,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=TuningHypothesis(
                phase=TunePhase.WIDE_SWEEP,
                parameter_key=f"service.directive.knob_{index}",
                parameter_name=f"knob_{index}",
                domain="service_config",
                tuning_layer=TuningLayer.SERVICE,
                proposed_value=str(index),
                source=CandidateSource.SERVICE_DIRECTIVE,
                apply_mode=ApplyMode.RELOAD,
                rationale="test rationale",
            ),
            status=HypothesisStatus.ACCEPTED if index % 2 else HypothesisStatus.INCONCLUSIVE,
            evaluation_summary=f"summary {index}",
        )
        for index in range(1, 8)
    )

    lines = format_compact_history_lines(history)

    assert any("older_history_summary=count=4" in line for line in lines)
    assert any("recent_history:" in line for line in lines)
    assert any("iteration=7" in line for line in lines)


def test_prior_run_memory_is_compact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tune_context = build_tune_context()
    knowledge_base = KnowledgeBase(tmp_path / "artifacts" / "test_prompt_kb.sqlite")
    knowledge_base.record_run(
        run_id="prior-run",
        preflight=tune_context.preflight,
        service_name=tune_context.onboard.service_name,
        benchmark_target=tune_context.baseline.benchmark_target,
    )
    knowledge_base.finalize_run(
        run_id="prior-run",
        stop_reason="converged",
        best_score=0.25,
        best_iteration=2,
        best_config={"service.directive.access_log": "off"},
    )
    tune_context = tune_context.__class__(
        preflight=tune_context.preflight,
        onboard=tune_context.onboard,
        snapshot=tune_context.snapshot,
        baseline=tune_context.baseline,
        benchmark_config=tune_context.benchmark_config,
        artifacts=RuntimeArtifacts(
            session_id="current-run",
            session_directory=tmp_path / "artifacts" / "current-run",
        ),
        host_profile=tune_context.host_profile,
        knowledge_base=knowledge_base,
    )

    summary = format_prior_run_memory(tune_context)

    assert "best_score" not in summary
    assert "score=25.00%" in summary
    assert "best=service.directive.access_log=off" in summary


def test_blocked_prior_pairs_deduplicate_history() -> None:
    history = (
        HypothesisRecord(
            iteration_number=1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=TuningHypothesis(
                phase=TunePhase.WIDE_SWEEP,
                parameter_key="sysctl.net.core.somaxconn",
                parameter_name="net.core.somaxconn",
                domain="kernel_sysctl",
                tuning_layer=TuningLayer.KERNEL,
                proposed_value="8192",
                source=CandidateSource.SERVICE_SYSCTL,
                apply_mode=ApplyMode.RELOAD,
                rationale="test",
            ),
            status=HypothesisStatus.ACCEPTED,
            evaluation_summary="accepted",
        ),
        HypothesisRecord(
            iteration_number=2,
            phase=TunePhase.DOMAIN_FOCUS,
            hypothesis=TuningHypothesis(
                phase=TunePhase.DOMAIN_FOCUS,
                parameter_key="sysctl.net.core.somaxconn",
                parameter_name="net.core.somaxconn",
                domain="kernel_sysctl",
                tuning_layer=TuningLayer.KERNEL,
                proposed_value="8192",
                source=CandidateSource.SERVICE_SYSCTL,
                apply_mode=ApplyMode.RELOAD,
                rationale="test",
            ),
            status=HypothesisStatus.INCONCLUSIVE,
            evaluation_summary="flat",
        ),
    )

    lines = format_blocked_prior_pairs(history)

    assert lines == ["- sysctl.net.core.somaxconn=8192"]


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
    assert "alternate_recommendations=" in prompt
    assert "Selected runtime config snippet:" in prompt
    assert "Selected service YAML reference snippet:" in prompt
    assert "Blocked prior parameter/value pairs:" in prompt
    assert "triage autofix is already resolved before this prompt" in prompt
    assert "only choose from 'Selectable candidates'" in prompt
    assert "do not repeat a parameter/value pair" in prompt
    assert "do not invent unsupported knobs mentioned only in signal text" in prompt
    assert "rollback_plan" in prompt
