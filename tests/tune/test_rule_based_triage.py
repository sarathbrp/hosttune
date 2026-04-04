from dataclasses import replace
from pathlib import Path

from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.application.rule_based_triage import RuleBasedTriage, TriageRulesLoader
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import CandidateAvailability, TunePhase

from tests.tune.test_candidate_catalog_builder import FakeExecutor, build_tune_context


def build_hypothesis_context() -> HypothesisContext:
    context = build_tune_context()
    built = CandidateCatalogBuilder().build(context, FakeExecutor())
    return HypothesisContext(
        tune_context=context,
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


def test_rules_loader_reads_sections() -> None:
    ruleset = TriageRulesLoader().load(Path("triage-rules.yaml"))
    assert "hardware_topology" in ruleset.sections
    assert "application_discovery_sanity" in ruleset.sections


def test_triage_returns_no_action_when_ruleset_is_empty(tmp_path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text("hardware_topology: []\n", encoding="utf-8")
    result = RuleBasedTriage(TriageRulesLoader().load(rules_path)).evaluate(
        build_hypothesis_context()
    )
    assert result.autofix_action is None
    assert result.recommended_action is None
    assert result.escalation_reason.startswith("no deterministic quick fix matched")
    assert "service.directive.worker_processes" in result.safe_candidate_subset


def test_triage_can_autofix_sendfile_when_disabled() -> None:
    base = build_hypothesis_context()
    candidates = tuple(
        replace(candidate, current_value="off")
        if candidate.parameter_key == "service.directive.sendfile"
        else candidate
        for candidate in base.candidates
    )
    context = HypothesisContext(
        tune_context=base.tune_context,
        phase=base.phase,
        iteration_number=base.iteration_number,
        candidates=candidates,
        deferred_candidates=base.deferred_candidates,
        history=base.history,
        active_parameter_keys=base.active_parameter_keys,
        best_parameter_values=base.best_parameter_values,
    )

    result = RuleBasedTriage(TriageRulesLoader().load(Path("triage-rules.yaml"))).evaluate(context)

    assert result.autofix_action is not None
    assert result.autofix_action.parameter_key == "service.directive.sendfile"
    assert result.autofix_action.proposed_value == "on"


def test_triage_emits_signal_for_dual_numa_host() -> None:
    result = RuleBasedTriage(TriageRulesLoader().load(Path("triage-rules.yaml"))).evaluate(
        build_hypothesis_context()
    )
    assert any(rule.rule_id == "signal_dual_numa_high_core_host" for rule in result.triggered_rules)


def test_triage_hides_fallback_signal_when_recommendation_exists() -> None:
    result = RuleBasedTriage(TriageRulesLoader().load(Path("triage-rules.yaml"))).evaluate(
        build_hypothesis_context()
    )
    assert result.recommended_action is not None
    assert not any(rule.rule_id == "fallback_to_llm" for rule in result.triggered_rules)


def test_triage_preserves_alternate_recommendations(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rules_yaml = """
kernel_os_baseline:
  - id: recommend_somaxconn_high_core
    enabled: true
    action: recommend
    kind: candidate_floor
    candidate_key: sysctl.net.core.somaxconn
    min_logical_cores: 32
    floor_value: 8192
application_discovery_sanity:
  - id: nginx_align_worker_rlimit_nofile
    enabled: true
    action: recommend
    kind: align_worker_rlimit_to_unit_limit
    service_name: nginx
    candidate_key: service.directive.worker_rlimit_nofile
"""
    rules_path = tmp_path / "alternate-rules.yaml"
    rules_path.write_text(rules_yaml, encoding="utf-8")
    base = build_hypothesis_context()
    candidates = tuple(
        replace(candidate, current_value="1024")
        if candidate.parameter_key in {
            "sysctl.net.core.somaxconn",
            "service.directive.worker_rlimit_nofile",
        }
        else replace(candidate, current_value="32768")
        if candidate.parameter_key == "systemd.unit.limit_nofile"
        else candidate
        for candidate in base.candidates
    )
    context = HypothesisContext(
        tune_context=base.tune_context,
        phase=base.phase,
        iteration_number=base.iteration_number,
        candidates=candidates,
        deferred_candidates=base.deferred_candidates,
        history=base.history,
        active_parameter_keys=base.active_parameter_keys,
        best_parameter_values=base.best_parameter_values,
    )
    result = RuleBasedTriage(TriageRulesLoader().load(rules_path)).evaluate(context)
    assert result.recommended_action is not None
    assert result.alternate_recommendations
    assert any(
        item.parameter_key == "service.directive.worker_rlimit_nofile"
        for item in result.alternate_recommendations
    )


def test_triage_recommends_candidate_floor_when_below_threshold(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rules_yaml = """
kernel_os_baseline:
  - id: recommend_somaxconn_high_core
    enabled: true
    action: recommend
    kind: candidate_floor
    candidate_key: sysctl.net.core.somaxconn
    min_logical_cores: 32
    floor_value: 8192
"""
    rules_path = tmp_path / "somaxconn-rules.yaml"
    rules_path.write_text(rules_yaml, encoding="utf-8")
    base = build_hypothesis_context()
    candidates = tuple(
        replace(candidate, current_value="1024")
        if candidate.parameter_key == "sysctl.net.core.somaxconn"
        else candidate
        for candidate in base.candidates
    )
    context = HypothesisContext(
        tune_context=base.tune_context,
        phase=base.phase,
        iteration_number=base.iteration_number,
        candidates=candidates,
        deferred_candidates=base.deferred_candidates,
        history=base.history,
        active_parameter_keys=base.active_parameter_keys,
        best_parameter_values=base.best_parameter_values,
    )

    result = RuleBasedTriage(TriageRulesLoader().load(rules_path)).evaluate(context)

    assert result.recommended_action is not None
    assert result.recommended_action.parameter_key == "sysctl.net.core.somaxconn"
    assert result.recommended_action.proposed_value == "8192"


def test_triage_can_recommend_worker_rlimit_alignment(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rules_yaml = """
application_discovery_sanity:
  - id: nginx_align_worker_rlimit_nofile
    enabled: true
    action: recommend
    kind: align_worker_rlimit_to_unit_limit
    service_name: nginx
    candidate_key: service.directive.worker_rlimit_nofile
"""
    rules_path = tmp_path / "worker-rlimit-rules.yaml"
    rules_path.write_text(rules_yaml, encoding="utf-8")
    base = build_hypothesis_context()
    candidates = tuple(
        replace(candidate, current_value="1024")
        if candidate.parameter_key == "service.directive.worker_rlimit_nofile"
        else replace(candidate, current_value="32768")
        if candidate.parameter_key == "systemd.unit.limit_nofile"
        else candidate
        for candidate in base.candidates
    )
    context = HypothesisContext(
        tune_context=base.tune_context,
        phase=base.phase,
        iteration_number=base.iteration_number,
        candidates=candidates,
        deferred_candidates=base.deferred_candidates,
        history=base.history,
        active_parameter_keys=base.active_parameter_keys,
        best_parameter_values=base.best_parameter_values,
    )

    result = RuleBasedTriage(TriageRulesLoader().load(rules_path)).evaluate(context)

    assert result.recommended_action is not None
    assert result.recommended_action.parameter_key == "service.directive.worker_rlimit_nofile"
    assert result.recommended_action.proposed_value == "32768"
