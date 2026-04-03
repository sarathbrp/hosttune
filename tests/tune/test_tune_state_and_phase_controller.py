from onboard.domain.models import ApplyMode, DirectiveValueType, PriorityTier
from preflight.domain.models import EngagementPolicy
from tune.application.phase_controller import PhaseController
from tune.application.result_evaluator import ResultEvaluator
from tune.domain.evaluation_models import EvaluationDecision, EvaluationResult, WorkloadEvaluation
from tune.domain.hypothesis_models import (
    CandidateParameter,
    CandidateSource,
    HypothesisRecord,
    HypothesisStatus,
    TunePhase,
    TuningHypothesis,
)
from tune.domain.iteration_record import TuneIterationRecord
from tune.domain.tune_state import BestKnownConfiguration, TuneState

from tests.tune.test_candidate_catalog_builder import FakeExecutor, build_tune_context
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tests.tune.test_benchmark_executor import build_validation_result
from tests.tune.test_result_evaluator import build_benchmark_result
from tune.domain.apply_models import AppliedChange
from tune.domain.tuning_layer import TuningLayer


def test_tune_state_tracks_iterations_since_best_update() -> None:
    state = TuneState.initialize(5)
    assert state.iterations_since_best_update == 0
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="service.directive.worker_processes",
        parameter_name="worker_processes",
        domain="service_config",
        tuning_layer=TuningLayer.SERVICE,
        proposed_value="56",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=ApplyMode.RELOAD,
        rationale="test",
    )
    applied_change = AppliedChange(
        hypothesis=hypothesis,
        target_path="/etc/nginx/nginx.conf",
        previous_value="112",
        applied_value="56",
        apply_mode=ApplyMode.RELOAD,
        apply_command="python3 -c ...",
        rollback_command="python3 -c ...",
    )
    win_eval = EvaluationResult(
        benchmark_result=build_benchmark_result(
            homepage_rps=1200.0,
            small_rps=1000.0,
            stable=True,
        ),
        decision=EvaluationDecision.ACCEPT,
        summary="accepted",
        primary_metric="requests_per_second",
        variance_threshold=0.05,
        guardrails_held=True,
        drift_detected=False,
        workload_evaluations=(
            WorkloadEvaluation(
                workload_name="homepage",
                baseline_requests_per_second=1000.0,
                current_requests_per_second=1200.0,
                relative_change=0.2,
                above_noise_floor=True,
            ),
        ),
        missing_guardrails=(),
    )
    flat_eval = EvaluationResult(
        benchmark_result=build_benchmark_result(
            homepage_rps=1000.0,
            small_rps=900.0,
            stable=True,
        ),
        decision=EvaluationDecision.INCONCLUSIVE,
        summary="flat",
        primary_metric="requests_per_second",
        variance_threshold=0.05,
        guardrails_held=True,
        drift_detected=False,
        workload_evaluations=(
            WorkloadEvaluation(
                workload_name="homepage",
                baseline_requests_per_second=1000.0,
                current_requests_per_second=1000.0,
                relative_change=0.0,
                above_noise_floor=False,
            ),
        ),
        missing_guardrails=(),
    )
    state.record_iteration(
        TuneIterationRecord(
            iteration_number=1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=hypothesis,
            applied_change=applied_change,
            validation_result=build_validation_result(),
            benchmark_result=win_eval.benchmark_result,
            evaluation_result=win_eval,
            attribution_verification=None,
            active_parameter_keys=("service.directive.worker_processes",),
            started_at_utc="2026-04-03T00:00:00+00:00",
            completed_at_utc="2026-04-03T00:01:00+00:00",
            duration_seconds=60.0,
        ),
        HypothesisRecord(
            iteration_number=1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=hypothesis,
            status=HypothesisStatus.ACCEPTED,
        ),
    )
    assert state.iterations_since_best_update == 0
    state.record_iteration(
        TuneIterationRecord(
            iteration_number=2,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=hypothesis,
            applied_change=applied_change,
            validation_result=build_validation_result(),
            benchmark_result=flat_eval.benchmark_result,
            evaluation_result=flat_eval,
            attribution_verification=None,
            active_parameter_keys=("service.directive.worker_processes",),
            started_at_utc="2026-04-03T00:00:00+00:00",
            completed_at_utc="2026-04-03T00:02:00+00:00",
            duration_seconds=60.0,
        ),
        HypothesisRecord(
            iteration_number=2,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=hypothesis,
            status=HypothesisStatus.INCONCLUSIVE,
        ),
    )
    assert state.iterations_since_best_update == 1


def test_tune_state_allocates_small_budget_cleanly() -> None:
    state = TuneState.initialize(1)

    assert state.remaining_budget[TunePhase.WIDE_SWEEP] == 1
    assert sum(state.remaining_budget.values()) == 1


def test_phase_controller_domain_focus_includes_winning_tuning_layer_across_domains() -> None:
    alpha = CandidateParameter(
        parameter_key="service.directive.alpha",
        domain="zone_a",
        tuning_layer=TuningLayer.SERVICE,
        parameter_name="alpha",
        source=CandidateSource.SERVICE_DIRECTIVE,
        value_type=DirectiveValueType.INTEGER,
        apply_mode=ApplyMode.RELOAD,
        priority_tier=PriorityTier.HIGH,
        allowed_values=(),
        forbidden_values=(),
        min_value=1,
        max_value=10,
        rationale_hint="test",
    )
    beta = CandidateParameter(
        parameter_key="service.directive.beta",
        domain="zone_b",
        tuning_layer=TuningLayer.SERVICE,
        parameter_name="beta",
        source=CandidateSource.SERVICE_DIRECTIVE,
        value_type=DirectiveValueType.INTEGER,
        apply_mode=ApplyMode.RELOAD,
        priority_tier=PriorityTier.HIGH,
        allowed_values=(),
        forbidden_values=(),
        min_value=1,
        max_value=10,
        rationale_hint="test",
    )
    gamma = CandidateParameter(
        parameter_key="sysctl.net.example",
        domain="kernel_sysctl",
        tuning_layer=TuningLayer.KERNEL,
        parameter_name="net.example",
        source=CandidateSource.SERVICE_SYSCTL,
        value_type=DirectiveValueType.STRING,
        apply_mode=ApplyMode.RELOAD,
        priority_tier=PriorityTier.HIGH,
        allowed_values=(),
        forbidden_values=(),
        min_value=None,
        max_value=None,
        rationale_hint="test",
    )
    candidates = (alpha, beta, gamma)
    state = TuneState.initialize(10)
    state.history = [
        HypothesisRecord(
            iteration_number=1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=TuningHypothesis(
                phase=TunePhase.WIDE_SWEEP,
                parameter_key=alpha.parameter_key,
                parameter_name=alpha.parameter_name,
                domain=alpha.domain,
                tuning_layer=alpha.tuning_layer,
                proposed_value="5",
                source=alpha.source,
                apply_mode=alpha.apply_mode,
                rationale="test",
            ),
            status=HypothesisStatus.ACCEPTED,
        )
    ]
    filtered = PhaseController().filter_candidates(TunePhase.DOMAIN_FOCUS, state, candidates)
    keys = {c.parameter_key for c in filtered}
    assert keys == {"service.directive.alpha", "service.directive.beta"}


def test_phase_controller_domain_focus_without_accepts_returns_full_catalog() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context, FakeExecutor())
    state = TuneState.initialize(10)
    filtered = PhaseController().filter_candidates(TunePhase.DOMAIN_FOCUS, state, candidates)
    assert filtered == candidates


def test_phase_controller_advances_after_wide_sweep_candidates_exhausted() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context)
    state = TuneState.initialize(10)
    state.history = [
        HypothesisRecord(
            iteration_number=index + 1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=TuningHypothesis(
                phase=TunePhase.WIDE_SWEEP,
                parameter_key=candidate.parameter_key,
                parameter_name=candidate.parameter_name,
                domain=candidate.domain,
                tuning_layer=candidate.tuning_layer,
                proposed_value="1",
                source=CandidateSource.SERVICE_SYSCTL,
                apply_mode=ApplyMode.RELOAD,
                rationale="test",
            ),
            status=HypothesisStatus.REJECTED,
        )
        for index, candidate in enumerate(candidates)
    ]

    phase = PhaseController().determine_phase(state, candidates)

    assert phase is TunePhase.DOMAIN_FOCUS


def test_phase_controller_wide_sweep_prioritizes_high_tier_first() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context)
    state = TuneState.initialize(10)

    filtered = PhaseController().filter_candidates(TunePhase.WIDE_SWEEP, state, candidates)

    assert filtered
    assert all(candidate.priority_tier is PriorityTier.HIGH for candidate in filtered)
    assert len({candidate.domain for candidate in filtered}) >= 2


def test_phase_controller_wide_sweep_prefers_untried_tuning_layer_within_high_tier() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context, FakeExecutor())
    high = [c for c in candidates if c.priority_tier is PriorityTier.HIGH]
    service_high = next(c for c in high if c.tuning_layer is TuningLayer.SERVICE)
    assert any(c.tuning_layer is not TuningLayer.SERVICE for c in high)
    state = TuneState.initialize(30)
    state.history = [
        HypothesisRecord(
            iteration_number=1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=_hypothesis_from_wide_sweep_candidate(service_high),
            status=HypothesisStatus.REJECTED,
        )
    ]
    filtered = PhaseController().filter_candidates(TunePhase.WIDE_SWEEP, state, candidates)
    assert filtered
    assert all(c.priority_tier is PriorityTier.HIGH for c in filtered)
    assert all(c.tuning_layer is not TuningLayer.SERVICE for c in filtered)


def test_phase_controller_wide_sweep_medium_prefers_fresh_layer_when_service_already_touched() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context, FakeExecutor())
    high = [c for c in candidates if c.priority_tier is PriorityTier.HIGH]
    high_sorted = sorted(high, key=lambda c: c.parameter_key)
    state = TuneState.initialize(40)
    state.history = [
        HypothesisRecord(
            iteration_number=i + 1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=_hypothesis_from_wide_sweep_candidate(c),
            status=HypothesisStatus.REJECTED,
        )
        for i, c in enumerate(high_sorted)
    ]
    medium = [c for c in candidates if c.priority_tier is PriorityTier.MEDIUM]
    network_medium = [c for c in medium if c.tuning_layer is TuningLayer.NETWORK]
    assert network_medium, "fixture needs a MEDIUM-tier network ring candidate"
    filtered = PhaseController().filter_candidates(TunePhase.WIDE_SWEEP, state, candidates)
    assert filtered
    assert all(c.priority_tier is PriorityTier.MEDIUM for c in filtered)
    assert any(c.tuning_layer is TuningLayer.NETWORK for c in filtered)
    assert all(c.tuning_layer is not TuningLayer.SERVICE for c in filtered)


def _hypothesis_from_wide_sweep_candidate(
    candidate,
    *,
    proposed_value: str = "1",
) -> TuningHypothesis:
    return TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key=candidate.parameter_key,
        parameter_name=candidate.parameter_name,
        domain=candidate.domain,
        tuning_layer=candidate.tuning_layer,
        proposed_value=proposed_value,
        source=candidate.source,
        apply_mode=candidate.apply_mode,
        rationale="test",
    )


def test_phase_controller_wide_sweep_offers_low_tier_after_high_medium_exhausted_no_accept() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context, FakeExecutor())
    non_low = [c for c in candidates if c.priority_tier is not PriorityTier.LOW]
    low_keys = {c.parameter_key for c in candidates if c.priority_tier is PriorityTier.LOW}
    assert low_keys
    state = TuneState.initialize(30)
    state.history = [
        HypothesisRecord(
            iteration_number=i + 1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=_hypothesis_from_wide_sweep_candidate(c),
            status=HypothesisStatus.REJECTED,
        )
        for i, c in enumerate(non_low)
    ]
    filtered = PhaseController().filter_candidates(TunePhase.WIDE_SWEEP, state, candidates)
    assert filtered
    assert {c.parameter_key for c in filtered}.issubset(low_keys)


def test_phase_controller_wide_sweep_skips_low_after_wide_sweep_accept_on_high_tier() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context, FakeExecutor())
    high_first = next(c for c in candidates if c.priority_tier is PriorityTier.HIGH)
    non_low = [c for c in candidates if c.priority_tier is not PriorityTier.LOW]
    ordered = sorted(
        non_low,
        key=lambda c: (0 if c.parameter_key == high_first.parameter_key else 1, c.parameter_key),
    )
    state = TuneState.initialize(30)
    state.history = [
        HypothesisRecord(
            iteration_number=i + 1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=_hypothesis_from_wide_sweep_candidate(c),
            status=(
                HypothesisStatus.ACCEPTED
                if c.parameter_key == high_first.parameter_key
                else HypothesisStatus.REJECTED
            ),
        )
        for i, c in enumerate(ordered)
    ]
    filtered = PhaseController().filter_candidates(TunePhase.WIDE_SWEEP, state, candidates)
    assert filtered == ()


def test_tune_state_updates_scoreboard_from_iteration() -> None:
    context = build_tune_context()
    state = TuneState.initialize(3)
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="service.directive.worker_processes",
        parameter_name="worker_processes",
        domain="service_config",
        tuning_layer=TuningLayer.SERVICE,
        proposed_value="56",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=ApplyMode.RELOAD,
        rationale="test",
    )
    applied_change = AppliedChange(
        hypothesis=hypothesis,
        target_path="/etc/nginx/nginx.conf",
        previous_value="112",
        applied_value="56",
        apply_mode=ApplyMode.RELOAD,
        apply_command="python3 -c ...",
        rollback_command="python3 -c ...",
    )
    benchmark_result = build_benchmark_result(
        homepage_rps=1200.0,
        small_rps=1000.0,
        stable=True,
    )
    evaluation_result = EvaluationResult(
        benchmark_result=benchmark_result,
        decision=EvaluationDecision.ACCEPT,
        summary="accepted",
        primary_metric="requests_per_second",
        variance_threshold=0.05,
        guardrails_held=True,
        drift_detected=False,
        workload_evaluations=(
            WorkloadEvaluation(
                workload_name="homepage",
                baseline_requests_per_second=1000.0,
                current_requests_per_second=1200.0,
                relative_change=0.2,
                above_noise_floor=True,
            ),
            WorkloadEvaluation(
                workload_name="small",
                baseline_requests_per_second=900.0,
                current_requests_per_second=1000.0,
                relative_change=0.1111111111,
                above_noise_floor=True,
            ),
        ),
        missing_guardrails=(),
    )
    record = TuneIterationRecord(
        iteration_number=1,
        phase=TunePhase.WIDE_SWEEP,
        hypothesis=hypothesis,
        applied_change=applied_change,
        validation_result=build_validation_result(),
        benchmark_result=benchmark_result,
        evaluation_result=evaluation_result,
        attribution_verification=None,
        active_parameter_keys=("service.directive.worker_processes",),
        started_at_utc="2026-04-03T00:00:00+00:00",
        completed_at_utc="2026-04-03T00:01:00+00:00",
        duration_seconds=60.0,
    )
    history_record = HypothesisRecord(
        iteration_number=1,
        phase=TunePhase.WIDE_SWEEP,
        hypothesis=hypothesis,
        status=HypothesisStatus.ACCEPTED,
        evaluation_summary="accepted",
    )

    state.record_iteration(record, history_record)

    parameter_score = state.scoreboard.parameter_scores["service.directive.worker_processes"]
    assert parameter_score.accepted_count == 1
    assert parameter_score.evaluated_count == 1
    assert parameter_score.average_relative_change > 0.15
    domain_score = state.scoreboard.domain_scores["service_config"]
    assert domain_score.accepted_count == 1
    assert "service.directive.worker_processes" in domain_score.parameter_keys
    workload_score = state.scoreboard.workload_scores["homepage"]
    assert workload_score.best_parameter_key == "service.directive.worker_processes"
    assert workload_score.win_count == 1


def _measured_iteration_record(
    *,
    iteration_number: int,
    hypothesis: TuningHypothesis,
    context,
) -> TuneIterationRecord:
    benchmark_result = build_benchmark_result(
        homepage_rps=1000.0,
        small_rps=900.0,
        stable=True,
    )
    evaluation = ResultEvaluator().evaluate(context, benchmark_result)
    assert evaluation.decision is not EvaluationDecision.ACCEPT
    return TuneIterationRecord(
        iteration_number=iteration_number,
        phase=TunePhase.WIDE_SWEEP,
        hypothesis=hypothesis,
        applied_change=None,
        validation_result=build_validation_result(),
        benchmark_result=benchmark_result,
        evaluation_result=evaluation,
        attribution_verification=None,
        active_parameter_keys=(),
        started_at_utc="2026-04-03T00:00:00+00:00",
        completed_at_utc="2026-04-03T00:01:00+00:00",
        duration_seconds=60.0,
    )


def test_phase_controller_stops_after_convergence_without_signal() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context)
    state = TuneState.initialize(10)
    high_priority_candidates = [
        candidate for candidate in candidates if candidate.priority_tier is PriorityTier.HIGH
    ]
    state.history = [
        HypothesisRecord(
            iteration_number=index + 1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=TuningHypothesis(
                phase=TunePhase.WIDE_SWEEP,
                parameter_key=candidate.parameter_key,
                parameter_name=candidate.parameter_name,
                domain=candidate.domain,
                tuning_layer=candidate.tuning_layer,
                proposed_value="2" if candidate.parameter_key == "service.directive.worker_processes" else "8192",
                source=CandidateSource.SERVICE_DIRECTIVE
                if candidate.parameter_key.startswith("service.directive.")
                else CandidateSource.SERVICE_SYSCTL,
                apply_mode=ApplyMode.RELOAD,
                rationale="test",
            ),
            status=HypothesisStatus.REJECTED,
            evaluation_summary="below noise floor",
        )
        for index, candidate in enumerate(high_priority_candidates)
    ]
    sample_hypothesis = state.history[-1].hypothesis
    state.iteration_records = [
        _measured_iteration_record(
            iteration_number=100 + i,
            hypothesis=sample_hypothesis,
            context=context,
        )
        for i in range(2)
    ]

    assert PhaseController(convergence_no_signal_limit=2).should_stop(state, candidates) is True


def test_phase_controller_convergence_not_before_best_stable_when_best_exists() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context)
    state = TuneState.initialize(10)
    high_priority_candidates = [
        candidate for candidate in candidates if candidate.priority_tier is PriorityTier.HIGH
    ]
    state.history = [
        HypothesisRecord(
            iteration_number=index + 1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=TuningHypothesis(
                phase=TunePhase.WIDE_SWEEP,
                parameter_key=candidate.parameter_key,
                parameter_name=candidate.parameter_name,
                domain=candidate.domain,
                tuning_layer=candidate.tuning_layer,
                proposed_value="2",
                source=CandidateSource.SERVICE_DIRECTIVE,
                apply_mode=ApplyMode.RELOAD,
                rationale="test",
            ),
            status=HypothesisStatus.REJECTED,
        )
        for index, candidate in enumerate(high_priority_candidates)
    ]
    sample_hypothesis = state.history[-1].hypothesis
    state.iteration_records = [
        _measured_iteration_record(
            iteration_number=200 + i,
            hypothesis=sample_hypothesis,
            context=context,
        )
        for i in range(3)
    ]
    state.best_configuration = BestKnownConfiguration(
        score=0.01,
        parameter_values={"service.directive.worker_processes": "56"},
        iteration_number=1,
    )
    state.iterations_since_best_update = 1
    controller = PhaseController(convergence_no_signal_limit=2, convergence_best_stability_limit=2)
    assert controller.should_stop(state, candidates) is False
    state.iterations_since_best_update = 2
    assert controller.should_stop(state, candidates) is True
