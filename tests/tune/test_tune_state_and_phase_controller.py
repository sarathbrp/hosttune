from dataclasses import replace

from preflight.domain.models import EngagementPolicy
from tune.application.phase_controller import PhaseController
from tune.domain.evaluation_models import EvaluationDecision, EvaluationResult, WorkloadEvaluation
from tune.domain.hypothesis_models import CandidateSource, HypothesisRecord, HypothesisStatus, TunePhase, TuningHypothesis
from tune.domain.iteration_record import TuneIterationRecord
from tune.domain.tune_state import TuneState

from tests.tune.test_candidate_catalog_builder import build_tune_context
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from onboard.domain.models import ApplyMode
from tests.tune.test_benchmark_executor import build_validation_result
from tests.tune.test_result_evaluator import build_benchmark_result
from tune.domain.apply_models import AppliedChange


def test_tune_state_allocates_small_budget_cleanly() -> None:
    state = TuneState.initialize(1)

    assert state.remaining_budget[TunePhase.WIDE_SWEEP] == 1
    assert sum(state.remaining_budget.values()) == 1


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


def test_tune_state_updates_scoreboard_from_iteration() -> None:
    context = build_tune_context()
    state = TuneState.initialize(3)
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="service.directive.worker_processes",
        parameter_name="worker_processes",
        domain="service_config",
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
