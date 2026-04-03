import json
from datetime import UTC, datetime

from onboard.domain.models import ApplyMode
from preflight.domain.runtime_artifacts import RuntimeArtifacts
from tune.application.result_evaluator import ResultEvaluator
from tune.application.tune_recorder import TuneRecorder
from tune.domain.apply_models import AppliedChange
from tune.domain.hypothesis_models import CandidateSource, TunePhase, TuningHypothesis
from tune.domain.scoreboard_models import (
    DomainImpactScore,
    ParameterImpactScore,
    TuneScoreboard,
    WorkloadImpactScore,
)
from tune.domain.iteration_record import TuneIterationRecord

from tests.tune.test_benchmark_executor import build_validation_result
from tests.tune.test_candidate_catalog_builder import build_tune_context
from tests.tune.test_result_evaluator import build_benchmark_result


def test_tune_recorder_writes_jsonl_iteration_record(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base_context = build_tune_context()
    artifacts = RuntimeArtifacts(
        session_id="abc123def456",
        session_directory=tmp_path / "artifacts" / "abc123def456",
    )
    artifacts.session_directory.mkdir(parents=True, exist_ok=True)
    context = base_context.__class__(
        preflight=base_context.preflight,
        onboard=base_context.onboard,
        snapshot=base_context.snapshot,
        baseline=base_context.baseline,
        benchmark_config=base_context.benchmark_config,
        artifacts=artifacts,
    )
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="sysctl.net.core.somaxconn",
        parameter_name="net.core.somaxconn",
        domain="kernel_sysctl",
        proposed_value="65535",
        source=CandidateSource.SERVICE_SYSCTL,
        apply_mode=ApplyMode.RELOAD,
        rationale="Increase backlog.",
    )
    applied_change = AppliedChange(
        hypothesis=hypothesis,
        target_path="net.core.somaxconn",
        previous_value="4096",
        applied_value="65535",
        apply_mode=ApplyMode.RELOAD,
        apply_command="sysctl -w net.core.somaxconn=65535",
        rollback_command="sysctl -w net.core.somaxconn=4096",
    )
    validation_result = build_validation_result()
    benchmark_result = build_benchmark_result(
        homepage_rps=1200.0,
        small_rps=1000.0,
        stable=True,
    )
    evaluation_result = ResultEvaluator().evaluate(context, benchmark_result)
    started_at = datetime.now(UTC)
    completed_at = datetime.now(UTC)
    record = TuneIterationRecord(
        iteration_number=1,
        phase=TunePhase.WIDE_SWEEP,
        hypothesis=hypothesis,
        applied_change=applied_change,
        validation_result=validation_result,
        benchmark_result=benchmark_result,
        evaluation_result=evaluation_result,
        active_parameter_keys=("sysctl.net.core.somaxconn",),
        started_at_utc=started_at.isoformat(),
        completed_at_utc=completed_at.isoformat(),
        duration_seconds=1.25,
    )

    file_path = TuneRecorder().record(context, record)
    payload = json.loads(file_path.read_text(encoding="utf-8").splitlines()[0])

    assert file_path.name == "tune_iterations_abc123def456.jsonl"
    assert payload["session_id"] == "abc123def456"
    assert payload["iteration_number"] == 1
    assert payload["phase"] == "wide_sweep"
    assert payload["record"]["evaluation_result"]["decision"] == "accept"
    assert artifacts.stage_files["tune_iterations"] == file_path


def test_tune_recorder_writes_scoreboard_snapshot(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base_context = build_tune_context()
    artifacts = RuntimeArtifacts(
        session_id="abc123def456",
        session_directory=tmp_path / "artifacts" / "abc123def456",
    )
    artifacts.session_directory.mkdir(parents=True, exist_ok=True)
    context = base_context.__class__(
        preflight=base_context.preflight,
        onboard=base_context.onboard,
        snapshot=base_context.snapshot,
        baseline=base_context.baseline,
        benchmark_config=base_context.benchmark_config,
        artifacts=artifacts,
    )
    scoreboard = TuneScoreboard(
        parameter_scores={
            "service.directive.worker_processes": ParameterImpactScore(
                parameter_key="service.directive.worker_processes",
                domain="service_config",
                evaluated_count=1,
                accepted_count=1,
                average_relative_change=0.2,
                best_relative_change=0.2,
            )
        },
        domain_scores={
            "service_config": DomainImpactScore(
                domain="service_config",
                evaluated_count=1,
                accepted_count=1,
                average_relative_change=0.2,
                parameter_keys=("service.directive.worker_processes",),
            )
        },
        workload_scores={
            "homepage": WorkloadImpactScore(
                workload_name="homepage",
                best_parameter_key="service.directive.worker_processes",
                best_relative_change=0.3,
                worst_parameter_key="service.directive.worker_processes",
                worst_relative_change=0.3,
                win_count=1,
            )
        },
    )

    file_path = TuneRecorder().record_scoreboard(context, scoreboard)
    payload = json.loads(file_path.read_text(encoding="utf-8"))

    assert file_path.name == "tune_scoreboard_abc123def456.json"
    assert payload["session_id"] == "abc123def456"
    assert payload["scoreboard"]["domain_scores"]["service_config"]["accepted_count"] == 1
    assert artifacts.stage_files["tune_scoreboard"] == file_path
