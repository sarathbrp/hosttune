from dataclasses import replace

from baseline.domain.models import WorkloadBenchmarkResult
from preflight.domain.models import CommandResult
from tune.application.attribution_verifier import AttributionVerifier
from tune.application.benchmark_executor import TuneBenchmarkExecutor
from tune.application.health_validator import HealthValidator
from tune.domain.apply_models import AppliedChange
from tune.domain.benchmark_models import BenchmarkWorkloadSummary, TuneBenchmarkResult
from tune.domain.hypothesis_models import CandidateSource, TunePhase, TuningHypothesis
from tune.domain.tuning_layer import tuning_layer_for_parameter_key

from tests.tune.test_benchmark_executor import BenchmarkExecutorDouble, build_validation_result
from tests.tune.test_candidate_catalog_builder import build_tune_context
from tests.tune.test_result_evaluator import build_benchmark_result
from onboard.domain.models import ApplyMode


class TargetExecutorDouble:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        if command.startswith("systemctl is-active"):
            return CommandResult(command=command, exit_code=0, stdout="active", stderr="")
        if "curl -sS" in command:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="200\n__HOSTTUNE_BODY__\n<html>ok</html>",
                stderr="",
            )
        if command.startswith("python3 -c "):
            return CommandResult(command=command, exit_code=0, stdout="", stderr="")
        return CommandResult(command=command, exit_code=0, stdout="", stderr="")


def test_attribution_verifier_confirms_change_when_revert_drops_score() -> None:
    context = build_tune_context()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="service.directive.worker_processes",
        parameter_name="worker_processes",
        domain="service_config",
        tuning_layer=tuning_layer_for_parameter_key("service.directive.worker_processes"),
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
        apply_command="python3 -c 'apply'",
        rollback_command="python3 -c 'rollback'",
    )
    accepted_benchmark_result = build_benchmark_result(
        homepage_rps=1200.0,
        small_rps=1000.0,
        stable=True,
    )
    benchmark_executor = BenchmarkExecutorDouble(
        [
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1000.0, "total": 10000},
                        "latency": {"avg": "2.0ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 900.0, "total": 9000},
                        "latency": {"avg": "1.5ms"},
                    }
                },
            }
        ]
    )

    result = AttributionVerifier(
        benchmark_executor=TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None),
        health_validator=HealthValidator(),
    ).verify(
        context=context,
        iteration_number=1,
        applied_change=applied_change,
        accepted_benchmark_result=accepted_benchmark_result,
        target_executor=TargetExecutorDouble(),
        benchmark_runner_executor=benchmark_executor,
    )

    assert result.verified is True
    assert result.reverted_benchmark_result is not None
    assert result.average_drop > context.baseline.expected_variance


def test_attribution_verifier_uses_material_gain_workloads_for_verification() -> None:
    base_context = build_tune_context()
    context = replace(
        base_context,
        baseline=replace(
            base_context.baseline,
            workload_results=(
                WorkloadBenchmarkResult("homepage", "/tmp/homepage.json", 368268.81, 0, 2.77),
                WorkloadBenchmarkResult("small", "/tmp/small.json", 362168.89, 0, 2.84),
                WorkloadBenchmarkResult("medium", "/tmp/medium.json", 1396.85, 0, 237.99),
                WorkloadBenchmarkResult("large", "/tmp/large.json", 185.84, 0, 492.16),
                WorkloadBenchmarkResult("mixed", "/tmp/mixed.json", 2264.51, 0, 152.58),
            ),
            expected_variance=0.1,
        ),
    )
    hypothesis = TuningHypothesis(
        phase=TunePhase.BOUNDARY_PUSH,
        parameter_key="service.directive.worker_rlimit_nofile",
        parameter_name="worker_rlimit_nofile",
        domain="runtime",
        tuning_layer=tuning_layer_for_parameter_key("service.directive.worker_rlimit_nofile"),
        proposed_value="524288",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=ApplyMode.RELOAD,
        rationale="test",
    )
    applied_change = AppliedChange(
        hypothesis=hypothesis,
        target_path="/etc/nginx/nginx.conf",
        previous_value="1024",
        applied_value="524288",
        apply_mode=ApplyMode.RELOAD,
        apply_command="python3 -c 'apply'",
        rollback_command="python3 -c 'rollback'",
    )
    accepted_benchmark_result = TuneBenchmarkResult(
        validation_result=build_validation_result(),
        benchmark_command="benchmark.sh hosttune",
        run_count=1,
        stable=True,
        variance_threshold=0.1,
        workload_summaries=(
            BenchmarkWorkloadSummary("homepage", (), 632903.82, 0, 1.64, 0.0, True),
            BenchmarkWorkloadSummary("small", (), 1737748.83, 0, 0.38, 0.0, True),
            BenchmarkWorkloadSummary("medium", (), 1396.42, 0, 237.68, 0.0, True),
            BenchmarkWorkloadSummary("large", (), 185.74, 0, 521.72, 0.0, True),
            BenchmarkWorkloadSummary("mixed", (), 2276.18, 0, 159.80, 0.0, True),
        ),
    )
    benchmark_executor = BenchmarkExecutorDouble(
        [
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 560085.54, "total": 16858357},
                        "latency": {"avg": "1.87ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 1736823.54, "total": 104382739},
                        "latency": {"avg": "0.38ms"},
                    }
                },
                "medium": {
                    "results": {
                        "requests": {"per_sec": 1396.90, "total": 83953},
                        "latency": {"avg": "240.46ms"},
                    }
                },
                "large": {
                    "results": {
                        "requests": {"per_sec": 185.84, "total": 11169},
                        "latency": {"avg": "519.45ms"},
                    }
                },
                "mixed": {
                    "results": {
                        "requests": {"per_sec": 2222.72, "total": 133585},
                        "latency": {"avg": "148.25ms"},
                    }
                },
            }
        ]
    )

    result = AttributionVerifier(
        benchmark_executor=TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None),
        health_validator=HealthValidator(),
    ).verify(
        context=context,
        iteration_number=7,
        applied_change=applied_change,
        accepted_benchmark_result=accepted_benchmark_result,
        target_executor=TargetExecutorDouble(),
        benchmark_runner_executor=benchmark_executor,
    )

    assert result.verified is True
    assert "compared_workloads=2" in result.summary
    assert "material_gain_workloads=2" in result.summary
    assert "max_drop=0.1151" in result.summary
