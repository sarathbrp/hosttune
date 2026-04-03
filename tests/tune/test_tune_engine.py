from dataclasses import replace

from preflight.domain.models import CommandResult
from preflight.domain.runtime_artifacts import RuntimeArtifacts
from preflight.interfaces.execution_logger import ExecutionLogger
from tune.application.apply_coordinator import ApplyCoordinator, NginxDirectiveApplier, SysctlApplier
from tune.application.benchmark_executor import TuneBenchmarkExecutor
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.application.health_validator import HealthValidator
from tune.application.hypothesis_generator import DeterministicHypothesisGenerator
from tune.application.phase_controller import PhaseController
from tune.application.result_evaluator import ResultEvaluator
from tune.application.rollback_coordinator import RollbackCoordinator
from tune.application.tune_engine import TuneEngine
from tune.application.tune_recorder import TuneRecorder
from tune.domain.hypothesis_models import (
    CandidateSource,
    HypothesisStatus,
    ModelUsage,
    TunePhase,
    TuningHypothesis,
)

from tests.tune.test_benchmark_executor import BenchmarkExecutorDouble
from tests.tune.test_candidate_catalog_builder import build_tune_context


class TargetExecutorDouble:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.sysctl_value = "4096"
        self.directive_value = "112"

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        if command.startswith("sysctl -n"):
            return CommandResult(command=command, exit_code=0, stdout=self.sysctl_value, stderr="")
        if command.startswith("sysctl -w"):
            self.sysctl_value = command.split("=", maxsplit=1)[1]
            return CommandResult(command=command, exit_code=0, stdout=self.sysctl_value, stderr="")
        if command.startswith("systemctl is-active"):
            return CommandResult(command=command, exit_code=0, stdout="active", stderr="")
        if "curl -sS" in command:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="200\n__HOSTTUNE_BODY__\n<html>ok</html>",
                stderr="",
            )
        if command.startswith("grep -E"):
            return CommandResult(
                command=command,
                exit_code=0,
                stdout=f"worker_processes {self.directive_value};",
                stderr="",
            )
        if command.startswith("python3 -c "):
            if command.endswith("/etc/nginx/nginx.conf worker_processes 56"):
                self.directive_value = "56"
            if command.endswith("/etc/nginx/nginx.conf worker_processes 112"):
                self.directive_value = "112"
            return CommandResult(command=command, exit_code=0, stdout="", stderr="")
        return CommandResult(command=command, exit_code=0, stdout="", stderr="")


class CaptureLogger(ExecutionLogger):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def stage_start(self, name: str) -> None:
        self.messages.append(f"start:{name}")

    def stage_detail(self, stage: str, message: str) -> None:
        self.messages.append(f"{stage}:{message}")

    def stage_end(self, name: str) -> None:
        self.messages.append(f"end:{name}")


class ModelBackedHypothesisGeneratorDouble:
    def generate(self, context):  # type: ignore[no-untyped-def]
        candidate = context.candidates[0]
        return TuningHypothesis(
            phase=TunePhase.WIDE_SWEEP,
            parameter_key=candidate.parameter_key,
            parameter_name=candidate.parameter_name,
            domain=candidate.domain,
            proposed_value="56",
            source=CandidateSource.SERVICE_DIRECTIVE,
            apply_mode=candidate.apply_mode,
            rationale="Test model-backed hypothesis.",
            model_usage=ModelUsage(
                model_name="/models/test-model",
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
            ),
        )


def test_tune_engine_runs_single_iteration_and_records_accept(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base_context = build_tune_context()
    context = replace(
        base_context,
        preflight=replace(
            base_context.preflight,
            policy=replace(base_context.preflight.policy, max_iterations=1),
        ),
        artifacts=RuntimeArtifacts(
            session_id="abc123def456",
            session_directory=tmp_path / "artifacts" / "abc123def456",
        ),
    )
    context.artifacts.session_directory.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    target_executor = TargetExecutorDouble()
    benchmark_executor = BenchmarkExecutorDouble(
        [
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1200.0, "total": 12000},
                        "latency": {"avg": "2.0ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 1000.0, "total": 10000},
                        "latency": {"avg": "1.5ms"},
                    }
                },
            },
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1210.0, "total": 12100},
                        "latency": {"avg": "2.0ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 1005.0, "total": 10050},
                        "latency": {"avg": "1.5ms"},
                    }
                },
            },
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1195.0, "total": 11950},
                        "latency": {"avg": "2.0ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 995.0, "total": 9950},
                        "latency": {"avg": "1.5ms"},
                    }
                },
            },
        ]
    )
    logger = CaptureLogger()

    state = TuneEngine(
        candidate_catalog_builder=CandidateCatalogBuilder(),
        phase_controller=PhaseController(),
        hypothesis_generator=DeterministicHypothesisGenerator(),
        apply_coordinator=ApplyCoordinator(
            service_directive_applier=NginxDirectiveApplier(),
            sysctl_applier=SysctlApplier(),
        ),
        health_validator=HealthValidator(),
        benchmark_executor=TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None),
        result_evaluator=ResultEvaluator(),
        rollback_coordinator=RollbackCoordinator(),
        recorder=TuneRecorder(),
        logger=logger,
    ).run(
        context=context,
        target_executor=target_executor,
        benchmark_executor=benchmark_executor,
    )

    assert state.total_iterations == 1
    assert state.history[0].status is HypothesisStatus.ACCEPTED
    assert "tune_iterations" in context.artifacts.stage_files  # type: ignore[union-attr]
    assert any("Hypothesis:" in message for message in logger.messages)
    assert any("Apply:" in message for message in logger.messages)
    assert any("Validate:" in message for message in logger.messages)
    assert any("Benchmark:" in message for message in logger.messages)
    assert any("Evaluate:" in message for message in logger.messages)


def test_tune_engine_logs_model_token_summary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base_context = build_tune_context()
    context = replace(
        base_context,
        preflight=replace(
            base_context.preflight,
            policy=replace(base_context.preflight.policy, max_iterations=1),
        ),
        artifacts=RuntimeArtifacts(
            session_id="abc123def456",
            session_directory=tmp_path / "artifacts" / "abc123def456",
        ),
    )
    context.artifacts.session_directory.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    target_executor = TargetExecutorDouble()
    benchmark_executor = BenchmarkExecutorDouble(
        [
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1200.0, "total": 12000},
                        "latency": {"avg": "2.0ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 1000.0, "total": 10000},
                        "latency": {"avg": "1.5ms"},
                    }
                },
            },
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1210.0, "total": 12100},
                        "latency": {"avg": "2.0ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 1005.0, "total": 10050},
                        "latency": {"avg": "1.5ms"},
                    }
                },
            },
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1195.0, "total": 11950},
                        "latency": {"avg": "2.0ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 995.0, "total": 9950},
                        "latency": {"avg": "1.5ms"},
                    }
                },
            },
        ]
    )
    logger = CaptureLogger()

    TuneEngine(
        candidate_catalog_builder=CandidateCatalogBuilder(),
        phase_controller=PhaseController(),
        hypothesis_generator=ModelBackedHypothesisGeneratorDouble(),
        apply_coordinator=ApplyCoordinator(
            service_directive_applier=NginxDirectiveApplier(),
            sysctl_applier=SysctlApplier(),
        ),
        health_validator=HealthValidator(),
        benchmark_executor=TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None),
        result_evaluator=ResultEvaluator(),
        rollback_coordinator=RollbackCoordinator(),
        recorder=TuneRecorder(),
        logger=logger,
    ).run(
        context=context,
        target_executor=target_executor,
        benchmark_executor=benchmark_executor,
    )

    assert any("Hypothesis tokens:" in message for message in logger.messages)
