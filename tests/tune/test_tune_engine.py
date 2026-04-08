import json
import sqlite3
from dataclasses import replace

from preflight.domain.models import CommandResult
from preflight.domain.runtime_artifacts import RuntimeArtifacts
from preflight.infrastructure.knowledge_base import KnowledgeBase
from preflight.interfaces.execution_logger import ExecutionLogger
from tune.application.apply_coordinator import (
    ApplyCoordinator,
    NetworkRingApplier,
    NginxDirectiveApplier,
    PrlimitApplier,
    SysctlApplier,
    SystemdUnitLimitApplier,
)
from tune.application.benchmark_executor import TuneBenchmarkExecutor
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.application.health_validator import HealthValidator
from tune.application.hypothesis_generator import DeterministicHypothesisGenerator
from tune.application.phase_controller import PhaseController
from tune.application.pre_apply_validator import PreApplyValidator
from tune.application.result_evaluator import ResultEvaluator
from tune.application.rollback_coordinator import RollbackCoordinator
from tune.application.tune_engine import TuneEngine, _normalize_parameter_group_hypotheses
from tune.application.tune_recorder import TuneRecorder
from tune.domain.evaluation_models import AttributionVerificationResult
from tune.domain.hypothesis_models import (
    CandidateSource,
    HypothesisStatus,
    ModelUsage,
    TunePhase,
    TuningHypothesis,
)
from tune.domain.validation_models import ValidationCheck, ValidationResult

from tests.tune.test_benchmark_executor import BenchmarkExecutorDouble
from tests.tune.test_candidate_catalog_builder import build_tune_context


class TargetExecutorDouble:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.sysctl_value = "4096"
        self.directive_value = "112"
        self.nofile_soft = "8192"
        self.nofile_hard = "1048576"

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
        if command.startswith("cat "):
            return CommandResult(command=command, exit_code=0, stdout="12345\n", stderr="")
        if "awk " in command and "/proc/" in command and "limits" in command:
            if "$4, $5" in command:
                stdout = f"{self.nofile_soft} {self.nofile_hard}\n"
            else:
                stdout = f"{self.nofile_soft}\n"
            return CommandResult(
                command=command,
                exit_code=0,
                stdout=stdout,
                stderr="",
            )
        if command.startswith("prlimit "):
            if "--nofile=" in command:
                value = command.split("--nofile=", maxsplit=1)[1].split()[0]
                if ":" in value:
                    soft, hard = value.split(":", maxsplit=1)
                else:
                    soft, hard = value, value
                self.nofile_soft = soft
                self.nofile_hard = hard
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
        return (
            TuningHypothesis(
                phase=TunePhase.WIDE_SWEEP,
                parameter_key=candidate.parameter_key,
                parameter_name=candidate.parameter_name,
                domain=candidate.domain,
                tuning_layer=candidate.tuning_layer,
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
            ),
        )


class FailingHealthValidator:
    def validate_baseline(self, context, executor):  # type: ignore[no-untyped-def]
        _ = context
        _ = executor
        return (
            ValidationCheck(name="systemd_active", passed=True, detail="active"),
            ValidationCheck(name="health_probe", passed=False, detail="status=500 body_match=True"),
        )

    def validate(self, context, applied_change, executor) -> ValidationResult:  # type: ignore[no-untyped-def]
        _ = context
        _ = executor
        return ValidationResult(
            applied_change=applied_change,
            healthy=False,
            checks=(
                ValidationCheck(name="systemd_active", passed=True, detail="active"),
                ValidationCheck(name="health_probe", passed=False, detail="status=500 body_match=True"),
                ValidationCheck(name="effective_value", passed=True, detail="ok"),
            ),
        )


class GatePassFailIterationValidator:
    def validate_baseline(self, context, executor):  # type: ignore[no-untyped-def]
        _ = context
        _ = executor
        return (
            ValidationCheck(name="systemd_active", passed=True, detail="active"),
            ValidationCheck(name="health_probe", passed=True, detail="status=200 body_match=True"),
        )

    def validate(self, context, applied_change, executor) -> ValidationResult:  # type: ignore[no-untyped-def]
        _ = context
        _ = executor
        return ValidationResult(
            applied_change=applied_change,
            healthy=False,
            checks=(
                ValidationCheck(name="systemd_active", passed=True, detail="active"),
                ValidationCheck(name="health_probe", passed=False, detail="status=500 body_match=True"),
                ValidationCheck(name="effective_value", passed=True, detail="ok"),
            ),
        )


class VerifiedAttributionVerifier:
    def verify(  # type: ignore[no-untyped-def]
        self,
        context,
        iteration_number,
        applied_change,
        accepted_benchmark_result,
        target_executor,
        benchmark_runner_executor,
    ) -> AttributionVerificationResult:
        _ = context
        _ = iteration_number
        _ = applied_change
        _ = accepted_benchmark_result
        _ = target_executor
        _ = benchmark_runner_executor
        return AttributionVerificationResult(
            verified=True,
            summary="average_drop=0.1000; threshold=0.0500; verified=True",
            reverted_benchmark_result=None,
            average_drop=0.1,
        )


def test_tune_engine_runs_single_iteration_and_records_accept(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base_context = build_tune_context()
    knowledge_base = KnowledgeBase(tmp_path / "artifacts" / "knowledge_base.sqlite")
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
        knowledge_base=knowledge_base,
    )
    context.artifacts.session_directory.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    context.artifacts.stage_files["knowledge_base"] = knowledge_base.path  # type: ignore[union-attr]
    knowledge_base.record_run(
        run_id="abc123def456",
        preflight=context.preflight,
        service_name=context.onboard.service_name,
        benchmark_target=context.baseline.benchmark_target,
    )
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
            network_ring_applier=NetworkRingApplier(),
            runtime_limit_applier=PrlimitApplier(),
            systemd_unit_limit_applier=SystemdUnitLimitApplier(),
        ),
        pre_apply_validator=PreApplyValidator(),
        health_validator=HealthValidator(),
        benchmark_executor=TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None),
        attribution_verifier=VerifiedAttributionVerifier(),
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
    assert any("tuning_layer=" in message for message in logger.messages)
    assert any(
        "Apply:" in message or "Applied changes" in message for message in logger.messages
    )
    assert any("Validate:" in message for message in logger.messages)
    assert any(
        "Benchmark:" in message or "Benchmark run summary:" in message
        for message in logger.messages
    )
    assert any("Evaluate:" in message for message in logger.messages)
    event_types = [
        row[0]
        for row in sqlite3.connect(knowledge_base.path)
        .execute(
            "SELECT event_type FROM events WHERE run_id=? ORDER BY id ASC",
            ("abc123def456",),
        )
        .fetchall()
    ]
    assert "change_applied" in event_types
    assert "benchmark_completed" in event_types
    assert "evaluation_completed" in event_types
    chosen_key = state.history[0].hypothesis.parameter_key
    chosen_value = state.history[0].hypothesis.proposed_value
    evaluation_payload = json.loads(
        sqlite3.connect(knowledge_base.path)
        .execute(
            """
            SELECT payload_json
            FROM events
            WHERE run_id=? AND event_type='evaluation_completed'
            ORDER BY id ASC
            LIMIT 1
            """,
            ("abc123def456",),
        )
        .fetchone()[0]
    )
    assert evaluation_payload["parameter_key"] == chosen_key
    assert evaluation_payload["proposed_value"] == chosen_value
    assert evaluation_payload["applied_parameter_values"] == [
        {
            "parameter_key": chosen_key,
            "proposed_value": chosen_value,
        }
    ]


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
            network_ring_applier=NetworkRingApplier(),
            runtime_limit_applier=PrlimitApplier(),
            systemd_unit_limit_applier=SystemdUnitLimitApplier(),
        ),
        pre_apply_validator=PreApplyValidator(),
        health_validator=HealthValidator(),
        benchmark_executor=TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None),
        attribution_verifier=VerifiedAttributionVerifier(),
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


def test_tune_engine_logs_benchmark_skipped_reason(tmp_path) -> None:  # type: ignore[no-untyped-def]
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
            }
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
            network_ring_applier=NetworkRingApplier(),
            runtime_limit_applier=PrlimitApplier(),
            systemd_unit_limit_applier=SystemdUnitLimitApplier(),
        ),
        pre_apply_validator=PreApplyValidator(),
        health_validator=GatePassFailIterationValidator(),
        benchmark_executor=TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None),
        attribution_verifier=VerifiedAttributionVerifier(),
        result_evaluator=ResultEvaluator(),
        rollback_coordinator=RollbackCoordinator(),
        recorder=TuneRecorder(),
        logger=logger,
    ).run(
        context=context,
        target_executor=target_executor,
        benchmark_executor=benchmark_executor,
    )

    assert state.history[0].status is HypothesisStatus.FAILED_VALIDATION
    assert any(
        "Rollback (all):" in message and "reason=validation_failed" in message
        for message in logger.messages
    )
    assert any(
        "health_probe passed=False" in message and "status=500 body_match=True" in message
        for message in logger.messages
    )


def test_tune_engine_fails_fast_when_pre_tune_health_gate_fails(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
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
    logger = CaptureLogger()

    try:
        TuneEngine(
            candidate_catalog_builder=CandidateCatalogBuilder(),
            phase_controller=PhaseController(),
            hypothesis_generator=DeterministicHypothesisGenerator(),
            apply_coordinator=ApplyCoordinator(
                service_directive_applier=NginxDirectiveApplier(),
                sysctl_applier=SysctlApplier(),
                network_ring_applier=NetworkRingApplier(),
                runtime_limit_applier=PrlimitApplier(),
                systemd_unit_limit_applier=SystemdUnitLimitApplier(),
            ),
            pre_apply_validator=PreApplyValidator(),
            health_validator=FailingHealthValidator(),
            benchmark_executor=TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None),
            attribution_verifier=VerifiedAttributionVerifier(),
            result_evaluator=ResultEvaluator(),
            rollback_coordinator=RollbackCoordinator(),
            recorder=TuneRecorder(),
            logger=logger,
        ).run(
            context=context,
            target_executor=TargetExecutorDouble(),
            benchmark_executor=BenchmarkExecutorDouble(
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
                    }
                ]
            ),
        )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("Expected pre-tune health gate failure")

    assert "Pre-tune health gate failed" in message
    assert any(msg == "start:tune" for msg in logger.messages)


def test_tune_engine_rejects_forbidden_value_before_apply(tmp_path) -> None:  # type: ignore[no-untyped-def]
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
    logger = CaptureLogger()

    class ForbiddenValueGenerator:
        def generate(self, context):  # type: ignore[no-untyped-def]
            candidate = next(
                item
                for item in context.candidates
                if item.parameter_key == "service.directive.worker_processes"
            )
            return (
                TuningHypothesis(
                    phase=TunePhase.WIDE_SWEEP,
                    parameter_key=candidate.parameter_key,
                    parameter_name=candidate.parameter_name,
                    domain=candidate.domain,
                    tuning_layer=candidate.tuning_layer,
                    proposed_value="1",
                    source=CandidateSource.SERVICE_DIRECTIVE,
                    apply_mode=candidate.apply_mode,
                    rationale="Propose forbidden value.",
                ),
            )

    state = TuneEngine(
        candidate_catalog_builder=CandidateCatalogBuilder(),
        phase_controller=PhaseController(),
        hypothesis_generator=ForbiddenValueGenerator(),
        apply_coordinator=ApplyCoordinator(
            service_directive_applier=NginxDirectiveApplier(),
            sysctl_applier=SysctlApplier(),
            network_ring_applier=NetworkRingApplier(),
            runtime_limit_applier=PrlimitApplier(),
            systemd_unit_limit_applier=SystemdUnitLimitApplier(),
        ),
        pre_apply_validator=PreApplyValidator(),
        health_validator=HealthValidator(),
        benchmark_executor=TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None),
        attribution_verifier=VerifiedAttributionVerifier(),
        result_evaluator=ResultEvaluator(),
        rollback_coordinator=RollbackCoordinator(),
        recorder=TuneRecorder(),
        logger=logger,
    ).run(
        context=context,
        target_executor=target_executor,
        benchmark_executor=BenchmarkExecutorDouble(
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
                }
            ]
        ),
    )

    assert state.history[0].status is HypothesisStatus.REJECTED_PRE_APPLY
    assert state.iteration_records[0].applied_change is None
    assert not any(command.startswith("python3 -c ") for command in target_executor.commands)
    assert any("Pre-apply rejection" in message for message in logger.messages)


def test_parameter_group_normalization_enforces_members_together() -> None:
    from onboard.domain.models import ApplyMode, DirectiveValueType, PriorityTier
    from tune.domain.tuning_layer import TuningLayer

    def _directive_candidate(parameter_key: str, parameter_name: str, current_value: str) -> object:
        from tune.domain.hypothesis_models import CandidateParameter

        return CandidateParameter(
            parameter_key=parameter_key,
            domain="service_config",
            tuning_layer=TuningLayer.SERVICE,
            parameter_name=parameter_name,
            source=CandidateSource.SERVICE_DIRECTIVE,
            value_type=DirectiveValueType.ENUM,
            apply_mode=ApplyMode.RELOAD,
            priority_tier=PriorityTier.MEDIUM,
            allowed_values=("on", "off"),
            forbidden_values=(),
            min_value=None,
            max_value=None,
            rationale_hint="test",
            current_value=current_value,
        )

    candidates = (
        _directive_candidate("service.directive.sendfile", "sendfile", "off"),
        _directive_candidate("service.directive.tcp_nopush", "tcp_nopush", "off"),
        _directive_candidate("service.directive.tcp_nodelay", "tcp_nodelay", "off"),
    )
    hypotheses = (
        TuningHypothesis(
            phase=TunePhase.OPTIMIZE,
            parameter_key="service.directive.tcp_nopush",
            parameter_name="tcp_nopush",
            domain="service_config",
            tuning_layer=TuningLayer.SERVICE,
            proposed_value="off",
            source=CandidateSource.SERVICE_DIRECTIVE,
            apply_mode=ApplyMode.RELOAD,
            rationale="test",
        ),
    )
    groups = (
        (
            "nginx_static_io_trio",
            (
                ("service.directive.sendfile", "on"),
                ("service.directive.tcp_nopush", "on"),
                ("service.directive.tcp_nodelay", "on"),
            ),
        ),
    )

    normalized = _normalize_parameter_group_hypotheses(hypotheses, candidates, groups)

    assert {h.parameter_key for h in normalized} == {
        "service.directive.sendfile",
        "service.directive.tcp_nopush",
        "service.directive.tcp_nodelay",
    }
    assert all(h.proposed_value == "on" for h in normalized)
