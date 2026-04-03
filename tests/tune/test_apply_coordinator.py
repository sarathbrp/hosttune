from preflight.domain.models import CommandResult
from tune.application.apply_coordinator import (
    ApplyCoordinator,
    NginxDirectiveApplier,
    SysctlApplier,
)
from tune.domain.hypothesis_models import CandidateSource, TunePhase, TuningHypothesis
from onboard.domain.models import ApplyMode

from tests.tune.test_candidate_catalog_builder import build_tune_context


class FakeExecutor:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        if command.startswith("grep -E"):
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="worker_processes 112;",
                stderr="",
            )
        if command.startswith("sysctl -n"):
            return CommandResult(command=command, exit_code=0, stdout="4096", stderr="")
        return CommandResult(command=command, exit_code=0, stdout="", stderr="")


def test_sysctl_applier_builds_apply_and_rollback() -> None:
    context = build_tune_context()
    executor = FakeExecutor()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="sysctl.net.core.somaxconn",
        parameter_name="net.core.somaxconn",
        domain="kernel_sysctl",
        proposed_value="65535",
        source=CandidateSource.SERVICE_SYSCTL,
        apply_mode=ApplyMode.RELOAD,
        rationale="Increase listen queue capacity.",
    )

    applied = SysctlApplier().apply(context, hypothesis, executor)

    assert applied.target_path == "net.core.somaxconn"
    assert applied.previous_value == "4096"
    assert applied.applied_value == "65535"
    assert applied.rollback_command == "sysctl -w net.core.somaxconn=4096"


def test_nginx_directive_applier_builds_apply_and_rollback() -> None:
    context = build_tune_context()
    executor = FakeExecutor()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="service.directive.worker_processes",
        parameter_name="worker_processes",
        domain="service_config",
        proposed_value="56",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=ApplyMode.RELOAD,
        rationale="Reduce workers to a balanced subset of CPUs.",
    )

    applied = NginxDirectiveApplier().apply(context, hypothesis, executor)

    assert applied.target_path == "/etc/nginx/nginx.conf"
    assert applied.previous_value == "112"
    assert applied.applied_value == "56"
    assert applied.apply_command.startswith("python3 -c ")
    assert applied.rollback_command.startswith("python3 -c ")
    assert applied.rollback_command.endswith(
        "/etc/nginx/nginx.conf worker_processes 112"
    )


def test_apply_coordinator_routes_by_parameter_prefix() -> None:
    context = build_tune_context()
    executor = FakeExecutor()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="sysctl.net.core.somaxconn",
        parameter_name="net.core.somaxconn",
        domain="kernel_sysctl",
        proposed_value="65535",
        source=CandidateSource.SERVICE_SYSCTL,
        apply_mode=ApplyMode.RELOAD,
        rationale="Increase listen queue capacity.",
    )

    applied = ApplyCoordinator(
        service_directive_applier=NginxDirectiveApplier(),
        sysctl_applier=SysctlApplier(),
    ).apply(context, hypothesis, executor)

    assert applied.hypothesis.parameter_key == "sysctl.net.core.somaxconn"
