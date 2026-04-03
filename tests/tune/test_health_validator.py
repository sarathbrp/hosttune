from onboard.domain.models import ApplyMode
from preflight.domain.models import CommandResult
from tune.application.health_validator import HealthValidator
from tune.domain.apply_models import AppliedChange
from tune.domain.hypothesis_models import CandidateSource, TunePhase, TuningHypothesis

from tests.tune.test_candidate_catalog_builder import build_tune_context


class HealthyExecutor:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        if command.startswith("systemctl is-active"):
            return CommandResult(command=command, exit_code=0, stdout="active", stderr="")
        if command == "nginx -t":
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="",
                stderr="nginx: configuration file /etc/nginx/nginx.conf test is successful",
            )
        if "curl -sS" in command:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="200\n__HOSTTUNE_BODY__\n<html>ok</html>",
                stderr="",
            )
        if command.startswith("sysctl -n"):
            return CommandResult(command=command, exit_code=0, stdout="65535", stderr="")
        if command.startswith("grep -E"):
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="worker_processes 56;",
                stderr="",
            )
        return CommandResult(command=command, exit_code=0, stdout="", stderr="")


def build_sysctl_change() -> AppliedChange:
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
    return AppliedChange(
        hypothesis=hypothesis,
        target_path="net.core.somaxconn",
        previous_value="4096",
        applied_value="65535",
        apply_mode=ApplyMode.RELOAD,
        apply_command="sysctl -w net.core.somaxconn=65535",
        rollback_command="sysctl -w net.core.somaxconn=4096",
    )


def build_directive_change() -> AppliedChange:
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="service.directive.worker_processes",
        parameter_name="worker_processes",
        domain="service_config",
        proposed_value="56",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=ApplyMode.RELOAD,
        rationale="Reduce workers.",
    )
    return AppliedChange(
        hypothesis=hypothesis,
        target_path="/etc/nginx/nginx.conf",
        previous_value="112",
        applied_value="56",
        apply_mode=ApplyMode.RELOAD,
        apply_command="perl -0pi -e ...",
        rollback_command="perl -0pi -e ...",
    )


def test_health_validator_accepts_healthy_sysctl_change() -> None:
    validator = HealthValidator()
    result = validator.validate(
        context=build_tune_context(),
        applied_change=build_sysctl_change(),
        executor=HealthyExecutor(),
    )

    assert result.healthy is True
    assert all(check.passed for check in result.checks)


def test_health_validator_accepts_healthy_directive_change() -> None:
    validator = HealthValidator()
    result = validator.validate(
        context=build_tune_context(),
        applied_change=build_directive_change(),
        executor=HealthyExecutor(),
    )

    assert result.healthy is True
    assert {check.name for check in result.checks} == {
        "systemd_active",
        "health_probe",
        "config_syntax",
        "effective_value",
    }


def test_health_validator_flags_unhealthy_service() -> None:
    class UnhealthyExecutor(HealthyExecutor):
        def run(self, command: str) -> CommandResult:
            if command.startswith("systemctl is-active"):
                return CommandResult(command=command, exit_code=3, stdout="failed", stderr="")
            return super().run(command)

    validator = HealthValidator()
    result = validator.validate(
        context=build_tune_context(),
        applied_change=build_sysctl_change(),
        executor=UnhealthyExecutor(),
    )

    assert result.healthy is False
    assert any(check.name == "systemd_active" and not check.passed for check in result.checks)


def test_health_validator_runs_baseline_checks() -> None:
    validator = HealthValidator()
    checks = validator.validate_baseline(
        context=build_tune_context(),
        executor=HealthyExecutor(),
    )

    assert {check.name for check in checks} == {"systemd_active", "health_probe"}
    assert all(check.passed for check in checks)


def test_health_validator_flags_nginx_config_syntax_failure() -> None:
    class BrokenSyntaxExecutor(HealthyExecutor):
        def run(self, command: str) -> CommandResult:
            if command == "nginx -t":
                return CommandResult(
                    command=command,
                    exit_code=1,
                    stdout="",
                    stderr="nginx: [emerg] invalid number of arguments",
                )
            return super().run(command)

    validator = HealthValidator()
    result = validator.validate(
        context=build_tune_context(),
        applied_change=build_directive_change(),
        executor=BrokenSyntaxExecutor(),
    )

    assert result.healthy is False
    assert any(check.name == "config_syntax" and not check.passed for check in result.checks)
