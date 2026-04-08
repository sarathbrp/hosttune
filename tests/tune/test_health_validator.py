from onboard.domain.models import ApplyMode
from preflight.domain.models import CommandResult
from tune.application.health_validator import HealthValidator
from tune.domain.apply_models import AppliedChange
from tune.domain.hypothesis_models import CandidateSource, TunePhase, TuningHypothesis
from tune.domain.tuning_layer import tuning_layer_for_parameter_key

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
        if "ethtool -g" in command:
            return CommandResult(command=command, exit_code=0, stdout="1024", stderr="")
        if command.startswith("grep -E"):
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="worker_processes 56;",
                stderr="",
            )
        if "systemctl show" in command and "LimitNPROC" in command:
            return CommandResult(command=command, exit_code=0, stdout="8000\n", stderr="")
        if "systemctl show" in command and "CPUQuotaPerSecUSec" in command:
            return CommandResult(command=command, exit_code=0, stdout="1250000\n", stderr="")
        if "systemctl show" in command and "CPUQuota" in command:
            return CommandResult(command=command, exit_code=0, stdout="125%\n", stderr="")
        return CommandResult(command=command, exit_code=0, stdout="", stderr="")


def build_sysctl_change() -> AppliedChange:
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="sysctl.net.core.somaxconn",
        parameter_name="net.core.somaxconn",
        domain="kernel_sysctl",
        tuning_layer=tuning_layer_for_parameter_key("sysctl.net.core.somaxconn"),
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
        tuning_layer=tuning_layer_for_parameter_key("service.directive.worker_processes"),
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
        apply_command="python3 -c ...",
        rollback_command="python3 -c ...",
    )


def build_network_change() -> AppliedChange:
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="network.ring.rx",
        parameter_name="rx",
        domain="network",
        tuning_layer=tuning_layer_for_parameter_key("network.ring.rx"),
        proposed_value="1024",
        source=CandidateSource.PLATFORM_CAPABILITY,
        apply_mode=ApplyMode.RELOAD,
        rationale="Increase receive ring buffer.",
    )
    return AppliedChange(
        hypothesis=hypothesis,
        target_path="eth0:rx",
        previous_value="511",
        applied_value="1024",
        apply_mode=ApplyMode.RELOAD,
        apply_command="ethtool -G eth0 rx 1024",
        rollback_command="ethtool -G eth0 rx 511",
    )


def build_systemd_unit_limit_change() -> AppliedChange:
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="systemd.unit.limit_nproc",
        parameter_name="limit_nproc",
        domain="runtime",
        tuning_layer=tuning_layer_for_parameter_key("systemd.unit.limit_nproc"),
        proposed_value="8000",
        source=CandidateSource.SYSTEMD_UNIT_LIMIT,
        apply_mode=ApplyMode.RESTART,
        rationale="Raise unit NPROC.",
    )
    return AppliedChange(
        hypothesis=hypothesis,
        target_path="nginx.service:LimitNPROC",
        previous_value="1000",
        applied_value="8000",
        apply_mode=ApplyMode.RESTART,
        apply_command="systemctl set-property ...",
        rollback_command="systemctl set-property ...",
    )


def build_systemd_cgroup_control_change() -> AppliedChange:
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="systemd.cgroup.cpu_quota_percent",
        parameter_name="cpu_quota_percent",
        domain="runtime",
        tuning_layer=tuning_layer_for_parameter_key("systemd.cgroup.cpu_quota_percent"),
        proposed_value="125",
        source=CandidateSource.SYSTEMD_CGROUP_CONTROL,
        apply_mode=ApplyMode.RESTART,
        rationale="Raise unit CPUQuota.",
    )
    return AppliedChange(
        hypothesis=hypothesis,
        target_path="nginx.service:CPUQuota",
        previous_value="50",
        applied_value="125",
        apply_mode=ApplyMode.RESTART,
        apply_command="systemctl set-property ...",
        rollback_command="systemctl set-property ...",
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


def test_health_validator_accepts_healthy_network_change() -> None:
    validator = HealthValidator()
    result = validator.validate(
        context=build_tune_context(),
        applied_change=build_network_change(),
        executor=HealthyExecutor(),
    )

    assert result.healthy is True
    assert any(check.name == "effective_value" and check.detail == "observed=1024" for check in result.checks)


def test_health_validator_accepts_healthy_systemd_unit_limit_change() -> None:
    validator = HealthValidator()
    result = validator.validate(
        context=build_tune_context(),
        applied_change=build_systemd_unit_limit_change(),
        executor=HealthyExecutor(),
    )

    assert result.healthy is True
    assert any(
        check.name == "effective_value" and check.detail == "observed=8000" for check in result.checks
    )


def test_health_validator_accepts_healthy_systemd_cgroup_control_change() -> None:
    validator = HealthValidator()
    result = validator.validate(
        context=build_tune_context(),
        applied_change=build_systemd_cgroup_control_change(),
        executor=HealthyExecutor(),
    )

    assert result.healthy is True
    assert any(
        check.name == "effective_value" and check.detail == "observed=125%" for check in result.checks
    )


def test_health_validator_reads_cpu_quota_from_per_sec_usec_property() -> None:
    class CgroupReadableOnlyExecutor(HealthyExecutor):
        def run(self, command: str) -> CommandResult:
            if "systemctl show" in command and "CPUQuotaPerSecUSec" in command:
                return CommandResult(command=command, exit_code=0, stdout="4s\n", stderr="")
            if "systemctl show" in command and "CPUQuota" in command:
                return CommandResult(command=command, exit_code=0, stdout="\n", stderr="")
            return super().run(command)

    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="systemd.cgroup.cpu_quota_percent",
        parameter_name="cpu_quota_percent",
        domain="runtime",
        tuning_layer=tuning_layer_for_parameter_key("systemd.cgroup.cpu_quota_percent"),
        proposed_value="400",
        source=CandidateSource.SYSTEMD_CGROUP_CONTROL,
        apply_mode=ApplyMode.RESTART,
        rationale="Raise unit CPUQuota.",
    )
    applied_change = AppliedChange(
        hypothesis=hypothesis,
        target_path="nginx.service:CPUQuota",
        previous_value="15",
        applied_value="400",
        apply_mode=ApplyMode.RESTART,
        apply_command="systemctl set-property ...",
        rollback_command="systemctl set-property ...",
    )

    result = HealthValidator().validate(
        context=build_tune_context(),
        applied_change=applied_change,
        executor=CgroupReadableOnlyExecutor(),
    )

    assert result.healthy is True
    assert any(
        check.name == "effective_value" and check.detail == "observed=400%" for check in result.checks
    )
