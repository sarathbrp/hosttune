import subprocess
import tempfile
from pathlib import Path

from preflight.domain.models import CommandResult
from tune.application.apply_coordinator import (
    ApplyCoordinator,
    NetworkRingApplier,
    NginxDirectiveApplier,
    PrlimitApplier,
    SysctlApplier,
    SystemdUnitLimitApplier,
)
from tune.domain.hypothesis_models import CandidateSource, TunePhase, TuningHypothesis
from tune.domain.tuning_layer import tuning_layer_for_parameter_key
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
        if "ethtool -g" in command:
            return CommandResult(command=command, exit_code=0, stdout="511", stderr="")
        if command.startswith("cat "):
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="12345\n",
                stderr="",
            )
        if "awk " in command and "/proc/" in command and "limits" in command:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="8192 1048576\n",
                stderr="",
            )
        if command.startswith("prlimit "):
            return CommandResult(command=command, exit_code=0, stdout="", stderr="")
        if "systemctl show" in command and "LimitNPROC" in command:
            return CommandResult(command=command, exit_code=0, stdout="1000\n", stderr="")
        if "systemctl set-property" in command:
            return CommandResult(command=command, exit_code=0, stdout="", stderr="")
        return CommandResult(command=command, exit_code=0, stdout="", stderr="")


class MissingDirectiveExecutor(FakeExecutor):
    def run(self, command: str) -> CommandResult:
        if command.startswith("grep -E") and "multi_accept" in command:
            self.commands.append(command)
            return CommandResult(command=command, exit_code=1, stdout="", stderr="")
        return super().run(command)


def test_sysctl_applier_builds_apply_and_rollback() -> None:
    context = build_tune_context()
    executor = FakeExecutor()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="sysctl.net.core.somaxconn",
        parameter_name="net.core.somaxconn",
        domain="kernel_sysctl",
        tuning_layer=tuning_layer_for_parameter_key("sysctl.net.core.somaxconn"),
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
        tuning_layer=tuning_layer_for_parameter_key("service.directive.worker_processes"),
        proposed_value="56",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=ApplyMode.RELOAD,
        rationale="Reduce workers to a balanced subset of CPUs.",
    )

    applied = NginxDirectiveApplier().apply(context, hypothesis, executor)

    assert applied.target_path == "/etc/nginx/nginx.conf"
    assert applied.previous_value == "112"
    assert applied.applied_value == "56"
    # apply_command = python3 edit && systemctl reload nginx
    assert applied.apply_command.startswith("python3 -c ")
    assert applied.apply_command.endswith("&& systemctl reload nginx")
    # rollback_command = python3 restore && systemctl reload nginx
    assert applied.rollback_command.startswith("python3 -c ")
    assert "worker_processes 112" in applied.rollback_command
    assert applied.rollback_command.endswith("&& systemctl reload nginx")


def test_nginx_directive_applier_inserts_missing_events_directive() -> None:
    context = build_tune_context()
    executor = MissingDirectiveExecutor()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="service.directive.multi_accept",
        parameter_name="multi_accept",
        domain="service_config",
        tuning_layer=tuning_layer_for_parameter_key("service.directive.multi_accept"),
        proposed_value="on",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=ApplyMode.RELOAD,
        rationale="Enable multi_accept for higher connection accept throughput.",
    )

    applied = NginxDirectiveApplier().apply(context, hypothesis, executor)

    assert applied.previous_value == "__absent__"
    assert applied.applied_value == "on"
    assert applied.apply_command.startswith("python3 -c ")
    assert "multi_accept" in applied.apply_command
    assert applied.rollback_command.startswith("python3 -c ")
    assert "multi_accept" in applied.rollback_command


def test_nginx_insert_command_executes_for_worker_cpu_affinity() -> None:
    config_text = """user nginx;
worker_processes auto;

events {
    worker_connections 1024;
}

http {
    sendfile on;
}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "nginx.conf"
        config_path.write_text(config_text)
        command = NginxDirectiveApplier()._build_insert_command(
            str(config_path),
            "worker_cpu_affinity",
            "auto",
        )

        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
        updated = config_path.read_text()
        assert "worker_cpu_affinity auto;\n" in updated
        assert updated.index("worker_cpu_affinity auto;") < updated.index("events {")


def test_apply_coordinator_routes_by_parameter_prefix() -> None:
    context = build_tune_context()
    executor = FakeExecutor()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="sysctl.net.core.somaxconn",
        parameter_name="net.core.somaxconn",
        domain="kernel_sysctl",
        tuning_layer=tuning_layer_for_parameter_key("sysctl.net.core.somaxconn"),
        proposed_value="65535",
        source=CandidateSource.SERVICE_SYSCTL,
        apply_mode=ApplyMode.RELOAD,
        rationale="Increase listen queue capacity.",
    )

    applied = ApplyCoordinator(
        service_directive_applier=NginxDirectiveApplier(),
        sysctl_applier=SysctlApplier(),
        network_ring_applier=NetworkRingApplier(),
        runtime_limit_applier=PrlimitApplier(),
        systemd_unit_limit_applier=SystemdUnitLimitApplier(),
    ).apply(context, hypothesis, executor)

    assert applied.hypothesis.parameter_key == "sysctl.net.core.somaxconn"


def test_prlimit_applier_applies_nofile_soft() -> None:
    context = build_tune_context()
    executor = FakeExecutor()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="runtime.prlimit.nofile_soft",
        parameter_name="nofile_soft",
        domain="runtime",
        tuning_layer=tuning_layer_for_parameter_key("runtime.prlimit.nofile_soft"),
        proposed_value="65536",
        source=CandidateSource.RUNTIME_PRLIMIT,
        apply_mode=ApplyMode.RELOAD,
        rationale="Raise process soft NOFILE for the nginx master.",
    )

    applied = PrlimitApplier().apply(context, hypothesis, executor)

    assert applied.target_path == "pid=12345:nofile"
    assert applied.previous_value == "8192"
    assert applied.applied_value == "65536"
    assert applied.apply_command == "prlimit --pid 12345 --nofile=65536:1048576"
    assert applied.rollback_command == "prlimit --pid 12345 --nofile=8192:1048576"


def test_apply_coordinator_routes_runtime_prlimit() -> None:
    context = build_tune_context()
    executor = FakeExecutor()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="runtime.prlimit.nofile_soft",
        parameter_name="nofile_soft",
        domain="runtime",
        tuning_layer=tuning_layer_for_parameter_key("runtime.prlimit.nofile_soft"),
        proposed_value="65536",
        source=CandidateSource.RUNTIME_PRLIMIT,
        apply_mode=ApplyMode.RELOAD,
        rationale="Raise NOFILE soft limit.",
    )

    applied = ApplyCoordinator(
        service_directive_applier=NginxDirectiveApplier(),
        sysctl_applier=SysctlApplier(),
        network_ring_applier=NetworkRingApplier(),
        runtime_limit_applier=PrlimitApplier(),
        systemd_unit_limit_applier=SystemdUnitLimitApplier(),
    ).apply(context, hypothesis, executor)

    assert applied.hypothesis.parameter_key == "runtime.prlimit.nofile_soft"


def test_systemd_unit_limit_applier_sets_property_and_restarts() -> None:
    context = build_tune_context()
    executor = FakeExecutor()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="systemd.unit.limit_nproc",
        parameter_name="limit_nproc",
        domain="runtime",
        tuning_layer=tuning_layer_for_parameter_key("systemd.unit.limit_nproc"),
        proposed_value="8000",
        source=CandidateSource.SYSTEMD_UNIT_LIMIT,
        apply_mode=ApplyMode.RESTART,
        rationale="Raise unit NPROC for worker fan-out.",
    )

    applied = SystemdUnitLimitApplier().apply(context, hypothesis, executor)

    assert applied.target_path == "nginx.service:LimitNPROC"
    assert applied.previous_value == "1000"
    assert applied.applied_value == "8000"
    assert "systemctl set-property" in applied.apply_command
    assert "LimitNPROC=8000" in applied.apply_command
    assert "systemctl daemon-reload" in applied.apply_command
    assert "systemctl restart nginx" in applied.apply_command


def test_apply_coordinator_routes_systemd_unit_limit() -> None:
    context = build_tune_context()
    executor = FakeExecutor()
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="systemd.unit.limit_nproc",
        parameter_name="limit_nproc",
        domain="runtime",
        tuning_layer=tuning_layer_for_parameter_key("systemd.unit.limit_nproc"),
        proposed_value="8000",
        source=CandidateSource.SYSTEMD_UNIT_LIMIT,
        apply_mode=ApplyMode.RESTART,
        rationale="Raise NPROC.",
    )

    applied = ApplyCoordinator(
        service_directive_applier=NginxDirectiveApplier(),
        sysctl_applier=SysctlApplier(),
        network_ring_applier=NetworkRingApplier(),
        runtime_limit_applier=PrlimitApplier(),
        systemd_unit_limit_applier=SystemdUnitLimitApplier(),
    ).apply(context, hypothesis, executor)

    assert applied.hypothesis.parameter_key == "systemd.unit.limit_nproc"


def test_network_ring_applier_builds_apply_and_rollback() -> None:
    context = build_tune_context()
    executor = FakeExecutor()
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

    applied = NetworkRingApplier().apply(context, hypothesis, executor)

    assert applied.target_path == "eth0:rx"
    assert applied.previous_value == "511"
    assert applied.applied_value == "1024"
    assert applied.apply_command == "ethtool -G eth0 rx 1024"
    assert applied.rollback_command == "ethtool -G eth0 rx 511"
