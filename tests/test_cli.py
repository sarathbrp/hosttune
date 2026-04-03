from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseline.domain.models import BaselineResult
from preflight.infrastructure.config_loader import LoadedConfig
from onboard.domain.models import OnboardResult
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from onboard.domain.models import CompatibilityReport
from preflight import cli
from preflight.domain.models import (
    BenchmarkResult,
    CapabilityFlag,
    CapabilityMap,
    CommandResult,
    CpuInfo,
    DiscoverySnapshot,
    KernelInfo,
    LocalTargetConfig,
    MemoryInfo,
    NetworkInfo,
    PlatformInfo,
    SshTargetConfig,
    StorageInfo,
)
from snapshot.domain.models import SnapshotResult

from tests.onboard.test_service_definition_validator import build_valid_definition


class FakeExecutor:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        return CommandResult(command=command, exit_code=0, stdout="123.5", stderr="")


def test_shell_benchmark_runner_normalizes_output() -> None:
    runner = cli.ShellBenchmarkRunner("printf '123.5'")
    executor = FakeExecutor()

    result = runner.run(executor)

    assert result.primary_metric_name == "score"
    assert result.primary_metric_value == 123.5
    assert executor.commands == ["printf '123.5'"]


def test_build_executor_returns_local_executor() -> None:
    executor = cli.build_executor(LocalTargetConfig())

    assert executor.__class__.__name__ == "LocalCommandExecutor"


def test_build_executor_returns_ssh_executor() -> None:
    executor = cli.build_executor(
        SshTargetConfig(host="example", user="tester", private_key_path=Path("/tmp/id_rsa"))
    )

    assert executor.__class__.__name__ == "SshCommandExecutor"


def test_build_discovery_runner_builds_typed_probes() -> None:
    runner = cli.build_discovery_runner(None)

    assert runner.platform_probe.name == "platform"
    assert runner.cpu_probe.name == "cpu"
    assert runner.memory_probe.name == "memory"
    assert runner.kernel_probe.name == "kernel"
    assert runner.network_probe.name == "network"
    assert runner.storage_probe.name == "storage"


def test_build_instance_exposes_preflight_slot() -> None:
    instance = cli.build_instance()

    assert instance.preflight is None
    assert instance.onboard is None
    assert instance.snapshot is None
    assert instance.baseline is None


def test_main_renders_snapshot(monkeypatch, capsys, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("target:\n  mode: local\n", encoding="utf-8")

    class FakeRunner:
        def run(self, executor, target, policy):  # type: ignore[no-untyped-def]
            _ = executor
            return DiscoverySnapshot(
                target=target,
                policy=policy,
                platform_summary="bare_metal_linux",
                platform=PlatformInfo(
                    hostname="node-a",
                    operating_system="RHEL",
                    kernel_version="5.14.0",
                    virtualization_type="none",
                    is_container=False,
                ),
                cpu=CpuInfo(
                    architecture="x86_64",
                    logical_cores=16,
                    threads_per_core=2,
                    cores_per_socket=8,
                    sockets=1,
                    numa_nodes=2,
                    hyperthreading_enabled=True,
                ),
                memory=MemoryInfo(
                    total_memory_kib=1024,
                    swap_total_kib=512,
                    hugepages_total=8,
                    transparent_hugepages_mode="always [madvise] never",
                ),
                kernel=KernelInfo(
                    sysctl_writable=True,
                    selinux_mode="Permissive",
                    tuned_profile="throughput-performance",
                ),
                network=NetworkInfo(
                    interface_name="eth0",
                    driver_name="ixgbe",
                    firmware_version="1.0.0",
                    rx_ring_current=512,
                    rx_ring_max=4096,
                    tx_ring_current=512,
                    tx_ring_max=4096,
                    combined_queues=8,
                    ring_buffer_tuning_supported=True,
                ),
                storage=StorageInfo(
                    device_name="sda",
                    device_type="ssd",
                    scheduler="[mq-deadline] none",
                    scheduler_meaningful=True,
                ),
                capability_map=CapabilityMap(
                    flags=(CapabilityFlag(name="irq_affinity", available=True, detail="supported"),)
                ),
            )

    @dataclass(frozen=True)
    class FakeOnboardState:
        onboard: OnboardResult
        snapshot: SnapshotResult
        baseline: BaselineResult

    fake_definition = ServiceDefinitionValidator().validate(build_valid_definition())
    fake_state = FakeOnboardState(
        onboard=OnboardResult(
            service_name="nginx",
            service=fake_definition,
            compatibility=CompatibilityReport(compatible=True, findings=()),
        ),
        snapshot=SnapshotResult(
            service_name="nginx",
            snapshot_directory="/var/tmp/hosttune/snapshots/nginx",
            captured_paths=("/etc/nginx/nginx.conf",),
            runtime_state_output="nginx -T",
            process_state={"pid_file": "1234"},
            restore_sequence=("systemctl restart nginx",),
        ),
        baseline=BaselineResult(
            service_name="nginx",
            benchmark_command="printf '1.0'",
            benchmark_result=BenchmarkResult(
                command="printf '1.0'",
                exit_code=0,
                primary_metric_name="score",
                primary_metric_value=1.0,
                raw_output="1.0",
            ),
            expected_variance=0.05,
            warmup_seconds=10,
            guardrail_metrics=("p95_latency",),
        ),
    )

    class FakeInstance:
        def __init__(self) -> None:
            self.config_loader = type(
                "FakeConfigLoader",
                (),
                {
                    "load": lambda _self, _path: LoadedConfig(
                        target=LocalTargetConfig(),
                        policy=None,  # type: ignore[arg-type]
                        service_name="nginx",
                        benchmark_command="printf '1.0'",
                    )
                },
            )()
            self.preflight = None
            self.onboard = None
            self.snapshot = None
            self.baseline = None

        def load_preflight(self, _config_path: Path) -> DiscoverySnapshot:
            self.preflight = FakeRunner().run(None, LocalTargetConfig(), None)
            return self.preflight

        def load_onboard(self, _config_path: Path) -> OnboardResult:
            self.onboard = fake_state.onboard
            return self.onboard

        def load_snapshot(self, _config_path: Path) -> SnapshotResult:
            self.snapshot = fake_state.snapshot
            return self.snapshot

        def load_baseline(self, _config_path: Path) -> BaselineResult:
            self.baseline = fake_state.baseline
            return self.baseline

    monkeypatch.setattr(cli, "build_instance", lambda: FakeInstance())
    monkeypatch.setattr("sys.argv", ["preflight", str(config_path)])

    exit_code = cli.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"platform_summary": "bare_metal_linux"' in output
    assert '"hostname": "node-a"' in output
    assert '"service_name": "nginx"' in output
    assert '"snapshot_directory": "/var/tmp/hosttune/snapshots/nginx"' in output
    assert '"benchmark_command": "printf \'1.0\'"' in output


def test_main_builds_benchmark_runner_when_command_present(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
target:
  mode: local
benchmark:
  command: "printf '1.0'"
""".strip(),
        encoding="utf-8",
    )

    class FakeRunner:
        def __init__(self) -> None:
            self.benchmark_runner_type = None

        def run(self, executor, target, policy):  # type: ignore[no-untyped-def]
            _ = executor
            return DiscoverySnapshot(
                target=target,
                policy=policy,
                platform_summary="bare_metal_linux",
                platform=PlatformInfo(
                    hostname="node-a",
                    operating_system="RHEL",
                    kernel_version="5.14.0",
                    virtualization_type="none",
                    is_container=False,
                ),
                cpu=CpuInfo(
                    architecture="x86_64",
                    logical_cores=16,
                    threads_per_core=2,
                    cores_per_socket=8,
                    sockets=1,
                    numa_nodes=2,
                    hyperthreading_enabled=True,
                ),
                memory=MemoryInfo(
                    total_memory_kib=1024,
                    swap_total_kib=512,
                    hugepages_total=8,
                    transparent_hugepages_mode="always [madvise] never",
                ),
                kernel=KernelInfo(
                    sysctl_writable=True,
                    selinux_mode="Permissive",
                    tuned_profile="throughput-performance",
                ),
                network=NetworkInfo(
                    interface_name="eth0",
                    driver_name="ixgbe",
                    firmware_version="1.0.0",
                    rx_ring_current=512,
                    rx_ring_max=4096,
                    tx_ring_current=512,
                    tx_ring_max=4096,
                    combined_queues=8,
                    ring_buffer_tuning_supported=True,
                ),
                storage=StorageInfo(
                    device_name="sda",
                    device_type="ssd",
                    scheduler="[mq-deadline] none",
                    scheduler_meaningful=True,
                ),
                capability_map=CapabilityMap(flags=()),
            )

    fake_runner = FakeRunner()
    fake_definition = ServiceDefinitionValidator().validate(build_valid_definition())
    fake_onboard = OnboardResult(
        service_name="nginx",
        service=fake_definition,
        compatibility=CompatibilityReport(compatible=True, findings=()),
    )
    fake_snapshot = SnapshotResult(
        service_name="nginx",
        snapshot_directory="/var/tmp/hosttune/snapshots/nginx",
        captured_paths=("/etc/nginx/nginx.conf",),
        runtime_state_output="nginx -T",
        process_state={"pid_file": "1234"},
        restore_sequence=("systemctl restart nginx",),
    )
    fake_baseline = BaselineResult(
        service_name="nginx",
        benchmark_command="printf '1.0'",
        benchmark_result=BenchmarkResult(
            command="printf '1.0'",
            exit_code=0,
            primary_metric_name="score",
            primary_metric_value=1.0,
            raw_output="1.0",
        ),
        expected_variance=0.05,
        warmup_seconds=10,
        guardrail_metrics=("p95_latency",),
    )

    class FakeInstance:
        def __init__(self) -> None:
            self.config_loader = type(
                "FakeConfigLoader",
                (),
                {
                    "load": lambda _self, _path: LoadedConfig(
                        target=LocalTargetConfig(),
                        policy=None,  # type: ignore[arg-type]
                        service_name="nginx",
                        benchmark_command="printf '1.0'",
                    )
                },
            )()
            self.preflight = None
            self.onboard = None
            self.snapshot = None
            self.baseline = None

        def load_preflight(self, _config_path: Path) -> DiscoverySnapshot:
            fake_runner.benchmark_runner_type = "ShellBenchmarkRunner"
            self.preflight = fake_runner.run(None, LocalTargetConfig(), None)
            return self.preflight

        def load_onboard(self, _config_path: Path) -> OnboardResult:
            self.onboard = fake_onboard
            return self.onboard

        def load_snapshot(self, _config_path: Path) -> SnapshotResult:
            self.snapshot = fake_snapshot
            return self.snapshot

        def load_baseline(self, _config_path: Path) -> BaselineResult:
            self.baseline = fake_baseline
            return self.baseline

    monkeypatch.setattr(cli, "build_instance", lambda: FakeInstance())
    monkeypatch.setattr("sys.argv", ["preflight", str(config_path)])

    exit_code = cli.main()
    _ = capsys.readouterr()

    assert exit_code == 0
    assert fake_runner.benchmark_runner_type == "ShellBenchmarkRunner"
