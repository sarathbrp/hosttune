from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseline.domain.models import BaselineResult, BenchmarkConfig, WorkloadBenchmarkResult
from onboard.domain.models import CompatibilityReport, OnboardResult
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight import cli
from preflight.domain.models import (
    BenchmarkResult,
    CapabilityFlag,
    CapabilityMap,
    CgroupInfo,
    CommandResult,
    CpuInfo,
    DiscoverySnapshot,
    EngagementPolicy,
    IrqInfo,
    KernelInfo,
    LocalTargetConfig,
    MemoryInfo,
    NetworkInfo,
    PlatformInfo,
    StorageInfo,
)
from preflight.infrastructure.config_loader import LoadedConfig
from snapshot.domain.models import SnapshotResult
from tune.domain.tune_state import TuneState

from tests.onboard.test_service_definition_validator import build_valid_definition


class FakeExecutor:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        return CommandResult(command=command, exit_code=0, stdout="123.5", stderr="")


def build_snapshot() -> DiscoverySnapshot:
    return DiscoverySnapshot(
        target=LocalTargetConfig(),
        policy=EngagementPolicy(
            allow_reload=True,
            allow_restart=False,
            allow_reboot=False,
            rollback_required=True,
            max_iterations=3,
            benchmark_stability_threshold=0.1,
        ),
        platform_summary="bare_metal_linux",
        platform=PlatformInfo(
            hostname="node-a",
            operating_system="RHEL",
            kernel_version="5.14.0",
            virtualization_type="none",
            is_container=False,
        ),
        cpu=CpuInfo("x86_64", 16, 2, 8, 1, 2, True),
        memory=MemoryInfo(1024, 512, 8, "always [madvise] never"),
        kernel=KernelInfo(True, "Permissive", "throughput-performance"),
        network=NetworkInfo("eth0", "ixgbe", "1.0.0", 512, 4096, 512, 4096, 8, True),
        storage=StorageInfo("sda", "ssd", "[mq-deadline] none", True),
        irq=IrqInfo(irqbalance_active=False, nic_irq_cpu_summary="unknown"),
        cgroup=CgroupInfo(cgroup_version="unknown", cpu_controller_available=False, memory_controller_available=False),
        capability_map=CapabilityMap(
            flags=(CapabilityFlag(name="irq_affinity", available=True, detail="supported"),)
        ),
    )


def build_baseline() -> BaselineResult:
    return BaselineResult(
        service_name="nginx",
        benchmark_command="TARGET_HOST=10.1.90.178 /root/hackathon-tools/benchmark.sh hosttune",
        benchmark_target="10.1.90.178",
        workload_results=(
            WorkloadBenchmarkResult(
                workload_name="homepage",
                result_path="/root/hackathon-results/hosttune_homepage.json",
                requests_per_second=1234.5,
                total_requests=9999,
                average_latency_ms=4.2,
            ),
        ),
        expected_variance=0.05,
        warmup_seconds=10,
        guardrail_metrics=("p95_latency",),
        comparison_output="homepage improved by 3%",
    )


def build_tune_state() -> TuneState:
    state = TuneState.initialize(2)
    state.total_iterations = 1
    return state


def test_shell_benchmark_runner_normalizes_output() -> None:
    runner = cli.ShellBenchmarkRunner("printf '123.5'")
    executor = FakeExecutor()

    result = runner.run(executor)

    assert result == BenchmarkResult(
        command="printf '123.5'",
        exit_code=0,
        primary_metric_name="score",
        primary_metric_value=123.5,
        raw_output="123.5",
    )
    assert executor.commands == ["printf '123.5'"]


def test_build_executor_returns_local_executor() -> None:
    executor = cli.build_executor(LocalTargetConfig())

    assert executor.__class__.__name__ == "LocalCommandExecutor"


def test_build_discovery_runner_builds_typed_probes() -> None:
    runner = cli.build_discovery_runner(None)

    assert runner.platform_probe.name == "platform"
    assert runner.cpu_probe.name == "cpu"
    assert runner.memory_probe.name == "memory"
    assert runner.kernel_probe.name == "kernel"
    assert runner.network_probe.name == "network"
    assert runner.storage_probe.name == "storage"


def test_build_instance_exposes_stage_slots() -> None:
    instance = cli.build_instance()

    assert instance.preflight is None
    assert instance.onboard is None
    assert instance.snapshot is None
    assert instance.baseline is None


def test_build_baseline_runner_uses_benchmark_config() -> None:
    config = BenchmarkConfig(
        runner_target=LocalTargetConfig(),
        contestant_name="hosttune",
        script_path="/root/hackathon-tools/benchmark.sh",
        results_directory="/root/hackathon-results",
        workloads=("homepage",),
        compare_script_path="/root/hackathon-tools/compare-results.sh",
    )

    runner = cli.build_baseline_runner(config)

    assert runner.benchmark_config == config


def test_build_instance_verbose_enables_logger() -> None:
    instance = cli.build_instance(verbose=True)

    assert instance.logger.__class__.__name__ == "VerboseExecutionLogger"


def test_build_instance_debug_enables_debug_logger() -> None:
    instance = cli.build_instance(debug=True)

    assert instance.logger.__class__.__name__ == "DebugExecutionLogger"


def test_main_renders_combined_runtime(monkeypatch, capsys, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("target:\n  mode: local\n", encoding="utf-8")
    fake_definition = ServiceDefinitionValidator().validate(build_valid_definition())

    @dataclass
    class FakeInstance:
        config_loader: object
        preflight: DiscoverySnapshot | None = None
        onboard: OnboardResult | None = None
        snapshot: SnapshotResult | None = None
        baseline: BaselineResult | None = None
        tune: TuneState | None = None

        def load_preflight(self, _config_path: Path) -> DiscoverySnapshot:
            self.preflight = build_snapshot()
            return self.preflight

        def load_onboard(self, _config_path: Path) -> OnboardResult:
            self.onboard = OnboardResult(
                service_name="nginx",
                service=fake_definition,
                compatibility=CompatibilityReport(compatible=True, findings=()),
            )
            return self.onboard

        def load_snapshot(self, _config_path: Path) -> SnapshotResult:
            self.snapshot = SnapshotResult(
                service_name="nginx",
                snapshot_directory="/var/tmp/hosttune/snapshots/nginx",
                captured_paths=("/etc/nginx/nginx.conf",),
                runtime_state_output="nginx -T",
                process_state={"pid_file": "1234"},
                restore_sequence=("systemctl restart nginx",),
            )
            return self.snapshot

        def load_baseline(self, _config_path: Path) -> BaselineResult:
            self.baseline = build_baseline()
            return self.baseline

        def run_tune(self, _config_path: Path, _tune_engine: object) -> TuneState:
            self.tune = build_tune_state()
            return self.tune

        def build_tune_context(self) -> object:
            return object()

    fake_config_loader = type(
        "FakeConfigLoader",
        (),
        {
            "load": lambda _self, _path: LoadedConfig(
                target=LocalTargetConfig(),
                policy=build_snapshot().policy,
                service_name="nginx",
                benchmark_config=BenchmarkConfig(
                    runner_target=LocalTargetConfig(),
                    contestant_name="hosttune",
                    script_path="/root/hackathon-tools/benchmark.sh",
                    results_directory="/root/hackathon-results",
                    workloads=("homepage",),
                    compare_script_path="/root/hackathon-tools/compare-results.sh",
                ),
            )
        },
    )()
    monkeypatch.setattr(
        cli,
        "build_instance",
        lambda verbose=False, debug=False: FakeInstance(config_loader=fake_config_loader),
    )
    monkeypatch.setattr(cli, "build_tune_engine", lambda logger=None: object())
    monkeypatch.setattr("sys.argv", ["preflight", str(config_path)])

    exit_code = cli.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Preflight" in output
    assert "Onboard" in output
    assert "Snapshot" in output
    assert "Baseline" in output
    assert "Tune" in output
    assert "Target: 10.1.90.178" in output


def test_main_returns_clean_error_for_tune_failure(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("target:\n  mode: local\n", encoding="utf-8")
    fake_definition = ServiceDefinitionValidator().validate(build_valid_definition())

    @dataclass
    class FakeInstance:
        config_loader: object
        preflight: DiscoverySnapshot | None = None
        onboard: OnboardResult | None = None
        snapshot: SnapshotResult | None = None
        baseline: BaselineResult | None = None

        def load_preflight(self, _config_path: Path) -> DiscoverySnapshot:
            self.preflight = build_snapshot()
            return self.preflight

        def load_onboard(self, _config_path: Path) -> OnboardResult:
            self.onboard = OnboardResult(
                service_name="nginx",
                service=fake_definition,
                compatibility=CompatibilityReport(compatible=True, findings=()),
            )
            return self.onboard

        def load_snapshot(self, _config_path: Path) -> SnapshotResult:
            self.snapshot = SnapshotResult(
                service_name="nginx",
                snapshot_directory="/var/tmp/hosttune/snapshots/nginx",
                captured_paths=("/etc/nginx/nginx.conf",),
                runtime_state_output="nginx -T",
                process_state={"pid_file": "1234"},
                restore_sequence=("systemctl restart nginx",),
            )
            return self.snapshot

        def load_baseline(self, _config_path: Path) -> BaselineResult:
            self.baseline = build_baseline()
            return self.baseline

        def run_tune(self, _config_path: Path, _tune_engine: object) -> TuneState:
            raise ValueError("Pre-tune health gate failed: health_probe: status=500")

    fake_config_loader = type(
        "FakeConfigLoader",
        (),
        {
            "load": lambda _self, _path: LoadedConfig(
                target=LocalTargetConfig(),
                policy=build_snapshot().policy,
                service_name="nginx",
                benchmark_config=BenchmarkConfig(
                    runner_target=LocalTargetConfig(),
                    contestant_name="hosttune",
                    script_path="/root/hackathon-tools/benchmark.sh",
                    results_directory="/root/hackathon-results",
                    workloads=("homepage",),
                    compare_script_path="/root/hackathon-tools/compare-results.sh",
                ),
            )
        },
    )()
    monkeypatch.setattr(
        cli,
        "build_instance",
        lambda verbose=False, debug=False: FakeInstance(config_loader=fake_config_loader),
    )
    monkeypatch.setattr(cli, "build_tune_engine", lambda logger=None: object())
    monkeypatch.setattr("sys.argv", ["preflight", str(config_path)])

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "Error: Pre-tune health gate failed: health_probe: status=500" in captured.err


def test_main_defaults_to_config_yaml_and_accepts_short_verbose(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("target:\n  mode: local\n", encoding="utf-8")

    class FakeInstance:
        def __init__(self) -> None:
            self.config_loader = type(
                "FakeConfigLoader",
                (),
                {
                    "load": lambda _self, _path: LoadedConfig(
                        target=LocalTargetConfig(),
                        policy=build_snapshot().policy,
                        service_name="nginx",
                        benchmark_config=None,
                    )
                },
            )()

        def load_preflight(self, _config_path: Path) -> DiscoverySnapshot:
            return build_snapshot()

        def load_onboard(self, _config_path: Path) -> OnboardResult:
            return OnboardResult(
                service_name="nginx",
                service=ServiceDefinitionValidator().validate(build_valid_definition()),
                compatibility=CompatibilityReport(compatible=True, findings=()),
            )

        def load_snapshot(self, _config_path: Path) -> SnapshotResult:
            return SnapshotResult(
                service_name="nginx",
                snapshot_directory="/var/tmp/hosttune/snapshots/nginx",
                captured_paths=("/etc/nginx/nginx.conf",),
                runtime_state_output=None,
                process_state={},
                restore_sequence=(),
            )

    called: dict[str, object] = {}

    def fake_build_instance(verbose: bool = False, debug: bool = False) -> FakeInstance:
        called["verbose"] = verbose
        called["debug"] = debug
        return FakeInstance()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_instance", fake_build_instance)
    monkeypatch.setattr("sys.argv", ["preflight", "-v"])

    exit_code = cli.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert called["verbose"] is True
    assert called["debug"] is False
    assert "Preflight" in output


def test_main_debug_enables_debug_instance(monkeypatch, capsys, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("target:\n  mode: local\n", encoding="utf-8")

    class FakeInstance:
        def __init__(self) -> None:
            self.config_loader = type(
                "FakeConfigLoader",
                (),
                {
                    "load": lambda _self, _path: LoadedConfig(
                        target=LocalTargetConfig(),
                        policy=build_snapshot().policy,
                        service_name="nginx",
                        benchmark_config=None,
                    )
                },
            )()

        def load_preflight(self, _config_path: Path) -> DiscoverySnapshot:
            return build_snapshot()

        def load_onboard(self, _config_path: Path) -> OnboardResult:
            return OnboardResult(
                service_name="nginx",
                service=ServiceDefinitionValidator().validate(build_valid_definition()),
                compatibility=CompatibilityReport(compatible=True, findings=()),
            )

        def load_snapshot(self, _config_path: Path) -> SnapshotResult:
            return SnapshotResult(
                service_name="nginx",
                snapshot_directory="/var/tmp/hosttune/snapshots/nginx",
                captured_paths=("/etc/nginx/nginx.conf",),
                runtime_state_output=None,
                process_state={},
                restore_sequence=(),
            )

    called: dict[str, object] = {}

    def fake_build_instance(verbose: bool = False, debug: bool = False) -> FakeInstance:
        called["verbose"] = verbose
        called["debug"] = debug
        return FakeInstance()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_instance", fake_build_instance)
    monkeypatch.setattr("sys.argv", ["preflight", "--debug"])

    exit_code = cli.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert called["verbose"] is True
    assert called["debug"] is True
    assert "Preflight" in output
