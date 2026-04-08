from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from baseline.domain.models import BaselineResult, BenchmarkConfig, WorkloadBenchmarkResult
from host_profile.domain.models import (
    HostPerformanceHierarchy,
    HostPerformanceHierarchyGroup,
    HostPerformanceHierarchyParameter,
    HostProfile,
    HostProfileIdentity,
    HostTunableSurface,
)
from onboard.domain.models import CompatibilityReport, OnboardResult
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.application.hosttune_instance import HostTuneInstance
from preflight.domain.kernel_sysctl_profile import PREFLIGHT_SYSCTL_KEYS
from preflight.domain.models import (
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
from preflight.infrastructure.config_loader import ConfigLoader, LoadedConfig
from preflight.infrastructure.runtime_artifact_store import RuntimeArtifactStore
from snapshot.domain.models import SnapshotResult
from tune.domain.tune_context import TuneContext
from tune.domain.tune_state import TuneState

from tests.onboard.test_service_definition_validator import build_valid_definition


class OnboardSysctlEnrichExecutor:
    """load_onboard runs sysctl reads for contract-only relevant_sysctls (e.g. ip_local_port_range)."""

    def run(self, command: str) -> CommandResult:
        if (
            "for k in " in command
            and "sysctl -n" in command
            and "ip_local_port_range" in command
        ):
            return CommandResult(
                command,
                0,
                "net.ipv4.ip_local_port_range=32768\t60999\n",
                "",
            )
        return CommandResult(command, 0, "", "")


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
                hugepages_total=0,
                transparent_hugepages_mode="[always] madvise never",
            ),
            kernel=KernelInfo(
                sysctl_writable=True,
                selinux_mode="Enforcing",
                tuned_profile="unknown",
            ),
            network=NetworkInfo(
                interface_name="eth0",
                driver_name="ixgbe",
                firmware_version="1.2.3",
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
            irq=IrqInfo(irqbalance_active=True, nic_irq_cpu_summary="0-7"),
            cgroup=CgroupInfo(
                cgroup_version="v2",
                cpu_controller_available=True,
                memory_controller_available=True,
            ),
            capability_map=CapabilityMap(flags=()),
        )


class FakeOnboardRunner:
    def run(self, service_name, preflight, executor):  # type: ignore[no-untyped-def]
        definition = ServiceDefinitionValidator().validate(build_valid_definition())
        _ = executor
        return OnboardResult(
            service_name=service_name,
            service=definition,
            compatibility=CompatibilityReport(compatible=True, findings=()),
        )


class FakeConfigLoader(ConfigLoader):
    def load(self, path: Path) -> LoadedConfig:
        _ = path
        return LoadedConfig(
            target=LocalTargetConfig(),
            policy=EngagementPolicy(
                allow_reload=False,
                allow_restart=False,
                allow_reboot=False,
                rollback_required=True,
                max_iterations=10,
                benchmark_stability_threshold=0.1,
            ),
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


def test_instance_stores_preflight_snapshot(tmp_path: Path) -> None:
    artifact_store = RuntimeArtifactStore(base_directory=tmp_path / "artifacts")
    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_config: None,  # type: ignore[arg-type]
        executor_factory=lambda target: object(),  # type: ignore[arg-type]
        artifact_store=artifact_store,
    )

    snapshot = instance.load_preflight(Path("config.yaml"))

    assert instance.preflight is snapshot
    assert instance.preflight is not None
    assert instance.preflight.cpu.logical_cores == 16
    assert instance.preflight.storage.device_name == "sda"
    assert instance.artifacts is not None
    assert "preflight" in instance.artifacts.stage_files


def test_instance_stores_onboard_result(tmp_path: Path) -> None:
    class FakeOnboardRunner:
        def run(self, service_name, preflight, executor):  # type: ignore[no-untyped-def]
            definition = ServiceDefinitionValidator().validate(build_valid_definition())
            _ = preflight
            _ = executor
            return OnboardResult(
                service_name=service_name,
                service=definition,
                compatibility=CompatibilityReport(compatible=True, findings=()),
            )

    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_config: None,  # type: ignore[arg-type]
        executor_factory=lambda _target: OnboardSysctlEnrichExecutor(),
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )

    instance.load_preflight(Path("config.yaml"))
    result = instance.load_onboard(Path("config.yaml"))

    assert instance.onboard is result
    assert instance.onboard is not None
    assert instance.onboard.service_name == "nginx"


def test_load_onboard_enriches_preflight_sysctl_profile_with_contract_only_keys(
    tmp_path: Path,
) -> None:
    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_config: None,  # type: ignore[arg-type]
        executor_factory=lambda _target: OnboardSysctlEnrichExecutor(),
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )
    instance.load_preflight(Path("config.yaml"))
    assert instance.preflight is not None
    assert instance.preflight.kernel.sysctl_profile == ()
    instance.load_onboard(Path("config.yaml"))
    by_name = dict(instance.preflight.kernel.sysctl_profile)
    assert by_name["net.ipv4.ip_local_port_range"] == "32768\t60999"
    assert len(instance.preflight.kernel.sysctl_profile) == len(PREFLIGHT_SYSCTL_KEYS) + 1


def test_instance_stores_snapshot_and_baseline_results(tmp_path: Path) -> None:
    class FakeSnapshotRunner:
        def run(self, service, executor):  # type: ignore[no-untyped-def]
            _ = service
            _ = executor
            return SnapshotResult(
                service_name="nginx",
                snapshot_directory="/var/tmp/hosttune/snapshots/nginx",
                captured_paths=("/etc/nginx/nginx.conf",),
                runtime_state_output="nginx -T",
                process_state={"pid_file": "1234"},
                restore_sequence=("systemctl restart nginx",),
            )

    class FakeBaselineRunner:
        def run(self, service, executor, dut_target):  # type: ignore[no-untyped-def]
            _ = service
            _ = executor
            _ = dut_target
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

    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: FakeSnapshotRunner(),
        baseline_runner_factory=lambda benchmark_config: FakeBaselineRunner(),
        executor_factory=lambda _target: OnboardSysctlEnrichExecutor(),
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )

    instance.load_preflight(Path("config.yaml"))
    instance.load_onboard(Path("config.yaml"))
    snapshot_result = instance.load_snapshot(Path("config.yaml"))
    baseline_result = instance.load_baseline(Path("config.yaml"))

    assert instance.snapshot is snapshot_result
    assert instance.baseline is baseline_result
    assert instance.snapshot is not None
    assert instance.baseline is not None


def test_instance_writes_stage_jsonl_artifacts(tmp_path: Path) -> None:
    artifact_store = RuntimeArtifactStore(base_directory=tmp_path / "artifacts")
    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_config: None,  # type: ignore[arg-type]
        executor_factory=lambda _target: OnboardSysctlEnrichExecutor(),
        artifact_store=artifact_store,
    )

    instance.load_preflight(Path("config.yaml"))
    instance.load_onboard(Path("config.yaml"))

    assert instance.artifacts is not None
    preflight_file = instance.artifacts.stage_files["preflight"]
    onboard_file = instance.artifacts.stage_files["onboard"]

    preflight_lines = preflight_file.read_text(encoding="utf-8").splitlines()
    assert len(preflight_lines) == 2
    preflight_record = json.loads(preflight_lines[0])
    enriched_preflight = json.loads(preflight_lines[1])
    onboard_record = json.loads(onboard_file.read_text(encoding="utf-8").splitlines()[0])

    assert len(instance.artifacts.session_id) == RuntimeArtifactStore.SESSION_ID_LENGTH
    assert preflight_file.name == f"preflight_{instance.artifacts.session_id}.jsonl"
    assert onboard_file.name == f"onboard_{instance.artifacts.session_id}.jsonl"
    assert preflight_record["stage"] == "preflight"
    assert preflight_record["payload"]["platform_summary"] == "bare_metal_linux"
    assert enriched_preflight["stage"] == "preflight"
    profile = {pair[0]: pair[1] for pair in enriched_preflight["payload"]["kernel"]["sysctl_profile"]}
    assert "32768" in profile.get("net.ipv4.ip_local_port_range", "")
    assert onboard_record["stage"] == "onboard"
    assert onboard_record["payload"]["service_name"] == "nginx"
    assert "knowledge_base" in instance.artifacts.stage_files


def test_instance_records_stage_events_in_knowledge_base(tmp_path: Path) -> None:
    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_config: None,  # type: ignore[arg-type]
        executor_factory=lambda _target: OnboardSysctlEnrichExecutor(),
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )

    instance.load_preflight(Path("config.yaml"))
    instance.load_onboard(Path("config.yaml"))

    assert instance.artifacts is not None
    with sqlite3.connect(instance.artifacts.stage_files["knowledge_base"]) as connection:
        rows = connection.execute(
            "SELECT event_type FROM events WHERE run_id=? ORDER BY id ASC",
            (instance.artifacts.session_id,),
        ).fetchall()
    assert [row[0] for row in rows] == ["preflight_completed", "onboard_completed"]


def test_instance_builds_tune_context(tmp_path: Path) -> None:
    class FakeSnapshotRunner:
        def run(self, service, executor):  # type: ignore[no-untyped-def]
            _ = service
            _ = executor
            return SnapshotResult(
                service_name="nginx",
                snapshot_directory="/var/tmp/hosttune/snapshots/nginx",
                captured_paths=("/etc/nginx/nginx.conf",),
                runtime_state_output="nginx -T",
                process_state={"pid_file": "1234"},
                restore_sequence=("systemctl restart nginx",),
            )

    class FakeBaselineRunner:
        def run(self, service, executor, dut_target):  # type: ignore[no-untyped-def]
            _ = service
            _ = executor
            _ = dut_target
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

    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: FakeSnapshotRunner(),
        baseline_runner_factory=lambda benchmark_config: FakeBaselineRunner(),
        executor_factory=lambda _target: OnboardSysctlEnrichExecutor(),
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )

    instance.load_preflight(Path("config.yaml"))
    instance.load_onboard(Path("config.yaml"))
    instance.load_snapshot(Path("config.yaml"))
    instance.load_baseline(Path("config.yaml"))

    context = instance.build_tune_context()

    assert isinstance(context, TuneContext)
    assert context.preflight is instance.preflight
    assert context.onboard is instance.onboard
    assert context.snapshot is instance.snapshot
    assert context.baseline is instance.baseline
    assert context.benchmark_config is instance.benchmark_config
    assert context.artifacts is instance.artifacts


def test_instance_rejects_incomplete_tune_context(tmp_path: Path) -> None:
    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_config: None,  # type: ignore[arg-type]
        executor_factory=lambda target: object(),  # type: ignore[arg-type]
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )

    with pytest.raises(ValueError, match="Preflight must be loaded"):
        instance.build_tune_context()


def test_instance_runs_tune_and_persists_artifact(tmp_path: Path) -> None:
    class FakeSnapshotRunner:
        def run(self, service, executor):  # type: ignore[no-untyped-def]
            _ = service
            _ = executor
            return SnapshotResult(
                service_name="nginx",
                snapshot_directory="/var/tmp/hosttune/snapshots/nginx",
                captured_paths=("/etc/nginx/nginx.conf",),
                runtime_state_output="nginx -T",
                process_state={"pid_file": "1234"},
                restore_sequence=("systemctl restart nginx",),
            )

    class FakeBaselineRunner:
        def run(self, service, executor, dut_target):  # type: ignore[no-untyped-def]
            _ = service
            _ = executor
            _ = dut_target
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

    class FakeTuneEngine:
        def run(self, context, target_executor, benchmark_executor):  # type: ignore[no-untyped-def]
            _ = context
            _ = target_executor
            _ = benchmark_executor
            state = TuneState.initialize(2)
            state.total_iterations = 1
            return state

    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: FakeSnapshotRunner(),
        baseline_runner_factory=lambda benchmark_config: FakeBaselineRunner(),
        executor_factory=lambda _target: OnboardSysctlEnrichExecutor(),
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )

    instance.load_preflight(Path("config.yaml"))
    instance.load_onboard(Path("config.yaml"))
    instance.load_snapshot(Path("config.yaml"))
    instance.load_baseline(Path("config.yaml"))

    result = instance.run_tune(Path("config.yaml"), FakeTuneEngine())

    assert instance.tune is result
    assert instance.artifacts is not None
    assert "tune" in instance.artifacts.stage_files


class EnvDiagnosticExecutor:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        if "systemctl show nginx.service -p CPUQuota" in command:
            return CommandResult(command=command, exit_code=0, stdout="CPUQuota=15%\n", stderr="")
        if "systemctl show nginx.service -p MemoryMax" in command:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="MemoryMax=268435456\n",
                stderr="",
            )
        if "Max open files" in command:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="Max open files            1024                 1024                 files\n",
                stderr="",
            )
        if "smp_affinity_list" in command:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="0\n0\n",
                stderr="",
            )
        return CommandResult(command=command, exit_code=0, stdout="", stderr="")


def _build_env_hierarchy_host_profile(
    *,
    include_limits: bool = False,
    include_irq_affinity: bool = False,
) -> HostProfile:
    parameters = [
        HostPerformanceHierarchyParameter(
            name="CPUQuota",
            target_perf="none",
            inspect_cmd="systemctl show nginx.service -p CPUQuota",
        ),
        HostPerformanceHierarchyParameter(
            name="MemoryMax",
            target_perf="none",
            inspect_cmd="systemctl show nginx.service -p MemoryMax",
        ),
    ]
    if include_limits:
        parameters.append(
            HostPerformanceHierarchyParameter(
                name="LimitNOFILE",
                target_perf="1048576",
                inspect_cmd="cat /proc/$(pgrep -n nginx)/limits | grep 'Max open files'",
            )
        )
    if include_irq_affinity:
        parameters.append(
            HostPerformanceHierarchyParameter(
                name="NIC_IRQ_affinity",
                target_perf="balanced (all cores)",
                inspect_cmd=(
                    "cat /proc/irq/$(grep eno /proc/interrupts | awk '{print $1}' | tr -d ':')/smp_affinity_list"
                ),
            )
        )
    hierarchy = HostPerformanceHierarchy(
        version="1.0",
        description="host env hierarchy",
        groups=(
            HostPerformanceHierarchyGroup(
                group_id="1_systemd_resource_limits",
                description="systemd limits",
                parameters=tuple(parameters),
            ),
        ),
    )
    return HostProfile(
        identity=HostProfileIdentity(
            name="rhel-9",
            platform="rhel",
            version="9",
            variant=None,
        ),
        tunable_surface=HostTunableSurface(
            network_queues=None,
            cpu_governor=None,
            host_sysctls=(),
            environment_blockers=(),
            performance_hierarchy=hierarchy,
        ),
    )


class EnvCleanupEnabledConfigLoader(FakeConfigLoader):
    def load(self, path: Path) -> LoadedConfig:
        loaded = super().load(path)
        return LoadedConfig(
            target=loaded.target,
            policy=replace(loaded.policy, allow_environment_cleanup=True),
            service_name=loaded.service_name,
            benchmark_config=loaded.benchmark_config,
            host_profile_name=loaded.host_profile_name,
        )


class EnvCleanupDisabledConfigLoader(FakeConfigLoader):
    def load(self, path: Path) -> LoadedConfig:
        loaded = super().load(path)
        return LoadedConfig(
            target=loaded.target,
            policy=replace(loaded.policy, allow_environment_cleanup=False),
            service_name=loaded.service_name,
            benchmark_config=loaded.benchmark_config,
            host_profile_name=loaded.host_profile_name,
        )


def test_env_diagnostic_applies_hierarchy_fixes_when_cleanup_allowed(tmp_path: Path) -> None:
    executor = EnvDiagnosticExecutor()
    instance = HostTuneInstance(
        config_loader=EnvCleanupEnabledConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_config: None,  # type: ignore[arg-type]
        executor_factory=lambda _target: executor,
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )
    instance.load_preflight(Path("config.yaml"))
    instance.host_profile = _build_env_hierarchy_host_profile()

    instance.clear_environment_blockers(Path("config.yaml"))

    assert any("systemctl show nginx.service -p CPUQuota" in cmd for cmd in executor.commands)
    assert any("systemctl show nginx.service -p MemoryMax" in cmd for cmd in executor.commands)
    assert any("systemctl set-property nginx.service CPUQuota=" in cmd for cmd in executor.commands)
    assert any("systemctl set-property nginx.service MemoryMax=" in cmd for cmd in executor.commands)


def test_env_diagnostic_does_not_apply_hierarchy_fixes_when_cleanup_disabled(
    tmp_path: Path,
) -> None:
    executor = EnvDiagnosticExecutor()
    instance = HostTuneInstance(
        config_loader=EnvCleanupDisabledConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_config: None,  # type: ignore[arg-type]
        executor_factory=lambda _target: executor,
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )
    instance.load_preflight(Path("config.yaml"))
    instance.host_profile = _build_env_hierarchy_host_profile()

    instance.clear_environment_blockers(Path("config.yaml"))

    assert any("systemctl show nginx.service -p CPUQuota" in cmd for cmd in executor.commands)
    assert any("systemctl show nginx.service -p MemoryMax" in cmd for cmd in executor.commands)
    assert not any("systemctl set-property nginx.service CPUQuota=" in cmd for cmd in executor.commands)
    assert not any("systemctl set-property nginx.service MemoryMax=" in cmd for cmd in executor.commands)


def test_env_diagnostic_uses_dropin_for_limitnofile_hierarchy_fix(tmp_path: Path) -> None:
    executor = EnvDiagnosticExecutor()
    instance = HostTuneInstance(
        config_loader=EnvCleanupEnabledConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_config: None,  # type: ignore[arg-type]
        executor_factory=lambda _target: executor,
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )
    instance.load_preflight(Path("config.yaml"))
    instance.host_profile = _build_env_hierarchy_host_profile(include_limits=True)

    instance.clear_environment_blockers(Path("config.yaml"))

    assert any("Max open files" in cmd for cmd in executor.commands)
    assert any("zz_hosttune_limit_nofile.conf" in cmd for cmd in executor.commands)
    assert any("grep -q '^LimitNOFILE='" in cmd for cmd in executor.commands)
    assert any("rm -f" in cmd for cmd in executor.commands)
    assert not any(
        "systemctl set-property nginx.service LimitNOFILE=" in cmd
        for cmd in executor.commands
    )


def test_env_diagnostic_fixes_nic_irq_affinity_via_irqbalance(tmp_path: Path) -> None:
    executor = EnvDiagnosticExecutor()
    instance = HostTuneInstance(
        config_loader=EnvCleanupEnabledConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_config: None,  # type: ignore[arg-type]
        executor_factory=lambda _target: executor,
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )
    instance.load_preflight(Path("config.yaml"))
    instance.host_profile = _build_env_hierarchy_host_profile(include_irq_affinity=True)

    instance.clear_environment_blockers(Path("config.yaml"))

    assert any("smp_affinity_list" in cmd for cmd in executor.commands)
    assert any(
        "systemctl enable --now irqbalance && systemctl restart irqbalance" in cmd
        for cmd in executor.commands
    )
