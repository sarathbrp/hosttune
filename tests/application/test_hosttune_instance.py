from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline.domain.models import BaselineResult, BenchmarkConfig, WorkloadBenchmarkResult
from onboard.domain.models import CompatibilityReport, OnboardResult
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.application.hosttune_instance import HostTuneInstance
from preflight.domain.models import (
    CapabilityMap,
    CpuInfo,
    DiscoverySnapshot,
    EngagementPolicy,
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

from tests.onboard.test_service_definition_validator import build_valid_definition


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
        executor_factory=lambda target: object(),  # type: ignore[arg-type]
        artifact_store=RuntimeArtifactStore(base_directory=tmp_path / "artifacts"),
    )

    instance.load_preflight(Path("config.yaml"))
    result = instance.load_onboard(Path("config.yaml"))

    assert instance.onboard is result
    assert instance.onboard is not None
    assert instance.onboard.service_name == "nginx"


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
        executor_factory=lambda target: object(),  # type: ignore[arg-type]
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
        executor_factory=lambda target: object(),  # type: ignore[arg-type]
        artifact_store=artifact_store,
    )

    instance.load_preflight(Path("config.yaml"))
    instance.load_onboard(Path("config.yaml"))

    assert instance.artifacts is not None
    preflight_file = instance.artifacts.stage_files["preflight"]
    onboard_file = instance.artifacts.stage_files["onboard"]

    preflight_record = json.loads(preflight_file.read_text(encoding="utf-8").splitlines()[0])
    onboard_record = json.loads(onboard_file.read_text(encoding="utf-8").splitlines()[0])

    assert len(instance.artifacts.session_id) == RuntimeArtifactStore.SESSION_ID_LENGTH
    assert preflight_file.name == f"preflight_{instance.artifacts.session_id}.jsonl"
    assert onboard_file.name == f"onboard_{instance.artifacts.session_id}.jsonl"
    assert preflight_record["stage"] == "preflight"
    assert preflight_record["payload"]["platform_summary"] == "bare_metal_linux"
    assert onboard_record["stage"] == "onboard"
    assert onboard_record["payload"]["service_name"] == "nginx"


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
        executor_factory=lambda target: object(),  # type: ignore[arg-type]
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
