from __future__ import annotations

from pathlib import Path

from baseline.domain.models import BaselineResult
from onboard.domain.models import CompatibilityReport, OnboardResult
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.application.hosttune_instance import HostTuneInstance
from preflight.domain.models import (
    BenchmarkResult,
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
from snapshot.domain.models import SnapshotResult

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
            benchmark_command="printf '1.0'",
        )


def test_instance_stores_preflight_snapshot() -> None:
    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: None,  # type: ignore[arg-type]
        baseline_runner_factory=lambda benchmark_command: None,  # type: ignore[arg-type]
        executor_factory=lambda target: object(),  # type: ignore[arg-type]
    )

    snapshot = instance.load_preflight(Path("config.yaml"))

    assert instance.preflight is snapshot
    assert instance.preflight is not None
    assert instance.preflight.cpu.logical_cores == 16
    assert instance.preflight.storage.device_name == "sda"


def test_instance_stores_onboard_result() -> None:
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
        baseline_runner_factory=lambda benchmark_command: None,  # type: ignore[arg-type]
        executor_factory=lambda target: object(),  # type: ignore[arg-type]
    )

    instance.load_preflight(Path("config.yaml"))
    result = instance.load_onboard(Path("config.yaml"))

    assert instance.onboard is result
    assert instance.onboard is not None
    assert instance.onboard.service_name == "nginx"


def test_instance_stores_snapshot_and_baseline_results() -> None:
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
        def run(self, service, executor):  # type: ignore[no-untyped-def]
            _ = service
            _ = executor
            return BaselineResult(
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

    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        onboard_runner_factory=lambda: FakeOnboardRunner(),
        snapshot_runner_factory=lambda: FakeSnapshotRunner(),
        baseline_runner_factory=lambda benchmark_command: FakeBaselineRunner(),
        executor_factory=lambda target: object(),  # type: ignore[arg-type]
    )

    instance.load_preflight(Path("config.yaml"))
    instance.load_onboard(Path("config.yaml"))
    snapshot_result = instance.load_snapshot(Path("config.yaml"))
    baseline_result = instance.load_baseline(Path("config.yaml"))

    assert instance.snapshot is snapshot_result
    assert instance.baseline is baseline_result
    assert instance.snapshot is not None
    assert instance.baseline is not None
