from preflight.domain.models import (
    CapabilityFlag,
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
from preflight.interfaces.console_reporter import ConsoleReporter
from snapshot.domain.models import SnapshotResult
from baseline.domain.models import BaselineResult
from preflight.domain.models import BenchmarkResult
from onboard.domain.models import CompatibilityReport, OnboardResult
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator

from tests.onboard.test_service_definition_validator import build_valid_definition


def test_console_reporter_serializes_snapshot() -> None:
    snapshot = DiscoverySnapshot(
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
        capability_map=CapabilityMap(flags=(CapabilityFlag("irq_affinity", True, "supported"),)),
    )

    rendered = ConsoleReporter().render(snapshot)

    assert '"platform_summary": "bare_metal_linux"' in rendered
    assert '"hostname": "node-a"' in rendered
    assert '"irq_affinity"' in rendered


def test_console_reporter_serializes_full_runtime() -> None:
    reporter = ConsoleReporter()
    snapshot = DiscoverySnapshot(
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
        capability_map=CapabilityMap(flags=()),
    )
    onboard = OnboardResult(
        service_name="nginx",
        service=ServiceDefinitionValidator().validate(build_valid_definition()),
        compatibility=CompatibilityReport(compatible=True, findings=()),
    )
    runtime_snapshot = SnapshotResult(
        service_name="nginx",
        snapshot_directory="/var/tmp/hosttune",
        captured_paths=("/etc/nginx/nginx.conf",),
        runtime_state_output="nginx -T",
        process_state={"pid_file": "1234"},
        restore_sequence=("systemctl restart nginx",),
    )
    baseline = BaselineResult(
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

    rendered = reporter.render_runtime(snapshot, onboard, runtime_snapshot, baseline)

    assert '"onboard"' in rendered
    assert '"snapshot"' in rendered
    assert '"baseline"' in rendered
