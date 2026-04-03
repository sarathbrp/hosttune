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
