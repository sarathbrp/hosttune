from preflight.domain.capability_builder import CapabilityMapBuilder
from preflight.domain.models import (
    CgroupInfo,
    CpuInfo,
    IrqInfo,
    KernelInfo,
    MemoryInfo,
    NetworkInfo,
    PlatformInfo,
    StorageInfo,
)


def test_capability_builder_marks_bare_metal_numa_as_available() -> None:
    capability_map = CapabilityMapBuilder().build(
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
            hugepages_total=16,
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
        irq=IrqInfo(irqbalance_active=True, nic_irq_cpu_summary="0-7"),
        cgroup=CgroupInfo(
            cgroup_version="v2",
            cpu_controller_available=True,
            memory_controller_available=True,
        ),
    )

    flags = {flag.name: flag for flag in capability_map.flags}

    assert flags["numa_tuning"].available is True
    assert flags["kernel_sysctl_tuning"].available is True
    assert flags["network_ring_buffer_tuning"].available is True
    assert flags["runtime_prlimit_tuning"].available is True
    assert flags["systemd_unit_limit_tuning"].available is True
    assert flags["storage_scheduler_tuning"].available is True


def test_capability_builder_restricts_container_tuning() -> None:
    capability_map = CapabilityMapBuilder().build(
        platform=PlatformInfo(
            hostname="node-a",
            operating_system="RHEL",
            kernel_version="5.14.0",
            virtualization_type="docker",
            is_container=True,
        ),
        cpu=CpuInfo(
            architecture="x86_64",
            logical_cores=8,
            threads_per_core=1,
            cores_per_socket=8,
            sockets=1,
            numa_nodes=1,
            hyperthreading_enabled=False,
        ),
        memory=MemoryInfo(
            total_memory_kib=1024,
            swap_total_kib=0,
            hugepages_total=0,
            transparent_hugepages_mode="unknown",
        ),
        kernel=KernelInfo(
            sysctl_writable=False,
            selinux_mode="Enforcing",
            tuned_profile="unknown",
        ),
        network=NetworkInfo(
            interface_name="eth0",
            driver_name="virtio_net",
            firmware_version="unknown",
            rx_ring_current=256,
            rx_ring_max=256,
            tx_ring_current=256,
            tx_ring_max=256,
            combined_queues=2,
            ring_buffer_tuning_supported=False,
        ),
        storage=StorageInfo(
            device_name="nvme0n1",
            device_type="nvme",
            scheduler="[none] mq-deadline",
            scheduler_meaningful=False,
        ),
        irq=IrqInfo(irqbalance_active=False, nic_irq_cpu_summary="unknown"),
        cgroup=CgroupInfo(
            cgroup_version="v1",
            cpu_controller_available=False,
            memory_controller_available=False,
        ),
    )

    flags = {flag.name: flag for flag in capability_map.flags}

    assert flags["cpu_topology_tuning"].available is False
    assert flags["hugepages_tuning"].available is False
    assert flags["kernel_sysctl_tuning"].available is False
    assert flags["network_ring_buffer_tuning"].available is False
    assert flags["runtime_prlimit_tuning"].available is False
    assert flags["systemd_unit_limit_tuning"].available is False
    assert flags["storage_scheduler_tuning"].available is False
