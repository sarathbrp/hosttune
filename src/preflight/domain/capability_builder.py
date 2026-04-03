from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import (
    CapabilityFlag,
    CapabilityMap,
    CgroupInfo,
    CpuInfo,
    IrqInfo,
    KernelInfo,
    MemoryInfo,
    NetworkInfo,
    PlatformInfo,
    StorageInfo,
)


@dataclass(frozen=True)
class CapabilityMapBuilder:
    def build(
        self,
        platform: PlatformInfo,
        cpu: CpuInfo,
        memory: MemoryInfo,
        kernel: KernelInfo,
        network: NetworkInfo,
        storage: StorageInfo,
        irq: IrqInfo,
        cgroup: CgroupInfo,
    ) -> CapabilityMap:
        flags = (
            CapabilityFlag(
                name="numa_tuning",
                available=cpu.numa_nodes > 1 and platform.virtualization_type == "none",
                detail=(
                    f"numa_nodes={cpu.numa_nodes}, "
                    f"virtualization={platform.virtualization_type}"
                ),
            ),
            CapabilityFlag(
                name="cpu_topology_tuning",
                available=cpu.logical_cores > 0 and not platform.is_container,
                detail=(
                    f"logical_cores={cpu.logical_cores}, "
                    f"hyperthreading_enabled={cpu.hyperthreading_enabled}"
                ),
            ),
            CapabilityFlag(
                name="hugepages_tuning",
                available=memory.hugepages_total >= 0 and not platform.is_container,
                detail=(
                    f"hugepages_total={memory.hugepages_total}, "
                    f"transparent_hugepages_mode={memory.transparent_hugepages_mode}"
                ),
            ),
            CapabilityFlag(
                name="swap_tuning",
                available=memory.swap_total_kib > 0 and kernel.sysctl_writable,
                detail=(
                    f"swap_total_kib={memory.swap_total_kib}, "
                    f"sysctl_writable={kernel.sysctl_writable}"
                ),
            ),
            CapabilityFlag(
                name="kernel_sysctl_tuning",
                available=kernel.sysctl_writable,
                detail=f"selinux_mode={kernel.selinux_mode}, tuned_profile={kernel.tuned_profile}",
            ),
            CapabilityFlag(
                name="runtime_prlimit_tuning",
                available=not platform.is_container,
                detail="prlimit NOFILE soft limit when service snapshot defines pid_file",
            ),
            CapabilityFlag(
                name="systemd_unit_limit_tuning",
                available=not platform.is_container,
                detail="systemctl set-property LimitNOFILE/LimitNPROC on the service unit",
            ),
            CapabilityFlag(
                name="network_ring_buffer_tuning",
                available=network.ring_buffer_tuning_supported
                and (
                    network.rx_ring_max > network.rx_ring_current
                    or network.tx_ring_max > network.tx_ring_current
                ),
                detail=(
                    f"interface={network.interface_name}, driver={network.driver_name}, "
                    f"rx={network.rx_ring_current}/{network.rx_ring_max}, "
                    f"tx={network.tx_ring_current}/{network.tx_ring_max}"
                ),
            ),
            CapabilityFlag(
                name="network_queue_tuning",
                available=network.combined_queues > 0 and not platform.is_container,
                detail=(
                    f"interface={network.interface_name}, "
                    f"combined_queues={network.combined_queues}"
                ),
            ),
            CapabilityFlag(
                name="storage_scheduler_tuning",
                available=storage.scheduler_meaningful,
                detail=(
                    f"device={storage.device_name}, device_type={storage.device_type}, "
                    f"scheduler={storage.scheduler}"
                ),
            ),
            CapabilityFlag(
                name="irq_affinity_tuning",
                available=not platform.is_container and irq.nic_irq_cpu_summary != "unknown",
                detail=(
                    f"irqbalance_active={irq.irqbalance_active}, "
                    f"nic_irq_cpus={irq.nic_irq_cpu_summary}"
                ),
            ),
            CapabilityFlag(
                name="cgroup_resource_control",
                available=cgroup.cgroup_version == "v2" and not platform.is_container,
                detail=(
                    f"cgroup_version={cgroup.cgroup_version}, "
                    f"cpu_controller={cgroup.cpu_controller_available}, "
                    f"memory_controller={cgroup.memory_controller_available}"
                ),
            ),
            CapabilityFlag(
                name="reboot_permitted",
                available=True,
                detail="reboot permission is controlled by engagement policy, not discovery",
            ),
        )
        return CapabilityMap(flags=flags)
