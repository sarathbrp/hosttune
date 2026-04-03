from __future__ import annotations

from pathlib import Path

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
            benchmark_command=None,
        )


def test_instance_stores_preflight_snapshot() -> None:
    instance = HostTuneInstance(
        config_loader=FakeConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: FakeRunner(),
        executor_factory=lambda target: object(),  # type: ignore[arg-type]
    )

    snapshot = instance.load_preflight(Path("config.yaml"))

    assert instance.preflight is snapshot
    assert instance.preflight is not None
    assert instance.preflight.cpu.logical_cores == 16
    assert instance.preflight.storage.device_name == "sda"
