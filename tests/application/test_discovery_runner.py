from preflight.application.discovery_runner import DiscoveryRunner
from preflight.domain.capability_builder import CapabilityMapBuilder
from preflight.domain.models import (
    BenchmarkResult,
    CommandResult,
    CpuInfo,
    EngagementPolicy,
    KernelInfo,
    LocalTargetConfig,
    MemoryInfo,
    NetworkInfo,
    PlatformInfo,
    StorageInfo,
)


class FakeExecutor:
    def run(self, command: str) -> CommandResult:
        return CommandResult(command=command, exit_code=0, stdout="unused", stderr="")


class FakeBenchmarkRunner:
    def run(self, _executor: FakeExecutor) -> BenchmarkResult:
        return BenchmarkResult(
            command="./benchmark.sh",
            exit_code=0,
            primary_metric_name="throughput",
            primary_metric_value=42.0,
            raw_output="42.0",
        )


class FakePlatformProbe:
    def collect(self, _executor: FakeExecutor) -> PlatformInfo:
        return PlatformInfo(
            hostname="node-a",
            operating_system="RHEL",
            kernel_version="5.14.0",
            virtualization_type="none",
            is_container=False,
        )


class FakeCpuProbe:
    def collect(self, _executor: FakeExecutor) -> CpuInfo:
        return CpuInfo(
            architecture="x86_64",
            logical_cores=16,
            threads_per_core=2,
            cores_per_socket=8,
            sockets=1,
            numa_nodes=2,
            hyperthreading_enabled=True,
        )


class FakeMemoryProbe:
    def collect(self, _executor: FakeExecutor) -> MemoryInfo:
        return MemoryInfo(
            total_memory_kib=1024,
            swap_total_kib=512,
            hugepages_total=8,
            transparent_hugepages_mode="always [madvise] never",
        )


class FakeKernelProbe:
    def collect(self, _executor: FakeExecutor) -> KernelInfo:
        return KernelInfo(
            sysctl_writable=True,
            selinux_mode="Permissive",
            tuned_profile="throughput-performance",
        )


class FakeNetworkProbe:
    def collect(self, _executor: FakeExecutor) -> NetworkInfo:
        return NetworkInfo(
            interface_name="eth0",
            driver_name="ixgbe",
            firmware_version="1.0.0",
            rx_ring_current=512,
            rx_ring_max=4096,
            tx_ring_current=512,
            tx_ring_max=4096,
            combined_queues=8,
            ring_buffer_tuning_supported=True,
        )


class FakeStorageProbe:
    def collect(self, _executor: FakeExecutor) -> StorageInfo:
        return StorageInfo(
            device_name="sda",
            device_type="ssd",
            scheduler="[mq-deadline] none",
            scheduler_meaningful=True,
        )


def test_runner_builds_snapshot() -> None:
    executor = FakeExecutor()
    runner = DiscoveryRunner(
        platform_probe=FakePlatformProbe(),
        cpu_probe=FakeCpuProbe(),
        memory_probe=FakeMemoryProbe(),
        kernel_probe=FakeKernelProbe(),
        network_probe=FakeNetworkProbe(),
        storage_probe=FakeStorageProbe(),
        capability_builder=CapabilityMapBuilder(),
        benchmark_runner=FakeBenchmarkRunner(),
    )
    policy = EngagementPolicy(
        allow_reload=True,
        allow_restart=True,
        allow_reboot=False,
        rollback_required=True,
        max_iterations=4,
        benchmark_stability_threshold=0.1,
    )

    snapshot = runner.run(executor=executor, target=LocalTargetConfig(), policy=policy)

    assert snapshot.platform_summary == "bare_metal_linux"
    assert snapshot.platform.hostname == "node-a"
    assert snapshot.platform.operating_system == "RHEL"
    assert snapshot.platform.kernel_version == "5.14.0"
    assert snapshot.network.driver_name == "ixgbe"
    assert snapshot.storage.device_name == "sda"
    assert snapshot.benchmark_result is not None
    assert len(snapshot.capability_map.flags) == 10
