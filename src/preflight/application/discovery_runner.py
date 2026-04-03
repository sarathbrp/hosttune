from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.capability_builder import CapabilityMapBuilder
from preflight.domain.models import (
    BenchmarkRunner,
    CommandExecutor,
    DiscoverySnapshot,
    EngagementPolicy,
    TargetConfig,
)
from preflight.infrastructure.probes.cpu_probe import CpuProbe
from preflight.infrastructure.probes.kernel_probe import KernelProbe
from preflight.infrastructure.probes.memory_probe import MemoryProbe
from preflight.infrastructure.probes.network_probe import NetworkProbe
from preflight.infrastructure.probes.platform_probe import PlatformProbe
from preflight.infrastructure.probes.storage_probe import StorageProbe


@dataclass
class DiscoveryRunner:
    platform_probe: PlatformProbe
    cpu_probe: CpuProbe
    memory_probe: MemoryProbe
    kernel_probe: KernelProbe
    network_probe: NetworkProbe
    storage_probe: StorageProbe
    capability_builder: CapabilityMapBuilder
    benchmark_runner: BenchmarkRunner | None = None

    def run(
        self,
        executor: CommandExecutor,
        target: TargetConfig,
        policy: EngagementPolicy,
    ) -> DiscoverySnapshot:
        platform = self.platform_probe.collect(executor)
        cpu = self.cpu_probe.collect(executor)
        memory = self.memory_probe.collect(executor)
        kernel = self.kernel_probe.collect(executor)
        network = self.network_probe.collect(executor)
        storage = self.storage_probe.collect(executor)
        capability_map = self.capability_builder.build(
            platform=platform,
            cpu=cpu,
            memory=memory,
            kernel=kernel,
            network=network,
            storage=storage,
        )
        benchmark_result = None
        if self.benchmark_runner is not None:
            benchmark_result = self.benchmark_runner.run(executor)

        return DiscoverySnapshot(
            target=target,
            policy=policy,
            platform=platform,
            cpu=cpu,
            memory=memory,
            kernel=kernel,
            network=network,
            storage=storage,
            capability_map=capability_map,
            benchmark_result=benchmark_result,
            raw_probe_results=None,
        )
