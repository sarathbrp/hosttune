from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.capability_builder import CapabilityMapBuilder
from preflight.domain.models import (
    BenchmarkRunner,
    CommandExecutor,
    DiscoverySnapshot,
    EngagementPolicy,
    PlatformInfo,
    TargetConfig,
)
from preflight.infrastructure.probes.cgroup_probe import CgroupProbe
from preflight.infrastructure.probes.cpu_probe import CpuProbe
from preflight.infrastructure.probes.irq_probe import IrqProbe
from preflight.infrastructure.probes.kernel_probe import KernelProbe
from preflight.infrastructure.probes.memory_probe import MemoryProbe
from preflight.infrastructure.probes.network_probe import NetworkProbe
from preflight.infrastructure.probes.platform_probe import PlatformProbe
from preflight.infrastructure.probes.storage_probe import StorageProbe
from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger


@dataclass
class DiscoveryRunner:
    platform_probe: PlatformProbe
    cpu_probe: CpuProbe
    memory_probe: MemoryProbe
    kernel_probe: KernelProbe
    network_probe: NetworkProbe
    storage_probe: StorageProbe
    irq_probe: IrqProbe
    cgroup_probe: CgroupProbe
    capability_builder: CapabilityMapBuilder
    benchmark_runner: BenchmarkRunner | None = None
    logger: ExecutionLogger = NullExecutionLogger()

    def run(
        self,
        executor: CommandExecutor,
        target: TargetConfig,
        policy: EngagementPolicy,
    ) -> DiscoverySnapshot:
        self.logger.stage_detail("preflight", "Collecting platform information")
        platform = self.platform_probe.collect(executor)
        self.logger.stage_detail("preflight", "Collecting CPU topology")
        cpu = self.cpu_probe.collect(executor)
        self.logger.stage_detail("preflight", "Collecting memory information")
        memory = self.memory_probe.collect(executor)
        self.logger.stage_detail("preflight", "Collecting kernel information")
        kernel = self.kernel_probe.collect(executor)
        self.logger.stage_detail("preflight", "Collecting network information")
        network = self.network_probe.collect(executor)
        self.logger.stage_detail("preflight", "Collecting storage information")
        storage = self.storage_probe.collect(executor)
        self.logger.stage_detail("preflight", "Collecting IRQ information")
        irq = self.irq_probe.collect(executor)
        self.logger.stage_detail("preflight", "Collecting cgroup information")
        cgroup = self.cgroup_probe.collect(executor)
        self.logger.stage_detail("preflight", "Building capability map")
        capability_map = self.capability_builder.build(
            platform=platform,
            cpu=cpu,
            memory=memory,
            kernel=kernel,
            network=network,
            storage=storage,
            irq=irq,
            cgroup=cgroup,
        )
        benchmark_result = None
        if self.benchmark_runner is not None:
            benchmark_result = self.benchmark_runner.run(executor)

        return DiscoverySnapshot(
            target=target,
            policy=policy,
            platform_summary=self._build_platform_summary(platform),
            platform=platform,
            cpu=cpu,
            memory=memory,
            kernel=kernel,
            network=network,
            storage=storage,
            irq=irq,
            cgroup=cgroup,
            capability_map=capability_map,
            benchmark_result=benchmark_result,
            raw_probe_results=None,
        )

    def _build_platform_summary(self, platform: PlatformInfo) -> str:
        if platform.is_container:
            return "containerized_linux"
        if platform.virtualization_type not in {"", "none", "unknown"}:
            return f"virtual_machine:{platform.virtualization_type}"
        if platform.operating_system == "unknown":
            return "non_linux_or_unsupported_platform"
        return "bare_metal_linux"
