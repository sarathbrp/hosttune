from __future__ import annotations

import argparse
from pathlib import Path

from preflight.application.discovery_runner import DiscoveryRunner
from preflight.domain.capability_builder import CapabilityMapBuilder
from preflight.domain.models import BenchmarkResult, CommandExecutor, LocalTargetConfig, SshTargetConfig
from preflight.infrastructure.config_loader import ConfigLoader
from preflight.infrastructure.executors.local_executor import LocalCommandExecutor
from preflight.infrastructure.executors.ssh_executor import SshCommandExecutor
from preflight.infrastructure.parsers.cpu_parser import CpuParser
from preflight.infrastructure.parsers.kernel_parser import KernelParser
from preflight.infrastructure.parsers.memory_parser import MemoryParser
from preflight.infrastructure.parsers.network_parser import NetworkParser
from preflight.infrastructure.parsers.platform_parser import PlatformParser
from preflight.infrastructure.parsers.storage_parser import StorageParser
from preflight.infrastructure.probes.cpu_probe import CpuProbe
from preflight.infrastructure.probes.kernel_probe import KernelProbe
from preflight.infrastructure.probes.memory_probe import MemoryProbe
from preflight.infrastructure.probes.network_probe import NetworkProbe
from preflight.infrastructure.probes.platform_probe import PlatformProbe
from preflight.infrastructure.probes.storage_probe import StorageProbe
from preflight.interfaces.console_reporter import ConsoleReporter


class ShellBenchmarkRunner:
    def __init__(self, command: str) -> None:
        self._command = command

    def run(self, executor: CommandExecutor) -> BenchmarkResult:
        result = executor.run(self._command)
        primary_value = float(result.stdout.splitlines()[0]) if result.stdout else 0.0
        return BenchmarkResult(
            command=self._command,
            exit_code=result.exit_code,
            primary_metric_name="score",
            primary_metric_value=primary_value,
            raw_output=result.stdout,
        )


def build_executor(target: LocalTargetConfig | SshTargetConfig) -> CommandExecutor:
    if isinstance(target, LocalTargetConfig):
        return LocalCommandExecutor()
    return SshCommandExecutor(target)


def build_discovery_runner(benchmark_command: str | None) -> DiscoveryRunner:
    benchmark_runner = ShellBenchmarkRunner(benchmark_command) if benchmark_command is not None else None
    return DiscoveryRunner(
        platform_probe=PlatformProbe(parser=PlatformParser()),
        cpu_probe=CpuProbe(parser=CpuParser()),
        memory_probe=MemoryProbe(parser=MemoryParser()),
        kernel_probe=KernelProbe(parser=KernelParser()),
        network_probe=NetworkProbe(parser=NetworkParser()),
        storage_probe=StorageProbe(parser=StorageParser()),
        capability_builder=CapabilityMapBuilder(),
        benchmark_runner=benchmark_runner,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run step1 discovery and optional baseline benchmark.")
    parser.add_argument("config", type=Path, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    loaded = ConfigLoader().load(args.config)
    runner = build_discovery_runner(loaded.benchmark_command)
    snapshot = runner.run(
        executor=build_executor(loaded.target),
        target=loaded.target,
        policy=loaded.policy,
    )
    print(ConsoleReporter().render(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
