from __future__ import annotations

import argparse
from pathlib import Path

from baseline.application.baseline_runner import BaselineRunner
from onboard.application.onboard_runner import OnboardRunner
from onboard.infrastructure.service_compatibility_evaluator import ServiceCompatibilityEvaluator
from onboard.infrastructure.service_definition_loader import ServiceDefinitionLoader
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.application.discovery_runner import DiscoveryRunner
from preflight.application.hosttune_instance import HostTuneInstance
from preflight.domain.capability_builder import CapabilityMapBuilder
from preflight.domain.models import (
    BenchmarkResult,
    CommandExecutor,
    LocalTargetConfig,
    SshTargetConfig,
)
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
from snapshot.application.snapshot_runner import SnapshotRunner

SERVICE_REGISTRY_PATH = Path("service-monitor")


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
    benchmark_runner = (
        ShellBenchmarkRunner(benchmark_command) if benchmark_command is not None else None
    )
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


def build_instance() -> HostTuneInstance:
    return HostTuneInstance(
        config_loader=ConfigLoader(),
        discovery_runner_factory=build_discovery_runner,
        onboard_runner_factory=build_onboard_runner,
        snapshot_runner_factory=build_snapshot_runner,
        baseline_runner_factory=build_baseline_runner,
        executor_factory=build_executor,
    )


def build_onboard_runner() -> OnboardRunner:
    return OnboardRunner(
        loader=ServiceDefinitionLoader(registry_path=SERVICE_REGISTRY_PATH),
        validator=ServiceDefinitionValidator(),
        evaluator=ServiceCompatibilityEvaluator(),
    )


def build_snapshot_runner() -> SnapshotRunner:
    return SnapshotRunner()


def build_baseline_runner(benchmark_command: str) -> BaselineRunner:
    return BaselineRunner(benchmark_runner=ShellBenchmarkRunner(benchmark_command))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run step1 discovery and optional baseline benchmark."
    )
    parser.add_argument("config", type=Path, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    instance = build_instance()
    loaded_config = instance.config_loader.load(args.config)
    preflight = instance.load_preflight(args.config)
    onboard = instance.load_onboard(args.config)
    snapshot = instance.load_snapshot(args.config)
    baseline = None
    if loaded_config.benchmark_command is not None:
        baseline = instance.load_baseline(args.config)
    print(
        ConsoleReporter().render_runtime(
            preflight=preflight,
            onboard=onboard,
            snapshot=snapshot,
            baseline=baseline,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
