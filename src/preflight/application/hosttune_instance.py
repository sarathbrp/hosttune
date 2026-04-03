from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from baseline.application.baseline_runner import BaselineRunner
from baseline.domain.models import BaselineResult, BenchmarkConfig
from onboard.application.onboard_runner import OnboardRunner
from onboard.domain.models import OnboardResult
from preflight.application.discovery_runner import DiscoveryRunner
from preflight.domain.models import (
    CommandExecutor,
    DiscoverySnapshot,
    LocalTargetConfig,
    SshTargetConfig,
)
from preflight.infrastructure.config_loader import ConfigLoader, LoadedConfig
from snapshot.application.snapshot_runner import SnapshotRunner
from snapshot.domain.models import SnapshotResult

ExecutorFactory = Callable[[LocalTargetConfig | SshTargetConfig], CommandExecutor]
DiscoveryRunnerFactory = Callable[[str | None], DiscoveryRunner]
OnboardRunnerFactory = Callable[[], OnboardRunner]
SnapshotRunnerFactory = Callable[[], SnapshotRunner]
BaselineRunnerFactory = Callable[[BenchmarkConfig], BaselineRunner]


@dataclass
class HostTuneInstance:
    config_loader: ConfigLoader
    discovery_runner_factory: DiscoveryRunnerFactory
    onboard_runner_factory: OnboardRunnerFactory
    snapshot_runner_factory: SnapshotRunnerFactory
    baseline_runner_factory: BaselineRunnerFactory
    executor_factory: ExecutorFactory
    preflight: DiscoverySnapshot | None = None
    onboard: OnboardResult | None = None
    snapshot: SnapshotResult | None = None
    baseline: BaselineResult | None = None

    def load_preflight(self, config_path: Path) -> DiscoverySnapshot:
        loaded_config = self.config_loader.load(config_path)
        snapshot = self._run_preflight(loaded_config)
        self.preflight = snapshot
        return snapshot

    def load_onboard(self, config_path: Path) -> OnboardResult:
        if self.preflight is None:
            msg = "Preflight must be loaded before onboard."
            raise ValueError(msg)
        loaded_config = self.config_loader.load(config_path)
        executor = self.executor_factory(loaded_config.target)
        runner = self.onboard_runner_factory()
        result = runner.run(
            service_name=loaded_config.service_name,
            preflight=self.preflight,
            executor=executor,
        )
        self.onboard = result
        return result

    def load_snapshot(self, config_path: Path) -> SnapshotResult:
        if self.onboard is None:
            msg = "Onboard must be loaded before snapshot."
            raise ValueError(msg)
        loaded_config = self.config_loader.load(config_path)
        executor = self.executor_factory(loaded_config.target)
        runner = self.snapshot_runner_factory()
        result = runner.run(self.onboard.service, executor)
        self.snapshot = result
        return result

    def load_baseline(self, config_path: Path) -> BaselineResult:
        if self.onboard is None:
            msg = "Onboard must be loaded before baseline."
            raise ValueError(msg)
        loaded_config = self.config_loader.load(config_path)
        if loaded_config.benchmark_config is None:
            msg = "benchmark must be configured before baseline."
            raise ValueError(msg)
        benchmark_executor = self.executor_factory(loaded_config.benchmark_config.runner_target)
        runner = self.baseline_runner_factory(loaded_config.benchmark_config)
        result = runner.run(self.onboard.service, benchmark_executor, loaded_config.target)
        self.baseline = result
        return result

    def _run_preflight(self, loaded_config: LoadedConfig) -> DiscoverySnapshot:
        runner = self.discovery_runner_factory(None)
        executor = self.executor_factory(loaded_config.target)
        return runner.run(
            executor=executor,
            target=loaded_config.target,
            policy=loaded_config.policy,
        )
