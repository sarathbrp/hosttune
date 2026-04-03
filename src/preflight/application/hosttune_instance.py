from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from preflight.application.discovery_runner import DiscoveryRunner
from preflight.domain.models import (
    CommandExecutor,
    DiscoverySnapshot,
    LocalTargetConfig,
    SshTargetConfig,
)
from preflight.infrastructure.config_loader import ConfigLoader, LoadedConfig

ExecutorFactory = Callable[[LocalTargetConfig | SshTargetConfig], CommandExecutor]
DiscoveryRunnerFactory = Callable[[str | None], DiscoveryRunner]


@dataclass
class HostTuneInstance:
    config_loader: ConfigLoader
    discovery_runner_factory: DiscoveryRunnerFactory
    executor_factory: ExecutorFactory
    preflight: DiscoverySnapshot | None = None

    def load_preflight(self, config_path: Path) -> DiscoverySnapshot:
        loaded_config = self.config_loader.load(config_path)
        snapshot = self._run_preflight(loaded_config)
        self.preflight = snapshot
        return snapshot

    def _run_preflight(self, loaded_config: LoadedConfig) -> DiscoverySnapshot:
        runner = self.discovery_runner_factory(loaded_config.benchmark_command)
        executor = self.executor_factory(loaded_config.target)
        return runner.run(
            executor=executor,
            target=loaded_config.target,
            policy=loaded_config.policy,
        )
