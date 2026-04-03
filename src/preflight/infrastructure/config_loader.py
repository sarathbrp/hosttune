from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from baseline.domain.models import BenchmarkConfig
from preflight.domain.models import (
    EngagementPolicy,
    LocalTargetConfig,
    SshTargetConfig,
    TargetConfig,
)


@dataclass(frozen=True)
class LoadedConfig:
    target: TargetConfig
    policy: EngagementPolicy
    service_name: str
    benchmark_config: BenchmarkConfig | None


class ConfigLoader:
    def load(self, path: Path) -> LoadedConfig:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            msg = "Configuration root must be a mapping."
            raise ValueError(msg)

        target = self._load_target(cast(dict[str, Any], content.get("target", {})))
        policy = self._load_policy(cast(dict[str, Any], content.get("policy", {})))
        service = cast(dict[str, Any], content.get("service", {}))
        service_name = service.get("name")
        if not isinstance(service_name, str) or service_name == "":
            msg = "service.name must be a non-empty string."
            raise ValueError(msg)
        benchmark = content.get("benchmark", {})
        if benchmark is None:
            benchmark = {}
        if not isinstance(benchmark, dict):
            msg = "benchmark must be a mapping when provided."
            raise ValueError(msg)
        benchmark_config = self._load_benchmark(cast(dict[str, Any], benchmark))

        return LoadedConfig(
            target=target,
            policy=policy,
            service_name=service_name,
            benchmark_config=benchmark_config,
        )

    def _load_target(self, data: dict[str, Any]) -> TargetConfig:
        mode = data.get("mode", "local")
        if mode == "local":
            return LocalTargetConfig()
        if mode != "ssh":
            msg = f"Unsupported target mode: {mode!r}"
            raise ValueError(msg)

        required = ["host", "user", "private_key_path"]
        missing = [field_name for field_name in required if field_name not in data]
        if missing:
            msg = f"Missing SSH target fields: {', '.join(missing)}"
            raise ValueError(msg)

        return SshTargetConfig(
            host=str(data["host"]),
            user=str(data["user"]),
            private_key_path=Path(str(data["private_key_path"])),
            port=int(data.get("port", 22)),
            connect_timeout_seconds=int(data.get("connect_timeout_seconds", 5)),
        )

    def _load_policy(self, data: dict[str, Any]) -> EngagementPolicy:
        return EngagementPolicy(
            allow_reload=bool(data.get("allow_reload", False)),
            allow_restart=bool(data.get("allow_restart", False)),
            allow_reboot=bool(data.get("allow_reboot", False)),
            rollback_required=bool(data.get("rollback_required", True)),
            max_iterations=int(data.get("max_iterations", 10)),
            benchmark_stability_threshold=float(data.get("benchmark_stability_threshold", 0.10)),
        )

    def _load_benchmark(self, data: dict[str, Any]) -> BenchmarkConfig | None:
        if not data:
            return None

        runner_data = cast(dict[str, Any], data.get("runner", {}))
        if not runner_data:
            msg = "benchmark.runner must be configured when benchmark is enabled."
            raise ValueError(msg)

        contestant_name = data.get("contestant_name")
        script_path = data.get("script_path")
        results_directory = data.get("results_directory")
        workloads = data.get("workloads")
        compare_script_path = data.get("compare_script_path")

        if not isinstance(contestant_name, str) or contestant_name == "":
            msg = "benchmark.contestant_name must be a non-empty string."
            raise ValueError(msg)
        if not isinstance(script_path, str) or script_path == "":
            msg = "benchmark.script_path must be a non-empty string."
            raise ValueError(msg)
        if not isinstance(results_directory, str) or results_directory == "":
            msg = "benchmark.results_directory must be a non-empty string."
            raise ValueError(msg)
        if not isinstance(workloads, list) or not workloads:
            msg = "benchmark.workloads must be a non-empty list of strings."
            raise ValueError(msg)
        if any(not isinstance(workload, str) or workload == "" for workload in workloads):
            msg = "benchmark.workloads must contain only non-empty strings."
            raise ValueError(msg)
        if compare_script_path is not None and not isinstance(compare_script_path, str):
            msg = "benchmark.compare_script_path must be a string when provided."
            raise ValueError(msg)

        return BenchmarkConfig(
            runner_target=self._load_target(runner_data),
            contestant_name=contestant_name,
            script_path=script_path,
            results_directory=results_directory,
            workloads=tuple(str(workload) for workload in workloads),
            compare_script_path=compare_script_path,
        )
