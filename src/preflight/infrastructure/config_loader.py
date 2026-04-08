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
    host_profile_name: str | None = None
    prompt_compression: bool = False
    kb_batch_apply: bool = False
    skip_marginal_attribution: bool = False
    marginal_attribution_multiplier: float = 2.0
    use_unified_resolver: bool = False
    dependency_graph_path: str = "tuning-dependency-graph.yaml"
    dspy_compiled_path: str | None = None
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "hosttune"
    skip_attribution: bool = False
    log_level: str = "INFO"
    stopping_marginal_gain_threshold: float = 0.03
    stopping_marginal_gain_iterations: int = 2
    stopping_historical_best_pct: float = 0.85
    stopping_telemetry_stop_enabled: bool = True


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

        host_profile_section = content.get("host_profile") or {}
        host_profile_name: str | None = None
        if isinstance(host_profile_section, dict):
            raw_name = host_profile_section.get("name")
            host_profile_name = str(raw_name) if isinstance(raw_name, str) and raw_name else None

        tune_section = content.get("tune") or {}
        prompt_compression = bool(tune_section.get("prompt_compression", False))
        kb_batch_apply = bool(tune_section.get("kb_batch_apply", False))
        skip_marginal_attribution = bool(
            tune_section.get("skip_marginal_attribution", False)
        )
        marginal_attribution_multiplier = float(
            tune_section.get("marginal_attribution_multiplier", 2.0)
        )
        use_unified_resolver = bool(
            tune_section.get("use_unified_resolver", False)
        )
        dependency_graph_path = str(
            tune_section.get(
                "dependency_graph_path", "tuning-dependency-graph.yaml"
            )
        )
        raw_compiled = tune_section.get("dspy_compiled_path")
        dspy_compiled_path = str(raw_compiled) if raw_compiled else None

        mlflow_section = content.get("mlflow") or {}
        mlflow_enabled = bool(mlflow_section.get("enabled", False))
        mlflow_tracking_uri = str(mlflow_section.get("tracking_uri", "http://localhost:5000"))
        mlflow_experiment_name = str(mlflow_section.get("experiment_name", "hosttune"))
        skip_attribution = bool(tune_section.get("skip_attribution", False))
        log_level = str(content.get("log_level", "INFO")).upper()

        stopping_section = tune_section.get("stopping") or {}
        stopping_marginal_gain_threshold = float(stopping_section.get("marginal_gain_threshold", 0.03))
        stopping_marginal_gain_iterations = int(stopping_section.get("marginal_gain_iterations", 2))
        stopping_historical_best_pct = float(stopping_section.get("historical_best_pct", 0.85))
        stopping_telemetry_stop_enabled = bool(stopping_section.get("telemetry_stop_enabled", True))

        return LoadedConfig(
            target=target,
            policy=policy,
            service_name=service_name,
            benchmark_config=benchmark_config,
            host_profile_name=host_profile_name,
            prompt_compression=prompt_compression,
            kb_batch_apply=kb_batch_apply,
            skip_marginal_attribution=skip_marginal_attribution,
            marginal_attribution_multiplier=marginal_attribution_multiplier,
            use_unified_resolver=use_unified_resolver,
            dependency_graph_path=dependency_graph_path,
            dspy_compiled_path=dspy_compiled_path,
            mlflow_enabled=mlflow_enabled,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            skip_attribution=skip_attribution,
            log_level=log_level,
            stopping_marginal_gain_threshold=stopping_marginal_gain_threshold,
            stopping_marginal_gain_iterations=stopping_marginal_gain_iterations,
            stopping_historical_best_pct=stopping_historical_best_pct,
            stopping_telemetry_stop_enabled=stopping_telemetry_stop_enabled,
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
            allow_environment_cleanup=bool(data.get("allow_environment_cleanup", False)),
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
        cooling_period_seconds = data.get("cooling_period_seconds", 30)

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
        if not isinstance(cooling_period_seconds, int) or cooling_period_seconds < 0:
            msg = "benchmark.cooling_period_seconds must be a non-negative integer."
            raise ValueError(msg)

        return BenchmarkConfig(
            runner_target=self._load_target(runner_data),
            contestant_name=contestant_name,
            script_path=script_path,
            results_directory=results_directory,
            workloads=tuple(str(workload) for workload in workloads),
            compare_script_path=compare_script_path,
            cooling_period_seconds=cooling_period_seconds,
        )
