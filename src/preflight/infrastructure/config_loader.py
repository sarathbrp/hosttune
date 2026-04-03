from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

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
    benchmark_command: str | None


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
        benchmark_command = benchmark.get("command") if isinstance(benchmark, dict) else None
        if benchmark_command is not None and not isinstance(benchmark_command, str):
            msg = "benchmark.command must be a string when provided."
            raise ValueError(msg)

        return LoadedConfig(
            target=target,
            policy=policy,
            service_name=service_name,
            benchmark_command=benchmark_command,
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
