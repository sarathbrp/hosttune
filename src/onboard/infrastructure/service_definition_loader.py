from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


class ServiceDefinitionLoader:
    def __init__(self, registry_path: Path) -> None:
        self._registry_path = registry_path

    def load(self, service_name: str) -> dict[str, Any]:
        path = self._registry_path / f"{service_name}.yaml"
        if not path.exists():
            msg = f"Service definition not found: {path}"
            raise ValueError(msg)
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            msg = f"Service definition must be a mapping: {path}"
            raise ValueError(msg)
        definition = cast(dict[str, Any], content)
        return self._merge_perf_hierarchy_compat(
            service_name=service_name,
            definition=definition,
        )

    def _merge_perf_hierarchy_compat(
        self,
        *,
        service_name: str,
        definition: dict[str, Any],
    ) -> dict[str, Any]:
        """Compat bridge during migration from *-perf-hierarchy.yaml to merged YAML.

        If tunable_surface.performance_hierarchy is missing in the main service file
        but a sidecar <service>-perf-hierarchy.yaml exists, inject it into the loaded
        definition so downstream validation sees a single merged schema.
        """
        tunable_surface = definition.get("tunable_surface")
        if not isinstance(tunable_surface, dict):
            return definition
        if "performance_hierarchy" in tunable_surface:
            return definition
        sidecar_path = self._registry_path / f"{service_name}-perf-hierarchy.yaml"
        if not sidecar_path.exists():
            return definition
        sidecar = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, dict):
            msg = f"Performance hierarchy sidecar must be a mapping: {sidecar_path}"
            raise ValueError(msg)
        merged_surface = {**tunable_surface, "performance_hierarchy": sidecar}
        return {**definition, "tunable_surface": merged_surface}
