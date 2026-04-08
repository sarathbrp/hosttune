from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from host_profile.domain.models import HostProfile
from host_profile.infrastructure.host_profile_validator import HostProfileValidator

HOST_PROFILES_DIR = Path(__file__).parent.parent.parent.parent / "host-profiles"


@dataclass(frozen=True)
class HostProfileLoader:
    profiles_dir: Path = HOST_PROFILES_DIR
    validator: HostProfileValidator = HostProfileValidator()

    def load(self, name: str) -> HostProfile:
        path = self.profiles_dir / f"{name}.yaml"
        if not path.exists():
            msg = f"Host profile {name!r} not found at {path}"
            raise ValueError(msg)
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        merged = self._merge_perf_hierarchy_compat(name=name, definition=raw)
        return self.validator.validate(merged)

    def _merge_perf_hierarchy_compat(
        self,
        *,
        name: str,
        definition: dict[str, Any],
    ) -> dict[str, Any]:
        """Compat bridge for host profile sidecar hierarchy migration.

        If tunable_surface.performance_hierarchy is missing in the main profile
        but a sidecar <profile>-perf-hierarchy.yaml exists, inject it so the
        validator receives a single merged profile shape.
        """
        tunable_surface = definition.get("tunable_surface")
        if not isinstance(tunable_surface, dict):
            return definition
        if "performance_hierarchy" in tunable_surface:
            return definition
        sidecar_path = self.profiles_dir / f"{name}-perf-hierarchy.yaml"
        if not sidecar_path.exists():
            return definition
        sidecar = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, dict):
            msg = f"Performance hierarchy sidecar must be a mapping: {sidecar_path}"
            raise ValueError(msg)
        merged_surface = {**tunable_surface, "performance_hierarchy": sidecar}
        return {**definition, "tunable_surface": merged_surface}
