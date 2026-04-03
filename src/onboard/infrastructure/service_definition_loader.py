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
        return cast(dict[str, Any], content)
