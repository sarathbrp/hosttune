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
        return self.validator.validate(raw)
