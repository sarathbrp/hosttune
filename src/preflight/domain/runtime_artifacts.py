from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RuntimeArtifacts:
    session_id: str
    session_directory: Path
    stage_files: dict[str, Path] = field(default_factory=dict)
