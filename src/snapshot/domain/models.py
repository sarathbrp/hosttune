from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SnapshotResult:
    service_name: str
    snapshot_directory: str
    captured_paths: tuple[str, ...]
    runtime_state_output: str | None
    process_state: dict[str, str]
    restore_sequence: tuple[str, ...]
