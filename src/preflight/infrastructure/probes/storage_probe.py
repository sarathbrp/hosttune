from __future__ import annotations

import re
from dataclasses import dataclass

from preflight.domain.models import CommandExecutor, CommandResult, StorageInfo
from preflight.infrastructure.parsers.storage_parser import StorageParser
from preflight.infrastructure.probes.base import BaseProbe


@dataclass(frozen=True)
class StorageProbe(BaseProbe):
    parser: StorageParser

    @property
    def name(self) -> str:
        return "storage"

    def collect(self, executor: CommandExecutor) -> StorageInfo:
        device = executor.run(
            "root_source=$(findmnt -n -o SOURCE /); "
            'root_real=$(realpath "$root_source" 2>/dev/null || printf \'%s\' "$root_source"); '
            'resolved_disk=$(lsblk -sno NAME,TYPE "$root_real" 2>/dev/null | '
            "awk '$2==\"disk\" {print $1; exit}'); "
            'if [ -n "$resolved_disk" ]; then '
            "printf '%s' \"$resolved_disk\"; "
            "else "
            'basename "$root_real"; '
            "fi"
        )
        device_name = self._normalize_device_name(device.stdout)
        return self.parser.parse(
            device_name=CommandResult(
                command=device.command,
                exit_code=device.exit_code,
                stdout=device_name,
                stderr=device.stderr,
            ),
            rotational=executor.run(
                f"cat /sys/block/{device_name}/queue/rotational 2>/dev/null || true"
            ),
            scheduler=executor.run(
                f"cat /sys/block/{device_name}/queue/scheduler 2>/dev/null || true"
            ),
            readahead=executor.run(
                f"blockdev --getra /dev/{device_name} 2>/dev/null || printf 'unknown'"
            ),
        )

    def _normalize_device_name(self, raw_value: str) -> str:
        cleaned = raw_value.strip().removeprefix("├─").removeprefix("└─").strip()
        if not cleaned or not re.fullmatch(r"[a-zA-Z0-9._-]+", cleaned):
            return "unknown"
        return cleaned
