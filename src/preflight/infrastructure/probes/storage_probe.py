from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandExecutor, StorageInfo
from preflight.infrastructure.parsers.storage_parser import StorageParser
from preflight.infrastructure.probes.base import BaseProbe


@dataclass(frozen=True)
class StorageProbe(BaseProbe):
    parser: StorageParser

    @property
    def name(self) -> str:
        return "storage"

    def collect(self, executor: CommandExecutor) -> StorageInfo:
        device = executor.run("lsblk -ndo PKNAME $(findmnt -n -o SOURCE /) 2>/dev/null | head -n 1")
        device_name = device.stdout or "unknown"
        return self.parser.parse(
            device_name=device,
            rotational=executor.run(
                f"cat /sys/block/{device_name}/queue/rotational 2>/dev/null || true"
            ),
            scheduler=executor.run(
                f"cat /sys/block/{device_name}/queue/scheduler 2>/dev/null || true"
            ),
        )
