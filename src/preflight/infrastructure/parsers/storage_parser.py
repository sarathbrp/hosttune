from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandResult, StorageInfo


@dataclass(frozen=True)
class StorageParser:
    def parse(
        self,
        device_name: CommandResult,
        rotational: CommandResult,
        scheduler: CommandResult,
    ) -> StorageInfo:
        device = device_name.stdout or "unknown"
        rotational_value = rotational.stdout.strip()
        scheduler_value = scheduler.stdout.strip() or "unknown"
        device_type = self._resolve_device_type(device, rotational_value)
        return StorageInfo(
            device_name=device,
            device_type=device_type,
            scheduler=scheduler_value,
            scheduler_meaningful=device_type not in {"nvme", "unknown"},
        )

    def _resolve_device_type(self, device_name: str, rotational: str) -> str:
        if device_name.startswith("nvme"):
            return "nvme"
        if device_name.startswith("vd"):
            return "virtio"
        if rotational == "1":
            return "rotational"
        if rotational == "0":
            return "ssd"
        return "unknown"
