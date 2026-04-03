from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandResult, MemoryInfo


@dataclass(frozen=True)
class MemoryParser:
    def parse(
        self,
        meminfo: CommandResult,
        hugepages: CommandResult,
        thp_mode: CommandResult,
    ) -> MemoryInfo:
        meminfo_values = self._parse_key_value_lines(meminfo.stdout)
        hugepages_total = self._safe_int(hugepages.stdout)
        return MemoryInfo(
            total_memory_kib=self._extract_kib(meminfo_values.get("MemTotal", "0 kB")),
            swap_total_kib=self._extract_kib(meminfo_values.get("SwapTotal", "0 kB")),
            hugepages_total=hugepages_total,
            transparent_hugepages_mode=thp_mode.stdout or "unknown",
        )

    def _parse_key_value_lines(self, output: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", maxsplit=1)
            parsed[key.strip()] = value.strip()
        return parsed

    def _extract_kib(self, value: str) -> int:
        return self._safe_int(value.split()[0])

    def _safe_int(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return 0
