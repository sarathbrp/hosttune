from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandResult, CpuInfo


@dataclass(frozen=True)
class CpuParser:
    def parse(self, lscpu: CommandResult) -> CpuInfo:
        values = self._parse_key_value_lines(lscpu.stdout)
        threads_per_core = self._to_int(values.get("Thread(s) per core", "0"))
        return CpuInfo(
            architecture=values.get("Architecture", "unknown"),
            logical_cores=self._to_int(values.get("CPU(s)", "0")),
            threads_per_core=threads_per_core,
            cores_per_socket=self._to_int(values.get("Core(s) per socket", "0")),
            sockets=self._to_int(values.get("Socket(s)", "0")),
            numa_nodes=self._to_int(values.get("NUMA node(s)", "0")),
            hyperthreading_enabled=threads_per_core > 1,
        )

    def _parse_key_value_lines(self, output: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", maxsplit=1)
            parsed[key.strip()] = value.strip()
        return parsed

    def _to_int(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return 0
