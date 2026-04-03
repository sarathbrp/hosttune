from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandExecutor, MemoryInfo
from preflight.infrastructure.parsers.memory_parser import MemoryParser
from preflight.infrastructure.probes.base import BaseProbe


@dataclass(frozen=True)
class MemoryProbe(BaseProbe):
    parser: MemoryParser

    @property
    def name(self) -> str:
        return "memory"

    def collect(self, executor: CommandExecutor) -> MemoryInfo:
        return self.parser.parse(
            meminfo=executor.run("cat /proc/meminfo"),
            hugepages=executor.run("cat /proc/sys/vm/nr_hugepages"),
            thp_mode=executor.run("cat /sys/kernel/mm/transparent_hugepage/enabled"),
        )
