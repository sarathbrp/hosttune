from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandExecutor, CpuInfo
from preflight.infrastructure.parsers.cpu_parser import CpuParser
from preflight.infrastructure.probes.base import BaseProbe


@dataclass(frozen=True)
class CpuProbe(BaseProbe):
    parser: CpuParser

    @property
    def name(self) -> str:
        return "cpu"

    def collect(self, executor: CommandExecutor) -> CpuInfo:
        return self.parser.parse(executor.run("lscpu"))
