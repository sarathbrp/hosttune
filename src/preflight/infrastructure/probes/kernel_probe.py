from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.kernel_sysctl_profile import sysctl_profile_read_command
from preflight.domain.models import CommandExecutor, KernelInfo
from preflight.infrastructure.parsers.kernel_parser import KernelParser
from preflight.infrastructure.probes.base import BaseProbe


@dataclass(frozen=True)
class KernelProbe(BaseProbe):
    parser: KernelParser

    @property
    def name(self) -> str:
        return "kernel"

    def collect(self, executor: CommandExecutor) -> KernelInfo:
        return self.parser.parse(
            sysctl_probe=executor.run("test -w /proc/sys/vm/swappiness && printf 'writable'"),
            selinux_mode=executor.run("getenforce || printf 'unknown'"),
            tuned_profile=executor.run("tuned-adm active | awk -F': ' 'NR==1 {print $2}' || true"),
            sysctl_profile_dump=executor.run(sysctl_profile_read_command()),
        )
