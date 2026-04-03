from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CgroupInfo, CommandExecutor
from preflight.infrastructure.parsers.cgroup_parser import CgroupParser
from preflight.infrastructure.probes.base import BaseProbe


@dataclass(frozen=True)
class CgroupProbe(BaseProbe):
    parser: CgroupParser

    @property
    def name(self) -> str:
        return "cgroup"

    def collect(self, executor: CommandExecutor) -> CgroupInfo:
        # "tmpfs" → cgroup v1, "cgroup2fs" → cgroup v2
        cgroup_fs_type = executor.run("stat -fc %T /sys/fs/cgroup 2>/dev/null || printf 'unknown'")
        # Available controllers are listed only on v2; empty/missing on v1.
        controllers = executor.run("cat /sys/fs/cgroup/cgroup.controllers 2>/dev/null || true")
        return self.parser.parse(
            cgroup_fs_type=cgroup_fs_type,
            controllers=controllers,
        )
