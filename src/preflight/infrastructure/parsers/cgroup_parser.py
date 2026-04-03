from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CgroupInfo, CommandResult


@dataclass(frozen=True)
class CgroupParser:
    def parse(
        self,
        cgroup_fs_type: CommandResult,
        controllers: CommandResult,
    ) -> CgroupInfo:
        fs_type = cgroup_fs_type.stdout.strip()
        if fs_type == "cgroup2fs":
            version = "v2"
        elif fs_type == "tmpfs":
            version = "v1"
        else:
            version = "unknown"
        ctrl_set = set(controllers.stdout.lower().split())
        return CgroupInfo(
            cgroup_version=version,
            cpu_controller_available="cpu" in ctrl_set,
            memory_controller_available="memory" in ctrl_set,
        )
