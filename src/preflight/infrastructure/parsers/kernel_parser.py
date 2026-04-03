from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandResult, KernelInfo


@dataclass(frozen=True)
class KernelParser:
    def parse(
        self,
        sysctl_probe: CommandResult,
        selinux_mode: CommandResult,
        tuned_profile: CommandResult,
    ) -> KernelInfo:
        return KernelInfo(
            sysctl_writable=sysctl_probe.exit_code == 0 and sysctl_probe.stdout == "writable",
            selinux_mode=selinux_mode.stdout or "unknown",
            tuned_profile=tuned_profile.stdout or "unknown",
        )
