from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.kernel_sysctl_profile import PREFLIGHT_SYSCTL_KEYS
from preflight.domain.models import CommandResult, KernelInfo


@dataclass(frozen=True)
class KernelParser:
    def parse(
        self,
        sysctl_probe: CommandResult,
        selinux_mode: CommandResult,
        tuned_profile: CommandResult,
        sysctl_profile_dump: CommandResult,
    ) -> KernelInfo:
        return KernelInfo(
            sysctl_writable=sysctl_probe.exit_code == 0 and sysctl_probe.stdout == "writable",
            selinux_mode=selinux_mode.stdout or "unknown",
            tuned_profile=tuned_profile.stdout or "unknown",
            sysctl_profile=self.parse_sysctl_profile_stdout(
                sysctl_profile_dump.stdout,
                keys=PREFLIGHT_SYSCTL_KEYS,
            ),
        )

    @staticmethod
    def parse_sysctl_profile_stdout(
        stdout: str,
        *,
        keys: tuple[str, ...] = PREFLIGHT_SYSCTL_KEYS,
    ) -> tuple[tuple[str, str], ...]:
        parsed: dict[str, str] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            parsed[name] = value.strip()
        return tuple((key, parsed.get(key, "")) for key in keys)
