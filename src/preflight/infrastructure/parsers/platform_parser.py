from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandResult, PlatformInfo


@dataclass(frozen=True)
class PlatformParser:
    def parse(
        self,
        hostname: CommandResult,
        os_release: CommandResult,
        kernel: CommandResult,
        virtualization: CommandResult,
        container: CommandResult,
    ) -> PlatformInfo:
        virtualization_type = virtualization.stdout or "none"
        is_container = container.exit_code == 0 and container.stdout == "container"
        return PlatformInfo(
            hostname=hostname.stdout or "unknown",
            operating_system=os_release.stdout or "unknown",
            kernel_version=kernel.stdout or "unknown",
            virtualization_type=virtualization_type,
            is_container=is_container,
        )
