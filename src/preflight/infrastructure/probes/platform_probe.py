from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandExecutor, CommandResult, PlatformInfo
from preflight.infrastructure.parsers.platform_parser import PlatformParser
from preflight.infrastructure.probes.base import BaseProbe


@dataclass(frozen=True)
class PlatformProbe(BaseProbe):
    parser: PlatformParser

    @property
    def name(self) -> str:
        return "platform"

    def collect(self, executor: CommandExecutor) -> PlatformInfo:
        hostname = executor.run("hostname")
        os_release = executor.run(". /etc/os-release && printf '%s' \"$PRETTY_NAME\"")
        kernel = executor.run("uname -r")
        virtualization = executor.run("systemd-detect-virt || true")
        container = executor.run("test -f /.dockerenv && printf 'container'")
        return self.parser.parse(
            hostname=hostname,
            os_release=os_release,
            kernel=kernel,
            virtualization=virtualization,
            container=container,
        )

    def collect_raw(self, executor: CommandExecutor) -> dict[str, CommandResult]:
        return {
            "hostname": executor.run("hostname"),
            "os_release": executor.run(". /etc/os-release && printf '%s' \"$PRETTY_NAME\""),
            "kernel": executor.run("uname -r"),
            "virtualization": executor.run("systemd-detect-virt || true"),
            "container": executor.run("test -f /.dockerenv && printf 'container'"),
        }
