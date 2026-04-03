from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandExecutor, NetworkInfo
from preflight.infrastructure.parsers.network_parser import NetworkParser
from preflight.infrastructure.probes.base import BaseProbe


@dataclass(frozen=True)
class NetworkProbe(BaseProbe):
    parser: NetworkParser

    @property
    def name(self) -> str:
        return "network"

    def collect(self, executor: CommandExecutor) -> NetworkInfo:
        interface = executor.run("ip route | awk '/default/ {print $5; exit}'")
        interface_name = interface.stdout or "unknown"
        return self.parser.parse(
            interface_name=interface,
            driver_info=executor.run(f"ethtool -i {interface_name} || true"),
            ring_info=executor.run(f"ethtool -g {interface_name} || true"),
            queue_info=executor.run(f"ethtool -l {interface_name} || true"),
        )
