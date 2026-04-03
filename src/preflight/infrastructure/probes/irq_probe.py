from __future__ import annotations

import shlex
from dataclasses import dataclass

from preflight.domain.models import CommandExecutor, IrqInfo
from preflight.infrastructure.parsers.irq_parser import IrqParser
from preflight.infrastructure.probes.base import BaseProbe


@dataclass(frozen=True)
class IrqProbe(BaseProbe):
    parser: IrqParser

    @property
    def name(self) -> str:
        return "irq"

    def collect(self, executor: CommandExecutor) -> IrqInfo:
        irqbalance_status = executor.run(
            "systemctl is-active irqbalance 2>/dev/null || printf 'inactive'"
        )
        interface = executor.run("ip route | awk '/default/ {print $5; exit}'")
        interface_name = interface.stdout.strip() or "unknown"
        # Read smp_affinity_list for all IRQs associated with the default NIC.
        # Extract IRQ numbers from /proc/interrupts lines that match the interface name,
        # then collect unique CPU ranges from each IRQ's smp_affinity_list.
        safe_iface = shlex.quote(interface_name)
        nic_irq_cpu_list = executor.run(
            "iface=" + safe_iface + "; "
            'awk -v nic="$iface" \'$0 ~ nic { split($1,a,":"); if (a[1]+0>0) print a[1]+0 }\' '
            "/proc/interrupts 2>/dev/null "
            "| sort -un "
            "| while IFS= read -r q; do cat /proc/irq/$q/smp_affinity_list 2>/dev/null; done "
            "| sort -u | tr '\\n' ',' | sed 's/,$//'"
        )
        return self.parser.parse(
            irqbalance_status=irqbalance_status,
            nic_irq_cpu_list=nic_irq_cpu_list,
        )
