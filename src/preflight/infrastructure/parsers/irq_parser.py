from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandResult, IrqInfo


@dataclass(frozen=True)
class IrqParser:
    def parse(
        self,
        irqbalance_status: CommandResult,
        nic_irq_cpu_list: CommandResult,
    ) -> IrqInfo:
        active = irqbalance_status.stdout.strip() == "active"
        cpu_summary = nic_irq_cpu_list.stdout.strip() or "unknown"
        return IrqInfo(
            irqbalance_active=active,
            nic_irq_cpu_summary=cpu_summary,
        )
