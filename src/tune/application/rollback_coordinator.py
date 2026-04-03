from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandExecutor
from tune.domain.apply_models import AppliedChange


@dataclass
class RollbackCoordinator:
    def rollback(self, applied_change: AppliedChange, executor: CommandExecutor) -> None:
        result = executor.run(applied_change.rollback_command)
        if result.exit_code != 0:
            msg = f"Rollback failed: {result.stderr or result.stdout}"
            raise ValueError(msg)
