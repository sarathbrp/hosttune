from __future__ import annotations

from abc import ABC, abstractmethod

from preflight.domain.models import CommandResult


class BaseCommandExecutor(ABC):
    @abstractmethod
    def run(self, command: str) -> CommandResult:
        """Run a command and return the structured result."""
