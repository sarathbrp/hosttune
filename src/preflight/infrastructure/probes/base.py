from __future__ import annotations

from abc import ABC, abstractmethod

from preflight.domain.models import CommandExecutor


class BaseProbe(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable probe name."""

    @abstractmethod
    def collect(self, executor: CommandExecutor) -> object:
        """Collect and normalize one discovery concern."""
