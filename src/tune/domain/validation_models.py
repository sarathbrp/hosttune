from __future__ import annotations

from dataclasses import dataclass

from tune.domain.apply_models import AppliedChange


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    applied_change: AppliedChange
    healthy: bool
    checks: tuple[ValidationCheck, ...]
