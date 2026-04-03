from __future__ import annotations

from dataclasses import dataclass

from onboard.domain.models import ApplyMode
from tune.domain.hypothesis_models import TuningHypothesis


@dataclass(frozen=True)
class AppliedChange:
    hypothesis: TuningHypothesis
    target_path: str
    previous_value: str
    applied_value: str
    apply_mode: ApplyMode
    apply_command: str
    rollback_command: str
