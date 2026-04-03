from __future__ import annotations

from dataclasses import dataclass

from tune.domain.hypothesis_models import CandidateParameter, HypothesisRecord, TunePhase
from tune.domain.tune_context import TuneContext


@dataclass(frozen=True)
class HypothesisContext:
    tune_context: TuneContext
    phase: TunePhase
    iteration_number: int
    candidates: tuple[CandidateParameter, ...]
    history: tuple[HypothesisRecord, ...]
    active_parameter_keys: tuple[str, ...]
    best_parameter_values: tuple[tuple[str, str], ...]
