from __future__ import annotations

from dataclasses import dataclass

from tune.domain.hypothesis_models import CandidateParameter, HypothesisRecord, TunePhase


@dataclass(frozen=True)
class HypothesisContext:
    phase: TunePhase
    iteration_number: int
    candidates: tuple[CandidateParameter, ...]
    history: tuple[HypothesisRecord, ...]
