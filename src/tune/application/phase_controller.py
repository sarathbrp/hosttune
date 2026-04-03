from __future__ import annotations

from dataclasses import dataclass

from tune.domain.hypothesis_models import CandidateParameter, HypothesisStatus, TunePhase
from tune.domain.tune_state import TuneState

PHASE_SEQUENCE = (
    TunePhase.WIDE_SWEEP,
    TunePhase.DOMAIN_FOCUS,
    TunePhase.INTERACTION,
    TunePhase.BOUNDARY_PUSH,
    TunePhase.EXPLOIT,
    TunePhase.REBOOT_BATCH,
)


@dataclass
class PhaseController:
    def determine_phase(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> TunePhase:
        phase = state.current_phase
        while self._should_advance(phase, state, candidates):
            next_phase = self._next_phase(phase)
            if next_phase == phase:
                break
            phase = next_phase
        state.current_phase = phase
        return phase

    def filter_candidates(
        self,
        phase: TunePhase,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> tuple[CandidateParameter, ...]:
        if phase is TunePhase.WIDE_SWEEP:
            return candidates
        if phase is TunePhase.DOMAIN_FOCUS:
            winning_domains = {
                record.hypothesis.domain
                for record in state.history
                if record.status is HypothesisStatus.ACCEPTED
            }
            if winning_domains:
                return tuple(
                    candidate for candidate in candidates if candidate.domain in winning_domains
                )
        if phase is TunePhase.EXPLOIT and state.best_configuration is not None:
            winning_keys = set(state.best_configuration.parameter_values)
            if winning_keys:
                return tuple(
                    candidate for candidate in candidates if candidate.parameter_key in winning_keys
                )
        return candidates

    def should_stop(self, state: TuneState) -> bool:
        return sum(state.remaining_budget.values()) == 0

    def _should_advance(
        self,
        phase: TunePhase,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> bool:
        if state.remaining_budget[phase] == 0:
            return True
        phase_history = [record for record in state.history if record.phase is phase]
        tried_keys = {record.hypothesis.parameter_key for record in phase_history}
        accepted = [
            record for record in phase_history if record.status is HypothesisStatus.ACCEPTED
        ]
        if phase is TunePhase.WIDE_SWEEP:
            return len(tried_keys) >= len(candidates)
        if phase is TunePhase.DOMAIN_FOCUS:
            return len(phase_history) >= max(2, len(accepted))
        if phase is TunePhase.INTERACTION:
            return len(phase_history) >= 2
        if phase is TunePhase.BOUNDARY_PUSH:
            return len(phase_history) >= 2
        if phase is TunePhase.EXPLOIT:
            return len(phase_history) >= max(2, len(accepted))
        return False

    def _next_phase(self, phase: TunePhase) -> TunePhase:
        index = PHASE_SEQUENCE.index(phase)
        if index + 1 >= len(PHASE_SEQUENCE):
            return phase
        return PHASE_SEQUENCE[index + 1]
