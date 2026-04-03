from __future__ import annotations

from dataclasses import dataclass

from onboard.domain.models import PriorityTier
from tune.domain.evaluation_models import EvaluationDecision
from tune.domain.hypothesis_models import (
    CandidateAvailability,
    CandidateParameter,
    HypothesisRecord,
    HypothesisStatus,
    TunePhase,
)
from tune.domain.tune_state import TuneState
from tune.domain.tuning_layer import TuningLayer

PHASE_SEQUENCE = (
    TunePhase.WIDE_SWEEP,
    TunePhase.DOMAIN_FOCUS,
    TunePhase.INTERACTION,
    TunePhase.BOUNDARY_PUSH,
    TunePhase.EXPLOIT,
    TunePhase.REBOOT_BATCH,
)


def active_catalog_candidates(
    candidates: tuple[CandidateParameter, ...],
) -> tuple[CandidateParameter, ...]:
    return tuple(c for c in candidates if c.availability is CandidateAvailability.ACTIVE)


@dataclass
class PhaseController:
    convergence_no_signal_limit: int = 3
    convergence_best_stability_limit: int = 2

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
        *,
        allow_reboot: bool = False,
    ) -> tuple[CandidateParameter, ...]:
        if phase is TunePhase.REBOOT_BATCH:
            if not allow_reboot:
                return ()
            return tuple(c for c in candidates if c.availability is CandidateAvailability.DEFERRED)
        active = active_catalog_candidates(candidates)
        if phase is TunePhase.WIDE_SWEEP:
            return self._filter_by_priority_tier(state, active)
        if phase is TunePhase.DOMAIN_FOCUS:
            return self._filter_domain_focus(state, active)
        if phase is TunePhase.EXPLOIT and state.best_configuration is not None:
            winning_keys = set(state.best_configuration.parameter_values)
            if winning_keys:
                return tuple(c for c in active if c.parameter_key in winning_keys)
        return active

    def _filter_domain_focus(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> tuple[CandidateParameter, ...]:
        """
        Keep candidates in domains that produced an accept, or on any tuning layer that
        produced an accept (C7 — layer-aligned focus beyond the domain string alone).
        """
        positive = [
            record
            for record in state.history
            if record.status in (HypothesisStatus.ACCEPTED, HypothesisStatus.PROMISING)
        ]
        if not positive:
            return candidates
        winning_domains = {record.hypothesis.domain for record in positive}
        winning_layers = {record.hypothesis.tuning_layer for record in positive}
        return tuple(
            candidate
            for candidate in candidates
            if candidate.domain in winning_domains or candidate.tuning_layer in winning_layers
        )

    def should_stop(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> bool:
        if sum(state.remaining_budget.values()) == 0:
            return True
        return self._has_converged(state, active_catalog_candidates(candidates))

    def _should_advance(
        self,
        phase: TunePhase,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> bool:
        candidates = active_catalog_candidates(candidates)
        if state.remaining_budget[phase] == 0:
            return True
        phase_history = [record for record in state.history if record.phase is phase]
        tried_keys = {record.hypothesis.parameter_key for record in phase_history}
        positive_signal = [
            record
            for record in phase_history
            if record.status in (HypothesisStatus.ACCEPTED, HypothesisStatus.PROMISING)
        ]
        if phase is TunePhase.WIDE_SWEEP:
            mandate = self._wide_sweep_mandate_keys(candidates, phase_history)
            return tried_keys >= mandate
        if phase is TunePhase.DOMAIN_FOCUS:
            return len(phase_history) >= max(2, len(positive_signal))
        if phase is TunePhase.INTERACTION:
            return len(phase_history) >= 2
        if phase is TunePhase.BOUNDARY_PUSH:
            return len(phase_history) >= 2
        if phase is TunePhase.EXPLOIT:
            return len(phase_history) >= max(2, len(positive_signal))
        return False

    def _next_phase(self, phase: TunePhase) -> TunePhase:
        index = PHASE_SEQUENCE.index(phase)
        if index + 1 >= len(PHASE_SEQUENCE):
            return phase
        return PHASE_SEQUENCE[index + 1]

    def _wide_sweep_history(self, state: TuneState) -> list[HypothesisRecord]:
        return [record for record in state.history if record.phase is TunePhase.WIDE_SWEEP]

    def _wide_sweep_low_tier_budget_allowed(
        self,
        wide_sweep_history: list[HypothesisRecord],
        candidates: tuple[CandidateParameter, ...],
    ) -> bool:
        """
        LOW tier runs only after HIGH and MEDIUM are fully exercised in Wide Sweep with
        no accepted win in either tier (curated exploration budget).
        """
        tried_keys = {record.hypothesis.parameter_key for record in wide_sweep_history}
        accept_keys = {
            record.hypothesis.parameter_key
            for record in wide_sweep_history
            if record.status is HypothesisStatus.ACCEPTED
        }
        high_keys = {c.parameter_key for c in candidates if c.priority_tier is PriorityTier.HIGH}
        medium_keys = {
            c.parameter_key for c in candidates if c.priority_tier is PriorityTier.MEDIUM
        }

        def tier_exhausted_without_wide_sweep_win(tier_keys: set[str]) -> bool:
            if not tier_keys:
                return True
            if tier_keys - tried_keys:
                return False
            return not (accept_keys & tier_keys)

        return tier_exhausted_without_wide_sweep_win(
            high_keys
        ) and tier_exhausted_without_wide_sweep_win(medium_keys)

    def _wide_sweep_mandate_keys(
        self,
        candidates: tuple[CandidateParameter, ...],
        wide_sweep_history: list[HypothesisRecord],
    ) -> set[str]:
        keys = {
            c.parameter_key
            for c in candidates
            if c.priority_tier in (PriorityTier.HIGH, PriorityTier.MEDIUM)
        }
        if self._wide_sweep_low_tier_budget_allowed(wide_sweep_history, candidates):
            keys |= {c.parameter_key for c in candidates if c.priority_tier is PriorityTier.LOW}
        return keys

    def _wide_sweep_tried_layers(
        self,
        phase_history: list[HypothesisRecord],
    ) -> set[TuningLayer]:
        return {record.hypothesis.tuning_layer for record in phase_history}

    def _select_wide_sweep_tier_candidates(
        self,
        tier_candidates: tuple[CandidateParameter, ...],
        tried_domains: set[str],
        tried_layers: set[TuningLayer],
    ) -> tuple[CandidateParameter, ...]:
        """
        Prefer knobs on tuning layers not yet touched in Wide Sweep, then new domains,
        then any remaining tier member (matches prior domain-only behavior as fallback).
        """
        if not tier_candidates:
            return ()
        fresh_layer = tuple(
            candidate for candidate in tier_candidates if candidate.tuning_layer not in tried_layers
        )
        fresh_layer_fresh_domain = tuple(
            candidate for candidate in fresh_layer if candidate.domain not in tried_domains
        )
        if fresh_layer_fresh_domain:
            return fresh_layer_fresh_domain
        if fresh_layer:
            return fresh_layer
        fresh_domain = tuple(
            candidate for candidate in tier_candidates if candidate.domain not in tried_domains
        )
        if fresh_domain:
            return fresh_domain
        return tier_candidates

    def _filter_by_priority_tier(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> tuple[CandidateParameter, ...]:
        phase_history = self._wide_sweep_history(state)
        tried_keys = {record.hypothesis.parameter_key for record in phase_history}
        tried_domains = {record.hypothesis.domain for record in phase_history}
        tried_layers = self._wide_sweep_tried_layers(phase_history)
        low_allowed = self._wide_sweep_low_tier_budget_allowed(phase_history, candidates)

        for tier in (PriorityTier.HIGH, PriorityTier.MEDIUM):
            tier_candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.priority_tier is tier and candidate.parameter_key not in tried_keys
            )
            selected = self._select_wide_sweep_tier_candidates(
                tier_candidates,
                tried_domains,
                tried_layers,
            )
            if selected:
                return selected

        if low_allowed:
            tier_candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.priority_tier is PriorityTier.LOW
                and candidate.parameter_key not in tried_keys
            )
            selected = self._select_wide_sweep_tier_candidates(
                tier_candidates,
                tried_domains,
                tried_layers,
            )
            if selected:
                return selected

        return ()

    def _has_converged(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> bool:
        if not self._recent_measured_lack_clear_improvement(state):
            return False
        if not self._high_impact_domains_explored_without_signal(state, candidates):
            return False
        return self._best_configuration_stable(state)

    def _recent_measured_lack_clear_improvement(self, state: TuneState) -> bool:
        """True when last N benchmarked iterations had no ACCEPT."""
        measured = [
            record
            for record in state.iteration_records
            if record.evaluation_result is not None and record.benchmark_result is not None
        ]
        if len(measured) < self.convergence_no_signal_limit:
            return False
        recent = measured[-self.convergence_no_signal_limit :]
        return all(
            record.evaluation_result is not None
            and record.evaluation_result.decision is not EvaluationDecision.ACCEPT
            for record in recent
        )

    def _high_impact_domains_explored_without_signal(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> bool:
        """
        Every domain that contains a HIGH-tier candidate has had all such candidates tried,
        and no HIGH-tier hypothesis was ever accepted.
        """
        high_priority_keys = {
            candidate.parameter_key
            for candidate in candidates
            if candidate.priority_tier is PriorityTier.HIGH
        }
        if not high_priority_keys:
            return True
        high_impact_domains = {
            candidate.domain
            for candidate in candidates
            if candidate.priority_tier is PriorityTier.HIGH
        }
        tried_keys = {record.hypothesis.parameter_key for record in state.history}
        for domain in high_impact_domains:
            domain_high_keys = {
                candidate.parameter_key
                for candidate in candidates
                if candidate.priority_tier is PriorityTier.HIGH and candidate.domain == domain
            }
            if domain_high_keys - tried_keys:
                return False
        high_priority_history = [
            record
            for record in state.history
            if record.hypothesis.parameter_key in high_priority_keys
        ]
        return not any(
            record.status is HypothesisStatus.ACCEPTED for record in high_priority_history
        )

    def _best_configuration_stable(self, state: TuneState) -> bool:
        if state.best_configuration is None:
            return True
        return state.iterations_since_best_update >= self.convergence_best_stability_limit
