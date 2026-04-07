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
    TunePhase.KNOWLEDGE_DRIVEN,
    TunePhase.WIDE_SWEEP,
    TunePhase.DOMAIN_FOCUS,
    TunePhase.INTERACTION,
    TunePhase.BOUNDARY_PUSH,
    TunePhase.EXPLOIT,
    TunePhase.REBOOT_BATCH,
)

UNIFIED_PHASE_SEQUENCE = (
    TunePhase.RESOLVE,
    TunePhase.OPTIMIZE,
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
    consecutive_failure_limit: int = 3
    use_unified_resolver: bool = False

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
        # Unified resolver phases.
        if phase is TunePhase.RESOLVE:
            return active  # Resolver handles ordering.
        if phase is TunePhase.OPTIMIZE:
            # Only show candidates NOT actively applied by the resolver.
            resolved_keys = set(state.active_changes.keys())
            active = self._suppress_failed_keys(state, active)
            return tuple(
                c for c in active
                if c.parameter_key not in resolved_keys
            )
        # Suppress parameters that were tried with no positive signal.
        # EXPLOIT is exempt — it refines around the best config.
        if phase not in (
            TunePhase.KNOWLEDGE_DRIVEN,
            TunePhase.EXPLOIT,
            TunePhase.REBOOT_BATCH,
        ):
            active = self._suppress_failed_keys(state, active)
        if phase is TunePhase.KNOWLEDGE_DRIVEN:
            return self._filter_knowledge_driven(state, active)
        if phase is TunePhase.WIDE_SWEEP:
            return self._filter_by_priority_tier(state, active)
        if phase is TunePhase.DOMAIN_FOCUS:
            focused = self._filter_domain_focus(state, active)
            return self._suppress_positive_keys(state, focused)
        if phase is TunePhase.INTERACTION:
            return self._suppress_positive_keys(state, active)
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

    def _filter_knowledge_driven(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> tuple[CandidateParameter, ...]:
        """Select candidates with KB confidence >= 50%, ordered by confidence desc.

        Skips candidates already tried in this session. Returns empty when
        no KB-scored candidates remain, triggering advancement to WIDE_SWEEP.
        """
        tried_keys = {record.hypothesis.parameter_key for record in state.history}
        # confidence_scores are stored on the most recent HypothesisContext,
        # but we need them here. Use the state's iteration records to find them.
        # For now, filter candidates that are NOT yet tried and have HIGH/MEDIUM
        # priority (KB data drives the warm-start; this phase validates remaining
        # high-value candidates that weren't auto-applied).
        untried = tuple(
            c
            for c in candidates
            if c.parameter_key not in tried_keys and c.parameter_key not in state.active_changes
        )
        if not untried:
            return ()
        # Sort by priority tier (HIGH first) — KB confidence is reflected in
        # auto-apply (100%) and warm-start; this phase handles the rest.
        from onboard.domain.models import PriorityTier

        high = tuple(c for c in untried if c.priority_tier is PriorityTier.HIGH)
        medium = tuple(c for c in untried if c.priority_tier is PriorityTier.MEDIUM)
        return high or medium or ()

    def _suppress_failed_keys(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> tuple[CandidateParameter, ...]:
        """Remove candidates whose parameter key was tried with no positive signal.

        If every attempt for a key resulted in rejected/inconclusive/failed,
        suppress it to avoid wasting iterations on dead-end parameters.
        Keys with at least one accepted/promising attempt are kept.
        """
        positive_keys: set[str] = set()
        tried_keys: set[str] = set()
        for record in state.history:
            key = record.hypothesis.parameter_key
            if key == "__no_hypothesis__":
                continue
            tried_keys.add(key)
            if record.status in (HypothesisStatus.ACCEPTED, HypothesisStatus.PROMISING):
                positive_keys.add(key)
        failed_keys = tried_keys - positive_keys
        if not failed_keys:
            return candidates
        return tuple(
            c for c in candidates if c.parameter_key not in failed_keys
        )

    def _suppress_positive_keys(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> tuple[CandidateParameter, ...]:
        positive_keys = {
            record.hypothesis.parameter_key
            for record in state.history
            if record.status in (HypothesisStatus.ACCEPTED, HypothesisStatus.PROMISING)
        }
        if not positive_keys:
            return candidates
        return tuple(
            candidate for candidate in candidates if candidate.parameter_key not in positive_keys
        )

    def should_stop(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> bool:
        return self.stop_reason(state, candidates) is not None

    def stop_reason(
        self,
        state: TuneState,
        candidates: tuple[CandidateParameter, ...],
    ) -> str | None:
        if sum(state.remaining_budget.values()) == 0:
            return "budget_exhausted"
        if self._has_converged(state, active_catalog_candidates(candidates)):
            return "converged"
        if (
            state.best_configuration is None
            and self._consecutive_failures(state) >= self.consecutive_failure_limit
        ):
            return "consecutive_failures"
        return None

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
        # Unified resolver phases.
        if phase is TunePhase.RESOLVE:
            return len(phase_history) >= 1
        if phase is TunePhase.OPTIMIZE:
            if not candidates:
                return True
            return len(phase_history) >= 2 and not positive_signal
        # Adaptive: advance if all phase iterations failed (no signal).
        if (
            len(phase_history) >= 3
            and not positive_signal
            and phase
            not in (TunePhase.KNOWLEDGE_DRIVEN, TunePhase.WIDE_SWEEP)
        ):
            return True
        if phase is TunePhase.KNOWLEDGE_DRIVEN:
            return not self._filter_knowledge_driven(state, candidates)
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
        seq = (
            UNIFIED_PHASE_SEQUENCE
            if self.use_unified_resolver
            else PHASE_SEQUENCE
        )
        if phase not in seq:
            return phase
        index = seq.index(phase)
        if index + 1 >= len(seq):
            return phase
        return seq[index + 1]

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

    def _consecutive_failures(self, state: TuneState) -> int:
        """Count consecutive non-accept iterations from the tail of history."""
        count = 0
        for record in reversed(state.history):
            if record.status in (HypothesisStatus.ACCEPTED, HypothesisStatus.PROMISING):
                break
            if record.status in (HypothesisStatus.REJECTED, HypothesisStatus.FAILED_VALIDATION):
                count += 1
        return count

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
