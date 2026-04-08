from __future__ import annotations

import re
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
    # Configurable stopping thresholds (from config.yaml tune.stopping.*)
    marginal_gain_threshold: float = 0.03    # homepage gain < 3% = marginal
    marginal_gain_iterations: int = 2        # consecutive marginal iters before stop
    historical_best_pct: float = 0.85       # stop at 85% of KB historical best
    telemetry_stop_enabled: bool = True      # enable telemetry deterioration stop

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
        Keys that only saw mechanical failures (pre-apply rejection or validation
        failure with no benchmark/evaluation) stay eligible for retry.
        """
        positive_keys: set[str] = set()
        tried_keys: set[str] = set()
        statuses_by_key: dict[str, set[HypothesisStatus]] = {}
        for record in state.history:
            key = record.hypothesis.parameter_key
            if key == "__no_hypothesis__":
                continue
            tried_keys.add(key)
            statuses_by_key.setdefault(key, set()).add(record.status)
            if record.status in (HypothesisStatus.ACCEPTED, HypothesisStatus.PROMISING):
                positive_keys.add(key)
        mechanical_failure_only_keys = {
            key
            for key, statuses in statuses_by_key.items()
            if statuses.issubset(
                {
                    HypothesisStatus.FAILED_VALIDATION,
                    HypothesisStatus.REJECTED_PRE_APPLY,
                }
            )
        }
        failed_keys = tried_keys - positive_keys - mechanical_failure_only_keys
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
        # RESOLVE does not count against max_iterations — exclude it.
        non_resolve_budget = sum(
            v for phase, v in state.remaining_budget.items()
            if phase is not TunePhase.RESOLVE
        )
        if non_resolve_budget == 0:
            return "budget_exhausted"
        if self._has_converged(state, active_catalog_candidates(candidates)):
            return "converged"
        if (
            state.best_configuration is None
            and self._consecutive_failures(state) >= self.consecutive_failure_limit
        ):
            return "consecutive_failures"
        if self._is_marginal_gain_plateau(state):
            return "marginal_gain_plateau"
        if self._near_historical_best(state):
            return "near_historical_best"
        if self.telemetry_stop_enabled and self._telemetry_deteriorating(state):
            return "telemetry_deterioration"
        return None

    def _is_marginal_gain_plateau(self, state: TuneState) -> bool:
        """Stop if last N benchmarked iterations all had marginal homepage gain
        AND large/medium workloads showed no improvement.
        Only considers non-RESOLVE iterations — resolver gains are expected."""
        benchmarked = [
            r for r in state.iteration_records
            if r.evaluation_result is not None and r.phase is not TunePhase.RESOLVE
        ]
        if len(benchmarked) < self.marginal_gain_iterations:
            return False
        recent = benchmarked[-self.marginal_gain_iterations:]
        for rec in recent:
            evals = {w.workload_name: w.relative_change for w in rec.evaluation_result.workload_evaluations}
            hp_change = evals.get("homepage", 0.0)
            if hp_change >= self.marginal_gain_threshold:
                return False  # at least one recent iter had meaningful homepage gain
        # All recent iters had marginal homepage gain; also check large/medium are stuck
        for rec in recent:
            evals = {w.workload_name: w.relative_change for w in rec.evaluation_result.workload_evaluations}
            if evals.get("large", 0.0) >= self.marginal_gain_threshold:
                return False
            if evals.get("medium", 0.0) >= self.marginal_gain_threshold:
                return False
        return True

    def _near_historical_best(self, state: TuneState) -> bool:
        """Stop when homepage RPS reaches historical_best_pct of KB best.

        Only fires in OPTIMIZE/EXPLOIT — not during RESOLVE (which is
        deterministic setup and may already exceed the stale KB best).
        Requires at least 1 non-RESOLVE benchmarked iteration so we don't
        stop before the LLM has had a chance to explore.
        """
        if state.kb_best_homepage_rps <= 0 or state.best_configuration is None:
            return False
        # Only active after RESOLVE phase is done.
        if state.current_phase is TunePhase.RESOLVE:
            return False
        # Require at least 1 LLM iteration (non-resolve benchmarked).
        llm_benchmarked = [
            r for r in state.iteration_records
            if r.phase is not TunePhase.RESOLVE and r.evaluation_result is not None
        ]
        if not llm_benchmarked:
            return False
        for rec in reversed(state.iteration_records):
            if rec.evaluation_result is not None:
                for w in rec.evaluation_result.workload_evaluations:
                    if w.workload_name == "homepage":
                        return w.current_requests_per_second >= self.historical_best_pct * state.kb_best_homepage_rps
        return False

    def _telemetry_deteriorating(self, state: TuneState) -> bool:
        """Stop if 2 consecutive recent iterations show worsening telemetry signals."""
        benchmarked = [
            r for r in state.iteration_records
            if r.benchmark_result is not None and r.benchmark_result.runtime_telemetry
        ]
        if len(benchmarked) < 2:
            return False
        signals: list[dict[str, float]] = []
        for rec in benchmarked[-2:]:
            digest = "\n".join(
                s.softnet_stat + "\n" + s.vmstat_s
                for s in rec.benchmark_result.runtime_telemetry
                if s.softnet_stat or s.vmstat_s
            )
            sq_match = re.search(r"time_squeeze total:\s*([\d,]+)", digest)
            squeeze = int(sq_match.group(1).replace(",", "")) if sq_match else 0
            sys_match = re.search(r"cpu_us=\d+%\s+sy=(\d+)%", digest)
            cpu_sys = int(sys_match.group(1)) if sys_match else 0
            cs_match = re.search(r"cs=([\d,]+)/s", digest)
            cs = int(cs_match.group(1).replace(",", "")) if cs_match else 0
            signals.append({"squeeze": squeeze, "cpu_sys": cpu_sys, "cs": cs})
        if len(signals) < 2:
            return False
        bad_signals = 0
        if signals[1]["squeeze"] > signals[0]["squeeze"] * 1.5 and signals[1]["squeeze"] > 10_000:
            bad_signals += 1
        if signals[1]["cpu_sys"] > 25 and signals[1]["cpu_sys"] > signals[0]["cpu_sys"] * 1.3:
            bad_signals += 1
        if signals[1]["cs"] > 5_000_000 and signals[1]["cs"] > signals[0]["cs"] * 1.3:
            bad_signals += 1
        return bad_signals >= 2

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
            # Only count iterations that ran a benchmark — pre-apply rejections
            # (no-op checks, constraint violations) don't represent genuine LLM
            # hypothesis tests and should not trigger phase advancement.
            benchmarked = [
                r for r in phase_history
                if r.status is not HypothesisStatus.REJECTED_PRE_APPLY
            ]
            if len(benchmarked) >= 2 and not positive_signal:
                return True
            # Safety valve: avoid looping indefinitely on repeated pre-apply failures.
            return len(phase_history) >= 5 and not positive_signal
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
