from __future__ import annotations

from dataclasses import dataclass, field

from tune.domain.apply_models import AppliedChange
from tune.domain.evaluation_models import EvaluationDecision, EvaluationResult
from tune.domain.hypothesis_models import HypothesisRecord, TunePhase
from tune.domain.iteration_record import TuneIterationRecord
from tune.domain.scoreboard_models import (
    DomainImpactScore,
    ParameterImpactScore,
    TuneScoreboard,
    WorkloadImpactScore,
)


@dataclass(frozen=True)
class BestKnownConfiguration:
    score: float
    parameter_values: dict[str, str]
    iteration_number: int


@dataclass
class TuneState:
    current_phase: TunePhase
    total_iterations: int
    phase_iterations: dict[TunePhase, int]
    remaining_budget: dict[TunePhase, int]
    history: list[HypothesisRecord] = field(default_factory=list)
    iteration_records: list[TuneIterationRecord] = field(default_factory=list)
    active_changes: dict[str, AppliedChange] = field(default_factory=dict)
    best_configuration: BestKnownConfiguration | None = None
    # Counts completed iterations since best_configuration last improved (0 while no best).
    iterations_since_best_update: int = 0
    drift_detected: bool = False
    scoreboard: TuneScoreboard = field(default_factory=TuneScoreboard)
    stop_reason: str | None = None
    layer_statuses: dict[str, str] = field(default_factory=dict)

    def best_iteration_config_values(self) -> dict[str, str]:
        if self.best_configuration is None:
            return {}
        return dict(self.best_configuration.parameter_values)

    def final_retained_config_values(self) -> dict[str, str]:
        return {key: change.applied_value for key, change in self.active_changes.items()}

    @classmethod
    def initialize(
        cls: type[TuneState],
        max_iterations: int,
        *,
        use_unified_resolver: bool = False,
        allow_reboot: bool = False,
    ) -> TuneState:
        if use_unified_resolver:
            budgets = cls._allocate_unified_budget(max_iterations, allow_reboot=allow_reboot)
            initial_phase = TunePhase.RESOLVE
        else:
            budgets = cls._allocate_budget(max_iterations)
            initial_phase = TunePhase.KNOWLEDGE_DRIVEN
        return cls(
            current_phase=initial_phase,
            total_iterations=0,
            phase_iterations={phase: 0 for phase in TunePhase},
            remaining_budget=budgets,
        )

    @staticmethod
    def _allocate_budget(max_iterations: int) -> dict[TunePhase, int]:
        if max_iterations <= 0:
            return {phase: 0 for phase in TunePhase}
        if max_iterations < len(TunePhase):
            return {
                phase: 1 if index < max_iterations else 0 for index, phase in enumerate(TunePhase)
            }
        seeds = {
            TunePhase.KNOWLEDGE_DRIVEN: 3,
            TunePhase.WIDE_SWEEP: 2,
            TunePhase.DOMAIN_FOCUS: 2,
            TunePhase.INTERACTION: 1,
            TunePhase.BOUNDARY_PUSH: 1,
            TunePhase.EXPLOIT: 2,
            TunePhase.REBOOT_BATCH: 1,
        }
        total_seed = sum(seeds.values())
        budgets = {phase: 0 for phase in TunePhase}
        for phase, seed in seeds.items():
            budgets[phase] = max(1, round(max_iterations * seed / total_seed))
        while sum(budgets.values()) > max_iterations:
            for phase in seeds:
                if budgets[phase] > 1 and sum(budgets.values()) > max_iterations:
                    budgets[phase] -= 1
        while sum(budgets.values()) < max_iterations:
            budgets[TunePhase.EXPLOIT] += 1
        return budgets

    @staticmethod
    def _allocate_unified_budget(max_iterations: int, *, allow_reboot: bool = False) -> dict[TunePhase, int]:
        """Proportional budget: OPTIMIZE 50% / EXPLOIT 30% / REBOOT_BATCH 20%.

        RESOLVE does not count against max_iterations — it runs however many
        dependency graph layers require. The remaining budget is split:
          - OPTIMIZE (LLM exploration): 50%
          - EXPLOIT  (best-config refinement): 30%
          - REBOOT_BATCH (reboot-required params): 20%

        Scales automatically as max_iterations increases. Minimum 1 per active
        phase (REBOOT_BATCH gets 0 for very small budgets).
        """
        budgets = {phase: 0 for phase in TunePhase}
        if max_iterations <= 0:
            return budgets
        # RESOLVE is uncapped — stop_reason excludes it from exhaustion check.
        # RESOLVE runs on top of max_iterations — it's architectural setup,
        # not user-controllable exploration. RESOLVE budget is generous so it
        # never exhausts; stop_reason already excludes it from budget check.
        budgets[TunePhase.RESOLVE] = max_iterations
        # max_iterations controls ONLY LLM-driven phases (optimize/exploit/reboot).
        if max_iterations == 1:
            budgets[TunePhase.OPTIMIZE] = 1
            return budgets
        # Proportional split: OPTIMIZE 50% / EXPLOIT 30% / REBOOT 20%.
        # If reboot is not allowed by policy, skip REBOOT allocation and give
        # those iterations to OPTIMIZE so max_iterations = effective LLM iterations.
        exploit = max(1, round(max_iterations * 0.30))
        reboot  = max(0, round(max_iterations * 0.20)) if (max_iterations > 3 and allow_reboot) else 0
        optimize = max(1, max_iterations - exploit - reboot)
        # Clamp total to exact max_iterations.
        total = optimize + exploit + reboot
        if total > max_iterations:
            optimize -= (total - max_iterations)
        elif total < max_iterations:
            optimize += (max_iterations - total)
        budgets[TunePhase.OPTIMIZE]     = max(1, optimize)
        budgets[TunePhase.EXPLOIT]      = exploit
        budgets[TunePhase.REBOOT_BATCH] = reboot
        return budgets

    def rebalance_budget(self, pre_applied_count: int) -> None:
        """Redistribute remaining budget after pre-loop KB operations.

        When the KB batch applies many params, exploration is less valuable
        and exploitation should get the freed budget.
        """
        remaining = sum(self.remaining_budget.values())
        if remaining <= 0 or pre_applied_count < 3:
            return  # Only rebalance when KB applied a meaningful batch.
        # Shrink exploration phases, grow EXPLOIT.
        shrinkable = (
            TunePhase.WIDE_SWEEP,
            TunePhase.DOMAIN_FOCUS,
            TunePhase.INTERACTION,
            TunePhase.BOUNDARY_PUSH,
        )
        freed = 0
        for phase in shrinkable:
            current = self.remaining_budget[phase]
            new = max(1, current - 1)
            freed += current - new
            self.remaining_budget[phase] = new
        self.remaining_budget[TunePhase.EXPLOIT] += freed

    def record_iteration(
        self,
        record: TuneIterationRecord,
        history_record: HypothesisRecord,
    ) -> None:
        previous_best = self.best_configuration
        self.total_iterations += 1
        self.phase_iterations[record.phase] += 1
        self.remaining_budget[record.phase] = max(0, self.remaining_budget[record.phase] - 1)
        self.history.append(history_record)
        self.iteration_records.append(record)
        if record.evaluation_result is not None:
            self.drift_detected = self.drift_detected or record.evaluation_result.drift_detected
            self._update_best_configuration(record.evaluation_result, record)
            self._update_scoreboard(record)
        self._refresh_iterations_since_best_update(previous_best)

    def _update_best_configuration(
        self,
        evaluation_result: EvaluationResult,
        record: TuneIterationRecord,
    ) -> None:
        if evaluation_result.decision is not EvaluationDecision.ACCEPT:
            return
        if not evaluation_result.workload_evaluations:
            return
        score = sum(item.relative_change for item in evaluation_result.workload_evaluations) / len(
            evaluation_result.workload_evaluations
        )
        if self.best_configuration is not None and score <= self.best_configuration.score:
            return
        parameter_values = {
            key: change.applied_value for key, change in self.active_changes.items()
        }
        self.best_configuration = BestKnownConfiguration(
            score=score,
            parameter_values=parameter_values,
            iteration_number=record.iteration_number,
        )

    def _refresh_iterations_since_best_update(
        self,
        previous_best: BestKnownConfiguration | None,
    ) -> None:
        if self.best_configuration is None:
            self.iterations_since_best_update = 0
            return
        if previous_best is None:
            self.iterations_since_best_update = 0
            return
        if self.best_configuration.iteration_number != previous_best.iteration_number:
            self.iterations_since_best_update = 0
            return
        if self.best_configuration.score > previous_best.score + 1e-12:
            self.iterations_since_best_update = 0
            return
        self.iterations_since_best_update += 1

    def _update_scoreboard(self, record: TuneIterationRecord) -> None:
        evaluation_result = record.evaluation_result
        if evaluation_result is None or not evaluation_result.workload_evaluations:
            return

        parameter_key = record.hypothesis.parameter_key
        domain = record.hypothesis.domain
        average_change = sum(
            item.relative_change for item in evaluation_result.workload_evaluations
        ) / len(evaluation_result.workload_evaluations)

        parameter_score = self.scoreboard.parameter_scores.setdefault(
            parameter_key,
            ParameterImpactScore(parameter_key=parameter_key, domain=domain),
        )
        parameter_score.evaluated_count += 1
        parameter_score.average_relative_change = self._rolling_average(
            previous_average=parameter_score.average_relative_change,
            previous_count=parameter_score.evaluated_count - 1,
            new_value=average_change,
        )
        parameter_score.best_relative_change = max(
            parameter_score.best_relative_change,
            average_change,
        )
        parameter_score.worst_relative_change = min(
            parameter_score.worst_relative_change,
            average_change,
        )
        self._increment_decision_count(parameter_score, evaluation_result.decision)

        domain_score = self.scoreboard.domain_scores.setdefault(
            domain,
            DomainImpactScore(domain=domain),
        )
        domain_score.evaluated_count += 1
        domain_score.average_relative_change = self._rolling_average(
            previous_average=domain_score.average_relative_change,
            previous_count=domain_score.evaluated_count - 1,
            new_value=average_change,
        )
        self._increment_decision_count(domain_score, evaluation_result.decision)
        if parameter_key not in domain_score.parameter_keys:
            domain_score.parameter_keys = (*domain_score.parameter_keys, parameter_key)

        for workload_evaluation in evaluation_result.workload_evaluations:
            workload_score = self.scoreboard.workload_scores.setdefault(
                workload_evaluation.workload_name,
                WorkloadImpactScore(workload_name=workload_evaluation.workload_name),
            )
            if (
                workload_score.best_parameter_key is None
                or workload_evaluation.relative_change > workload_score.best_relative_change
            ):
                workload_score.best_parameter_key = parameter_key
                workload_score.best_relative_change = workload_evaluation.relative_change
            if (
                workload_score.worst_parameter_key is None
                or workload_evaluation.relative_change < workload_score.worst_relative_change
            ):
                workload_score.worst_parameter_key = parameter_key
                workload_score.worst_relative_change = workload_evaluation.relative_change
            if workload_evaluation.relative_change > 0.0:
                workload_score.win_count += 1
            if workload_evaluation.relative_change < 0.0:
                workload_score.loss_count += 1

    def _increment_decision_count(
        self,
        score: ParameterImpactScore | DomainImpactScore,
        decision: EvaluationDecision,
    ) -> None:
        if decision is EvaluationDecision.ACCEPT:
            score.accepted_count += 1
        elif decision is EvaluationDecision.PROMISING:
            score.promising_count += 1
        elif decision is EvaluationDecision.REJECT:
            score.rejected_count += 1
        else:
            score.inconclusive_count += 1

    def _rolling_average(
        self,
        previous_average: float,
        previous_count: int,
        new_value: float,
    ) -> float:
        return ((previous_average * previous_count) + new_value) / (previous_count + 1)
