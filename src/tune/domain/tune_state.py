from __future__ import annotations

from dataclasses import dataclass, field

from tune.domain.apply_models import AppliedChange
from tune.domain.evaluation_models import EvaluationResult
from tune.domain.hypothesis_models import HypothesisRecord, TunePhase
from tune.domain.iteration_record import TuneIterationRecord


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
    drift_detected: bool = False

    @classmethod
    def initialize(cls: type[TuneState], max_iterations: int) -> TuneState:
        budgets = cls._allocate_budget(max_iterations)
        return cls(
            current_phase=TunePhase.WIDE_SWEEP,
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
            TunePhase.WIDE_SWEEP: 3,
            TunePhase.DOMAIN_FOCUS: 2,
            TunePhase.INTERACTION: 1,
            TunePhase.BOUNDARY_PUSH: 1,
            TunePhase.EXPLOIT: 2,
            TunePhase.REBOOT_BATCH: 1,
        }
        total_seed = sum(seeds.values())
        budgets = {
            phase: max(1, round(max_iterations * seeds[phase] / total_seed)) for phase in TunePhase
        }
        while sum(budgets.values()) > max_iterations:
            for phase in TunePhase:
                if budgets[phase] > 1 and sum(budgets.values()) > max_iterations:
                    budgets[phase] -= 1
        while sum(budgets.values()) < max_iterations:
            budgets[TunePhase.EXPLOIT] += 1
        return budgets

    def record_iteration(
        self,
        record: TuneIterationRecord,
        history_record: HypothesisRecord,
    ) -> None:
        self.total_iterations += 1
        self.phase_iterations[record.phase] += 1
        self.remaining_budget[record.phase] = max(0, self.remaining_budget[record.phase] - 1)
        self.history.append(history_record)
        self.iteration_records.append(record)
        if record.evaluation_result is not None:
            self.drift_detected = self.drift_detected or record.evaluation_result.drift_detected
            self._update_best_configuration(record.evaluation_result, record)

    def _update_best_configuration(
        self,
        evaluation_result: EvaluationResult,
        record: TuneIterationRecord,
    ) -> None:
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
