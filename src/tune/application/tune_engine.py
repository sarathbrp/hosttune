from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from preflight.domain.models import CommandExecutor
from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from tune.application.apply_coordinator import ApplyCoordinator
from tune.application.attribution_verifier import AttributionVerifier
from tune.application.benchmark_executor import TuneBenchmarkExecutor
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.application.health_validator import HealthValidator
from tune.application.phase_controller import PhaseController
from tune.application.pre_apply_validator import PreApplyValidator
from tune.application.result_evaluator import ResultEvaluator
from tune.application.rollback_coordinator import RollbackCoordinator
from tune.application.tune_recorder import TuneRecorder
from tune.domain.benchmark_models import TuneBenchmarkResult
from tune.domain.evaluation_models import (
    AttributionVerificationResult,
    EvaluationDecision,
    EvaluationResult,
)
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import (
    CandidateAvailability,
    CandidateParameter,
    HypothesisRecord,
    HypothesisStatus,
    TunePhase,
    TuningHypothesis,
)
from tune.domain.iteration_record import TuneIterationRecord
from tune.domain.tune_context import TuneContext
from tune.domain.tune_state import TuneState
from tune.domain.validation_models import ValidationResult


class SupportsHypothesisGeneration(Protocol):
    def generate(self, context: HypothesisContext) -> TuningHypothesis:
        """Generate one validated tuning hypothesis."""


@dataclass
class TuneEngine:
    candidate_catalog_builder: CandidateCatalogBuilder
    phase_controller: PhaseController
    hypothesis_generator: SupportsHypothesisGeneration
    apply_coordinator: ApplyCoordinator
    pre_apply_validator: PreApplyValidator
    health_validator: HealthValidator
    benchmark_executor: TuneBenchmarkExecutor
    attribution_verifier: AttributionVerifier
    result_evaluator: ResultEvaluator
    rollback_coordinator: RollbackCoordinator
    recorder: TuneRecorder
    logger: ExecutionLogger = NullExecutionLogger()

    def run(
        self,
        context: TuneContext,
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
    ) -> TuneState:
        state = TuneState.initialize(context.preflight.policy.max_iterations)
        all_candidates = self.candidate_catalog_builder.build(context, target_executor)
        deferred_catalog = tuple(
            c for c in all_candidates if c.availability is CandidateAvailability.DEFERRED
        )
        self.logger.stage_start("tune")
        baseline_checks = self.health_validator.validate_baseline(context, target_executor)
        baseline_failed_checks = tuple(check for check in baseline_checks if not check.passed)
        if baseline_failed_checks:
            detail = ", ".join(f"{check.name}: {check.detail}" for check in baseline_failed_checks)
            self.logger.stage_detail(
                "tune",
                f"Pre-tune health gate failed ({detail})",
            )
            raise ValueError(f"Pre-tune health gate failed: {detail}")
        self.logger.stage_detail("tune", "Pre-tune health gate passed.")
        if deferred_catalog:
            self.logger.stage_detail(
                "tune",
                f"Catalog: {len(deferred_catalog)} deferred (reboot_batch) sysctl candidate(s).",
            )
        while not self.phase_controller.should_stop(state, all_candidates):
            previous_phase = state.current_phase
            phase = self.phase_controller.determine_phase(state, all_candidates)
            if phase is not previous_phase:
                self.logger.stage_detail(
                    "tune",
                    f"Phase advanced: {previous_phase.value} -> {phase.value}",
                )
            candidates = self.phase_controller.filter_candidates(
                phase,
                state,
                all_candidates,
                allow_reboot=context.preflight.policy.allow_reboot,
            )
            if not candidates:
                if phase is TunePhase.REBOOT_BATCH and deferred_catalog:
                    self.logger.stage_detail(
                        "tune",
                        (
                            f"REBOOT_BATCH phase has {len(deferred_catalog)} deferred "
                            "candidate(s) but engagement policy disallows reboot. "
                            "Set allow_reboot=true to unlock."
                        ),
                    )
                else:
                    self.logger.stage_detail(
                        "tune", "No eligible candidates remain for current phase."
                    )
                break
            iteration_number = state.total_iterations + 1
            self.logger.stage_detail("tune", f"Iteration {iteration_number} phase={phase.value}")
            previous_best_iteration = (
                None
                if state.best_configuration is None
                else state.best_configuration.iteration_number
            )
            record, history_record = self._run_iteration(
                context=context,
                state=state,
                phase=phase,
                iteration_number=iteration_number,
                candidates=candidates,
                deferred_candidates=deferred_catalog,
                target_executor=target_executor,
                benchmark_executor=benchmark_executor,
            )
            state.record_iteration(record, history_record)
            if (
                state.best_configuration is not None
                and state.best_configuration.iteration_number != previous_best_iteration
            ):
                self.logger.stage_detail(
                    "tune",
                    (
                        "Best config updated: "
                        f"iteration={state.best_configuration.iteration_number} "
                        f"score={state.best_configuration.score:.2%}"
                    ),
                )
            self.recorder.record(context, record)
            self.recorder.record_scoreboard(context, state.scoreboard)
        provisional_keys = {
            record.hypothesis.parameter_key
            for record in state.history
            if record.status is HypothesisStatus.PROMISING
            and record.hypothesis.parameter_key in state.active_changes
        }
        if provisional_keys:
            self.logger.stage_detail(
                "tune",
                (
                    f"Active changes include {len(provisional_keys)} provisional "
                    f"(unverified) parameter(s): {', '.join(sorted(provisional_keys))}"
                ),
            )
        self.logger.stage_end("tune")
        return state

    def _run_iteration(
        self,
        context: TuneContext,
        state: TuneState,
        phase: TunePhase,
        iteration_number: int,
        candidates: tuple[CandidateParameter, ...],
        deferred_candidates: tuple[CandidateParameter, ...],
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
    ) -> tuple[TuneIterationRecord, HypothesisRecord]:
        started_at = datetime.now(UTC)
        started_timer = perf_counter()
        hypothesis = self.hypothesis_generator.generate(
            HypothesisContext(
                tune_context=context,
                phase=phase,
                iteration_number=iteration_number,
                candidates=candidates,
                deferred_candidates=deferred_candidates,
                history=tuple(state.history),
                active_parameter_keys=tuple(sorted(state.active_changes)),
                best_parameter_values=(
                    tuple(sorted(state.best_configuration.parameter_values.items()))
                    if state.best_configuration is not None
                    else ()
                ),
            )
        )
        candidate = self._find_candidate(candidates, hypothesis.parameter_key)
        self._log_hypothesis(hypothesis, candidate, state)
        pre_apply_validation = self.pre_apply_validator.validate(candidate, hypothesis)
        if not pre_apply_validation.allowed:
            self.logger.stage_detail(
                "tune",
                (
                    "Pre-apply rejection: "
                    f"tuning_layer={candidate.tuning_layer.value} "
                    f"parameter={hypothesis.parameter_key} "
                    f"reason={pre_apply_validation.reason}"
                ),
            )
            completed_at = datetime.now(UTC)
            duration_seconds = perf_counter() - started_timer
            record = TuneIterationRecord(
                iteration_number=iteration_number,
                phase=phase,
                hypothesis=hypothesis,
                applied_change=None,
                validation_result=None,
                benchmark_result=None,
                evaluation_result=None,
                attribution_verification=None,
                active_parameter_keys=tuple(sorted(state.active_changes)),
                started_at_utc=started_at.isoformat(),
                completed_at_utc=completed_at.isoformat(),
                duration_seconds=duration_seconds,
            )
            history_record = HypothesisRecord(
                iteration_number=iteration_number,
                phase=phase,
                hypothesis=hypothesis,
                status=HypothesisStatus.REJECTED_PRE_APPLY,
                evaluation_summary=pre_apply_validation.reason,
            )
            return record, history_record
        try:
            applied_change = self.apply_coordinator.apply(context, hypothesis, target_executor)
        except ValueError as exc:
            completed_at = datetime.now(UTC)
            duration_seconds = perf_counter() - started_timer
            self.logger.stage_detail("tune", f"Apply failed: {exc}")
            record = TuneIterationRecord(
                iteration_number=iteration_number,
                phase=phase,
                hypothesis=hypothesis,
                applied_change=None,
                validation_result=None,
                benchmark_result=None,
                evaluation_result=None,
                attribution_verification=None,
                active_parameter_keys=tuple(sorted(state.active_changes)),
                started_at_utc=started_at.isoformat(),
                completed_at_utc=completed_at.isoformat(),
                duration_seconds=duration_seconds,
            )
            history_record = HypothesisRecord(
                iteration_number=iteration_number,
                phase=phase,
                hypothesis=hypothesis,
                status=HypothesisStatus.FAILED_VALIDATION,
                evaluation_summary=f"apply failed: {exc}",
            )
            return record, history_record
        self.logger.stage_detail(
            "tune",
            (
                "Apply: "
                f"tuning_layer={candidate.tuning_layer.value} "
                f"parameter={hypothesis.parameter_key} "
                f"previous={applied_change.previous_value} "
                f"applied={applied_change.applied_value} "
                f"mode={applied_change.apply_mode.value}"
            ),
        )
        validation_result = self.health_validator.validate(context, applied_change, target_executor)
        self._log_validation(validation_result)

        benchmark_result = None
        evaluation_result = None
        attribution_verification = None
        if not validation_result.healthy:
            failed_checks = (
                ", ".join(
                    f"{check.name}: {check.detail}"
                    for check in validation_result.checks
                    if not check.passed
                )
                or "unknown validation failure"
            )
            self.logger.stage_detail(
                "tune",
                f"Benchmark skipped: validation failed ({failed_checks})",
            )
            self.rollback_coordinator.rollback(applied_change, target_executor)
            self.logger.stage_detail(
                "tune",
                (
                    "Rollback: "
                    f"tuning_layer={candidate.tuning_layer.value} "
                    f"parameter={hypothesis.parameter_key} "
                    "reason=validation_failed"
                ),
            )
            status = HypothesisStatus.FAILED_VALIDATION
        else:
            benchmark_result = self.benchmark_executor.run(
                context=context,
                iteration_number=iteration_number,
                validation_result=validation_result,
                benchmark_executor=benchmark_executor,
            )
            self._log_benchmark(benchmark_result)
            evaluation_result = self.result_evaluator.evaluate(
                context, benchmark_result, phase=phase
            )
            if evaluation_result.decision is EvaluationDecision.ACCEPT:
                attribution_verification = self.attribution_verifier.verify(
                    context=context,
                    iteration_number=iteration_number,
                    applied_change=applied_change,
                    accepted_benchmark_result=benchmark_result,
                    target_executor=target_executor,
                    benchmark_runner_executor=benchmark_executor,
                )
                self._log_attribution_verification(attribution_verification)
                if attribution_verification.verified:
                    evaluation_result = replace(
                        evaluation_result,
                        attribution_verified=True,
                        attribution_summary=attribution_verification.summary,
                    )
                else:
                    evaluation_result = replace(
                        evaluation_result,
                        decision=EvaluationDecision.INCONCLUSIVE,
                        summary=(
                            f"{evaluation_result.summary}; "
                            f"attribution_unverified={attribution_verification.summary}"
                        ),
                        attribution_verified=False,
                        attribution_summary=attribution_verification.summary,
                    )
            self._log_evaluation(evaluation_result)
            status = self._resolve_status(evaluation_result)
            if status is HypothesisStatus.ACCEPTED:
                state.active_changes[hypothesis.parameter_key] = applied_change
                self.logger.stage_detail(
                    "tune",
                    (
                        "Decision: accepted; attribution verified; "
                        f"tuning_layer={candidate.tuning_layer.value} "
                        f"parameter={hypothesis.parameter_key}; "
                        "retaining change as active configuration."
                    ),
                )
            elif status is HypothesisStatus.PROMISING:
                state.active_changes[hypothesis.parameter_key] = applied_change
                self.logger.stage_detail(
                    "tune",
                    (
                        "Decision: promising (wide sweep directional signal); "
                        f"tuning_layer={candidate.tuning_layer.value} "
                        f"parameter={hypothesis.parameter_key}; "
                        "retaining change without attribution verification."
                    ),
                )
            elif (
                status is HypothesisStatus.INCONCLUSIVE
                and attribution_verification is not None
                and not attribution_verification.verified
            ):
                self.rollback_coordinator.rollback(applied_change, target_executor)
                self.logger.stage_detail(
                    "tune",
                    (
                        "Decision revised: attribution unverified; "
                        f"tuning_layer={candidate.tuning_layer.value} "
                        f"parameter={hypothesis.parameter_key} not retained; "
                        "rolled back."
                    ),
                )
            else:
                self.rollback_coordinator.rollback(applied_change, target_executor)
                self.logger.stage_detail(
                    "tune",
                    (
                        "Rollback: "
                        f"tuning_layer={candidate.tuning_layer.value} "
                        f"parameter={hypothesis.parameter_key} "
                        f"reason={evaluation_result.decision.value}"
                    ),
                )

        completed_at = datetime.now(UTC)
        duration_seconds = perf_counter() - started_timer
        active_parameter_keys = tuple(sorted(state.active_changes))
        record = TuneIterationRecord(
            iteration_number=iteration_number,
            phase=phase,
            hypothesis=hypothesis,
            applied_change=applied_change,
            validation_result=validation_result,
            benchmark_result=benchmark_result,
            evaluation_result=evaluation_result,
            attribution_verification=attribution_verification,
            active_parameter_keys=active_parameter_keys,
            started_at_utc=started_at.isoformat(),
            completed_at_utc=completed_at.isoformat(),
            duration_seconds=duration_seconds,
        )
        history_record = HypothesisRecord(
            iteration_number=iteration_number,
            phase=phase,
            hypothesis=hypothesis,
            status=status,
            evaluation_summary=evaluation_result.summary if evaluation_result is not None else None,
        )
        return record, history_record

    def _find_candidate(
        self,
        candidates: tuple[CandidateParameter, ...],
        parameter_key: str,
    ) -> CandidateParameter:
        for candidate in candidates:
            if candidate.parameter_key == parameter_key:
                return candidate
        msg = f"No candidate found for hypothesis parameter: {parameter_key}"
        raise ValueError(msg)

    def _resolve_status(self, evaluation_result: EvaluationResult) -> HypothesisStatus:
        if evaluation_result.decision is EvaluationDecision.ACCEPT:
            return HypothesisStatus.ACCEPTED
        if evaluation_result.decision is EvaluationDecision.PROMISING:
            return HypothesisStatus.PROMISING
        if evaluation_result.decision is EvaluationDecision.REJECT:
            return HypothesisStatus.REJECTED
        return HypothesisStatus.INCONCLUSIVE

    def _log_hypothesis(
        self,
        hypothesis: TuningHypothesis,
        candidate: CandidateParameter,
        state: TuneState,
    ) -> None:
        self.logger.stage_detail(
            "tune",
            (
                "Hypothesis: "
                f"phase={hypothesis.phase.value} "
                f"tuning_layer={candidate.tuning_layer.value} "
                f"domain={hypothesis.domain} "
                f"parameter={hypothesis.parameter_key} "
                f"value={hypothesis.proposed_value} "
                f"mode={hypothesis.apply_mode.value} "
                f"reason={hypothesis.rationale}"
            ),
        )
        if hypothesis.model_usage is not None:
            cumulative_input = hypothesis.model_usage.input_tokens + sum(
                record.hypothesis.model_usage.input_tokens
                for record in state.iteration_records
                if record.hypothesis.model_usage is not None
            )
            cumulative_output = hypothesis.model_usage.output_tokens + sum(
                record.hypothesis.model_usage.output_tokens
                for record in state.iteration_records
                if record.hypothesis.model_usage is not None
            )
            cumulative_total = hypothesis.model_usage.total_tokens + sum(
                record.hypothesis.model_usage.total_tokens
                for record in state.iteration_records
                if record.hypothesis.model_usage is not None
            )
            self.logger.stage_detail(
                "tune",
                (
                    "Hypothesis tokens: "
                    f"model={hypothesis.model_usage.model_name} "
                    f"input={hypothesis.model_usage.input_tokens} "
                    f"output={hypothesis.model_usage.output_tokens} "
                    f"total={hypothesis.model_usage.total_tokens} "
                    f"cumulative_input={cumulative_input} "
                    f"cumulative_output={cumulative_output} "
                    f"cumulative_total={cumulative_total}"
                ),
            )

    def _log_attribution_verification(
        self,
        verification_result: AttributionVerificationResult,
    ) -> None:
        self.logger.stage_detail(
            "tune",
            (
                "Attribution verification: "
                f"verified={verification_result.verified} "
                f"summary={verification_result.summary}"
            ),
        )

    def _log_validation(self, validation_result: ValidationResult) -> None:
        passed_checks = sum(1 for check in validation_result.checks if check.passed)
        total_checks = len(validation_result.checks)
        self.logger.stage_detail(
            "tune",
            f"Validate: healthy={validation_result.healthy} checks={passed_checks}/{total_checks}",
        )
        for check in validation_result.checks:
            self.logger.stage_detail(
                "tune",
                f"Validate check: {check.name} passed={check.passed} detail={check.detail}",
            )

    def _log_benchmark(self, benchmark_result: TuneBenchmarkResult) -> None:
        self.logger.stage_detail(
            "tune",
            (
                "Benchmark: "
                f"stable={benchmark_result.stable} "
                f"run_count={benchmark_result.run_count} "
                f"variance_threshold={benchmark_result.variance_threshold:.2%}"
            ),
        )
        for summary in benchmark_result.workload_summaries:
            self.logger.stage_detail(
                "tune",
                (
                    f"Benchmark workload: {summary.workload_name} "
                    f"rps={summary.median_requests_per_second:.2f} "
                    f"latency_ms={summary.median_latency_ms:.2f} "
                    f"variance={summary.relative_variance:.2%} "
                    f"stable={summary.stable}"
                ),
            )

    def _log_evaluation(self, evaluation_result: EvaluationResult) -> None:
        self.logger.stage_detail(
            "tune",
            (
                "Evaluate: "
                f"decision={evaluation_result.decision.value} "
                f"guardrails_held={evaluation_result.guardrails_held} "
                f"drift_detected={evaluation_result.drift_detected}"
            ),
        )
        self.logger.stage_detail("tune", f"Evaluate summary: {evaluation_result.summary}")
        for workload in evaluation_result.workload_evaluations:
            self.logger.stage_detail(
                "tune",
                (
                    f"Evaluate workload: {workload.workload_name} "
                    f"baseline_rps={workload.baseline_requests_per_second:.2f} "
                    f"current_rps={workload.current_requests_per_second:.2f} "
                    f"change={workload.relative_change:.2%} "
                    f"above_noise_floor={workload.above_noise_floor}"
                ),
            )
