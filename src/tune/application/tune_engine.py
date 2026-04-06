from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from onboard.domain.models import ApplyMode
from preflight.domain.models import CommandExecutor
from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from snapshot.domain.models import SnapshotResult
from tune.application.apply_coordinator import ApplyCoordinator
from tune.application.attribution_verifier import AttributionVerifier
from tune.application.benchmark_executor import TuneBenchmarkExecutor
from tune.application.benchmark_runtime_telemetry import format_runtime_telemetry_digest
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.application.health_validator import HealthValidator
from tune.application.phase_controller import PhaseController
from tune.application.pre_apply_validator import PreApplyValidator
from tune.application.result_evaluator import ResultEvaluator
from tune.application.rollback_coordinator import RollbackCoordinator
from tune.application.tune_recorder import TuneRecorder
from tune.domain.apply_models import AppliedChange
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
    CandidateSource,
    HypothesisRecord,
    HypothesisStatus,
    TunePhase,
    TuningHypothesis,
)
from tune.domain.iteration_record import TuneIterationRecord
from tune.domain.tune_context import TuneContext
from tune.domain.tune_state import TuneState
from tune.domain.tuning_layer import TuningLayer
from tune.domain.validation_models import ValidationResult


class SupportsHypothesisGeneration(Protocol):
    def generate(self, context: HypothesisContext) -> tuple[TuningHypothesis, ...]:
        """Generate one or more validated tuning hypotheses (one per domain)."""


def _refresh_snapshot_runtime_state(
    context: TuneContext,
    executor: CommandExecutor,
    logger: ExecutionLogger | None = None,
) -> SnapshotResult:
    """Re-run runtime_state_command so the LLM sees the current live config."""
    from dataclasses import replace as dc_replace

    cmd = context.onboard.service.snapshot.runtime_state_command
    if cmd is None:
        return context.snapshot
    result = executor.run(cmd)
    if result.exit_code != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        if logger is not None:
            logger.stage_detail(
                "tune",
                f"Snapshot refresh failed (exit={result.exit_code}): {detail}; "
                "using stale snapshot",
            )
        return context.snapshot
    return dc_replace(context.snapshot, runtime_state_output=result.stdout)


def _last_benchmark_runtime_telemetry_digest(
    iteration_records: list[TuneIterationRecord],
) -> str:
    for record in reversed(iteration_records):
        benchmark = record.benchmark_result
        if benchmark is not None and benchmark.runtime_telemetry:
            return format_runtime_telemetry_digest(
                benchmark.runtime_telemetry,
                max_chars_per_section=420,
            )
    return format_runtime_telemetry_digest((), max_chars_per_section=420)


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
        self._record_kb_event(
            context=context,
            component="tuning_executor",
            event_type="pre_tune_health_gate",
            payload={
                "passed": True,
                "checks": [
                    {"name": check.name, "passed": check.passed, "detail": check.detail}
                    for check in baseline_checks
                ],
            },
        )
        # KB-driven warm start: pre-apply best config from prior similar run.
        self._warm_start_from_prior_runs(context, state, all_candidates, target_executor)
        if deferred_catalog:
            self.logger.stage_detail(
                "tune",
                f"Catalog: {len(deferred_catalog)} deferred (reboot_batch) sysctl candidate(s).",
            )
        # KB-driven blocked pairs: load prior-run failures to avoid re-trying.
        prior_blocked = self._load_prior_blocked_pairs(context)
        if prior_blocked:
            self.logger.stage_detail(
                "tune",
                f"KB: loaded {len(prior_blocked)} blocked pairs from prior runs.",
            )
        else:
            self.logger.stage_detail("tune", "KB: no prior blocked pairs found.")
        prev_active_keys: set[str] = set()
        stop_reason = self.phase_controller.stop_reason(state, all_candidates)
        while stop_reason is None:
            # Rebuild catalog only when active changes changed (saves ~30s SSH per iteration).
            current_active_keys = set(state.active_changes)
            if current_active_keys != prev_active_keys:
                all_candidates = self.candidate_catalog_builder.build(context, target_executor)
                deferred_catalog = tuple(
                    c for c in all_candidates if c.availability is CandidateAvailability.DEFERRED
                )
                prev_active_keys = current_active_keys.copy()
            previous_phase = state.current_phase
            phase = self.phase_controller.determine_phase(state, all_candidates)
            if phase is not previous_phase:
                self.logger.stage_detail(
                    "tune",
                    f"Phase advanced: {previous_phase.value} -> {phase.value}",
                )
                self._record_kb_event(
                    context=context,
                    component="phase_controller",
                    event_type="phase_transition",
                    phase=phase,
                    payload={
                        "from_phase": previous_phase.value,
                        "to_phase": phase.value,
                    },
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
                    stop_reason = "reboot_blocked"
                else:
                    self.logger.stage_detail(
                        "tune", "No eligible candidates remain for current phase."
                    )
                    stop_reason = "no_candidates"
                    self._record_kb_event(
                        context=context,
                        component="phase_controller",
                        event_type="no_candidates",
                        phase=phase,
                        payload={"phase": phase.value},
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
                prior_blocked_pairs=tuple(prior_blocked),
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
                self._record_kb_event(
                    context=context,
                    component="best_config_tracker",
                    event_type="best_config_updated",
                    iteration_number=state.best_configuration.iteration_number,
                    phase=phase,
                    payload={
                        "score": state.best_configuration.score,
                        "parameter_values": state.best_configuration.parameter_values,
                        "workloads": []
                        if record.evaluation_result is None
                        else [
                            {
                                "workload_name": item.workload_name,
                                "relative_change": item.relative_change,
                                "current_requests_per_second": item.current_requests_per_second,
                            }
                            for item in record.evaluation_result.workload_evaluations
                        ],
                    },
                )
            self.recorder.record(context, record)
            self.recorder.record_scoreboard(context, state.scoreboard)
            stop_reason = self.phase_controller.stop_reason(state, all_candidates)
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
        state.stop_reason = stop_reason or "completed"
        self._restore_best_configuration(state, target_executor)
        self._record_kb_event(
            context=context,
            component="convergence_logic",
            event_type="stop_reason",
            payload={"stop_reason": state.stop_reason},
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
        prior_blocked_pairs: tuple[tuple[str, str], ...],
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
    ) -> tuple[TuneIterationRecord, HypothesisRecord]:
        started_at = datetime.now(UTC)
        started_timer = perf_counter()
        # Refresh runtime_state (e.g. nginx -T) so the LLM sees the live config,
        # not the stale snapshot captured before tuning started.
        live_snapshot = _refresh_snapshot_runtime_state(context, target_executor, self.logger)
        live_context = replace(context, snapshot=live_snapshot)
        hyp_context = HypothesisContext(
            tune_context=live_context,
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
            last_benchmark_runtime_telemetry_digest=_last_benchmark_runtime_telemetry_digest(
                state.iteration_records
            ),
            prior_blocked_pairs=prior_blocked_pairs,
        )
        try:
            hypotheses = self.hypothesis_generator.generate(hyp_context)
        except Exception as exc:
            # Catch broadly so a single bad LLM response or parse error doesn't crash
            # the entire session. Log the full exception type for debugging.
            self.logger.stage_detail(
                "tune",
                (
                    f"Hypothesis generation failed ({type(exc).__name__}): {exc} "
                    "— skipping iteration."
                ),
            )
            self._record_kb_event(
                context=context,
                component="hybrid_llm",
                event_type="hypothesis_generation_failed",
                iteration_number=iteration_number,
                phase=phase,
                payload={"error": str(exc)},
            )
            completed_at = datetime.now(UTC)
            duration_seconds = perf_counter() - started_timer
            # Build a placeholder record so the engine can continue to the next iteration.
            placeholder = TuningHypothesis(
                phase=phase,
                parameter_key="__no_hypothesis__",
                parameter_name="__no_hypothesis__",
                domain="none",
                tuning_layer=next(iter(candidates)).tuning_layer
                if candidates
                else TuningLayer.SERVICE,
                proposed_value="",
                source=next(iter(candidates)).source
                if candidates
                else CandidateSource.SERVICE_DIRECTIVE,
                apply_mode=ApplyMode.RELOAD,
                rationale=str(exc),
            )
            record = TuneIterationRecord(
                iteration_number=iteration_number,
                phase=phase,
                hypothesis=placeholder,
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
                hypothesis=placeholder,
                status=HypothesisStatus.FAILED_VALIDATION,
                evaluation_summary=f"hypothesis generation failed: {exc}",
            )
            return record, history_record

        # Primary hypothesis drives the iteration record; companions are logged and co-applied.
        primary = hypotheses[0]
        primary_candidate = self._find_candidate(candidates, primary.parameter_key)
        self._log_hypothesis(primary, primary_candidate, state)
        for companion in hypotheses[1:]:
            companion_candidate = self._find_candidate(candidates, companion.parameter_key)
            self._log_hypothesis(companion, companion_candidate, state)

        # Pre-apply validate all; skip invalid companions, reject if primary fails.
        valid: list[TuningHypothesis] = []
        primary_pre_apply = self.pre_apply_validator.validate(primary_candidate, primary)
        if not primary_pre_apply.allowed:
            self.logger.stage_detail(
                "tune",
                (
                    "Pre-apply rejection: "
                    f"tuning_layer={primary_candidate.tuning_layer.value} "
                    f"parameter={primary.parameter_key} "
                    f"reason={primary_pre_apply.reason}"
                ),
            )
            self._record_kb_event(
                context=context,
                component="tuning_executor",
                event_type="pre_apply_rejected",
                iteration_number=iteration_number,
                phase=phase,
                payload={
                    "parameter_key": primary.parameter_key,
                    "reason": primary_pre_apply.reason,
                },
            )
            completed_at = datetime.now(UTC)
            duration_seconds = perf_counter() - started_timer
            record = TuneIterationRecord(
                iteration_number=iteration_number,
                phase=phase,
                hypothesis=primary,
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
                hypothesis=primary,
                status=HypothesisStatus.REJECTED_PRE_APPLY,
                evaluation_summary=primary_pre_apply.reason,
            )
            return record, history_record
        valid.append(primary)
        for companion in hypotheses[1:]:
            companion_candidate = self._find_candidate(candidates, companion.parameter_key)
            companion_pre_apply = self.pre_apply_validator.validate(companion_candidate, companion)
            if companion_pre_apply.allowed:
                valid.append(companion)
            else:
                self.logger.stage_detail(
                    "tune",
                    (
                        "Companion pre-apply skipped: "
                        f"parameter={companion.parameter_key} "
                        f"reason={companion_pre_apply.reason}"
                    ),
                )

        # Apply all valid hypotheses; roll back all on any failure.
        applied_changes: dict[str, AppliedChange] = {}
        apply_error: Exception | None = None
        for h in valid:
            try:
                ac = self.apply_coordinator.apply(context, h, target_executor)
                applied_changes[h.parameter_key] = ac
                h_candidate = self._find_candidate(candidates, h.parameter_key)
                self.logger.stage_detail(
                    "tune",
                    (
                        "Apply: "
                        f"tuning_layer={h_candidate.tuning_layer.value} "
                        f"parameter={h.parameter_key} "
                        f"previous={ac.previous_value} "
                        f"applied={ac.applied_value} "
                        f"mode={ac.apply_mode.value}"
                    ),
                )
                self._record_kb_event(
                    context=context,
                    component="tuning_executor",
                    event_type="change_applied",
                    iteration_number=iteration_number,
                    phase=phase,
                    payload={
                        "parameter_key": h.parameter_key,
                        "previous_value": ac.previous_value,
                        "applied_value": ac.applied_value,
                        "apply_mode": ac.apply_mode.value,
                        "apply_command": ac.apply_command,
                    },
                )
            except Exception as exc:
                apply_error = exc
                self.logger.stage_detail("tune", f"Apply failed for {h.parameter_key}: {exc}")
                self._record_kb_event(
                    context=context,
                    component="tuning_executor",
                    event_type="apply_failed",
                    iteration_number=iteration_number,
                    phase=phase,
                    payload={"parameter_key": h.parameter_key, "error": str(exc)},
                )
                break

        if apply_error is not None:
            self._rollback_all(applied_changes, target_executor)
            completed_at = datetime.now(UTC)
            duration_seconds = perf_counter() - started_timer
            record = TuneIterationRecord(
                iteration_number=iteration_number,
                phase=phase,
                hypothesis=primary,
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
                hypothesis=primary,
                status=HypothesisStatus.FAILED_VALIDATION,
                evaluation_summary=f"apply failed: {apply_error}",
            )
            return record, history_record

        # Single health check and benchmark across all applied changes.
        primary_applied_change = applied_changes[primary.parameter_key]
        validation_result = self.health_validator.validate(
            context, primary_applied_change, target_executor
        )
        self._log_validation(validation_result)
        self._record_kb_event(
            context=context,
            component="tuning_executor",
            event_type="validation_completed",
            iteration_number=iteration_number,
            phase=phase,
            payload={
                "parameter_key": primary.parameter_key,
                "healthy": validation_result.healthy,
                "checks": [
                    {"name": check.name, "passed": check.passed, "detail": check.detail}
                    for check in validation_result.checks
                ],
            },
        )

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
            self._rollback_all(applied_changes, target_executor)
            self.logger.stage_detail(
                "tune",
                (
                    "Rollback (all): "
                    f"parameters={list(applied_changes)} "
                    "reason=validation_failed"
                ),
            )
            self._record_kb_event(
                context=context,
                component="tuning_executor",
                event_type="rollback_completed",
                iteration_number=iteration_number,
                phase=phase,
                payload={
                    "parameters": sorted(applied_changes),
                    "reason": "validation_failed",
                },
            )
            status = HypothesisStatus.FAILED_VALIDATION
        else:
            benchmark_result = self.benchmark_executor.run(
                context=context,
                iteration_number=iteration_number,
                validation_result=validation_result,
                benchmark_executor=benchmark_executor,
                telemetry_executor=target_executor,
            )
            self._log_benchmark(benchmark_result)
            self._record_kb_event(
                context=context,
                component="benchmark_runner",
                event_type="benchmark_completed",
                iteration_number=iteration_number,
                phase=phase,
                payload={
                    "stable": benchmark_result.stable,
                    "run_count": benchmark_result.run_count,
                    "variance_threshold": benchmark_result.variance_threshold,
                    "workloads": [
                        {
                            "workload_name": item.workload_name,
                            "median_requests_per_second": item.median_requests_per_second,
                            "median_total_requests": item.median_total_requests,
                            "median_latency_ms": item.median_latency_ms,
                            "relative_variance": item.relative_variance,
                            "stable": item.stable,
                        }
                        for item in benchmark_result.workload_summaries
                    ],
                },
            )
            evaluation_result = self.result_evaluator.evaluate(
                context, benchmark_result, phase=phase
            )
            self._record_kb_event(
                context=context,
                component="benchmark_runner",
                event_type="evaluation_completed",
                iteration_number=iteration_number,
                phase=phase,
                payload={
                    "decision": evaluation_result.decision.value,
                    "summary": evaluation_result.summary,
                    "guardrails_held": evaluation_result.guardrails_held,
                    "drift_detected": evaluation_result.drift_detected,
                    "workloads": [
                        {
                            "workload_name": item.workload_name,
                            "baseline_requests_per_second": item.baseline_requests_per_second,
                            "current_requests_per_second": item.current_requests_per_second,
                            "relative_change": item.relative_change,
                            "above_noise_floor": item.above_noise_floor,
                        }
                        for item in evaluation_result.workload_evaluations
                    ],
                },
            )
            if evaluation_result.decision is EvaluationDecision.ACCEPT:
                # Skip attribution for overwhelming gains (>50% avg improvement).
                # The signal is self-evident; save ~150s of rollback/re-benchmark.
                avg_change = sum(
                    w.relative_change for w in evaluation_result.workload_evaluations
                ) / max(len(evaluation_result.workload_evaluations), 1)
                if avg_change > 0.50:
                    self.logger.stage_detail(
                        "tune",
                        (
                            f"Attribution skipped: avg improvement {avg_change:.1%} "
                            "> 50% threshold; accepting without verification."
                        ),
                    )
                    attribution_verification = AttributionVerificationResult(
                        verified=True,
                        summary=f"skipped (overwhelming gain {avg_change:.1%})",
                        reverted_benchmark_result=None,
                        average_drop=avg_change,
                    )
                else:
                    attribution_verification = self.attribution_verifier.verify(
                        context=context,
                        iteration_number=iteration_number,
                        applied_change=primary_applied_change,
                        accepted_benchmark_result=benchmark_result,
                        target_executor=target_executor,
                        benchmark_runner_executor=benchmark_executor,
                    )
                self._log_attribution_verification(attribution_verification)
                self._record_kb_event(
                    context=context,
                    component="benchmark_runner",
                    event_type="attribution_verification_completed",
                    iteration_number=iteration_number,
                    phase=phase,
                    payload={
                        "verified": attribution_verification.verified,
                        "summary": attribution_verification.summary,
                        "average_drop": attribution_verification.average_drop,
                    },
                )
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
            keep = status in {HypothesisStatus.ACCEPTED, HypothesisStatus.PROMISING}
            inconclusive_unverified = (
                status is HypothesisStatus.INCONCLUSIVE
                and attribution_verification is not None
                and not attribution_verification.verified
            )
            # When rollback_required=false, retain changes even on REJECT/INCONCLUSIVE
            # so the operator can accumulate changes without reverting each one.
            # Attribution-unverified INCONCLUSIVE is always rolled back regardless —
            # it means we couldn't confirm the improvement is real.
            rollback_override = (
                not context.preflight.policy.rollback_required
                and not inconclusive_unverified
                and not keep
            )
            if rollback_override:
                keep = True
            if keep:
                for param_key, ac in applied_changes.items():
                    state.active_changes[param_key] = ac
                applied_keys = ", ".join(sorted(applied_changes))
                if rollback_override:
                    self.logger.stage_detail(
                        "tune",
                        (
                            f"Decision: {status.value}; "
                            f"parameters={applied_keys}; "
                            "retaining despite negative outcome "
                            "(rollback_required=false in policy)."
                        ),
                    )
                else:
                    self.logger.stage_detail(
                        "tune",
                        (
                            f"Decision: {status.value}; "
                            f"parameters={applied_keys}; "
                            "retaining all applied changes."
                        ),
                    )
            else:
                self._rollback_all(applied_changes, target_executor)
                reason = (
                    "attribution_unverified"
                    if inconclusive_unverified
                    else evaluation_result.decision.value
                )
                self.logger.stage_detail(
                    "tune",
                    ("Rollback (all): " f"parameters={list(applied_changes)} " f"reason={reason}"),
                )
                self._record_kb_event(
                    context=context,
                    component="tuning_executor",
                    event_type="rollback_completed",
                    iteration_number=iteration_number,
                    phase=phase,
                    payload={
                        "parameters": sorted(applied_changes),
                        "reason": reason,
                    },
                )

        completed_at = datetime.now(UTC)
        duration_seconds = perf_counter() - started_timer
        active_parameter_keys = tuple(sorted(state.active_changes))
        record = TuneIterationRecord(
            iteration_number=iteration_number,
            phase=phase,
            hypothesis=primary,
            applied_change=primary_applied_change,
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
            hypothesis=primary,
            status=status,
            evaluation_summary=evaluation_result.summary if evaluation_result is not None else None,
        )
        return record, history_record

    def _rollback_all(
        self,
        applied_changes: dict[str, AppliedChange],
        target_executor: CommandExecutor,
    ) -> None:
        failures: list[str] = []
        for param_key, ac in applied_changes.items():
            try:
                self.rollback_coordinator.rollback(ac, target_executor)
            except Exception as exc:
                failures.append(param_key)
                self.logger.stage_detail("tune", f"ROLLBACK FAILED for {param_key}: {exc}")
        if failures:
            self.logger.stage_detail(
                "tune",
                f"CRITICAL: partial rollback — still applied: {failures}",
            )

    def _warm_start_from_prior_runs(
        self,
        context: TuneContext,
        state: TuneState,
        all_candidates: tuple[CandidateParameter, ...],
        target_executor: CommandExecutor,
    ) -> None:
        """Pre-apply best config from prior similar run to skip rediscovery."""
        kb = getattr(context, "knowledge_base", None)
        artifacts = context.artifacts
        if kb is None or artifacts is None:
            return
        prior_config = kb.get_prior_best_config(
            service_name=context.onboard.service_name,
            cpu_logical_cores=context.preflight.cpu.logical_cores,
            numa_nodes=context.preflight.cpu.numa_nodes,
            platform_summary=context.preflight.platform_summary,
            nic_driver=context.preflight.network.driver_name,
            exclude_run_id=artifacts.session_id,
        )
        if not prior_config:
            self.logger.stage_detail("tune", "KB: no prior best config found.")
            return
        catalog_index = {c.parameter_key: c for c in all_candidates}
        skipped = [k for k in prior_config if k not in catalog_index]
        if skipped:
            self.logger.stage_detail(
                "tune",
                (
                    f"KB warm start: skipping {len(skipped)} param(s) not in current catalog: "
                    f"{sorted(skipped)}"
                ),
            )
        applicable = {k: v for k, v in prior_config.items() if k in catalog_index}
        if not applicable:
            self.logger.stage_detail(
                "tune", "KB warm start: no applicable params after catalog filter."
            )
            return
        self.logger.stage_detail(
            "tune",
            (
                f"KB warm start: applying {len(applicable)} parameter(s) from "
                f"prior best config: "
                f"{', '.join(f'{k}={v}' for k, v in sorted(applicable.items()))}"
            ),
        )
        applied_count = 0
        for param_key, param_value in applicable.items():
            candidate = catalog_index[param_key]
            hypothesis = TuningHypothesis(
                phase=state.current_phase,
                parameter_key=param_key,
                parameter_name=candidate.parameter_name,
                domain=candidate.domain,
                tuning_layer=candidate.tuning_layer,
                proposed_value=param_value,
                source=candidate.source,
                apply_mode=candidate.apply_mode,
                rationale="warm start from prior best config",
            )
            try:
                ac = self.apply_coordinator.apply(context, hypothesis, target_executor)
                state.active_changes[param_key] = ac
                applied_count += 1
            except Exception as exc:
                self.logger.stage_detail(
                    "tune",
                    f"KB warm start: failed to apply {param_key}={param_value}: {exc}",
                )
        if applied_count:
            self.logger.stage_detail(
                "tune",
                f"KB warm start: {applied_count}/{len(applicable)} applied successfully.",
            )

    def _load_prior_blocked_pairs(
        self,
        context: TuneContext,
    ) -> list[tuple[str, str]]:
        """Load parameter/value pairs that failed in prior similar runs."""
        kb = getattr(context, "knowledge_base", None)
        artifacts = context.artifacts
        if kb is None or artifacts is None:
            return []
        return kb.get_prior_blocked_pairs(
            service_name=context.onboard.service_name,
            cpu_logical_cores=context.preflight.cpu.logical_cores,
            numa_nodes=context.preflight.cpu.numa_nodes,
            platform_summary=context.preflight.platform_summary,
            nic_driver=context.preflight.network.driver_name,
            exclude_run_id=artifacts.session_id,
        )

    def _restore_best_configuration(
        self,
        state: TuneState,
        target_executor: CommandExecutor,
    ) -> None:
        """Ensure the target system is left at the best-known configuration on exit.

        1. Roll back any active change whose key is NOT in best config.
        2. Re-apply any best config value that differs from the current active value.
        """
        best = state.best_configuration
        if best is None:
            # No accepted iteration — roll back everything to baseline.
            if state.active_changes:
                self.logger.stage_detail(
                    "tune",
                    (
                        "No best configuration found; rolling back all "
                        f"{len(state.active_changes)} active change(s) to baseline."
                    ),
                )
                self._rollback_all(state.active_changes, target_executor)
                state.active_changes.clear()
            return

        best_values = best.parameter_values
        active = state.active_changes

        # Step 1: Roll back active changes not in the best config.
        to_rollback = {key: ac for key, ac in active.items() if key not in best_values}
        if to_rollback:
            self.logger.stage_detail(
                "tune",
                (
                    f"Rolling back {len(to_rollback)} change(s) not in best config: "
                    f"{sorted(to_rollback)}"
                ),
            )
            self._rollback_all(to_rollback, target_executor)
            for key in to_rollback:
                del active[key]

        # Step 2: Re-apply best config values that differ from current active.
        restored: list[str] = []
        for key, best_value in best_values.items():
            current_ac = active.get(key)
            if current_ac is not None and current_ac.applied_value == best_value:
                continue  # Already at best value.
            # Need to re-apply. Find the apply command from the best iteration
            # or from any iteration record that applied this key with this value.
            source_record = next(
                (
                    record
                    for record in state.iteration_records
                    if record.applied_change is not None
                    and record.applied_change.hypothesis.parameter_key == key
                    and record.applied_change.applied_value == best_value
                ),
                None,
            )
            if source_record is None or source_record.applied_change is None:
                self.logger.stage_detail(
                    "tune",
                    (
                        f"Cannot restore {key}={best_value}: "
                        "no matching apply record found in iteration history."
                    ),
                )
                continue
            apply_cmd = source_record.applied_change.apply_command
            self.logger.stage_detail(
                "tune",
                f"Restoring best config: {key}={best_value}",
            )
            result = target_executor.run(apply_cmd)
            if result.exit_code != 0:
                self.logger.stage_detail(
                    "tune",
                    (
                        f"RESTORE FAILED for {key}={best_value}: "
                        f"{result.stderr or result.stdout}"
                    ),
                )
            else:
                active[key] = source_record.applied_change
                restored.append(f"{key}={best_value}")

        if restored:
            self.logger.stage_detail(
                "tune",
                f"Best configuration restored: {', '.join(sorted(restored))}",
            )
        else:
            self.logger.stage_detail(
                "tune",
                "Best configuration already active; no re-apply needed.",
            )

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

    def _record_kb_event(
        self,
        *,
        context: TuneContext,
        component: str,
        event_type: str,
        payload: object,
        iteration_number: int | None = None,
        phase: TunePhase | None = None,
    ) -> None:
        artifacts = context.artifacts
        knowledge_base = context.knowledge_base
        if artifacts is None or knowledge_base is None:
            return
        knowledge_base.record_event(
            run_id=artifacts.session_id,
            component=component,
            event_type=event_type,
            payload=payload,
            iteration_number=iteration_number,
            phase=None if phase is None else phase.value,
            service_name=context.onboard.service_name,
        )

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
