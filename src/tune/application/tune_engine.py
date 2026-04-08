from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
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
from tune.application.rule_based_triage import RuleBasedTriage
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


def _current_workload_rps(state: TuneState) -> tuple[tuple[str, float], ...]:
    """RPS per workload from the most recent benchmarked iteration."""
    for record in reversed(state.iteration_records):
        if record.evaluation_result is not None:
            return tuple(
                (w.workload_name, w.current_requests_per_second)
                for w in record.evaluation_result.workload_evaluations
            )
    return ()


def _kb_best_workload_rps(context: TuneContext) -> tuple[tuple[str, float], ...]:
    """All-time best RPS per workload from the knowledge base."""
    kb = getattr(context, "knowledge_base", None)
    artifacts = context.artifacts
    if kb is None or artifacts is None:
        return ()
    best = kb.get_best_workload_rps(
        service_name=context.onboard.service_name,
        exclude_run_id=artifacts.session_id,
    )
    return tuple(sorted(best.items()))


def _normalize_parameter_group_hypotheses(
    hypotheses: tuple[TuningHypothesis, ...],
    candidates: tuple[CandidateParameter, ...],
    parameter_groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
) -> tuple[TuningHypothesis, ...]:
    """When one member is proposed, enforce all group members together."""
    if not hypotheses or not parameter_groups:
        return hypotheses

    proposed_keys = {h.parameter_key for h in hypotheses}

    candidate_by_key = {candidate.parameter_key: candidate for candidate in candidates}
    hypothesis_by_key = {hypothesis.parameter_key: hypothesis for hypothesis in hypotheses}
    primary = hypotheses[0]
    changed = False

    for _group_name, members in parameter_groups:
        member_keys = {key for key, _value in members}
        if member_keys.isdisjoint(proposed_keys):
            continue
        for key, target_value in members:
            candidate = candidate_by_key.get(key)
            if candidate is None:
                continue
            existing = hypothesis_by_key.get(key)
            if existing is not None:
                if existing.proposed_value != target_value:
                    hypothesis_by_key[key] = replace(
                        existing,
                        proposed_value=target_value,
                        rationale=(
                            f"{existing.rationale} | parameter-group normalization: "
                            "enforce grouped values together"
                        ),
                    )
                    changed = True
                continue
            if candidate.current_value == target_value:
                continue
            hypothesis_by_key[key] = TuningHypothesis(
                phase=primary.phase,
                parameter_key=candidate.parameter_key,
                parameter_name=candidate.parameter_name,
                domain=candidate.domain,
                tuning_layer=candidate.tuning_layer,
                proposed_value=target_value,
                source=candidate.source,
                apply_mode=candidate.apply_mode,
                rationale="parameter-group normalization: enforce grouped values together",
            )
            changed = True

    if not changed:
        return hypotheses

    normalized: list[TuningHypothesis] = []
    seen: set[str] = set()
    for hypothesis in hypotheses:
        normalized.append(hypothesis_by_key[hypothesis.parameter_key])
        seen.add(hypothesis.parameter_key)
    for _group_name, members in parameter_groups:
        for key, _value in members:
            if key in seen:
                continue
            extra = hypothesis_by_key.get(key)
            if extra is not None:
                normalized.append(extra)
                seen.add(key)
    return tuple(normalized)


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
    triage: RuleBasedTriage | None = None
    unified_resolver: object | None = None  # UnifiedResolver (avoids circular)
    kb_batch_apply: bool = False
    skip_marginal_attribution: bool = False
    marginal_attribution_multiplier: float = 2.0
    skip_attribution: bool = False
    logger: ExecutionLogger = NullExecutionLogger()
    compiled_path: Path | None = None
    auto_compile_threshold: int = 30
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "hosttune"

    def run(
        self,
        context: TuneContext,
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
    ) -> TuneState:
        kb_best = _kb_best_workload_rps(context)
        kb_best_homepage = dict(kb_best).get("homepage", 0.0)
        state = TuneState.initialize(
            context.preflight.policy.max_iterations,
            use_unified_resolver=self.unified_resolver is not None,
            allow_reboot=context.preflight.policy.allow_reboot,
            kb_best_homepage_rps=kb_best_homepage,
        )
        all_candidates = self.candidate_catalog_builder.build(context, target_executor)
        deferred_catalog = tuple(
            c for c in all_candidates if c.availability is CandidateAvailability.DEFERRED
        )
        self.logger.stage_start("tune")
        if self.mlflow_enabled:
            self._mlflow_start_run(context)
        baseline_checks = self.health_validator.validate_baseline(context, target_executor)
        baseline_failed_checks = tuple(check for check in baseline_checks if not check.passed)
        if baseline_failed_checks:
            detail = ", ".join(f"{check.name}: {check.detail}" for check in baseline_failed_checks)
            self.logger.stage_warning(
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
        if deferred_catalog:
            self.logger.stage_detail(
                "tune",
                f"Catalog: {len(deferred_catalog)} deferred (reboot_batch) sysctl candidate(s).",
            )
        prior_blocked = self._load_prior_blocked_pairs(context)
        if prior_blocked:
            self.logger.stage_detail(
                "tune",
                f"KB: loaded {len(prior_blocked)} blocked pairs from prior runs.",
            )
        else:
            self.logger.stage_detail("tune", "KB: no prior blocked pairs found.")
        confidence_scores = self._load_confidence_scores(context)
        if self.unified_resolver is not None:
            # --- UNIFIED PATH: layer-by-layer application ---
            from tune.application.unified_resolver import LayerStatus, UnifiedResolver

            resolver: UnifiedResolver = self.unified_resolver  # type: ignore[assignment]
            recipe_seq = self._load_recipe_fix_sequence(context)
            layer_hypotheses, layer_statuses = resolver.resolve(
                context=context,
                state=state,
                all_candidates=all_candidates,
                confidence_scores=confidence_scores,
                recipe_fix_sequence=recipe_seq,
                prior_blocked_pairs=prior_blocked,
            )
            # Batch all resolver layers into ONE benchmark instead of N.
            # The dependency graph guarantees layers are safe to apply together.
            from tune.application.format_table import resolver_apply_table
            all_resolver_hyps: list[TuningHypothesis] = []
            for layer_name, layer_hyps in layer_hypotheses:
                if not layer_hyps:
                    continue
                self.logger.stage_detail(
                    "tune",
                    resolver_apply_table(
                        layer_name,
                        [(h.parameter_key, h.proposed_value) for h in layer_hyps],
                    ),
                )
                all_resolver_hyps.extend(layer_hyps)
            pre_count = len(state.active_changes)
            if all_resolver_hyps:
                self._apply_resolver_layer(
                    context, state, all_resolver_hyps,
                    target_executor, benchmark_executor,
                    "resolver_batch",
                )
            total_applied = len(state.active_changes) - pre_count
            if total_applied > 0:
                all_candidates = self.candidate_catalog_builder.build(
                    context, target_executor
                )
                # Mark all contributing layers as fixed.
                for layer_name, layer_hyps in layer_hypotheses:
                    if layer_hyps:
                        layer_statuses[layer_name] = LayerStatus.FIXED.value
            state.layer_statuses = layer_statuses
            from tune.application.format_table import resolver_summary_table
            layer_param_counts = {name: len(hyps) for name, hyps in layer_hypotheses}
            self.logger.stage_detail(
                "tune",
                resolver_summary_table(total_applied, layer_statuses, layer_param_counts),
            )
        else:
            # --- LEGACY PATH ---
            recipe_applied = self._try_recipe_shortcut(
                context, state, all_candidates,
                target_executor, benchmark_executor,
            )
            if recipe_applied:
                pass
            elif self.kb_batch_apply:
                self._run_kb_batch_consolidated(
                    context, state, all_candidates,
                    confidence_scores,
                    target_executor, benchmark_executor,
                )
            else:
                self._warm_start_from_prior_runs(
                    context, state, all_candidates, target_executor
                )
                if confidence_scores:
                    self._auto_apply_high_confidence(
                        context, state, all_candidates,
                        confidence_scores, target_executor,
                    )
                    all_candidates = self.candidate_catalog_builder.build(
                        context, target_executor
                    )
                    deferred_catalog = tuple(
                        c for c in all_candidates
                        if c.availability is CandidateAvailability.DEFERRED
                    )
                self._run_knowledge_driven_batch(
                    context, state, all_candidates,
                    confidence_scores,
                    target_executor, benchmark_executor,
                )
            # Pre-loop autofix batch (legacy path only).
            self._apply_autofix_batch(
                context, state, all_candidates, prior_blocked,
                target_executor, benchmark_executor,
            )
        if state.active_changes:
            all_candidates = self.candidate_catalog_builder.build(
                context, target_executor
            )
            deferred_catalog = tuple(
                c
                for c in all_candidates
                if c.availability is CandidateAvailability.DEFERRED
            )
        # Track params applied in pre-loop batches — attribution is
        # unreliable for these because rollback of one param in a batch
        # shows minimal drop even when the param contributed to the gain.
        # This includes PROMISING (provisionally retained) params from
        # the batch — their batch benchmark already validated them.
        batch_applied_keys: set[str] = set(state.active_changes.keys())
        prev_active_keys: set[str] = set()
        consecutive_noeval_short: int = 0
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
                confidence_scores=confidence_scores,
                target_executor=target_executor,
                benchmark_executor=benchmark_executor,
                batch_applied_keys=batch_applied_keys,
                full_candidates=all_candidates,
            )
            state.record_iteration(record, history_record)
            self._record_best_config_update(
                context=context,
                state=state,
                previous_best_iteration=previous_best_iteration,
                record=record,
                phase=phase,
            )
            self.recorder.record(context, record)
            self.recorder.record_scoreboard(context, state.scoreboard)
            if self.mlflow_enabled:
                self._mlflow_log_iteration(record, history_record, state.total_iterations)
            # Track failed/inconclusive to prevent retry loops.
            if (
                history_record.status
                in (
                    HypothesisStatus.FAILED_VALIDATION,
                    HypothesisStatus.REJECTED_PRE_APPLY,
                    HypothesisStatus.INCONCLUSIVE,
                )
                and record.hypothesis.parameter_key != "__no_hypothesis__"
            ):
                prior_blocked.append(
                    (record.hypothesis.parameter_key, record.hypothesis.proposed_value)
                )
            # Detect service-dead pattern: consecutive no_eval with short duration.
            # Exclude REJECTED_PRE_APPLY — apply was rolled back, service is untouched.
            if (
                record.evaluation_result is None
                and record.duration_seconds < 30
                and history_record.status is HypothesisStatus.FAILED_VALIDATION
            ):
                consecutive_noeval_short += 1
            else:
                consecutive_noeval_short = 0
            if consecutive_noeval_short >= 2:
                self.logger.stage_warning(
                    "tune",
                    "Service appears dead: "
                    f"{consecutive_noeval_short} consecutive fast failures. "
                    "Attempting service restart.",
                )
                unit = context.onboard.service.identity.systemd_unit_name
                restart_result = target_executor.run(
                    f"systemctl restart {unit}"
                )
                if restart_result.exit_code == 0:
                    self.logger.stage_detail(
                        "tune", "Service restarted successfully."
                    )
                    consecutive_noeval_short = 0
                else:
                    self.logger.stage_warning(
                        "tune",
                        "Service restart failed: "
                        f"{restart_result.stderr.strip()}",
                    )
                    stop_reason = "service_dead"
                    break
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
        self._maybe_auto_compile(context)
        if self.mlflow_enabled:
            self._mlflow_log_session_end(context, state)
            try:
                import mlflow as _mlflow
                _mlflow.end_run()
            except Exception:
                pass
        return state

    # ── MLflow helpers ────────────────────────────────────────────────────────

    def _mlflow_start_run(self, context: TuneContext) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.mlflow_experiment_name)
            session_id = context.artifacts.session_id if context.artifacts else "unknown"
            mlflow.start_run(run_name=f"{context.onboard.service_name}-{session_id}")
            params: dict[str, str | int | float] = {
                "session_id": session_id,
                "service": context.onboard.service_name,
                "max_iterations": context.preflight.policy.max_iterations,
            }
            cpu = getattr(context.preflight, "cpu", None)
            if cpu:
                params["cpu_logical_cores"] = getattr(cpu, "logical_cores", "unknown")
                params["numa_nodes"] = getattr(cpu, "numa_nodes", "unknown")
            network = getattr(context.preflight, "network", None)
            if network:
                params["nic_driver"] = getattr(network, "driver", "unknown")
            platform = getattr(context.preflight, "platform", None)
            if platform:
                params["platform"] = getattr(platform, "platform_summary", "unknown")
            mlflow.log_params(params)
        except Exception as exc:
            self.logger.stage_detail("tune", f"MLflow start_run failed (non-fatal): {exc}")

    def _mlflow_log_iteration(
        self,
        record: TuneIterationRecord,
        history_record: HypothesisRecord,
        step: int,
    ) -> None:
        try:
            import mlflow
            metrics: dict[str, float] = {"duration_seconds": record.duration_seconds}
            if record.benchmark_result:
                for ws in record.benchmark_result.workload_summaries:
                    metrics[f"rps_{ws.workload_name}"] = ws.median_requests_per_second
                    metrics[f"latency_ms_{ws.workload_name}"] = ws.median_latency_ms
            if record.evaluation_result:
                evals = record.evaluation_result.workload_evaluations
                if evals:
                    metrics["avg_relative_change"] = sum(
                        w.relative_change for w in evals
                    ) / len(evals)
            mlflow.log_metrics(metrics, step=step)
            mlflow.set_tags({
                "last_phase": record.phase.value,
                "last_parameter": record.hypothesis.parameter_key,
                "last_decision": history_record.status.value,
            })
            # Log hypothesis rationale as text artifact for this step.
            if record.hypothesis.rationale:
                mlflow.log_text(
                    f"parameter: {record.hypothesis.parameter_key}\n"
                    f"value: {record.hypothesis.proposed_value}\n"
                    f"phase: {record.phase.value}\n"
                    f"decision: {history_record.status.value}\n\n"
                    f"rationale:\n{record.hypothesis.rationale}",
                    artifact_file=f"prompts/iter{step:03d}_hypothesis.txt",
                )
        except Exception as exc:
            self.logger.stage_detail("tune", f"MLflow log_iteration failed (non-fatal): {exc}")

    def _mlflow_log_session_end(self, context: TuneContext, state: TuneState) -> None:
        try:
            import mlflow
            metrics: dict[str, float] = {"total_iterations": float(state.total_iterations)}
            if state.best_configuration:
                metrics["best_score"] = state.best_configuration.score
                metrics["best_iteration"] = float(state.best_configuration.iteration_number)
            metrics.update(self._sum_session_tokens(context))
            mlflow.log_metrics(metrics)
            if context.artifacts:
                hyp_dir = context.artifacts.session_directory / "hypotheses"
                session_id = context.artifacts.session_id
                # Upload run summaries.
                for path in (
                    hyp_dir / f"tune_scoreboard_{session_id}.json",
                    hyp_dir / f"tune_iterations_{session_id}.jsonl",
                    hyp_dir / f"prompt_artifacts_{session_id}.jsonl",
                ):
                    if path.exists():
                        mlflow.log_artifact(str(path))
                # Upload all per-iteration hypothesis files (prompt + response + tokens).
                for hyp_file in sorted(hyp_dir.glob("iter*_hybrid_hypothesizer.json")):
                    mlflow.log_artifact(str(hyp_file), artifact_path="prompts")
        except Exception as exc:
            self.logger.stage_detail("tune", f"MLflow log_session_end failed (non-fatal): {exc}")

    def _sum_session_tokens(self, context: TuneContext) -> dict[str, float]:
        if context.artifacts is None:
            return {}
        prompt_file = (
            context.artifacts.session_directory
            / "hypotheses"
            / f"prompt_artifacts_{context.artifacts.session_id}.jsonl"
        )
        if not prompt_file.exists():
            return {}
        total_input = total_output = total_total = 0
        try:
            for line in prompt_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                tu = entry.get("token_usage") or {}
                total_input += tu.get("input_tokens", 0)
                total_output += tu.get("output_tokens", 0)
                total_total += tu.get("total_tokens", 0)
        except Exception:
            return {}
        return {
            "tokens_input_total": float(total_input),
            "tokens_output_total": float(total_output),
            "tokens_total_session": float(total_total),
        }

    def _maybe_auto_compile(self, context: TuneContext) -> None:
        """Compile DSPy hypothesis prompt once enough accepted examples accumulate.

        Skips silently if compiled_path is None, if the file already exists,
        or if artifacts are unavailable. Any compile failure is logged but non-fatal.
        """
        if self.compiled_path is None:
            return
        if self.compiled_path.exists():
            return  # already compiled; delete the file to trigger recompile
        artifacts = context.artifacts
        if artifacts is None:
            return

        sessions_dir = artifacts.session_directory.parent
        accepted_prompts: list[str] = []

        for session_dir in sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            hypotheses_dir = session_dir / "hypotheses"
            if not hypotheses_dir.exists():
                continue

            # Build accept/reject outcome map for this session.
            outcomes: dict[int, bool] = {}
            for iteration_file in hypotheses_dir.glob("tune_iterations_*.jsonl"):
                for line in iteration_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("phase") == "knowledge_driven":
                        continue  # KB path — no LLM call
                    eval_result = record.get("record", {}).get("evaluation_result")
                    if eval_result is None:
                        continue
                    iteration = record.get("iteration_number", -1)
                    outcomes[iteration] = eval_result.get("decision") == "ACCEPT"

            # Match accepted outcomes to their saved prompt artifacts.
            for artifact_file in hypotheses_dir.glob("iter*_hybrid_hypothesizer.json"):
                try:
                    data = json.loads(artifact_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                iteration = data.get("iteration")
                if not outcomes.get(iteration):
                    continue
                prompt = data.get("prompt", "")
                if prompt:
                    accepted_prompts.append(prompt)

        count = len(accepted_prompts)
        if count < self.auto_compile_threshold:
            self.logger.stage_detail(
                "tune",
                f"DSPy auto-compile: {count}/{self.auto_compile_threshold} accepted "
                "LLM examples collected — run more sessions to trigger optimization.",
            )
            return

        self.logger.stage_detail(
            "tune",
            f"DSPy auto-compile: {count} accepted examples — compiling hypothesis prompt...",
        )
        try:
            import dspy
            from dspy.teleprompt import BootstrapFewShot

            from tune.application.dspy_hypothesis_module import (
                HypothesisPredictor,
                reset_predictor,
            )

            trainset = [
                dspy.Example(context=prompt).with_inputs("context")
                for prompt in accepted_prompts
            ]

            def metric(
                example: dspy.Example,
                prediction: dspy.Prediction,
                trace: object = None,  # noqa: ARG001
            ) -> bool:
                return hasattr(prediction, "hypothesis") and prediction.hypothesis is not None

            optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
            compiled = optimizer.compile(HypothesisPredictor(), trainset=trainset)
            compiled.save(str(self.compiled_path))
            reset_predictor()  # reload singleton from disk on next call
            self.logger.stage_detail(
                "tune",
                f"DSPy auto-compile: saved to {self.compiled_path} — "
                "optimized prompts active from next session.",
            )
        except Exception as exc:
            self.logger.stage_detail(
                "tune",
                f"DSPy auto-compile failed (non-fatal): {type(exc).__name__}: {exc}",
            )

    def _run_iteration(
        self,
        context: TuneContext,
        state: TuneState,
        phase: TunePhase,
        iteration_number: int,
        candidates: tuple[CandidateParameter, ...],
        deferred_candidates: tuple[CandidateParameter, ...],
        prior_blocked_pairs: tuple[tuple[str, str], ...],
        confidence_scores: dict[str, tuple[int, int, float]],
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
        batch_applied_keys: set[str] | None = None,
        full_candidates: tuple[CandidateParameter, ...] = (),
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
            confidence_scores=tuple((k, t, a, c) for k, (t, a, c) in confidence_scores.items()),
            layer_statuses=tuple(sorted(state.layer_statuses.items())),
            current_workload_rps=_current_workload_rps(state),
            kb_best_workload_rps=_kb_best_workload_rps(context),
            full_candidates=full_candidates,
        )
        try:
            hypotheses = self.hypothesis_generator.generate(hyp_context)
            service_groups = tuple(
                (
                    group.name,
                    tuple((member.parameter_key, member.target_value) for member in group.members),
                )
                for group in live_context.onboard.service.tunable_surface.parameter_groups
            )
            hypotheses = _normalize_parameter_group_hypotheses(
                hypotheses,
                candidates,
                service_groups,
            )
        except Exception as exc:
            # Catch broadly so a single bad LLM response or parse error doesn't crash
            # the entire session. Log the full exception type for debugging.
            self.logger.stage_warning(
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
            from tune.application.format_table import pre_apply_rejection_table
            self.logger.stage_detail(
                "tune",
                pre_apply_rejection_table(
                    primary_candidate.tuning_layer.value,
                    primary.parameter_key,
                    primary_pre_apply.reason,
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
                from tune.application.format_table import apply_table
                self.logger.stage_detail(
                    "tune",
                    apply_table([(
                        h.parameter_key,
                        h_candidate.tuning_layer.value,
                        str(ac.previous_value),
                        str(ac.applied_value),
                        ac.apply_mode.value,
                    )]),
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
                from tune.application.format_table import apply_failed_table
                self.logger.stage_warning("tune", apply_failed_table(h.parameter_key, str(exc)))
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
                status=HypothesisStatus.REJECTED_PRE_APPLY,
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
            self.logger.stage_warning(
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
                    "parameter_key": primary.parameter_key,
                    "proposed_value": primary.proposed_value,
                    "applied_parameter_values": [
                        {
                            "parameter_key": hypothesis.parameter_key,
                            "proposed_value": hypothesis.proposed_value,
                        }
                        for hypothesis in valid
                    ],
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
                avg_change = sum(
                    w.relative_change for w in evaluation_result.workload_evaluations
                ) / max(len(evaluation_result.workload_evaluations), 1)
                variance_threshold = context.baseline.expected_variance
                _batch_keys = batch_applied_keys or set()
                # Batch masking is reliable with 3+ params; with only 2,
                # rollback of one still gives a measurable signal.
                # EXPLOIT phase always runs attribution — it explicitly changes
                # batch-applied params to new values, so rollback is meaningful.
                is_batch_param = (
                    primary.parameter_key in _batch_keys
                    and len(_batch_keys) >= 3
                    and phase is not TunePhase.EXPLOIT
                )
                if self.skip_attribution:
                    self.logger.stage_detail(
                        "tune",
                        f"Attribution skipped: skip_attribution=True "
                        f"avg_change={avg_change:.1%}; accepting directly.",
                    )
                    attribution_verification = AttributionVerificationResult(
                        verified=True,
                        summary="skipped (skip_attribution=True)",
                        reverted_benchmark_result=None,
                        average_drop=avg_change,
                    )
                elif is_batch_param:
                    # Batch-applied param: per-param attribution is
                    # unreliable because other batch params mask the drop.
                    self.logger.stage_detail(
                        "tune",
                        f"Attribution skipped: "
                        f"{primary.parameter_key} was batch-applied "
                        f"with {len(_batch_keys)} params; "
                        f"accepting (batch-verified).",
                    )
                    attribution_verification = AttributionVerificationResult(
                        verified=True,
                        summary=(
                            f"skipped (batch-applied with "
                            f"{len(_batch_keys)} params)"
                        ),
                        reverted_benchmark_result=None,
                        average_drop=avg_change,
                    )
                elif avg_change > 0.50:
                    # Skip attribution for overwhelming gains (>50%).
                    self.logger.stage_detail(
                        "tune",
                        f"Attribution skipped: avg improvement "
                        f"{avg_change:.1%} > 50%; accepting.",
                    )
                    attribution_verification = AttributionVerificationResult(
                        verified=True,
                        summary=f"skipped (overwhelming gain {avg_change:.1%})",
                        reverted_benchmark_result=None,
                        average_drop=avg_change,
                    )
                elif (
                    self.skip_marginal_attribution
                    and avg_change
                    < variance_threshold * self.marginal_attribution_multiplier
                ):
                    # Marginal gain within noise floor — skip attribution,
                    # downgrade to inconclusive to avoid 300s wasted.
                    marginal_ceiling = (
                        variance_threshold
                        * self.marginal_attribution_multiplier
                    )
                    self.logger.stage_detail(
                        "tune",
                        f"Attribution skipped: avg improvement "
                        f"{avg_change:.1%} < "
                        f"{self.marginal_attribution_multiplier:.0f}x "
                        f"variance ({marginal_ceiling:.1%}); "
                        f"marking inconclusive.",
                    )
                    evaluation_result = replace(
                        evaluation_result,
                        decision=EvaluationDecision.INCONCLUSIVE,
                        summary=(
                            f"{evaluation_result.summary}; "
                            f"marginal gain {avg_change:.1%} "
                            f"within noise floor"
                        ),
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
                if evaluation_result.decision is EvaluationDecision.ACCEPT:
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
                self.logger.stage_warning("tune", f"ROLLBACK FAILED for {param_key}: {exc}")
        if failures:
            self.logger.stage_warning(
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

    def _load_confidence_scores(
        self,
        context: TuneContext,
    ) -> dict[str, tuple[int, int, float]]:
        """Load per-parameter confidence from prior similar runs."""
        kb = getattr(context, "knowledge_base", None)
        artifacts = context.artifacts
        if kb is None or artifacts is None:
            return {}
        scores = kb.get_parameter_confidence_scores(
            service_name=context.onboard.service_name,
            cpu_logical_cores=context.preflight.cpu.logical_cores,
            numa_nodes=context.preflight.cpu.numa_nodes,
            platform_summary=context.preflight.platform_summary,
            nic_driver=context.preflight.network.driver_name,
            exclude_run_id=artifacts.session_id,
        )
        if scores:
            high = [k for k, (t, a, c) in scores.items() if c >= 1.0 and t >= 1]
            mid = [k for k, (t, a, c) in scores.items() if 0.5 <= c < 1.0]
            low = [k for k, (t, a, c) in scores.items() if c < 0.33 and t >= 2]
            self.logger.stage_detail(
                "tune",
                (
                    f"KB confidence: {len(high)} high (100%), "
                    f"{len(mid)} medium (50-99%), "
                    f"{len(low)} low (<33%)"
                ),
            )
        return scores

    def _auto_apply_high_confidence(
        self,
        context: TuneContext,
        state: TuneState,
        all_candidates: tuple[CandidateParameter, ...],
        confidence_scores: dict[str, tuple[int, int, float]],
        target_executor: CommandExecutor,
    ) -> None:
        """Auto-apply parameters with 100% confidence from KB.

        These have always been accepted in prior runs on similar hardware.
        Skip LLM and benchmark entirely — apply the best-known value directly.
        """
        catalog_index = {c.parameter_key: c for c in all_candidates}
        auto_applied: list[str] = []
        for param_key, (tests, _accepted, confidence) in confidence_scores.items():
            if confidence < 1.0 or tests < 1:
                continue
            if param_key in state.active_changes:
                continue  # Already applied (e.g., by warm start).
            candidate = catalog_index.get(param_key)
            if candidate is None:
                continue
            # Find the best-known value from the prior best config.
            kb = getattr(context, "knowledge_base", None)
            if kb is None or context.artifacts is None:
                continue
            prior_config = kb.get_prior_best_config(
                service_name=context.onboard.service_name,
                cpu_logical_cores=context.preflight.cpu.logical_cores,
                numa_nodes=context.preflight.cpu.numa_nodes,
                platform_summary=context.preflight.platform_summary,
                nic_driver=context.preflight.network.driver_name,
                exclude_run_id=context.artifacts.session_id,
            )
            if prior_config is None or param_key not in prior_config:
                continue
            best_value = prior_config[param_key]
            if candidate.current_value == best_value:
                continue  # Already at best value.
            hypothesis = TuningHypothesis(
                phase=state.current_phase,
                parameter_key=param_key,
                parameter_name=candidate.parameter_name,
                domain=candidate.domain,
                tuning_layer=candidate.tuning_layer,
                proposed_value=best_value,
                source=candidate.source,
                apply_mode=candidate.apply_mode,
                rationale=(
                    f"KB auto-apply: 100% confidence " f"({tests}/{tests} accepted in prior runs)"
                ),
            )
            try:
                ac = self.apply_coordinator.apply(context, hypothesis, target_executor)
                state.active_changes[param_key] = ac
                auto_applied.append(f"{param_key}={best_value}")
            except Exception as exc:
                self.logger.stage_detail(
                    "tune",
                    f"KB auto-apply failed for {param_key}={best_value}: {exc}",
                )
        if auto_applied:
            self.logger.stage_detail(
                "tune",
                (
                    f"KB auto-applied {len(auto_applied)} high-confidence "
                    f"parameter(s): {', '.join(sorted(auto_applied))}"
                ),
            )

    def _run_knowledge_driven_batch(
        self,
        context: TuneContext,
        state: TuneState,
        all_candidates: tuple[CandidateParameter, ...],
        confidence_scores: dict[str, tuple[int, int, float]],
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
    ) -> None:
        """Batch-apply all medium+ confidence KB candidates, benchmark once.

        Applies all candidates with KB confidence >= 50% in a single
        iteration. One benchmark validates the entire batch. This saves
        (N-1) × ~320s compared to per-candidate iterations.
        """
        if not confidence_scores:
            self.logger.stage_detail("tune", "KB batch: no confidence data; skipping.")
            return
        catalog_index = {c.parameter_key: c for c in all_candidates}
        # Fetch prior best config once (not per-candidate).
        kb = getattr(context, "knowledge_base", None)
        artifacts = context.artifacts
        prior_config: dict[str, str] = {}
        if kb is not None and artifacts is not None:
            prior_config = (
                kb.get_prior_best_config(
                    service_name=context.onboard.service_name,
                    cpu_logical_cores=context.preflight.cpu.logical_cores,
                    numa_nodes=context.preflight.cpu.numa_nodes,
                    platform_summary=context.preflight.platform_summary,
                    nic_driver=context.preflight.network.driver_name,
                    exclude_run_id=artifacts.session_id,
                )
                or {}
            )
        # Collect candidates with 50%+ confidence, not already active.
        batch: list[tuple[CandidateParameter, str, float]] = []
        for param_key, (tests, _accepted, confidence) in confidence_scores.items():
            if confidence < 0.50 or tests < 1:
                continue
            if param_key in state.active_changes:
                continue
            candidate = catalog_index.get(param_key)
            if candidate is None:
                continue
            proposed_value = prior_config.get(param_key)
            if proposed_value is None or proposed_value == candidate.current_value:
                continue
            batch.append((candidate, proposed_value, confidence))
        if not batch:
            self.logger.stage_detail("tune", "KB batch: no applicable candidates.")
            return
        # Sort by confidence desc.
        batch.sort(key=lambda x: x[2], reverse=True)
        self.logger.stage_detail(
            "tune",
            (
                f"KB batch: applying {len(batch)} candidate(s) in one iteration: "
                + ", ".join(f"{c.parameter_key}={v} ({conf:.0%})" for c, v, conf in batch)
            ),
        )
        hypotheses = [
            TuningHypothesis(
                phase=TunePhase.KNOWLEDGE_DRIVEN,
                parameter_key=candidate.parameter_key,
                parameter_name=candidate.parameter_name,
                domain=candidate.domain,
                tuning_layer=candidate.tuning_layer,
                proposed_value=proposed_value,
                source=candidate.source,
                apply_mode=candidate.apply_mode,
                rationale=f"KB batch: {confidence:.0%} confidence",
            )
            for candidate, proposed_value, confidence in batch
        ]
        self._apply_and_benchmark_batch(
            context, state, hypotheses, target_executor, benchmark_executor, "KB batch"
        )

    def _apply_autofix_batch(
        self,
        context: TuneContext,
        state: TuneState,
        all_candidates: tuple[CandidateParameter, ...],
        prior_blocked: list[tuple[str, str]],
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
    ) -> None:
        """Collect ALL triage autofixes and apply them in one batch.

        Instead of consuming one iteration per autofix, this applies
        sendfile=on, tcp_nopush=on, open_file_cache, limit_rate=0, etc.
        all at once with a single benchmark validation.
        """
        if self.triage is None:
            return
        # Build a minimal HypothesisContext for triage evaluation.
        hyp_context = HypothesisContext(
            tune_context=context,
            phase=state.current_phase,
            iteration_number=0,
            candidates=all_candidates,
            deferred_candidates=(),
            history=tuple(state.history),
            active_parameter_keys=tuple(sorted(state.active_changes)),
            best_parameter_values=(),
            prior_blocked_pairs=tuple(prior_blocked),
            confidence_scores=(),
        )
        autofixes = self.triage.collect_all_autofixes(hyp_context)
        if not autofixes:
            return
        catalog_index = {c.parameter_key: c for c in all_candidates}
        hypotheses: list[TuningHypothesis] = []
        seen_keys: set[str] = set()
        for param_key, proposed_value, reason in autofixes:
            candidate = catalog_index.get(param_key)
            if candidate is None:
                continue
            if param_key in state.active_changes or param_key in seen_keys:
                continue
            if candidate.current_value == proposed_value:
                continue
            seen_keys.add(param_key)
            hypotheses.append(
                TuningHypothesis(
                    phase=TunePhase.KNOWLEDGE_DRIVEN,
                    parameter_key=param_key,
                    parameter_name=candidate.parameter_name,
                    domain=candidate.domain,
                    tuning_layer=candidate.tuning_layer,
                    proposed_value=proposed_value,
                    source=candidate.source,
                    apply_mode=candidate.apply_mode,
                    rationale=f"Autofix batch: {reason}",
                )
            )
        if not hypotheses:
            return
        self.logger.stage_detail(
            "tune",
            "Autofix batch: applying "
            + ", ".join(
                f"{h.parameter_key}={h.proposed_value}"
                for h in hypotheses
            ),
        )
        self._apply_and_benchmark_batch(
            context, state, hypotheses,
            target_executor, benchmark_executor,
            "Autofix batch",
        )

    def _load_recipe_fix_sequence(
        self,
        context: TuneContext,
    ) -> list[dict[str, str]] | None:
        """Load recipe fix sequence without applying it (for unified resolver)."""
        from preflight.infrastructure.knowledge_base import (
            compute_degradation_fingerprint,
            host_fingerprint_for_snapshot,
        )

        kb = getattr(context, "knowledge_base", None)
        artifacts = context.artifacts
        if kb is None or artifacts is None:
            return None
        workload_results = context.baseline.workload_results
        if not workload_results:
            return None
        names, vector = compute_degradation_fingerprint(workload_results)
        fingerprint = host_fingerprint_for_snapshot(
            context.preflight, context.onboard.service_name
        )
        recipe = kb.lookup_degradation_recipe(
            service_name=context.onboard.service_name,
            host_fingerprint=fingerprint,
            current_fingerprint=vector,
            current_workload_names=names,
            exclude_run_id=artifacts.session_id,
        )
        if recipe is None:
            return None
        self.logger.stage_detail(
            "tune",
            f"Recipe found: run={recipe['run_id']} "
            f"similarity={recipe['similarity']:.2%} "
            f"score={recipe['best_score']:.2%}",
        )
        return recipe["fix_sequence"]

    def _try_recipe_shortcut(
        self,
        context: TuneContext,
        state: TuneState,
        all_candidates: tuple[CandidateParameter, ...],
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
    ) -> bool:
        """Apply a matching degradation recipe if one exists.

        Computes a fingerprint from the current baseline, looks up
        prior runs with similar degradation patterns, and if a match
        is found, batch-applies the entire fix sequence.

        Returns True if a recipe was applied, False otherwise.
        """
        from preflight.infrastructure.knowledge_base import (
            compute_degradation_fingerprint,
            host_fingerprint_for_snapshot,
        )

        kb = getattr(context, "knowledge_base", None)
        artifacts = context.artifacts
        if kb is None or artifacts is None:
            return False
        # Compute current baseline fingerprint.
        workload_results = context.baseline.workload_results
        if not workload_results:
            return False
        names, vector = compute_degradation_fingerprint(workload_results)
        fingerprint = host_fingerprint_for_snapshot(
            context.preflight, context.onboard.service_name
        )
        recipe = kb.lookup_degradation_recipe(
            service_name=context.onboard.service_name,
            host_fingerprint=fingerprint,
            current_fingerprint=vector,
            current_workload_names=names,
            exclude_run_id=artifacts.session_id,
        )
        if recipe is None:
            self.logger.stage_detail(
                "tune", "Recipe lookup: no matching degradation pattern."
            )
            return False
        fix_sequence = recipe["fix_sequence"]
        self.logger.stage_detail(
            "tune",
            f"Recipe match: run={recipe['run_id']} "
            f"similarity={recipe['similarity']:.2%} "
            f"score={recipe['best_score']:.2%} "
            f"fixes={len(fix_sequence)}",
        )
        self._record_kb_event(
            context=context,
            component="recipe_lookup",
            event_type="recipe_matched",
            payload={
                "matched_run_id": recipe["run_id"],
                "similarity": recipe["similarity"],
                "best_score": recipe["best_score"],
                "fix_count": len(fix_sequence),
            },
        )
        # Build hypotheses from the fix sequence.
        catalog_index = {c.parameter_key: c for c in all_candidates}
        hypotheses: list[TuningHypothesis] = []
        for step in fix_sequence:
            param_key = step["parameter_key"]
            candidate = catalog_index.get(param_key)
            if candidate is None:
                self.logger.stage_detail(
                    "tune",
                    f"Recipe: skipping {param_key} (not in catalog)",
                )
                continue
            if candidate.current_value == step["value"]:
                continue  # Already at recipe value.
            hypotheses.append(
                TuningHypothesis(
                    phase=TunePhase.KNOWLEDGE_DRIVEN,
                    parameter_key=param_key,
                    parameter_name=candidate.parameter_name,
                    domain=candidate.domain,
                    tuning_layer=candidate.tuning_layer,
                    proposed_value=step["value"],
                    source=candidate.source,
                    apply_mode=candidate.apply_mode,
                    rationale=(
                        f"Recipe shortcut from run {recipe['run_id']} "
                        f"(similarity={recipe['similarity']:.0%})"
                    ),
                )
            )
        if not hypotheses:
            self.logger.stage_detail(
                "tune", "Recipe: all fixes already applied or not in catalog."
            )
            return False
        self.logger.stage_detail(
            "tune",
            "Recipe: applying "
            + ", ".join(f"{h.parameter_key}={h.proposed_value}" for h in hypotheses),
        )
        self._apply_and_benchmark_batch(
            context, state, hypotheses,
            target_executor, benchmark_executor,
            "Recipe shortcut",
        )
        # Zero out exploration budget if recipe was accepted.
        if state.active_changes:
            for phase in TunePhase:
                if phase not in (TunePhase.EXPLOIT, TunePhase.REBOOT_BATCH):
                    state.remaining_budget[phase] = 0
            state.remaining_budget[TunePhase.EXPLOIT] = min(
                2, state.remaining_budget[TunePhase.EXPLOIT]
            )
            self.logger.stage_detail(
                "tune",
                f"Recipe applied: zeroed exploration budget, "
                f"EXPLOIT={state.remaining_budget[TunePhase.EXPLOIT]} "
                f"remaining.",
            )
            return True
        return False

    def _run_kb_batch_consolidated(
        self,
        context: TuneContext,
        state: TuneState,
        all_candidates: tuple[CandidateParameter, ...],
        confidence_scores: dict[str, tuple[int, int, float]],
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
    ) -> None:
        """Unified KB apply: merge warm-start + auto-apply + batch into one step.

        1. Collect ALL params from prior best config + 50%+ confidence scores.
        2. Apply them all at once (dedup by key, prefer prior best value).
        3. Run ONE benchmark to validate the batch.
        4. If regression → roll back all and fall through to normal tuning.
        """
        kb = getattr(context, "knowledge_base", None)
        artifacts = context.artifacts
        if kb is None or artifacts is None:
            self.logger.stage_detail("tune", "KB batch consolidated: no KB available.")
            return
        catalog_index = {c.parameter_key: c for c in all_candidates}
        # Fetch prior best config.
        prior_config = (
            kb.get_prior_best_config(
                service_name=context.onboard.service_name,
                cpu_logical_cores=context.preflight.cpu.logical_cores,
                numa_nodes=context.preflight.cpu.numa_nodes,
                platform_summary=context.preflight.platform_summary,
                nic_driver=context.preflight.network.driver_name,
                exclude_run_id=artifacts.session_id,
            )
            or {}
        )
        # Collect prior best config params that are in the catalog and differ from current.
        # Only prior_config provides proposed values; confidence_scores refine priority.
        merged: dict[str, tuple[str, float, str]] = {}  # key -> (value, confidence, source)
        for param_key, param_value in prior_config.items():
            candidate = catalog_index.get(param_key)
            if candidate is None:
                continue
            if candidate.current_value == param_value:
                continue
            conf = confidence_scores.get(param_key, (0, 0, 0.0))
            merged[param_key] = (param_value, conf[2], "prior_best")
        if not merged:
            self.logger.stage_detail("tune", "KB batch consolidated: no applicable params.")
            return
        self.logger.stage_detail(
            "tune",
            (
                f"KB batch consolidated: applying {len(merged)} param(s) in one go: "
                + ", ".join(f"{k}={v} ({c:.0%} {s})" for k, (v, c, s) in sorted(merged.items()))
            ),
        )
        hypotheses = [
            TuningHypothesis(
                phase=TunePhase.KNOWLEDGE_DRIVEN,
                parameter_key=param_key,
                parameter_name=catalog_index[param_key].parameter_name,
                domain=catalog_index[param_key].domain,
                tuning_layer=catalog_index[param_key].tuning_layer,
                proposed_value=proposed_value,
                source=catalog_index[param_key].source,
                apply_mode=catalog_index[param_key].apply_mode,
                rationale=f"KB consolidated batch: {confidence:.0%} confidence ({source})",
            )
            for param_key, (proposed_value, confidence, source) in sorted(
                merged.items(), key=lambda x: x[1][1], reverse=True
            )
        ]
        self._apply_and_benchmark_batch(
            context, state, hypotheses, target_executor, benchmark_executor, "KB consolidated"
        )

    def _apply_resolver_layer(
        self,
        context: TuneContext,
        state: TuneState,
        hypotheses: list[TuningHypothesis],
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
        layer_name: str,
    ) -> None:
        """Apply a resolver layer: keep INCONCLUSIVE, only rollback REJECT.

        Unlike _apply_and_benchmark_batch which rolls back on any
        non-accept, this keeps inconclusive changes applied so higher
        layers can build on them. The combined effect of multiple
        layers may push marginal gains over the acceptance threshold.
        """
        from datetime import UTC, datetime
        from time import perf_counter

        iteration_number = state.total_iterations + 1
        started_at = datetime.now(UTC)
        started_timer = perf_counter()
        applied: dict[str, AppliedChange] = {}
        primary_hypothesis = hypotheses[0]
        hypothesis_by_key = {hypothesis.parameter_key: hypothesis for hypothesis in hypotheses}
        for hypothesis in hypotheses:
            try:
                ac = self.apply_coordinator.apply(
                    context, hypothesis, target_executor
                )
                applied[hypothesis.parameter_key] = ac
                state.active_changes[hypothesis.parameter_key] = ac
                self._record_kb_event(
                    context=context,
                    component="tuning_executor",
                    event_type="change_applied",
                    iteration_number=iteration_number,
                    phase=primary_hypothesis.phase,
                    payload={
                        "parameter_key": hypothesis.parameter_key,
                        "previous_value": ac.previous_value,
                        "applied_value": ac.applied_value,
                        "apply_mode": ac.apply_mode.value,
                        "apply_command": ac.apply_command,
                    },
                )
            except Exception as exc:
                self.logger.stage_detail(
                    "tune",
                    f"Resolver {layer_name}: failed "
                    f"{hypothesis.parameter_key}="
                    f"{hypothesis.proposed_value}: {exc}",
                )
                self._record_kb_event(
                    context=context,
                    component="tuning_executor",
                    event_type="apply_failed",
                    iteration_number=iteration_number,
                    phase=primary_hypothesis.phase,
                    payload={
                        "parameter_key": hypothesis.parameter_key,
                        "error": str(exc),
                    },
                )
        if not applied:
            return
        primary_ac = next(iter(applied.values()))
        validation_result = self.health_validator.validate(
            context, primary_ac, target_executor
        )
        self._record_kb_event(
            context=context,
            component="tuning_executor",
            event_type="validation_completed",
            iteration_number=iteration_number,
            phase=primary_hypothesis.phase,
            payload={
                "parameter_key": primary_hypothesis.parameter_key,
                "healthy": validation_result.healthy,
                "checks": [
                    {"name": check.name, "passed": check.passed, "detail": check.detail}
                    for check in validation_result.checks
                ],
            },
        )
        if not validation_result.healthy:
            self.logger.stage_warning(
                "tune",
                f"Resolver {layer_name}: health check failed; "
                f"rolling back.",
            )
            self._rollback_all(applied, target_executor)
            for key in applied:
                state.active_changes.pop(key, None)
            self._record_kb_event(
                context=context,
                component="tuning_executor",
                event_type="rollback_completed",
                iteration_number=iteration_number,
                phase=primary_hypothesis.phase,
                payload={
                    "parameters": sorted(applied),
                    "reason": "validation_failed",
                },
            )
            return
        benchmark_result = self.benchmark_executor.run(
            context=context,
            iteration_number=iteration_number,
            validation_result=validation_result,
            benchmark_executor=benchmark_executor,
            telemetry_executor=target_executor,
        )
        self._log_benchmark(benchmark_result)
        batch_phase = primary_hypothesis.phase
        self._record_kb_event(
            context=context,
            component="benchmark_runner",
            event_type="benchmark_completed",
            iteration_number=iteration_number,
            phase=batch_phase,
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
            context, benchmark_result, phase=batch_phase
        )
        self._log_evaluation(evaluation_result)
        applied_parameter_values = [
            {
                "parameter_key": parameter_key,
                "proposed_value": hypothesis_by_key[parameter_key].proposed_value,
            }
            for parameter_key in sorted(applied)
            if parameter_key in hypothesis_by_key
        ]
        self._record_kb_event(
            context=context,
            component="benchmark_runner",
            event_type="evaluation_completed",
            iteration_number=iteration_number,
            phase=batch_phase,
            payload={
                "parameter_key": primary_hypothesis.parameter_key,
                "proposed_value": primary_hypothesis.proposed_value,
                "applied_parameter_values": applied_parameter_values,
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
        completed_at = datetime.now(UTC)
        duration_seconds = perf_counter() - started_timer
        status = self._resolve_status(evaluation_result)
        if status is HypothesisStatus.REJECTED:
            # Only roll back on clear rejection.
            self.logger.stage_detail(
                "tune",
                f"Resolver {layer_name}: REJECTED; rolling back.",
            )
            self._rollback_all(applied, target_executor)
            for key in applied:
                state.active_changes.pop(key, None)
            self._record_kb_event(
                context=context,
                component="tuning_executor",
                event_type="rollback_completed",
                iteration_number=iteration_number,
                phase=batch_phase,
                payload={
                    "parameters": sorted(applied),
                    "reason": "evaluation_rejected",
                },
            )
        else:
            # ACCEPTED, PROMISING, or INCONCLUSIVE — keep changes.
            label = status.value
            if status is HypothesisStatus.INCONCLUSIVE:
                label = "inconclusive (keeping — may help higher layers)"
            self.logger.stage_detail(
                "tune",
                f"Resolver {layer_name}: {label}; "
                f"retaining {len(applied)} param(s).",
            )
        record = TuneIterationRecord(
            iteration_number=iteration_number,
            phase=batch_phase,
            hypothesis=primary_hypothesis,
            applied_change=primary_ac,
            validation_result=validation_result,
            benchmark_result=benchmark_result,
            evaluation_result=evaluation_result,
            attribution_verification=None,
            active_parameter_keys=tuple(sorted(state.active_changes)),
            started_at_utc=started_at.isoformat(),
            completed_at_utc=completed_at.isoformat(),
            duration_seconds=duration_seconds,
        )
        history_record = HypothesisRecord(
            iteration_number=iteration_number,
            phase=batch_phase,
            hypothesis=primary_hypothesis,
            status=status,
            evaluation_summary=evaluation_result.summary,
        )
        previous_best_iteration = (
            None
            if state.best_configuration is None
            else state.best_configuration.iteration_number
        )
        state.record_iteration(record, history_record)
        self._record_best_config_update(
            context=context,
            state=state,
            previous_best_iteration=previous_best_iteration,
            record=record,
            phase=batch_phase,
        )
        self.recorder.record(context, record)
        self.recorder.record_scoreboard(context, state.scoreboard)
        self.logger.stage_detail(
            "tune",
            f"Resolver {layer_name} iteration {iteration_number}: "
            f"{len(applied)} params, {duration_seconds:.1f}s, "
            f"status={status.value}",
        )

    def _apply_and_benchmark_batch(
        self,
        context: TuneContext,
        state: TuneState,
        hypotheses: list[TuningHypothesis],
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
        log_prefix: str,
    ) -> None:
        """Apply a list of hypotheses, run one benchmark, and record the result."""
        iteration_number = state.total_iterations + 1
        started_at = datetime.now(UTC)
        started_timer = perf_counter()
        applied: dict[str, AppliedChange] = {}
        primary_hypothesis: TuningHypothesis | None = None
        hypothesis_by_key = {hypothesis.parameter_key: hypothesis for hypothesis in hypotheses}
        for hypothesis in hypotheses:
            if primary_hypothesis is None:
                primary_hypothesis = hypothesis
            try:
                ac = self.apply_coordinator.apply(context, hypothesis, target_executor)
                applied[hypothesis.parameter_key] = ac
                state.active_changes[hypothesis.parameter_key] = ac
                self._record_kb_event(
                    context=context,
                    component="tuning_executor",
                    event_type="change_applied",
                    iteration_number=iteration_number,
                    phase=hypothesis.phase,
                    payload={
                        "parameter_key": hypothesis.parameter_key,
                        "previous_value": ac.previous_value,
                        "applied_value": ac.applied_value,
                        "apply_mode": ac.apply_mode.value,
                        "apply_command": ac.apply_command,
                    },
                )
            except Exception as exc:
                self.logger.stage_detail(
                    "tune",
                    f"{log_prefix}: failed "
                    f"{hypothesis.parameter_key}={hypothesis.proposed_value}: {exc}",
                )
                self._record_kb_event(
                    context=context,
                    component="tuning_executor",
                    event_type="apply_failed",
                    iteration_number=iteration_number,
                    phase=hypothesis.phase,
                    payload={"parameter_key": hypothesis.parameter_key, "error": str(exc)},
                )
        if not applied or primary_hypothesis is None:
            self.logger.stage_detail("tune", f"{log_prefix}: no params applied successfully.")
            return
        primary_ac = next(iter(applied.values()))
        validation_result = self.health_validator.validate(context, primary_ac, target_executor)
        self._record_kb_event(
            context=context,
            component="tuning_executor",
            event_type="validation_completed",
            iteration_number=iteration_number,
            phase=primary_hypothesis.phase,
            payload={
                "parameter_key": primary_hypothesis.parameter_key,
                "healthy": validation_result.healthy,
                "checks": [
                    {"name": check.name, "passed": check.passed, "detail": check.detail}
                    for check in validation_result.checks
                ],
            },
        )
        if not validation_result.healthy:
            self.logger.stage_warning(
                "tune", f"{log_prefix}: health check failed; rolling back all."
            )
            self._rollback_all(applied, target_executor)
            for key in applied:
                state.active_changes.pop(key, None)
            self._record_kb_event(
                context=context,
                component="tuning_executor",
                event_type="rollback_completed",
                iteration_number=iteration_number,
                phase=primary_hypothesis.phase,
                payload={
                    "parameters": sorted(applied),
                    "reason": "validation_failed",
                },
            )
            return
        benchmark_result = self.benchmark_executor.run(
            context=context,
            iteration_number=iteration_number,
            validation_result=validation_result,
            benchmark_executor=benchmark_executor,
            telemetry_executor=target_executor,
        )
        self._log_benchmark(benchmark_result)
        batch_phase = primary_hypothesis.phase
        self._record_kb_event(
            context=context,
            component="benchmark_runner",
            event_type="benchmark_completed",
            iteration_number=iteration_number,
            phase=batch_phase,
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
            context, benchmark_result, phase=batch_phase
        )
        self._log_evaluation(evaluation_result)
        applied_parameter_values = [
            {
                "parameter_key": parameter_key,
                "proposed_value": hypothesis_by_key[parameter_key].proposed_value,
            }
            for parameter_key in sorted(applied)
            if parameter_key in hypothesis_by_key
        ]
        self._record_kb_event(
            context=context,
            component="benchmark_runner",
            event_type="evaluation_completed",
            iteration_number=iteration_number,
            phase=batch_phase,
            payload={
                "parameter_key": primary_hypothesis.parameter_key,
                "proposed_value": primary_hypothesis.proposed_value,
                "applied_parameter_values": applied_parameter_values,
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
        completed_at = datetime.now(UTC)
        duration_seconds = perf_counter() - started_timer
        status = self._resolve_status(evaluation_result)
        if status not in (HypothesisStatus.ACCEPTED, HypothesisStatus.PROMISING):
            self.logger.stage_detail(
                "tune", f"{log_prefix}: evaluation={status.value}; rolling back all."
            )
            self._rollback_all(applied, target_executor)
            for key in applied:
                state.active_changes.pop(key, None)
            status = HypothesisStatus.INCONCLUSIVE
            self._record_kb_event(
                context=context,
                component="tuning_executor",
                event_type="rollback_completed",
                iteration_number=iteration_number,
                phase=batch_phase,
                payload={
                    "parameters": sorted(applied),
                    "reason": "evaluation_not_accepted",
                },
            )
        else:
            self.logger.stage_detail(
                "tune",
                f"{log_prefix}: {status.value}; "
                f"retaining {len(applied)} param(s): {sorted(applied)}",
            )
        record = TuneIterationRecord(
            iteration_number=iteration_number,
            phase=batch_phase,
            hypothesis=primary_hypothesis,
            applied_change=primary_ac,
            validation_result=validation_result,
            benchmark_result=benchmark_result,
            evaluation_result=evaluation_result,
            attribution_verification=None,
            active_parameter_keys=tuple(sorted(state.active_changes)),
            started_at_utc=started_at.isoformat(),
            completed_at_utc=completed_at.isoformat(),
            duration_seconds=duration_seconds,
        )
        history_record = HypothesisRecord(
            iteration_number=iteration_number,
            phase=batch_phase,
            hypothesis=primary_hypothesis,
            status=status,
            evaluation_summary=evaluation_result.summary,
        )
        previous_best_iteration = (
            None
            if state.best_configuration is None
            else state.best_configuration.iteration_number
        )
        state.record_iteration(record, history_record)
        self._record_best_config_update(
            context=context,
            state=state,
            previous_best_iteration=previous_best_iteration,
            record=record,
            phase=batch_phase,
        )
        self.recorder.record(context, record)
        self.recorder.record_scoreboard(context, state.scoreboard)
        self.logger.stage_detail(
            "tune",
            f"{log_prefix} iteration {iteration_number}: "
            f"{len(applied)} params, {duration_seconds:.1f}s, "
            f"status={status.value}",
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

    def _record_best_config_update(
        self,
        *,
        context: TuneContext,
        state: TuneState,
        previous_best_iteration: int | None,
        record: TuneIterationRecord,
        phase: TunePhase,
    ) -> None:
        if state.best_configuration is None:
            return
        if state.best_configuration.iteration_number == previous_best_iteration:
            return
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
        from tune.application.format_table import benchmark_summary_table
        self.logger.stage_detail("tune", benchmark_summary_table(benchmark_result))

    def _log_evaluation(self, evaluation_result: EvaluationResult) -> None:
        from tune.application.format_table import evaluation_table
        decision = evaluation_result.decision.value
        logger_fn = (
            self.logger.stage_warning
            if decision in ("reject", "inconclusive")
            else self.logger.stage_detail
        )
        logger_fn("tune", evaluation_table(evaluation_result))
