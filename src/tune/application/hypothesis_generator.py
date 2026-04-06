from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from tune.application.hypothesis_prompt_layer import format_hybrid_hypothesis_prompt
from tune.application.rule_based_triage import RuleBasedTriage
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import (
    CandidateParameter,
    ModelCompletion,
    TuningHypothesis,
)


class HypothesisModelClient(Protocol):
    def complete(self, context: HypothesisContext) -> ModelCompletion:
        """Return a serialized hypothesis proposal plus optional model usage."""


@dataclass
class HypothesisPromptBuilder:
    """Assembles curated LLM prompts from HypothesisContext."""

    triage: RuleBasedTriage

    def build(self, context: HypothesisContext) -> str:
        return format_hybrid_hypothesis_prompt(
            context=context,
            triage=self.triage.evaluate(context),
        )


@dataclass
class LlmHypothesisGenerator:
    model_client: HypothesisModelClient
    triage: RuleBasedTriage | None = None
    logger: ExecutionLogger = NullExecutionLogger()

    def generate(self, context: HypothesisContext) -> tuple[TuningHypothesis, ...]:
        triage_result = self.triage.evaluate(context) if self.triage is not None else None
        if triage_result is not None:
            self._record_kb_event(
                context=context,
                component="triage_layer",
                event_type="triage_completed",
                payload={
                    "autofix_action": None
                    if triage_result.autofix_action is None
                    else {
                        "parameter_key": triage_result.autofix_action.parameter_key,
                        "proposed_value": triage_result.autofix_action.proposed_value,
                        "reason": triage_result.autofix_action.reason,
                    },
                    "recommended_action": None
                    if triage_result.recommended_action is None
                    else {
                        "parameter_key": triage_result.recommended_action.parameter_key,
                        "proposed_value": triage_result.recommended_action.proposed_value,
                        "reason": triage_result.recommended_action.reason,
                    },
                    "alternate_recommendations": [
                        {
                            "parameter_key": item.parameter_key,
                            "proposed_value": item.proposed_value,
                            "reason": item.reason,
                        }
                        for item in triage_result.alternate_recommendations
                    ],
                    "safe_candidate_subset": list(triage_result.safe_candidate_subset),
                    "suppressed_candidates": list(triage_result.suppressed_candidates),
                    "triggered_rules": [
                        {
                            "rule_id": item.rule_id,
                            "section": item.section,
                            "outcome": item.outcome,
                            "detail": item.detail,
                        }
                        for item in triage_result.triggered_rules
                    ],
                    "escalation_reason": triage_result.escalation_reason,
                },
            )
        if triage_result is not None and triage_result.autofix_action is not None:
            autofix = triage_result.autofix_action
            candidate = self._find_candidate(context, autofix.parameter_key)
            self.logger.stage_detail(
                "tune",
                (
                    "Triage autofix: "
                    f"parameter={autofix.parameter_key} "
                    f"value={autofix.proposed_value} "
                    f"reason={autofix.reason}"
                ),
            )
            self._record_kb_event(
                context=context,
                component="hybrid_llm",
                event_type="llm_skipped_autofix",
                payload={
                    "parameter_key": autofix.parameter_key,
                    "proposed_value": autofix.proposed_value,
                    "reason": autofix.reason,
                },
            )
            return (
                TuningHypothesis(
                    phase=context.phase,
                    parameter_key=candidate.parameter_key,
                    parameter_name=candidate.parameter_name,
                    domain=candidate.domain,
                    tuning_layer=candidate.tuning_layer,
                    proposed_value=autofix.proposed_value,
                    source=candidate.source,
                    apply_mode=candidate.apply_mode,
                    rationale=autofix.reason,
                    model_usage=None,
                    expected_benchmark_impact=(
                        "Deterministic baseline correction; expect cleaner benchmark behavior "
                        "rather than a model-predicted range change."
                    ),
                    rollback_plan=(
                        f"Restore {candidate.parameter_name} to its prior value and "
                        f"{candidate.apply_mode.value} the service/runtime."
                    ),
                ),
            )
        self._debug_log(
            "LLM call",
            f"phase={context.phase.value} iteration={context.iteration_number} "
            f"candidates={len(context.candidates)}",
        )
        self._record_kb_event(
            context=context,
            component="hybrid_llm",
            event_type="llm_prompt_requested",
            payload={
                "candidate_count": len(context.candidates),
                "deferred_candidate_count": len(context.deferred_candidates),
                "history_count": len(context.history),
                "active_parameter_keys": list(context.active_parameter_keys),
            },
        )
        response = self.model_client.complete(context)
        self._record_kb_event(
            context=context,
            component="hybrid_llm",
            event_type="llm_prompt_artifact_saved",
            payload={
                "artifact_path": response.artifact_path,
                "token_usage": None
                if response.usage is None
                else {
                    "model_name": response.usage.model_name,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            },
        )
        self._debug_log("LLM raw response", response.content)
        try:
            raw = json.loads(response.content)
        except json.JSONDecodeError as exc:
            snippet = response.content[:200]
            msg = f"LLM returned non-JSON response: {exc}; content: {snippet!r}"
            self._record_kb_event(
                context=context,
                component="hybrid_llm",
                event_type="llm_invalid_response",
                payload={
                    "error": msg,
                    "response_snippet": snippet,
                    "artifact_path": response.artifact_path,
                },
            )
            raise ValueError(msg) from exc
        try:
            if isinstance(raw, list):
                raise ValueError("LLM must return exactly one JSON object, not an array.")
            if not isinstance(raw, dict):
                raise ValueError("LLM must return a JSON object.")
            self._debug_log("LLM parsed payload", json.dumps(raw, sort_keys=True))
            parameter_key = self._require_string(raw, "parameter_key")
            proposed_value = self._require_string(raw, "proposed_value")
            tuning_layer = self._require_string(raw, "tuning_layer")
            apply_mode = self._require_string(raw, "apply_mode")
            rationale = self._require_string(raw, "rationale")
            expected_benchmark_impact = self._require_string(raw, "expected_benchmark_impact")
            rollback_plan = self._require_string(raw, "rollback_plan")
            candidate = self._find_candidate(context, parameter_key)
            self._validate_proposed_value(candidate, proposed_value)
            self._validate_against_history(context, candidate, proposed_value)
            self._validate_contract_fields(candidate, tuning_layer, apply_mode)
        except ValueError as exc:
            self._record_kb_event(
                context=context,
                component="hybrid_llm",
                event_type="llm_invalid_response",
                payload={
                    "error": str(exc),
                    "response": raw,
                    "artifact_path": response.artifact_path,
                },
            )
            raise
        self._record_kb_event(
            context=context,
            component="hybrid_llm",
            event_type="llm_proposal_selected",
            payload={
                "parameter_key": candidate.parameter_key,
                "proposed_value": proposed_value,
                "tuning_layer": tuning_layer,
                "apply_mode": apply_mode,
                "rationale": rationale,
                "expected_benchmark_impact": expected_benchmark_impact,
                "rollback_plan": rollback_plan,
                "artifact_path": response.artifact_path,
                "token_usage": None
                if response.usage is None
                else {
                    "model_name": response.usage.model_name,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            },
        )
        return (
            TuningHypothesis(
                phase=context.phase,
                parameter_key=candidate.parameter_key,
                parameter_name=candidate.parameter_name,
                domain=candidate.domain,
                tuning_layer=candidate.tuning_layer,
                proposed_value=proposed_value,
                source=candidate.source,
                apply_mode=candidate.apply_mode,
                rationale=rationale,
                model_usage=response.usage,
                expected_benchmark_impact=expected_benchmark_impact,
                rollback_plan=rollback_plan,
            ),
        )

    def _debug_log(self, title: str, content: str) -> None:
        if not self.logger.debug_enabled():
            return
        self.logger.stage_detail("tune", f"{title}:")
        self.logger.stage_detail("tune", content)

    def _find_candidate(
        self,
        context: HypothesisContext,
        parameter_key: str,
    ) -> CandidateParameter:
        for candidate in context.candidates:
            if candidate.parameter_key == parameter_key:
                return candidate
        msg = f"Model proposed unsupported parameter_key: {parameter_key}"
        raise ValueError(msg)

    def _validate_proposed_value(
        self,
        candidate: CandidateParameter,
        proposed_value: str,
    ) -> None:
        if candidate.allowed_values and proposed_value not in candidate.allowed_values:
            msg = (
                f"Model proposed unsupported value {proposed_value!r} "
                f"for {candidate.parameter_key}"
            )
            raise ValueError(msg)
        if candidate.min_value is None and candidate.max_value is None:
            return
        numeric_value = int(proposed_value)
        if candidate.min_value is not None and numeric_value < candidate.min_value:
            msg = f"Value {numeric_value} is below minimum for {candidate.parameter_key}"
            raise ValueError(msg)
        if candidate.max_value is not None and numeric_value > candidate.max_value:
            msg = f"Value {numeric_value} is above maximum for {candidate.parameter_key}"
            raise ValueError(msg)
        if candidate.current_value is not None and proposed_value == candidate.current_value:
            msg = (
                f"Model proposed no-op value {proposed_value!r} for {candidate.parameter_key} "
                f"(current_value_source={candidate.current_value_source})"
            )
            raise ValueError(msg)

    _BOUNDARY_PUSH_KEYS = frozenset(
        {
            "sysctl.net.core.somaxconn",
            "sysctl.net.core.netdev_max_backlog",
            "sysctl.net.ipv4.tcp_max_syn_backlog",
            "sysctl.net.core.rmem_max",
            "sysctl.net.core.wmem_max",
        }
    )

    def _validate_against_history(
        self,
        context: HypothesisContext,
        candidate: CandidateParameter,
        proposed_value: str,
    ) -> None:
        for record in context.history:
            prior = record.hypothesis
            if (
                prior.parameter_key == candidate.parameter_key
                and prior.proposed_value == proposed_value
            ):
                raise ValueError(
                    "Model proposed duplicate parameter/value pair "
                    f"{candidate.parameter_key}={proposed_value!r} already tried in "
                    f"iteration {record.iteration_number} "
                    f"(current_value_source={candidate.current_value_source})"
                )
        self._validate_diminishing_returns(context, candidate, proposed_value)

    def _validate_diminishing_returns(
        self,
        context: HypothesisContext,
        candidate: CandidateParameter,
        proposed_value: str,
    ) -> None:
        """Block re-escalation of boundary-push sysctls without prior verified gain."""
        if candidate.parameter_key not in self._BOUNDARY_PUSH_KEYS:
            return
        try:
            proposed_int = int(proposed_value)
        except ValueError:
            return
        prior_attempts = [
            record
            for record in context.history
            if record.hypothesis.parameter_key == candidate.parameter_key
        ]
        if not prior_attempts:
            return
        last_attempt = prior_attempts[-1]
        try:
            last_proposed_int = int(last_attempt.hypothesis.proposed_value)
        except ValueError:
            return
        # Only block if the new proposal is a further escalation (higher value).
        if proposed_int <= last_proposed_int:
            return
        # Allow re-escalation if the last attempt was accepted or promising.
        if last_attempt.status.value in ("accepted", "promising"):
            return
        raise ValueError(
            f"Diminishing-return suppression: {candidate.parameter_key} was "
            f"escalated to {last_proposed_int} in iteration "
            f"{last_attempt.iteration_number} with status={last_attempt.status.value}; "
            f"blocking further escalation to {proposed_int} without prior verified gain"
        )

    def _validate_contract_fields(
        self,
        candidate: CandidateParameter,
        tuning_layer: str,
        apply_mode: str,
    ) -> None:
        if tuning_layer != candidate.tuning_layer.value:
            raise ValueError(
                f"Model proposed mismatched tuning_layer {tuning_layer!r} "
                f"for {candidate.parameter_key}"
            )
        if apply_mode != candidate.apply_mode.value:
            raise ValueError(
                f"Model proposed mismatched apply_mode {apply_mode!r} "
                f"for {candidate.parameter_key}"
            )

    def _require_string(self, payload: dict[str, object], field_name: str) -> str:
        value = payload.get(field_name)
        if isinstance(value, bool):
            msg = f"Model response must include non-empty string field: {field_name}"
            raise ValueError(msg)
        if isinstance(value, int | float):
            return str(value)
        if not isinstance(value, str) or value == "":
            msg = f"Model response must include non-empty string field: {field_name}"
            raise ValueError(msg)
        return value

    def _record_kb_event(
        self,
        *,
        context: HypothesisContext,
        component: str,
        event_type: str,
        payload: object,
    ) -> None:
        artifacts = context.tune_context.artifacts
        knowledge_base = context.tune_context.knowledge_base
        if artifacts is None or knowledge_base is None:
            return
        knowledge_base.record_event(
            run_id=artifacts.session_id,
            iteration_number=context.iteration_number,
            phase=context.phase.value,
            component=component,
            event_type=event_type,
            service_name=context.tune_context.onboard.service_name,
            payload=payload,
        )


@dataclass
class DeterministicHypothesisGenerator:
    def generate(self, context: HypothesisContext) -> tuple[TuningHypothesis, ...]:
        tried_keys = {
            record.hypothesis.parameter_key
            for record in context.history
            if record.phase == context.phase
        }
        for candidate in context.candidates:
            if candidate.parameter_key in tried_keys:
                continue
            proposed_value = self._default_value(candidate)
            return (
                TuningHypothesis(
                    phase=context.phase,
                    parameter_key=candidate.parameter_key,
                    parameter_name=candidate.parameter_name,
                    domain=candidate.domain,
                    tuning_layer=candidate.tuning_layer,
                    proposed_value=proposed_value,
                    source=candidate.source,
                    apply_mode=candidate.apply_mode,
                    rationale=(
                        "Deterministic fallback selected the first untried allowed candidate "
                        f"for phase {context.phase.value}"
                    ),
                    model_usage=None,
                ),
            )
        msg = "No untried candidates remain for the current phase."
        raise ValueError(msg)

    def _default_value(self, candidate: CandidateParameter) -> str:
        if candidate.allowed_values:
            for allowed_value in candidate.allowed_values:
                if allowed_value not in candidate.forbidden_values:
                    return allowed_value
            msg = f"No allowed values remain for {candidate.parameter_key}"
            raise ValueError(msg)
        if candidate.min_value is not None and candidate.max_value is not None:
            midpoint = (candidate.min_value + candidate.max_value) // 2
            if str(midpoint) not in candidate.forbidden_values:
                return str(midpoint)
            for numeric_value in range(midpoint + 1, candidate.max_value + 1):
                if str(numeric_value) not in candidate.forbidden_values:
                    return str(numeric_value)
            for numeric_value in range(midpoint - 1, candidate.min_value - 1, -1):
                if str(numeric_value) not in candidate.forbidden_values:
                    return str(numeric_value)
            msg = f"No safe integer values remain for {candidate.parameter_key}"
            raise ValueError(msg)
        if candidate.min_value is not None:
            for numeric_value in range(candidate.min_value, candidate.min_value + 1000):
                if str(numeric_value) not in candidate.forbidden_values:
                    return str(numeric_value)
            msg = f"No safe values remain for {candidate.parameter_key}"
            raise ValueError(msg)
        return "1"
