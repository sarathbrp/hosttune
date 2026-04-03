from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from tune.application.benchmark_runtime_telemetry import (
    format_runtime_telemetry_digest,
    truncate_for_prompt,
)
from tune.application.hypothesis_prompt_layer import (
    format_baseline_digest_lines,
    format_candidate_line_for_llm,
    format_contract_digest_lines,
    format_limit_baseline_lines,
    format_preflight_digest_lines,
    hypothesis_prompt_layer_preamble,
)
from tune.application.snapshot_prompt_digest import format_snapshot_digest_for_prompt
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import (
    CandidateParameter,
    ModelCompletion,
    TunePhase,
    TuningHypothesis,
)


class HypothesisModelClient(Protocol):
    def complete(self, context: HypothesisContext) -> ModelCompletion:
        """Return a serialized hypothesis proposal plus optional model usage."""


@dataclass
class HypothesisPromptBuilder:
    """Assembles curated LLM prompts from HypothesisContext."""

    def build(self, context: HypothesisContext) -> str:
        tune_context = context.tune_context

        candidate_lines = [format_candidate_line_for_llm(c) for c in context.candidates]
        deferred_lines = [format_candidate_line_for_llm(c) for c in context.deferred_candidates]
        _hist_eval_max = 180
        history_lines = [
            (
                f"- iteration={record.iteration_number}; "
                f"phase={record.phase.value}; "
                f"parameter_key={record.hypothesis.parameter_key}; "
                f"value={record.hypothesis.proposed_value}; "
                f"status={record.status.value}; "
                f"evaluation={truncate_for_prompt(record.evaluation_summary or '', _hist_eval_max)}"
            )
            for record in context.history
        ]
        capability_lines = [
            f"- {flag.name}: {flag.detail}"
            for flag in tune_context.preflight.capability_map.flags
            if flag.available
        ]
        phase_objective = self._phase_objective(context.phase)
        best_config = (
            ", ".join(f"{key}={value}" for key, value in context.best_parameter_values) or "none"
        )
        active_changes = ", ".join(context.active_parameter_keys) or "none"
        _telemetry_max_section = 420
        telemetry_digest_trimmed = context.last_benchmark_runtime_telemetry_digest.strip()
        telemetry_body = (
            telemetry_digest_trimmed
            if telemetry_digest_trimmed
            else format_runtime_telemetry_digest((), max_chars_per_section=_telemetry_max_section)
        )
        sections = [
            "You are the hypothesis generator for HostTune.",
            "Select exactly one candidate parameter from the allowed list.",
            "Do not invent parameters, values, or apply modes outside the candidate list.",
            "Prefer unexplored or promising regions based on prior history and current phase.",
            "Return strict JSON with keys: parameter_key, proposed_value, rationale.",
            *hypothesis_prompt_layer_preamble(),
            f"Current phase: {context.phase.value}",
            f"Phase objective: {phase_objective}",
            f"Iteration number: {context.iteration_number}",
            "Preflight digest (structured discovery facts):",
            *format_preflight_digest_lines(tune_context.preflight),
            "Available tunable surfaces:",
            *(capability_lines or ["- none"]),
            "Onboard service contract (summary):",
            *format_contract_digest_lines(tune_context),
            "Limit baselines (prlimit/systemd-unit; check before no-op):",
            *format_limit_baseline_lines(context.candidates + context.deferred_candidates),
            "Snapshot digest (effective runtime view; truncated, not full dumps):",
            format_snapshot_digest_for_prompt(tune_context.snapshot),
            "Baseline digest:",
            *format_baseline_digest_lines(tune_context),
            (
                "Last benchmark runtime telemetry digest (target during load: ss -s, "
                "softnet_stat, ethtool -S; truncated per section). "
                "Use for domain hints — e.g. backlog/sysctl tuning only if counters show pressure."
            ),
            telemetry_body,
            "Current tune state:",
            f"- active_changes={active_changes}",
            f"- best_config={best_config}",
            "Selectable candidates (this phase):",
            *candidate_lines,
            (
                "Deferred candidates (reboot-required per service policy; listed for visibility; "
                "selectable only in reboot_batch when engagement allow_reboot is true):"
            ),
            *(deferred_lines or ["- none"]),
            "Prior hypothesis history:",
            *(history_lines or ["- none"]),
        ]
        return "\n".join(sections)

    def _phase_objective(self, phase: TunePhase) -> str:
        objectives = {
            TunePhase.WIDE_SWEEP: "Explore broadly across domains with maximum diversity.",
            TunePhase.DOMAIN_FOCUS: "Focus on domains that have shown positive signal.",
            TunePhase.INTERACTION: "Explore interactions between promising parameters.",
            TunePhase.BOUNDARY_PUSH: "Push promising parameters toward safe limits.",
            TunePhase.EXPLOIT: "Refine around the current best configuration.",
            TunePhase.REBOOT_BATCH: "Batch deferred reboot-required changes if any exist.",
        }
        return objectives[phase]


@dataclass
class LlmHypothesisGenerator:
    model_client: HypothesisModelClient
    logger: ExecutionLogger = NullExecutionLogger()

    def generate(self, context: HypothesisContext) -> TuningHypothesis:
        self._debug_log(
            "LLM call",
            f"phase={context.phase.value} iteration={context.iteration_number} "
            f"candidates={len(context.candidates)}",
        )
        response = self.model_client.complete(context)
        self._debug_log("LLM raw response", response.content)
        payload = json.loads(response.content)
        self._debug_log("LLM parsed payload", json.dumps(payload, sort_keys=True))
        parameter_key = self._require_string(payload, "parameter_key")
        proposed_value = self._require_string(payload, "proposed_value")
        rationale = self._require_string(payload, "rationale")
        candidate = self._find_candidate(context, parameter_key)
        self._validate_proposed_value(candidate, proposed_value)
        return TuningHypothesis(
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
            msg = f"Model proposed no-op value {proposed_value!r} for {candidate.parameter_key}"
            raise ValueError(msg)

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


@dataclass
class DeterministicHypothesisGenerator:
    def generate(self, context: HypothesisContext) -> TuningHypothesis:
        tried_keys = {
            record.hypothesis.parameter_key
            for record in context.history
            if record.phase == context.phase
        }
        for candidate in context.candidates:
            if candidate.parameter_key in tried_keys:
                continue
            proposed_value = self._default_value(candidate)
            return TuningHypothesis(
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
