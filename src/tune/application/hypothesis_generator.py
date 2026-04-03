from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import (
    CandidateParameter,
    ModelCompletion,
    TunePhase,
    TuningHypothesis,
)


class HypothesisModelClient(Protocol):
    def complete(self, prompt: str) -> ModelCompletion:
        """Return a serialized hypothesis proposal plus optional model usage."""


@dataclass
class HypothesisPromptBuilder:
    def build(self, context: HypothesisContext) -> str:
        tune_context = context.tune_context
        candidate_lines = [
            (
                f"- key={candidate.parameter_key}; "
                f"domain={candidate.domain}; "
                f"parameter={candidate.parameter_name}; "
                f"source={candidate.source.value}; "
                f"apply_mode={candidate.apply_mode.value}; "
                f"value_type={candidate.value_type.value}; "
                f"min={candidate.min_value}; "
                f"max={candidate.max_value}; "
                f"allowed={candidate.allowed_values}; "
                f"current={candidate.current_value}; "
                f"hint={candidate.rationale_hint}"
            )
            for candidate in context.candidates
        ]
        history_lines = [
            (
                f"- iteration={record.iteration_number}; "
                f"phase={record.phase.value}; "
                f"parameter_key={record.hypothesis.parameter_key}; "
                f"value={record.hypothesis.proposed_value}; "
                f"status={record.status.value}; "
                f"evaluation={record.evaluation_summary}"
            )
            for record in context.history
        ]
        capability_lines = [
            f"- {flag.name}: {flag.detail}"
            for flag in tune_context.preflight.capability_map.flags
            if flag.available
        ]
        workload_lines = [
            (
                f"- {workload.workload_name}: "
                f"rps={workload.requests_per_second:.2f}; "
                f"latency_ms={workload.average_latency_ms:.2f}; "
                f"total={workload.total_requests}"
            )
            for workload in tune_context.baseline.workload_results
        ]
        phase_objective = self._phase_objective(context.phase)
        best_config = (
            ", ".join(f"{key}={value}" for key, value in context.best_parameter_values) or "none"
        )
        active_changes = ", ".join(context.active_parameter_keys) or "none"
        allowed_directives = ", ".join(
            sorted(tune_context.onboard.service.tunable_surface.allowed_directives)
        )
        relevant_sysctls = ", ".join(tune_context.onboard.service.tunable_surface.relevant_sysctls)
        guardrails = ", ".join(tune_context.onboard.service.benchmark_hints.guardrail_metrics)
        interference = ", ".join(tune_context.onboard.service.benchmark_hints.interference_sources)
        sections = [
            "You are the hypothesis generator for HostTune.",
            "Select exactly one candidate parameter from the allowed list.",
            "Do not invent parameters, values, or apply modes outside the candidate list.",
            "Prefer unexplored or promising regions based on prior history and current phase.",
            "Return strict JSON with keys: parameter_key, proposed_value, rationale.",
            f"Current phase: {context.phase.value}",
            f"Phase objective: {phase_objective}",
            f"Iteration number: {context.iteration_number}",
            "Preflight summary:",
            f"- platform={tune_context.preflight.platform_summary}",
            (
                "- cpu="
                f"{tune_context.preflight.cpu.logical_cores} logical cores; "
                f"numa_nodes={tune_context.preflight.cpu.numa_nodes}; "
                f"hyperthreading={tune_context.preflight.cpu.hyperthreading_enabled}"
            ),
            (
                "- memory="
                f"swap_kib={tune_context.preflight.memory.swap_total_kib}; "
                f"hugepages_total={tune_context.preflight.memory.hugepages_total}; "
                f"thp={tune_context.preflight.memory.transparent_hugepages_mode}"
            ),
            (
                "- network="
                f"{tune_context.preflight.network.interface_name}; "
                f"driver={tune_context.preflight.network.driver_name}; "
                f"queues={tune_context.preflight.network.combined_queues}"
            ),
            "Available tunable surfaces:",
            *(capability_lines or ["- none"]),
            "Service contract summary:",
            f"- service={tune_context.onboard.service_name}",
            f"- allowed_directives={allowed_directives or 'none'}",
            f"- relevant_sysctls={relevant_sysctls or 'none'}",
            f"- health_probe={tune_context.onboard.service.health_check.probe_type.value}",
            f"- primary_metric={tune_context.onboard.service.benchmark_hints.primary_metric}",
            f"- guardrails={guardrails or 'none'}",
            f"- interference_sources={interference or 'none'}",
            "Baseline summary:",
            f"- target={tune_context.baseline.benchmark_target}",
            f"- expected_variance={tune_context.baseline.expected_variance:.2%}",
            f"- warmup_seconds={tune_context.baseline.warmup_seconds}",
            "Baseline workloads:",
            *(workload_lines or ["- none"]),
            "Current tune state:",
            f"- active_changes={active_changes}",
            f"- best_config={best_config}",
            "Allowed candidates:",
            *candidate_lines,
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
    prompt_builder: HypothesisPromptBuilder
    logger: ExecutionLogger = NullExecutionLogger()

    def generate(self, context: HypothesisContext) -> TuningHypothesis:
        prompt = self.prompt_builder.build(context)
        self._debug_log("LLM prompt", prompt)
        response = self.model_client.complete(prompt)
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
            msg = f"Model proposed no-op value {proposed_value!r} " f"for {candidate.parameter_key}"
            raise ValueError(msg)

    def _require_string(self, payload: dict[str, object], field_name: str) -> str:
        value = payload.get(field_name)
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
            return candidate.allowed_values[0]
        if candidate.min_value is not None and candidate.max_value is not None:
            midpoint = (candidate.min_value + candidate.max_value) // 2
            return str(midpoint)
        if candidate.min_value is not None:
            return str(candidate.min_value)
        return "1"
