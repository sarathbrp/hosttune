from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import CandidateParameter, TuningHypothesis


class HypothesisModelClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return a serialized hypothesis proposal."""


@dataclass
class HypothesisPromptBuilder:
    def build(self, context: HypothesisContext) -> str:
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
        sections = [
            "You must select exactly one candidate parameter from the allowed list.",
            "Return JSON with keys: parameter_key, proposed_value, rationale.",
            f"Current phase: {context.phase.value}",
            f"Iteration number: {context.iteration_number}",
            "Allowed candidates:",
            *candidate_lines,
            "Prior hypothesis history:",
            *(history_lines or ["- none"]),
        ]
        return "\n".join(sections)


@dataclass
class LlmHypothesisGenerator:
    model_client: HypothesisModelClient
    prompt_builder: HypothesisPromptBuilder

    def generate(self, context: HypothesisContext) -> TuningHypothesis:
        prompt = self.prompt_builder.build(context)
        response = self.model_client.complete(prompt)
        payload = json.loads(response)
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
        )

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
