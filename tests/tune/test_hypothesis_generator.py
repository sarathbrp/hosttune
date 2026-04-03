import json

import pytest

from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.application.hypothesis_generator import (
    DeterministicHypothesisGenerator,
    HypothesisPromptBuilder,
    LlmHypothesisGenerator,
)
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import (
    HypothesisRecord,
    HypothesisStatus,
    TunePhase,
)

from tests.tune.test_candidate_catalog_builder import build_tune_context


class FakeModelClient:
    def __init__(self, response: dict[str, str]) -> None:
        self._response = response

    def complete(self, prompt: str) -> str:
        assert "Allowed candidates:" in prompt
        return json.dumps(self._response)


def build_hypothesis_context() -> HypothesisContext:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context)
    return HypothesisContext(
        phase=TunePhase.WIDE_SWEEP,
        iteration_number=1,
        candidates=candidates,
        history=(),
    )


def test_llm_hypothesis_generator_accepts_allowed_candidate() -> None:
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(
            {
                "parameter_key": "service.directive.worker_processes",
                "proposed_value": "56",
                "rationale": "Match worker count to a balanced subset of logical cores.",
            }
        ),
        prompt_builder=HypothesisPromptBuilder(),
    )

    hypothesis = generator.generate(context)

    assert hypothesis.parameter_key == "service.directive.worker_processes"
    assert hypothesis.proposed_value == "56"
    assert hypothesis.phase == TunePhase.WIDE_SWEEP


def test_llm_hypothesis_generator_rejects_unknown_parameter() -> None:
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(
            {
                "parameter_key": "service.directive.unknown_knob",
                "proposed_value": "1",
                "rationale": "Try something unsupported.",
            }
        ),
        prompt_builder=HypothesisPromptBuilder(),
    )

    with pytest.raises(ValueError, match="unsupported parameter_key"):
        generator.generate(context)


def test_deterministic_hypothesis_generator_skips_tried_candidates() -> None:
    base_context = build_hypothesis_context()
    first_candidate = base_context.candidates[0]
    history = (
        HypothesisRecord(
            iteration_number=1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=DeterministicHypothesisGenerator().generate(base_context),
            status=HypothesisStatus.REJECTED,
            evaluation_summary="No improvement",
        ),
    )
    context = HypothesisContext(
        phase=TunePhase.WIDE_SWEEP,
        iteration_number=2,
        candidates=base_context.candidates,
        history=history,
    )

    hypothesis = DeterministicHypothesisGenerator().generate(context)

    assert hypothesis.parameter_key != first_candidate.parameter_key
