import json

import pytest

from preflight.interfaces.execution_logger import DebugExecutionLogger, VerboseExecutionLogger
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
    ModelCompletion,
    ModelUsage,
    TunePhase,
)

from tests.tune.test_candidate_catalog_builder import build_tune_context
from tests.tune.test_candidate_catalog_builder import FakeExecutor


class FakeModelClient:
    def __init__(self, response: dict[str, str]) -> None:
        self._response = response

    def complete(self, prompt: str) -> ModelCompletion:
        assert "Allowed candidates:" in prompt
        assert "Preflight summary:" in prompt
        assert "Service contract summary:" in prompt
        assert "Baseline summary:" in prompt
        assert "Current tune state:" in prompt
        assert "current=112" in prompt
        return ModelCompletion(
            content=json.dumps(self._response),
            usage=ModelUsage(
                model_name="/models/test-model",
                input_tokens=120,
                output_tokens=24,
                total_tokens=144,
            ),
        )


class CaptureDebugLogger(DebugExecutionLogger):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def stage_detail(self, stage: str, message: str) -> None:
        self.messages.append(f"{stage}:{message}")


def build_hypothesis_context() -> HypothesisContext:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context, FakeExecutor())
    return HypothesisContext(
        tune_context=context,
        phase=TunePhase.WIDE_SWEEP,
        iteration_number=1,
        candidates=candidates,
        history=(),
        active_parameter_keys=(),
        best_parameter_values=(),
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
    assert hypothesis.model_usage is not None
    assert hypothesis.model_usage.input_tokens == 120


def test_llm_hypothesis_generator_logs_prompt_and_response_in_debug() -> None:
    context = build_hypothesis_context()
    logger = CaptureDebugLogger()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(
            {
                "parameter_key": "service.directive.worker_processes",
                "proposed_value": "56",
                "rationale": "Match worker count to a balanced subset of logical cores.",
            }
        ),
        prompt_builder=HypothesisPromptBuilder(),
        logger=logger,
    )

    generator.generate(context)

    assert any("LLM prompt:" in message for message in logger.messages)
    assert any("LLM raw response:" in message for message in logger.messages)
    assert any("LLM parsed payload:" in message for message in logger.messages)


def test_llm_hypothesis_generator_does_not_log_prompt_in_verbose() -> None:
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
        logger=VerboseExecutionLogger(),
    )

    hypothesis = generator.generate(context)

    assert hypothesis.parameter_key == "service.directive.worker_processes"


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


def test_llm_hypothesis_generator_rejects_noop_value() -> None:
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(
            {
                "parameter_key": "service.directive.worker_processes",
                "proposed_value": "112",
                "rationale": "Repeat the current value.",
            }
        ),
        prompt_builder=HypothesisPromptBuilder(),
    )

    with pytest.raises(ValueError, match="no-op value"):
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
        tune_context=base_context.tune_context,
        phase=TunePhase.WIDE_SWEEP,
        iteration_number=2,
        candidates=base_context.candidates,
        history=history,
        active_parameter_keys=(),
        best_parameter_values=(),
    )

    hypothesis = DeterministicHypothesisGenerator().generate(context)

    assert hypothesis.parameter_key != first_candidate.parameter_key
