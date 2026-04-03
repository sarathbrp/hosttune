import json
from dataclasses import replace
from typing import cast

import pytest

from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.interfaces.execution_logger import DebugExecutionLogger, VerboseExecutionLogger
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.application.hypothesis_generator import (
    DeterministicHypothesisGenerator,
    HypothesisPromptBuilder,
    LlmHypothesisGenerator,
)
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import (
    CandidateAvailability,
    HypothesisRecord,
    HypothesisStatus,
    ModelCompletion,
    ModelUsage,
    TunePhase,
)
from tune.domain.tuning_layer import TuningLayer

from tests.onboard.test_service_definition_validator import build_valid_definition
from tests.tune.test_candidate_catalog_builder import FakeExecutor, build_tune_context


class FakeModelClient:
    def __init__(self, response: dict[str, object]) -> None:
        self._response = response

    def complete(self, context: HypothesisContext) -> ModelCompletion:
        # Build the full monolithic prompt to validate its contents.
        prompt = HypothesisPromptBuilder().build(context)
        assert "Selectable candidates (this phase):" in prompt
        assert "Deferred candidates" in prompt
        assert "Context policy:" in prompt
        assert "Preflight digest" in prompt
        assert "Onboard service contract" in prompt
        assert "Baseline digest" in prompt
        assert "Last benchmark runtime telemetry digest" in prompt
        assert "kernel_sysctl_profile=" in prompt
        assert "Current tune state:" in prompt
        assert "current=112" in prompt
        assert "forbidden=" not in prompt
        assert "priority=high" in prompt
        assert "runtime_limits=nofile_soft" in prompt
        assert "systemd_unit_limits=limit_nofile, limit_nproc" in prompt
        assert "cgroup_resource_controls=cpu_quota_percent, memory_max_mib" in prompt
        assert "Limit baselines" in prompt
        assert "runtime.prlimit.nofile_soft" in prompt
        assert "systemd.unit.limit_nproc" in prompt
        assert "Snapshot digest" in prompt
        assert "nginx -T" in prompt
        assert "process_state" in prompt
        return ModelCompletion(
            content=json.dumps([self._response]),
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
    built = CandidateCatalogBuilder().build(context, FakeExecutor())
    deferred = tuple(c for c in built if c.availability is CandidateAvailability.DEFERRED)
    active_only = tuple(c for c in built if c.availability is CandidateAvailability.ACTIVE)
    return HypothesisContext(
        tune_context=context,
        phase=TunePhase.WIDE_SWEEP,
        iteration_number=1,
        candidates=active_only,
        deferred_candidates=deferred,
        history=(),
        active_parameter_keys=(),
        best_parameter_values=(),
    )


def test_hypothesis_prompt_lists_deferred_sysctl_when_kernel_network_reboot() -> None:
    data = build_valid_definition()
    restart = cast(dict[str, object], data["restart"])
    categories = cast(dict[str, object], restart["change_categories"])
    restart = {
        **restart,
        "change_categories": {**categories, "kernel_network": "reboot"},
    }
    data = {**data, "restart": restart}
    service = ServiceDefinitionValidator().validate(data)
    base = build_tune_context()
    ctx = replace(base, onboard=replace(base.onboard, service=service))
    built = CandidateCatalogBuilder().build(ctx, FakeExecutor())
    deferred = [c for c in built if c.availability is CandidateAvailability.DEFERRED]
    assert deferred
    assert any(c.parameter_key == "sysctl.net.core.somaxconn" for c in deferred)
    active_only = tuple(c for c in built if c.availability is CandidateAvailability.ACTIVE)
    assert not any(c.parameter_key == "sysctl.net.core.somaxconn" for c in active_only)
    hctx = HypothesisContext(
        tune_context=ctx,
        phase=TunePhase.WIDE_SWEEP,
        iteration_number=1,
        candidates=active_only,
        deferred_candidates=tuple(deferred),
        history=(),
        active_parameter_keys=(),
        best_parameter_values=(),
    )
    prompt = HypothesisPromptBuilder().build(hctx)
    assert "availability=deferred" in prompt
    assert "sysctl.net.core.somaxconn" in prompt


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

    )

    hypotheses = generator.generate(context)
    hypothesis = hypotheses[0]

    assert hypothesis.parameter_key == "service.directive.worker_processes"
    assert hypothesis.proposed_value == "56"
    assert hypothesis.phase == TunePhase.WIDE_SWEEP
    assert hypothesis.tuning_layer is TuningLayer.SERVICE
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

        logger=logger,
    )

    generator.generate(context)

    assert any("LLM call:" in message for message in logger.messages)
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

        logger=VerboseExecutionLogger(),
    )

    hypotheses = generator.generate(context)
    hypothesis = hypotheses[0]

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

    )

    with pytest.raises(ValueError, match="invalid or empty"):
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

    )

    with pytest.raises(ValueError, match="invalid or empty"):
        generator.generate(context)


def test_llm_hypothesis_generator_passes_forbidden_value_to_pre_apply_gate() -> None:
    """Forbidden values are not shown in the prompt; PreApplyValidator rejects them."""
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(
            {
                "parameter_key": "service.directive.sendfile",
                "proposed_value": "off",
                "rationale": "Try a known-bad setting.",
            }
        ),

    )

    hypotheses = generator.generate(context)
    hypothesis = hypotheses[0]

    assert hypothesis.parameter_key == "service.directive.sendfile"
    assert hypothesis.proposed_value == "off"


def test_llm_hypothesis_generator_accepts_numeric_proposed_value() -> None:
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(
            {
                "parameter_key": "service.directive.worker_processes",
                "proposed_value": 56,
                "rationale": "Return a numeric JSON value.",
            }
        ),

    )

    hypotheses = generator.generate(context)
    hypothesis = hypotheses[0]

    assert hypothesis.proposed_value == "56"


def test_deterministic_hypothesis_generator_skips_tried_candidates() -> None:
    base_context = build_hypothesis_context()
    first_candidate = base_context.candidates[0]
    history = (
        HypothesisRecord(
            iteration_number=1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=DeterministicHypothesisGenerator().generate(base_context)[0],
            status=HypothesisStatus.REJECTED,
            evaluation_summary="No improvement",
        ),
    )
    context = HypothesisContext(
        tune_context=base_context.tune_context,
        phase=TunePhase.WIDE_SWEEP,
        iteration_number=2,
        candidates=base_context.candidates,
        deferred_candidates=base_context.deferred_candidates,
        history=history,
        active_parameter_keys=(),
        best_parameter_values=(),
    )

    hypothesis = DeterministicHypothesisGenerator().generate(context)[0]

    assert hypothesis.parameter_key != first_candidate.parameter_key
