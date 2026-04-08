import json
from dataclasses import replace
from pathlib import Path

import pytest

from preflight.domain.runtime_artifacts import RuntimeArtifacts
from preflight.infrastructure.knowledge_base import KnowledgeBase
from preflight.interfaces.execution_logger import DebugExecutionLogger, VerboseExecutionLogger
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.application.hypothesis_generator import (
    DeterministicHypothesisGenerator,
    HypothesisPromptBuilder,
    LlmHypothesisGenerator,
)
from tune.application.rule_based_triage import RuleBasedTriage, TriageRulesLoader
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import (
    CandidateAvailability,
    CandidateParameter,
    HypothesisRecord,
    HypothesisStatus,
    ModelCompletion,
    ModelUsage,
    TunePhase,
)
from tune.domain.tuning_layer import TuningLayer

from tests.tune.test_candidate_catalog_builder import FakeExecutor, build_tune_context


RULES_PATH = Path("triage-rules.yaml")


class FakeModelClient:
    def __init__(self, response: dict[str, object]) -> None:
        self._response = response

    def complete(self, context: HypothesisContext) -> ModelCompletion:
        prompt = HypothesisPromptBuilder(
            triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH))
        ).build(context)
        assert "Rule-based triage result:" in prompt
        assert "Selected runtime config snippet:" in prompt
        assert "Selected service YAML reference snippet:" in prompt
        assert "safe_candidate_subset=" in prompt
        assert "suppressed_candidates=" in prompt
        assert "Output rules:" in prompt
        assert "choose exactly one selectable candidate" in prompt
        assert "triage autofix is already resolved before this prompt" in prompt
        assert "do not invent unsupported knobs mentioned only in signal text" in prompt
        assert "current=112" in prompt
        assert "forbidden=" not in prompt
        assert "priority=high" in prompt
        assert "systemd.unit.limit_nproc" in prompt
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

    def debug_enabled(self) -> bool:
        return True

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


def _valid_response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "parameter_key": "service.directive.worker_processes",
        "proposed_value": "56",
        "tuning_layer": "service",
        "apply_mode": "reload",
        "rationale": "Match worker count to a balanced subset of logical cores.",
        "expected_benchmark_impact": "Moderate RPS uplift on homepage and small workloads.",
        "rollback_plan": "Restore worker_processes to the previous nginx.conf value and reload.",
    }
    payload.update(overrides)
    return payload


def test_llm_hypothesis_generator_accepts_allowed_candidate() -> None:
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(_valid_response()),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    hypotheses = generator.generate(context)
    hypothesis = hypotheses[0]

    assert hypothesis.parameter_key == "service.directive.worker_processes"
    assert hypothesis.proposed_value == "56"
    assert hypothesis.phase == TunePhase.WIDE_SWEEP
    assert hypothesis.tuning_layer is TuningLayer.SERVICE
    assert hypothesis.model_usage is not None
    assert hypothesis.model_usage.input_tokens == 120
    assert hypothesis.expected_benchmark_impact is not None
    assert hypothesis.rollback_plan is not None


def test_llm_hypothesis_generator_records_prompt_artifact_path_in_kb(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base = build_tune_context()
    artifacts = RuntimeArtifacts(
        session_id="abc123def456",
        session_directory=tmp_path / "artifacts" / "abc123def456",
    )
    artifacts.session_directory.mkdir(parents=True, exist_ok=True)
    knowledge_base = KnowledgeBase(tmp_path / "artifacts" / "knowledge_base.sqlite")
    tune_context = base.__class__(
        preflight=base.preflight,
        onboard=base.onboard,
        snapshot=base.snapshot,
        baseline=base.baseline,
        benchmark_config=base.benchmark_config,
        artifacts=artifacts,
        host_profile=base.host_profile,
        knowledge_base=knowledge_base,
    )
    built = CandidateCatalogBuilder().build(tune_context, FakeExecutor())
    context = HypothesisContext(
        tune_context=tune_context,
        phase=TunePhase.WIDE_SWEEP,
        iteration_number=1,
        candidates=tuple(c for c in built if c.availability is CandidateAvailability.ACTIVE),
        deferred_candidates=tuple(
            c for c in built if c.availability is CandidateAvailability.DEFERRED
        ),
        history=(),
        active_parameter_keys=(),
        best_parameter_values=(),
    )

    class ArtifactModelClient(FakeModelClient):
        def complete(self, context: HypothesisContext) -> ModelCompletion:
            completion = super().complete(context)
            return ModelCompletion(
                content=completion.content,
                usage=completion.usage,
                artifact_path="/tmp/fake_prompt_artifact.json",
            )

    generator = LlmHypothesisGenerator(
        model_client=ArtifactModelClient(_valid_response()),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    generator.generate(context)

    events = knowledge_base.get_run_events("abc123def456")
    artifact_events = [event for event in events if event["event_type"] == "llm_prompt_artifact_saved"]
    proposal_events = [event for event in events if event["event_type"] == "llm_proposal_selected"]
    assert artifact_events
    assert artifact_events[0]["payload"]["artifact_path"] == "/tmp/fake_prompt_artifact.json"
    assert proposal_events
    assert proposal_events[0]["payload"]["artifact_path"] == "/tmp/fake_prompt_artifact.json"


def test_llm_hypothesis_generator_logs_prompt_and_response_in_debug() -> None:
    context = build_hypothesis_context()
    logger = CaptureDebugLogger()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(_valid_response()),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
        logger=logger,
    )

    generator.generate(context)

    assert any("LLM call:" in message for message in logger.messages)
    assert any("LLM raw response:" in message for message in logger.messages)
    assert any("LLM parsed payload:" in message for message in logger.messages)


def test_llm_hypothesis_generator_does_not_log_prompt_in_verbose() -> None:
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(_valid_response()),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
        logger=VerboseExecutionLogger(),
    )

    hypotheses = generator.generate(context)
    hypothesis = hypotheses[0]

    assert hypothesis.parameter_key == "service.directive.worker_processes"


def test_llm_hypothesis_generator_rejects_unknown_parameter() -> None:
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(_valid_response(parameter_key="service.directive.unknown_knob")),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    with pytest.raises(ValueError, match="unsupported parameter_key"):
        generator.generate(context)


def test_llm_hypothesis_generator_rejects_noop_value() -> None:
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(_valid_response(proposed_value="112")),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    with pytest.raises(ValueError, match="no-op value .*current_value_source="):
        generator.generate(context)


def test_llm_hypothesis_generator_rejects_duplicate_parameter_value_pair() -> None:
    base_context = build_hypothesis_context()
    history = (
        HypothesisRecord(
            iteration_number=1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=DeterministicHypothesisGenerator().generate(base_context)[0],
            status=HypothesisStatus.ACCEPTED,
            evaluation_summary="accepted",
        ),
    )
    context = HypothesisContext(
        tune_context=base_context.tune_context,
        phase=TunePhase.DOMAIN_FOCUS,
        iteration_number=2,
        candidates=base_context.candidates,
        deferred_candidates=base_context.deferred_candidates,
        history=history,
        active_parameter_keys=(),
        best_parameter_values=(),
    )
    duplicate = history[0].hypothesis
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(
            _valid_response(
                parameter_key=duplicate.parameter_key,
                proposed_value=duplicate.proposed_value,
                tuning_layer=duplicate.tuning_layer.value,
                apply_mode=duplicate.apply_mode.value,
            )
        ),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    with pytest.raises(ValueError, match="duplicate parameter/value pair .*current_value_source="):
        generator.generate(context)


def test_llm_hypothesis_generator_repairs_forbidden_enum_value() -> None:
    base = build_hypothesis_context()
    class MinimalModelClient:
        def __init__(self, response: dict[str, object]) -> None:
            self._response = response

        def complete(self, context: HypothesisContext) -> ModelCompletion:
            _ = context
            return ModelCompletion(content=json.dumps(self._response))

    candidate = CandidateParameter(
        parameter_key="service.directive.tcp_nopush",
        domain="service_config",
        tuning_layer=TuningLayer.SERVICE,
        parameter_name="tcp_nopush",
        source=base.candidates[0].source,
        value_type=base.candidates[0].value_type,
        apply_mode=base.candidates[0].apply_mode,
        priority_tier=base.candidates[0].priority_tier,
        allowed_values=("on", "off"),
        forbidden_values=("off",),
        min_value=None,
        max_value=None,
        rationale_hint="test",
        current_value="off",
    )
    context = HypothesisContext(
        tune_context=base.tune_context,
        phase=base.phase,
        iteration_number=base.iteration_number,
        candidates=(candidate,),
        deferred_candidates=(),
        history=(),
        active_parameter_keys=(),
        best_parameter_values=(),
    )
    generator = LlmHypothesisGenerator(
        model_client=MinimalModelClient(
            _valid_response(
                parameter_key="service.directive.tcp_nopush",
                proposed_value="off",
                rationale="test forbidden repair",
            )
        ),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    hypothesis = generator.generate(context)[0]

    assert hypothesis.parameter_key == "service.directive.tcp_nopush"
    assert hypothesis.proposed_value == "on"


def test_llm_hypothesis_generator_repairs_noop_enum_value() -> None:
    base = build_hypothesis_context()
    class MinimalModelClient:
        def __init__(self, response: dict[str, object]) -> None:
            self._response = response

        def complete(self, context: HypothesisContext) -> ModelCompletion:
            _ = context
            return ModelCompletion(content=json.dumps(self._response))

    candidate = CandidateParameter(
        parameter_key="service.directive.multi_accept",
        domain="service_config",
        tuning_layer=TuningLayer.SERVICE,
        parameter_name="multi_accept",
        source=base.candidates[0].source,
        value_type=base.candidates[0].value_type,
        apply_mode=base.candidates[0].apply_mode,
        priority_tier=base.candidates[0].priority_tier,
        allowed_values=("on", "off"),
        forbidden_values=(),
        min_value=None,
        max_value=None,
        rationale_hint="test",
        current_value="on",
    )
    context = HypothesisContext(
        tune_context=base.tune_context,
        phase=base.phase,
        iteration_number=base.iteration_number,
        candidates=(candidate,),
        deferred_candidates=(),
        history=(),
        active_parameter_keys=(),
        best_parameter_values=(),
    )
    generator = LlmHypothesisGenerator(
        model_client=MinimalModelClient(
            _valid_response(
                parameter_key="service.directive.multi_accept",
                proposed_value="on",
                rationale="test noop repair",
            )
        ),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    hypothesis = generator.generate(context)[0]

    assert hypothesis.parameter_key == "service.directive.multi_accept"
    assert hypothesis.proposed_value == "off"


def test_llm_hypothesis_generator_rejects_array_payload() -> None:
    context = build_hypothesis_context()

    class ArrayModelClient:
        def complete(self, context: HypothesisContext) -> ModelCompletion:
            _ = context
            return ModelCompletion(content=json.dumps([_valid_response()]))

    generator = LlmHypothesisGenerator(model_client=ArrayModelClient())

    with pytest.raises(ValueError, match="exactly one JSON object"):
        generator.generate(context)


def test_llm_hypothesis_generator_rejects_mismatched_layer() -> None:
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(_valid_response(tuning_layer="kernel")),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    with pytest.raises(ValueError, match="mismatched tuning_layer"):
        generator.generate(context)


def test_llm_hypothesis_generator_accepts_numeric_proposed_value() -> None:
    context = build_hypothesis_context()
    generator = LlmHypothesisGenerator(
        model_client=FakeModelClient(_valid_response(proposed_value=56)),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    hypotheses = generator.generate(context)
    hypothesis = hypotheses[0]

    assert hypothesis.proposed_value == "56"


def test_llm_hypothesis_generator_short_circuits_for_autofix() -> None:
    base = build_hypothesis_context()
    context = HypothesisContext(
        tune_context=base.tune_context,
        phase=base.phase,
        iteration_number=base.iteration_number,
        candidates=tuple(
            replace(candidate, current_value="off")
            if candidate.parameter_key == "service.directive.sendfile"
            else candidate
            for candidate in base.candidates
        ),
        deferred_candidates=base.deferred_candidates,
        history=base.history,
        active_parameter_keys=base.active_parameter_keys,
        best_parameter_values=base.best_parameter_values,
    )

    class FailIfCalledModelClient:
        def complete(self, context: HypothesisContext) -> ModelCompletion:
            _ = context
            raise AssertionError("LLM should not be called for autofix")

    generator = LlmHypothesisGenerator(
        model_client=FailIfCalledModelClient(),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    hypothesis = generator.generate(context)[0]

    assert hypothesis.parameter_key == "service.directive.sendfile"
    assert hypothesis.proposed_value == "on"
    assert hypothesis.model_usage is None


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


def test_deterministic_hypothesis_generator_skips_unsupported_candidate_defaults() -> None:
    base_context = build_hypothesis_context()
    first = base_context.candidates[0]
    second = base_context.candidates[1]
    unsupported = replace(
        first,
        parameter_key="sysctl.net.core.somaxconn",
        parameter_name="net.core.somaxconn",
        allowed_values=(),
        forbidden_values=(),
        min_value=None,
        max_value=None,
    )
    supported = replace(
        second,
        parameter_key="service.directive.access_log",
        parameter_name="access_log",
        allowed_values=("off", "/var/log/nginx/access.log"),
        forbidden_values=(),
        min_value=None,
        max_value=None,
    )
    context = HypothesisContext(
        tune_context=base_context.tune_context,
        phase=TunePhase.WIDE_SWEEP,
        iteration_number=1,
        candidates=(unsupported, supported),
        deferred_candidates=(),
        history=(),
        active_parameter_keys=(),
        best_parameter_values=(),
    )

    hypothesis = DeterministicHypothesisGenerator().generate(context)[0]

    assert hypothesis.parameter_key == "service.directive.access_log"
    assert hypothesis.proposed_value == "off"


def test_diminishing_return_blocks_re_escalation_without_gain() -> None:
    """Escalating a boundary-push sysctl after REJECTED should be blocked."""
    from onboard.domain.models import ApplyMode
    from tune.domain.hypothesis_models import CandidateSource, TuningHypothesis
    from tune.domain.tuning_layer import tuning_layer_for_parameter_key

    base_context = build_hypothesis_context()
    somaxconn_candidate = next(
        (c for c in base_context.candidates if c.parameter_key == "sysctl.net.core.somaxconn"),
        None,
    )
    if somaxconn_candidate is None:
        pytest.skip("somaxconn candidate not in test catalog")

    prior_hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="sysctl.net.core.somaxconn",
        parameter_name="net.core.somaxconn",
        domain="kernel_sysctl",
        tuning_layer=tuning_layer_for_parameter_key("sysctl.net.core.somaxconn"),
        proposed_value="8192",
        source=CandidateSource.SERVICE_SYSCTL,
        apply_mode=ApplyMode.RELOAD,
        rationale="test escalation",
    )
    history = (
        HypothesisRecord(
            iteration_number=1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=prior_hypothesis,
            status=HypothesisStatus.REJECTED,
            evaluation_summary="No improvement",
        ),
    )

    class EscalatingModelClient:
        def complete(self, context: HypothesisContext) -> ModelCompletion:
            return ModelCompletion(
                content=json.dumps({
                    "parameter_key": "sysctl.net.core.somaxconn",
                    "proposed_value": "16384",
                    "rationale": "push higher",
                    "tuning_layer": "kernel",
                    "apply_mode": "reload",
                    "expected_benchmark_impact": "marginal",
                    "rollback_plan": "revert to 8192",
                }),
                usage=None,
            )

    context = HypothesisContext(
        tune_context=base_context.tune_context,
        phase=TunePhase.BOUNDARY_PUSH,
        iteration_number=2,
        candidates=base_context.candidates,
        deferred_candidates=base_context.deferred_candidates,
        history=history,
        active_parameter_keys=(),
        best_parameter_values=(),
    )
    generator = LlmHypothesisGenerator(
        model_client=EscalatingModelClient(),
        triage=RuleBasedTriage(TriageRulesLoader().load(RULES_PATH)),
    )

    with pytest.raises(ValueError, match="Diminishing-return suppression"):
        generator.generate(context)
