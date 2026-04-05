import json

from preflight.domain.runtime_artifacts import RuntimeArtifacts
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import CandidateAvailability, ModelUsage, TunePhase
from tune.infrastructure.langgraph_hypothesis_client import LangGraphHypothesisClient
from tune.infrastructure.model_config import ModelEndpointConfig
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder

from tests.tune.test_candidate_catalog_builder import FakeExecutor, build_tune_context


class _PromptBuilder:
    def build(self, context: HypothesisContext) -> str:
        _ = context
        return "PROMPT"


class _ClientDouble(LangGraphHypothesisClient):
    def _call_llm_with_usage(self, *, caller: str, system: str, prompt: str):  # type: ignore[override]
        _ = caller
        _ = system
        assert prompt == "PROMPT"
        return (
            json.dumps(
                {
                    "parameter_key": "service.directive.worker_processes",
                    "proposed_value": "56",
                    "tuning_layer": "service",
                    "apply_mode": "reload",
                    "rationale": "test",
                    "expected_benchmark_impact": "test",
                    "rollback_plan": "test",
                }
            ),
            ModelUsage(
                model_name="/models/test",
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
            ),
        )


def test_langgraph_client_saves_prompt_artifact_and_index(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base = build_tune_context()
    artifacts = RuntimeArtifacts(
        session_id="abc123def456",
        session_directory=tmp_path / "artifacts" / "abc123def456",
    )
    artifacts.session_directory.mkdir(parents=True, exist_ok=True)
    tune_context = base.__class__(
        preflight=base.preflight,
        onboard=base.onboard,
        snapshot=base.snapshot,
        baseline=base.baseline,
        benchmark_config=base.benchmark_config,
        artifacts=artifacts,
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
    client = _ClientDouble(
        config=ModelEndpointConfig(
            base_url="http://example/v1",
            api_key="test-key",
            model_name="/models/test",
        ),
        prompt_builder=_PromptBuilder(),
    )

    completion = client.complete(context)

    assert completion.artifact_path is not None
    artifact_path = artifacts.session_directory / "hypotheses" / "iter001_hybrid_hypothesizer.json"
    index_path = (
        artifacts.session_directory / "hypotheses" / "prompt_artifacts_abc123def456.jsonl"
    )
    assert artifact_path.exists()
    assert index_path.exists()
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    index_payload = json.loads(index_path.read_text(encoding="utf-8").splitlines()[0])
    assert artifact_payload["prompt"] == "PROMPT"
    assert artifact_payload["token_usage"]["total_tokens"] == 120
    assert index_payload["artifact_path"] == str(artifact_path)
    assert index_payload["token_usage"]["total_tokens"] == 120
    assert artifacts.stage_files["prompt_artifacts"] == index_path
