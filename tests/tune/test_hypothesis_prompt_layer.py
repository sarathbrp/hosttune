from onboard.domain.models import ApplyMode, DirectiveValueType, PriorityTier
from tune.application.hypothesis_prompt_layer import (
    format_candidate_line_for_llm,
    format_preflight_digest_lines,
    hypothesis_prompt_layer_preamble,
)
from tune.domain.hypothesis_models import CandidateParameter, CandidateSource
from tune.domain.tuning_layer import TuningLayer

from tests.tune.test_candidate_catalog_builder import build_tune_context


def test_preamble_mentions_curated_digests() -> None:
    lines = hypothesis_prompt_layer_preamble()
    assert any("curated digests" in line for line in lines)


def test_preflight_digest_includes_ring_and_storage_facts() -> None:
    ctx = build_tune_context()
    lines = format_preflight_digest_lines(ctx.preflight)
    joined = "\n".join(lines)
    assert "rings_rx=512/4096" in joined
    assert "rings_tx=512/4096" in joined
    assert "storage=" in joined


def test_candidate_line_truncates_long_hint() -> None:
    long_hint = "x" * 500
    candidate = CandidateParameter(
        parameter_key="k",
        domain="d",
        tuning_layer=TuningLayer.RUNTIME,
        parameter_name="p",
        source=CandidateSource.SERVICE_DIRECTIVE,
        value_type=DirectiveValueType.INTEGER,
        apply_mode=ApplyMode.RELOAD,
        priority_tier=PriorityTier.HIGH,
        allowed_values=(),
        forbidden_values=(),
        min_value=1,
        max_value=10,
        rationale_hint=long_hint,
        current_value="5",
    )
    line = format_candidate_line_for_llm(candidate)
    assert "truncated" in line
    assert len(line) < len(long_hint) + 200
