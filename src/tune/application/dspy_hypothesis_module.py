"""DSPy module for structured hypothesis generation.

Replaces raw OpenAI JSON prompting with a typed Signature + ChainOfThought module.
The module is a singleton so compiled demos (loaded from disk) persist across calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import dspy
import pydantic

if TYPE_CHECKING:
    from tune.infrastructure.model_config import ModelEndpointConfig


class HypothesisProposal(pydantic.BaseModel):
    """Structured output for one tuning hypothesis. Field names match the JSON contract."""

    parameter_key: str
    proposed_value: str
    tuning_layer: str
    apply_mode: str
    rationale: str
    expected_benchmark_impact: str
    rollback_plan: str


class ProposeHypothesis(dspy.Signature):
    """You are the single hybrid hypothesizer for HostTune.

    A deterministic rule-based triage layer has already inspected the host and
    service context. Use triage signal as a hard priority input, then reason
    across service, runtime, kernel, network, and platform layers to choose
    exactly one change from the selectable candidates.
    """

    context: str = dspy.InputField(
        desc=(
            "Full tuning context: host facts, triage result, baseline benchmarks, "
            "candidate parameters with constraints, and prior iteration history"
        )
    )
    hypothesis: HypothesisProposal = dspy.OutputField(
        desc=(
            "Exactly one tuning hypothesis. parameter_key must name a selectable candidate. "
            "tuning_layer and apply_mode must match the candidate exactly."
        )
    )


class HypothesisPredictor(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        # ChainOfThought adds a hidden reasoning step before emitting the structured output.
        self.predict = dspy.ChainOfThought(ProposeHypothesis)

    def forward(self, context: str) -> dspy.Prediction:
        return self.predict(context=context)


# Module-level singleton so compiled demos persist across calls within a process.
_PREDICTOR: HypothesisPredictor | None = None


def configure_dspy(config: ModelEndpointConfig) -> None:
    """Configure the global DSPy LM from model endpoint config."""
    lm = dspy.LM(
        f"openai/{config.model_name}",
        api_base=config.base_url,
        api_key=config.api_key,
        temperature=0.0,
    )
    dspy.configure(lm=lm)


def get_predictor(compiled_path: Path | None = None) -> HypothesisPredictor:
    """Return the module-level predictor, loading compiled demos if available."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = HypothesisPredictor()
        if compiled_path is not None and compiled_path.exists():
            _PREDICTOR.load(str(compiled_path))
    return _PREDICTOR


def reset_predictor() -> None:
    """Clear the singleton so the next call reloads from disk (e.g. after auto-compile)."""
    global _PREDICTOR
    _PREDICTOR = None


def call_predictor(
    context: str, compiled_path: Path | None = None
) -> tuple[HypothesisProposal, str | None]:
    """Call the predictor and return (HypothesisProposal, reasoning).

    reasoning is DSPy's ChainOfThought internal reasoning step — the model's
    step-by-step thinking before producing the structured output. May be None
    if the model didn't emit a reasoning field.
    """
    predictor = get_predictor(compiled_path)
    result = predictor(context=context)
    reasoning: str | None = getattr(result, "reasoning", None)
    return cast(HypothesisProposal, result.hypothesis), reasoning
