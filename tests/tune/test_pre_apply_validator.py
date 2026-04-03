from tune.application.pre_apply_validator import PreApplyValidator
from tune.domain.hypothesis_models import TuningHypothesis

from tests.tune.test_candidate_catalog_builder import FakeExecutor, build_tune_context
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.domain.hypothesis_models import CandidateSource, TunePhase


def build_candidate(parameter_key: str):  # type: ignore[no-untyped-def]
    context = build_tune_context()
    return next(
        candidate
        for candidate in CandidateCatalogBuilder().build(context, FakeExecutor())
        if candidate.parameter_key == parameter_key
    )


def test_pre_apply_validator_rejects_forbidden_value() -> None:
    candidate = build_candidate("service.directive.sendfile")
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key=candidate.parameter_key,
        parameter_name=candidate.parameter_name,
        domain=candidate.domain,
        tuning_layer=candidate.tuning_layer,
        proposed_value="off",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=candidate.apply_mode,
        rationale="Try known-bad value.",
    )

    outcome = PreApplyValidator().validate(candidate, hypothesis)

    assert outcome.allowed is False
    assert "forbidden" in outcome.reason


def test_pre_apply_validator_rejects_noop() -> None:
    candidate = build_candidate("service.directive.worker_processes")
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key=candidate.parameter_key,
        parameter_name=candidate.parameter_name,
        domain=candidate.domain,
        tuning_layer=candidate.tuning_layer,
        proposed_value="112",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=candidate.apply_mode,
        rationale="Repeat current value.",
    )

    outcome = PreApplyValidator().validate(candidate, hypothesis)

    assert outcome.allowed is False
    assert "no-op" in outcome.reason


def test_pre_apply_validator_rejects_below_minimum() -> None:
    candidate = build_candidate("service.directive.worker_processes")
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key=candidate.parameter_key,
        parameter_name=candidate.parameter_name,
        domain=candidate.domain,
        tuning_layer=candidate.tuning_layer,
        proposed_value="0",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=candidate.apply_mode,
        rationale="Try below minimum.",
    )

    outcome = PreApplyValidator().validate(candidate, hypothesis)

    assert outcome.allowed is False
    assert "below minimum" in outcome.reason


def test_pre_apply_validator_rejects_above_maximum() -> None:
    candidate = build_candidate("service.directive.worker_processes")
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key=candidate.parameter_key,
        parameter_name=candidate.parameter_name,
        domain=candidate.domain,
        tuning_layer=candidate.tuning_layer,
        proposed_value="999",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=candidate.apply_mode,
        rationale="Try above maximum.",
    )

    outcome = PreApplyValidator().validate(candidate, hypothesis)

    assert outcome.allowed is False
    assert "above maximum" in outcome.reason


def test_pre_apply_validator_rejects_non_integer_for_integer_type() -> None:
    candidate = build_candidate("service.directive.worker_processes")
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key=candidate.parameter_key,
        parameter_name=candidate.parameter_name,
        domain=candidate.domain,
        tuning_layer=candidate.tuning_layer,
        proposed_value="abc",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=candidate.apply_mode,
        rationale="Try non-integer.",
    )

    outcome = PreApplyValidator().validate(candidate, hypothesis)

    assert outcome.allowed is False
    assert "not an integer" in outcome.reason


def test_pre_apply_validator_accepts_valid_value() -> None:
    candidate = build_candidate("service.directive.worker_processes")
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key=candidate.parameter_key,
        parameter_name=candidate.parameter_name,
        domain=candidate.domain,
        tuning_layer=candidate.tuning_layer,
        proposed_value="56",
        source=CandidateSource.SERVICE_DIRECTIVE,
        apply_mode=candidate.apply_mode,
        rationale="Valid mid-range value.",
    )

    outcome = PreApplyValidator().validate(candidate, hypothesis)

    assert outcome.allowed is True
