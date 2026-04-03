from dataclasses import replace

from preflight.domain.models import EngagementPolicy
from tune.application.phase_controller import PhaseController
from tune.domain.hypothesis_models import CandidateSource, HypothesisRecord, HypothesisStatus, TunePhase, TuningHypothesis
from tune.domain.tune_state import TuneState

from tests.tune.test_candidate_catalog_builder import build_tune_context
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from onboard.domain.models import ApplyMode


def test_tune_state_allocates_small_budget_cleanly() -> None:
    state = TuneState.initialize(1)

    assert state.remaining_budget[TunePhase.WIDE_SWEEP] == 1
    assert sum(state.remaining_budget.values()) == 1


def test_phase_controller_advances_after_wide_sweep_candidates_exhausted() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context)
    state = TuneState.initialize(10)
    state.history = [
        HypothesisRecord(
            iteration_number=index + 1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=TuningHypothesis(
                phase=TunePhase.WIDE_SWEEP,
                parameter_key=candidate.parameter_key,
                parameter_name=candidate.parameter_name,
                domain=candidate.domain,
                proposed_value="1",
                source=CandidateSource.SERVICE_SYSCTL,
                apply_mode=ApplyMode.RELOAD,
                rationale="test",
            ),
            status=HypothesisStatus.REJECTED,
        )
        for index, candidate in enumerate(candidates)
    ]

    phase = PhaseController().determine_phase(state, candidates)

    assert phase is TunePhase.DOMAIN_FOCUS
