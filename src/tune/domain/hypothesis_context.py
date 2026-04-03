from __future__ import annotations

from dataclasses import dataclass

from tune.domain.hypothesis_models import CandidateParameter, HypothesisRecord, TunePhase
from tune.domain.tune_context import TuneContext


@dataclass(frozen=True)
class HypothesisContext:
    """Inputs for hypothesis generation; raw host data stays on `tune_context` (preflight/snapshot).

    The prompt builder derives curated digests from this context plus catalog rows (`candidates`).
    """

    tune_context: TuneContext
    phase: TunePhase
    iteration_number: int
    # Phase-filtered selectable rows (active catalog, or deferred-only in reboot_batch).
    candidates: tuple[CandidateParameter, ...]
    # Reboot-deferred catalog rows for prompt visibility; selectable only in reboot_batch.
    deferred_candidates: tuple[CandidateParameter, ...]
    history: tuple[HypothesisRecord, ...]
    active_parameter_keys: tuple[str, ...]
    best_parameter_values: tuple[tuple[str, str], ...]
    # Truncated telemetry digest from the last benchmarked iteration (built in tune_engine).
    last_benchmark_runtime_telemetry_digest: str = ""
