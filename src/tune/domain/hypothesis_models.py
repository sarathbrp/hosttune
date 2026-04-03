from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from onboard.domain.models import ApplyMode, DirectiveValueType


class TunePhase(StrEnum):
    WIDE_SWEEP = "wide_sweep"
    DOMAIN_FOCUS = "domain_focus"
    INTERACTION = "interaction"
    BOUNDARY_PUSH = "boundary_push"
    EXPLOIT = "exploit"
    REBOOT_BATCH = "reboot_batch"


class CandidateSource(StrEnum):
    SERVICE_DIRECTIVE = "service_directive"
    SERVICE_SYSCTL = "service_sysctl"
    PLATFORM_CAPABILITY = "platform_capability"


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED_VALIDATION = "failed_validation"
    BENCHMARKED = "benchmarked"
    ACCEPTED = "accepted"
    INCONCLUSIVE = "inconclusive"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class ModelUsage:
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ModelCompletion:
    content: str
    usage: ModelUsage | None = None


@dataclass(frozen=True)
class CandidateParameter:
    parameter_key: str
    domain: str
    parameter_name: str
    source: CandidateSource
    value_type: DirectiveValueType
    apply_mode: ApplyMode
    allowed_values: tuple[str, ...]
    min_value: int | None
    max_value: int | None
    rationale_hint: str


@dataclass(frozen=True)
class TuningHypothesis:
    phase: TunePhase
    parameter_key: str
    parameter_name: str
    domain: str
    proposed_value: str
    source: CandidateSource
    apply_mode: ApplyMode
    rationale: str
    model_usage: ModelUsage | None = None


@dataclass(frozen=True)
class HypothesisRecord:
    iteration_number: int
    phase: TunePhase
    hypothesis: TuningHypothesis
    status: HypothesisStatus
    evaluation_summary: str | None = None
