from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from onboard.domain.models import ApplyMode, DirectiveValueType, PriorityTier
from tune.domain.tuning_layer import TuningLayer


class TunePhase(StrEnum):
    """Tune loop phases. See phase controller for TuningLayer usage."""

    KNOWLEDGE_DRIVEN = "knowledge_driven"
    WIDE_SWEEP = "wide_sweep"
    DOMAIN_FOCUS = "domain_focus"
    INTERACTION = "interaction"
    BOUNDARY_PUSH = "boundary_push"
    EXPLOIT = "exploit"
    REBOOT_BATCH = "reboot_batch"
    # Unified resolver phases (coexist with legacy for backward compat).
    RESOLVE = "resolve"
    OPTIMIZE = "optimize"


class CandidateSource(StrEnum):
    SERVICE_DIRECTIVE = "service_directive"
    SERVICE_SYSCTL = "service_sysctl"
    PLATFORM_CAPABILITY = "platform_capability"
    RUNTIME_PRLIMIT = "runtime_prlimit"
    SYSTEMD_UNIT_LIMIT = "systemd_unit_limit"
    SYSTEMD_CGROUP_CONTROL = "systemd_cgroup_control"
    HOST_NIC_QUEUE = "host_nic_queue"  # ethtool -L combined N from host profile
    HOST_CPU_GOVERNOR = "host_cpu_governor"  # cpupower frequency-set -g from host profile
    HOST_SYSCTL = "host_sysctl"  # host-level sysctl from host profile


class CandidateAvailability(StrEnum):
    """Active = normal phases; deferred = reboot_batch when policy allows."""

    ACTIVE = "active"
    DEFERRED = "deferred"


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED_VALIDATION = "failed_validation"
    BENCHMARKED = "benchmarked"
    ACCEPTED = "accepted"
    PROMISING = "promising"
    INCONCLUSIVE = "inconclusive"
    ROLLED_BACK = "rolled_back"
    REJECTED_PRE_APPLY = "rejected_pre_apply"


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
    artifact_path: str | None = None


@dataclass(frozen=True)
class CandidateParameter:
    parameter_key: str
    domain: str
    tuning_layer: TuningLayer
    parameter_name: str
    source: CandidateSource
    value_type: DirectiveValueType
    apply_mode: ApplyMode
    priority_tier: PriorityTier
    allowed_values: tuple[str, ...]
    forbidden_values: tuple[str, ...]
    min_value: int | None
    max_value: int | None
    rationale_hint: str
    current_value: str | None = None
    current_value_source: str = "unknown"
    availability: CandidateAvailability = CandidateAvailability.ACTIVE


@dataclass(frozen=True)
class TuningHypothesis:
    phase: TunePhase
    parameter_key: str
    parameter_name: str
    domain: str
    tuning_layer: TuningLayer
    proposed_value: str
    source: CandidateSource
    apply_mode: ApplyMode
    rationale: str
    model_usage: ModelUsage | None = None
    expected_benchmark_impact: str | None = None
    rollback_plan: str | None = None


@dataclass(frozen=True)
class HypothesisRecord:
    iteration_number: int
    phase: TunePhase
    hypothesis: TuningHypothesis
    status: HypothesisStatus
    evaluation_summary: str | None = None
