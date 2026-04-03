from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParameterImpactScore:
    parameter_key: str
    domain: str
    evaluated_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    inconclusive_count: int = 0
    average_relative_change: float = 0.0
    best_relative_change: float = 0.0
    worst_relative_change: float = 0.0


@dataclass
class DomainImpactScore:
    domain: str
    evaluated_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    inconclusive_count: int = 0
    average_relative_change: float = 0.0
    parameter_keys: tuple[str, ...] = ()


@dataclass
class WorkloadImpactScore:
    workload_name: str
    best_parameter_key: str | None = None
    best_relative_change: float = 0.0
    worst_parameter_key: str | None = None
    worst_relative_change: float = 0.0
    win_count: int = 0
    loss_count: int = 0


@dataclass
class TuneScoreboard:
    parameter_scores: dict[str, ParameterImpactScore] = field(default_factory=dict)
    domain_scores: dict[str, DomainImpactScore] = field(default_factory=dict)
    workload_scores: dict[str, WorkloadImpactScore] = field(default_factory=dict)
