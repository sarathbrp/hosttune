from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriageRecommendation:
    rule_id: str
    parameter_key: str
    proposed_value: str
    reason: str


@dataclass(frozen=True)
class TriggeredRule:
    rule_id: str
    section: str
    outcome: str
    detail: str


@dataclass(frozen=True)
class TriageResult:
    autofix_action: TriageRecommendation | None
    recommended_action: TriageRecommendation | None
    alternate_recommendations: tuple[TriageRecommendation, ...]
    safe_candidate_subset: tuple[str, ...]
    suppressed_candidates: tuple[str, ...]
    triggered_rules: tuple[TriggeredRule, ...]
    non_triggered_summary: str
    reboot_required_flags: tuple[str, ...]
    escalation_reason: str
