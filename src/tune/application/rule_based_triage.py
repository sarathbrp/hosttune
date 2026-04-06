from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import CandidateParameter
from tune.domain.triage_models import TriageRecommendation, TriageResult, TriggeredRule


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return int(text)
    return None


@dataclass(frozen=True)
class TriageRuleset:
    sections: dict[str, tuple[dict[str, Any], ...]]


@dataclass
class TriageRulesLoader:
    def load(self, path: Path) -> TriageRuleset:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("triage rules must be a mapping")
        sections: dict[str, tuple[dict[str, Any], ...]] = {}
        for section, value in raw.items():
            if not isinstance(section, str):
                raise ValueError("triage rule section names must be strings")
            if not isinstance(value, list):
                raise ValueError(f"triage rules section {section!r} must be a list")
            parsed_rules: list[dict[str, Any]] = []
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    raise ValueError(f"triage rule {section}[{index}] must be a mapping")
                parsed_rules.append(item)
            sections[section] = tuple(parsed_rules)
        return TriageRuleset(sections=sections)


@dataclass
class RuleBasedTriage:
    ruleset: TriageRuleset

    def evaluate(self, context: HypothesisContext) -> TriageResult:
        candidates_by_key = {candidate.parameter_key: candidate for candidate in context.candidates}
        triggered: list[TriggeredRule] = []
        autofix: TriageRecommendation | None = None
        recommendations: list[TriageRecommendation] = []

        for section, rules in self.ruleset.sections.items():
            for rule in rules:
                if rule.get("enabled", True) is False:
                    continue
                if rule.get("only_if_no_action", False) and (
                    autofix is not None or recommendations
                ):
                    continue
                outcome = self._evaluate_rule(rule, section, context, candidates_by_key)
                if outcome is None:
                    continue
                triggered.append(outcome[0])
                if outcome[1] is None:
                    continue
                action = str(rule.get("action", "recommend"))
                if action == "autofix" and autofix is None:
                    autofix = outcome[1]
                elif action != "autofix":
                    recommendations.append(outcome[1])

        recommendation = recommendations[0] if recommendations else None
        alternate_recommendations = tuple(recommendations[1:])

        suppressed = tuple(sorted(self._suppressed_candidates(context, autofix, recommendation)))
        safe_subset = tuple(
            candidate.parameter_key
            for candidate in context.candidates
            if candidate.parameter_key not in suppressed
        )
        reboot_flags = tuple(
            candidate.parameter_key
            for candidate in context.deferred_candidates
            if candidate.apply_mode.value == "reboot"
        )
        escalation_reason = (
            f"triage_autofix={autofix.parameter_key}"
            if autofix is not None
            else f"triage_recommended={recommendation.parameter_key}"
            if recommendation is not None
            else "no deterministic quick fix matched; escalate to LLM for full correlation"
        )
        return TriageResult(
            autofix_action=autofix,
            recommended_action=recommendation,
            alternate_recommendations=alternate_recommendations,
            safe_candidate_subset=safe_subset,
            suppressed_candidates=suppressed,
            triggered_rules=tuple(triggered),
            non_triggered_summary=(
                f"triggered={len(triggered)} of "
                f"{sum(len(rules) for rules in self.ruleset.sections.values())} configured rules"
            ),
            reboot_required_flags=reboot_flags,
            escalation_reason=escalation_reason,
        )

    def _evaluate_rule(
        self,
        rule: dict[str, Any],
        section: str,
        context: HypothesisContext,
        candidates_by_key: dict[str, CandidateParameter],
    ) -> tuple[TriggeredRule, TriageRecommendation | None] | None:
        kind = rule.get("kind")
        if not isinstance(kind, str):
            return None
        rule_id = str(rule.get("id", f"{section}:{kind}"))
        if not self._matches_common_filters(rule, context):
            return None
        handler = {
            "host_profile_preferred_governor": self._rule_host_profile_preferred_governor,
            "service_current_not": self._rule_service_current_not,
            "align_worker_rlimit_to_unit_limit": self._rule_align_worker_rlimit_to_unit_limit,
            "queue_scale_up": self._rule_queue_scale_up,
            "candidate_floor": self._rule_candidate_floor,
            "candidate_ceiling": self._rule_candidate_ceiling,
            "candidate_scale_outlier": self._rule_candidate_scale_outlier,
            "host_fact_signal": self._rule_host_fact_signal,
            "environment_blocker": self._rule_environment_blocker,
        }.get(kind)
        if handler is None:
            return None
        outcome = handler(rule_id, section, rule, context, candidates_by_key)
        if outcome is None or outcome[1] is None:
            return outcome
        if str(rule.get("action", "recommend")) == "autofix":
            candidate = candidates_by_key.get(outcome[1].parameter_key)
            if candidate is None or not self.can_autofix(candidate, outcome[1].proposed_value):
                return None
        return outcome

    def _matches_common_filters(
        self,
        rule: dict[str, Any],
        context: HypothesisContext,
    ) -> bool:
        service_name = rule.get("service_name")
        if (
            isinstance(service_name, str)
            and service_name != context.tune_context.onboard.service_name
        ):
            return False
        min_cores = rule.get("min_logical_cores")
        if (
            isinstance(min_cores, int)
            and context.tune_context.preflight.cpu.logical_cores < min_cores
        ):
            return False
        min_numa_nodes = rule.get("min_numa_nodes")
        if (
            isinstance(min_numa_nodes, int)
            and context.tune_context.preflight.cpu.numa_nodes < min_numa_nodes
        ):
            return False
        min_ram_gb = rule.get("min_ram_gb")
        if isinstance(min_ram_gb, int):
            total_gb = context.tune_context.preflight.memory.total_memory_kib / (1024 * 1024)
            if total_gb < min_ram_gb:
                return False
        return True

    def _rule_host_profile_preferred_governor(
        self,
        rule_id: str,
        section: str,
        rule: dict[str, Any],
        context: HypothesisContext,
        candidates_by_key: dict[str, CandidateParameter],
    ) -> tuple[TriggeredRule, TriageRecommendation | None] | None:
        candidate_key = str(rule.get("candidate_key", ""))
        candidate = candidates_by_key.get(candidate_key)
        if candidate is None or context.tune_context.host_profile is None:
            return None
        preferred = context.tune_context.host_profile.tunable_surface.cpu_governor
        if preferred is None:
            return None
        proposed_value = preferred.preferred_governor
        if candidate.current_value == proposed_value:
            return None
        trigger = TriggeredRule(
            rule_id=rule_id,
            section=section,
            outcome="recommend",
            detail=(
                f"{candidate_key} should prefer host-profile governor {proposed_value} "
                f"on {context.tune_context.preflight.cpu.logical_cores} logical cores"
            ),
        )
        recommendation = TriageRecommendation(
            rule_id=rule_id,
            parameter_key=candidate_key,
            proposed_value=proposed_value,
            reason=trigger.detail,
        )
        return trigger, recommendation

    def _rule_service_current_not(
        self,
        rule_id: str,
        section: str,
        rule: dict[str, Any],
        context: HypothesisContext,
        candidates_by_key: dict[str, CandidateParameter],
    ) -> tuple[TriggeredRule, TriageRecommendation | None] | None:
        candidate_key = str(rule.get("candidate_key", ""))
        candidate = candidates_by_key.get(candidate_key)
        proposed_value = str(rule.get("proposed_value", ""))
        if candidate is None or proposed_value == "" or candidate.current_value == proposed_value:
            return None
        only_when_current_in = rule.get("only_when_current_in")
        if isinstance(only_when_current_in, list):
            allowed_currents = {str(value) for value in only_when_current_in}
            if candidate.current_value not in allowed_currents:
                return None
        required_interference = rule.get("interference_source_contains")
        if isinstance(required_interference, str):
            haystack = " ".join(
                context.tune_context.onboard.service.benchmark_hints.interference_sources
            )
            if required_interference not in haystack:
                return None
        trigger = TriggeredRule(
            rule_id=rule_id,
            section=section,
            outcome="recommend",
            detail=f"{candidate_key} is an obvious service-level quick fix toward {proposed_value}",
        )
        recommendation = TriageRecommendation(
            rule_id=rule_id,
            parameter_key=candidate_key,
            proposed_value=proposed_value,
            reason=trigger.detail,
        )
        return trigger, recommendation

    def _rule_align_worker_rlimit_to_unit_limit(
        self,
        rule_id: str,
        section: str,
        rule: dict[str, Any],
        context: HypothesisContext,
        candidates_by_key: dict[str, CandidateParameter],
    ) -> tuple[TriggeredRule, TriageRecommendation | None] | None:
        candidate_key = str(rule.get("candidate_key", "service.directive.worker_rlimit_nofile"))
        candidate = candidates_by_key.get(candidate_key)
        if candidate is None:
            return None
        unit_limit = next(
            (
                _to_int(limit_candidate.current_value)
                for limit_candidate in context.candidates + context.deferred_candidates
                if limit_candidate.parameter_key == "systemd.unit.limit_nofile"
            ),
            None,
        )
        current_value = _to_int(candidate.current_value)
        if unit_limit is None or current_value is None or current_value >= unit_limit:
            return None
        if candidate.max_value is not None:
            unit_limit = min(unit_limit, candidate.max_value)
        trigger = TriggeredRule(
            rule_id=rule_id,
            section=section,
            outcome="recommend",
            detail=(
                f"{candidate_key} is far below systemd LimitNOFILE; align runtime fd ceiling "
                f"to {unit_limit}"
            ),
        )
        recommendation = TriageRecommendation(
            rule_id=rule_id,
            parameter_key=candidate_key,
            proposed_value=str(unit_limit),
            reason=trigger.detail,
        )
        return trigger, recommendation

    def _rule_queue_scale_up(
        self,
        rule_id: str,
        section: str,
        rule: dict[str, Any],
        context: HypothesisContext,
        candidates_by_key: dict[str, CandidateParameter],
    ) -> tuple[TriggeredRule, TriageRecommendation | None] | None:
        candidate_key = str(rule.get("candidate_key", "network.queue.combined"))
        candidate = candidates_by_key.get(candidate_key)
        if candidate is None:
            return None
        current_value = _to_int(candidate.current_value)
        max_value = candidate.max_value
        if current_value is None or max_value is None or current_value >= max_value:
            return None
        target = min(max_value, context.tune_context.preflight.cpu.logical_cores)
        if target <= current_value:
            return None
        trigger = TriggeredRule(
            rule_id=rule_id,
            section=section,
            outcome="recommend",
            detail=(
                f"{candidate_key} can scale from {current_value} to {target} on this host "
                "without reboot"
            ),
        )
        recommendation = TriageRecommendation(
            rule_id=rule_id,
            parameter_key=candidate_key,
            proposed_value=str(target),
            reason=trigger.detail,
        )
        return trigger, recommendation

    def _rule_candidate_floor(
        self,
        rule_id: str,
        section: str,
        rule: dict[str, Any],
        context: HypothesisContext,
        candidates_by_key: dict[str, CandidateParameter],
    ) -> tuple[TriggeredRule, TriageRecommendation | None] | None:
        _ = context
        candidate_key = str(rule.get("candidate_key", ""))
        candidate = candidates_by_key.get(candidate_key)
        floor_value = rule.get("floor_value")
        if candidate is None or not isinstance(floor_value, int):
            return None
        current_value = _to_int(candidate.current_value)
        if current_value is None or current_value >= floor_value:
            return None
        proposed = floor_value
        if candidate.max_value is not None:
            proposed = min(proposed, candidate.max_value)
        if proposed <= current_value:
            return None
        trigger = TriggeredRule(
            rule_id=rule_id,
            section=section,
            outcome="recommend",
            detail=(
                f"{candidate_key} is below deterministic floor {floor_value} "
                f"for this host shape"
            ),
        )
        recommendation = TriageRecommendation(
            rule_id=rule_id,
            parameter_key=candidate_key,
            proposed_value=str(proposed),
            reason=trigger.detail,
        )
        return trigger, recommendation

    def _rule_candidate_scale_outlier(
        self,
        rule_id: str,
        section: str,
        rule: dict[str, Any],
        context: HypothesisContext,
        candidates_by_key: dict[str, CandidateParameter],
    ) -> tuple[TriggeredRule, TriageRecommendation | None] | None:
        candidate_key = str(rule.get("candidate_key", ""))
        candidate = candidates_by_key.get(candidate_key)
        if candidate is None:
            return None
        current_value = _to_int(candidate.current_value)
        min_recommended = rule.get("min_recommended_value")
        if not isinstance(min_recommended, int) or current_value is None:
            return None
        if current_value >= min_recommended:
            return None
        preferred_value = str(rule.get("preferred_value", "scale-aware value"))
        detail = str(
            rule.get(
                "detail",
                f"{candidate_key} appears undersized for a "
                f"{context.tune_context.preflight.cpu.logical_cores}-core host; "
                f"prefer {preferred_value}",
            )
        )
        return (
            TriggeredRule(
                rule_id=rule_id,
                section=section,
                outcome="signal",
                detail=detail,
            ),
            None,
        )

    def _rule_candidate_ceiling(
        self,
        rule_id: str,
        section: str,
        rule: dict[str, Any],
        context: HypothesisContext,
        candidates_by_key: dict[str, CandidateParameter],
    ) -> tuple[TriggeredRule, TriageRecommendation | None] | None:
        _ = context
        candidate_key = str(rule.get("candidate_key", ""))
        candidate = candidates_by_key.get(candidate_key)
        ceiling_value = rule.get("ceiling_value")
        if candidate is None or not isinstance(ceiling_value, int):
            return None
        current_value = _to_int(candidate.current_value)
        if current_value is None or current_value <= ceiling_value:
            return None
        proposed = ceiling_value
        if candidate.min_value is not None:
            proposed = max(proposed, candidate.min_value)
        if proposed >= current_value:
            return None
        trigger = TriggeredRule(
            rule_id=rule_id,
            section=section,
            outcome="recommend",
            detail=(
                f"{candidate_key} is above deterministic ceiling {ceiling_value} "
                f"for this host shape"
            ),
        )
        recommendation = TriageRecommendation(
            rule_id=rule_id,
            parameter_key=candidate_key,
            proposed_value=str(proposed),
            reason=trigger.detail,
        )
        return trigger, recommendation

    def _rule_host_fact_signal(
        self,
        rule_id: str,
        section: str,
        rule: dict[str, Any],
        context: HypothesisContext,
        candidates_by_key: dict[str, CandidateParameter],
    ) -> tuple[TriggeredRule, TriageRecommendation | None] | None:
        _ = context
        _ = candidates_by_key
        detail = rule.get("detail")
        if not isinstance(detail, str) or detail == "":
            return None
        return (
            TriggeredRule(
                rule_id=rule_id,
                section=section,
                outcome="signal",
                detail=detail,
            ),
            None,
        )

    def _rule_environment_blocker(
        self,
        rule_id: str,
        section: str,
        rule: dict[str, Any],
        context: HypothesisContext,
        candidates_by_key: dict[str, CandidateParameter],
    ) -> tuple[TriggeredRule, TriageRecommendation | None] | None:
        """Detect external throttles by probing system state via telemetry digest."""
        _ = candidates_by_key
        detail = rule.get("detail")
        if not isinstance(detail, str) or detail == "":
            return None
        telemetry = context.last_benchmark_runtime_telemetry_digest
        probe_command = rule.get("probe_command")
        signal_key = rule.get("signal_key", "")
        # Check telemetry digest for indicators when no live probe is available.
        # These keywords are produced by the telemetry collector (ss -s, softnet_stat).
        keyword_indicators: dict[str, tuple[str, ...]] = {
            "conntrack_pressure": ("conntrack", "nf_conntrack"),
            "firewall_connlimit": ("connlimit", "iptables", "nftables"),
            "tc_shaping": ("tc qdisc", "tbf", "htb", "netem"),
            "cgroup_cpu_throttle": ("nr_throttled", "cpu.stat"),
        }
        indicators = keyword_indicators.get(signal_key, ())
        if indicators and telemetry:
            telemetry_lower = telemetry.lower()
            if any(indicator in telemetry_lower for indicator in indicators):
                return (
                    TriggeredRule(
                        rule_id=rule_id,
                        section=section,
                        outcome="signal",
                        detail=f"[env_blocker] {detail}",
                    ),
                    None,
                )
        # If a probe_command is defined, the actual probe runs at the host level
        # via the preflight executor. The triage layer only detects based on
        # available telemetry; the probe_command is documented for operators
        # to run manually or for future live-probe integration.
        if not telemetry and probe_command:
            return (
                TriggeredRule(
                    rule_id=rule_id,
                    section=section,
                    outcome="signal",
                    detail=(
                        f"[env_blocker] {detail} "
                        f"(no telemetry available; run manually: {probe_command})"
                    ),
                ),
                None,
            )
        return None

    def _suppressed_candidates(
        self,
        context: HypothesisContext,
        autofix: TriageRecommendation | None,
        recommendation: TriageRecommendation | None,
    ) -> set[str]:
        suppressed: set[str] = set()
        current = (
            autofix.parameter_key
            if autofix is not None
            else recommendation.parameter_key
            if recommendation is not None
            else None
        )
        if current is None:
            return suppressed
        if current.startswith("service.directive.worker_rlimit_nofile"):
            suppressed.add("runtime.prlimit.nofile_soft")
        if current == "platform.cpu_governor.scaling_governor":
            suppressed.add("network.queue.combined")
        if current == "network.queue.combined":
            suppressed.add("platform.cpu_governor.scaling_governor")
        for candidate in context.deferred_candidates:
            if candidate.apply_mode.value == "reboot":
                suppressed.add(candidate.parameter_key)
        return suppressed

    def can_autofix(
        self,
        candidate: CandidateParameter,
        proposed_value: str,
    ) -> bool:
        if candidate.value_type.value != "enum":
            return False
        if candidate.allowed_values and proposed_value not in candidate.allowed_values:
            return False
        return candidate.current_value != proposed_value
