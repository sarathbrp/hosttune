"""Unified dependency-graph resolver for pre-loop parameter application.

Replaces the 3-step pre-loop (recipe lookup + KB batch + autofix batch)
with a single bottom-up pass through a layered dependency graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from tune.application.rule_based_triage import RuleBasedTriage
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import (
    CandidateParameter,
    TunePhase,
    TuningHypothesis,
)
from tune.domain.tune_context import TuneContext
from tune.domain.tune_state import TuneState

_log = logging.getLogger(__name__)


class LayerStatus(StrEnum):
    OK = "ok"
    FIXED = "fixed"
    LLM_DEFERRED = "llm_deferred"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class ResolvedParameter:
    parameter_key: str
    layer_name: str
    layer_order: int
    proposed_value: str
    resolution_source: str
    dependency_note: str


@dataclass
class UnifiedResolver:
    graph_path: Path
    triage: RuleBasedTriage | None = None
    logger: ExecutionLogger = NullExecutionLogger()

    def __post_init__(self) -> None:
        raw = yaml.safe_load(self.graph_path.read_text(encoding="utf-8"))
        self._layers: list[tuple[str, dict[str, Any]]] = sorted(
            (raw.get("layers") or {}).items(),
            key=lambda x: x[0],
        )

    def resolve(
        self,
        context: TuneContext,
        state: TuneState,
        all_candidates: tuple[CandidateParameter, ...],
        confidence_scores: dict[str, tuple[int, int, float]],
        recipe_fix_sequence: list[dict[str, str]] | None,
        prior_blocked_pairs: list[tuple[str, str]],
    ) -> tuple[list[tuple[str, list[TuningHypothesis]]], dict[str, str]]:
        """Walk the dependency graph bottom-up, resolve values, return hypotheses.

        Returns:
            layer_hypotheses: [(layer_name, hypotheses)] grouped by layer
            layer_statuses: {layer_name: LayerStatus.value} for LLM prompt
        """
        catalog_index = {c.parameter_key: c for c in all_candidates}
        recipe_values = self._build_recipe_values(recipe_fix_sequence)
        kb_best = self._build_kb_best_values(context)
        blocked_set = set(prior_blocked_pairs)
        preflight_facts = self._build_preflight_facts(context)
        triage_autofixes = self._build_triage_autofixes(
            context, state, all_candidates, prior_blocked_pairs
        )

        layer_hypotheses: list[tuple[str, list[TuningHypothesis]]] = []
        all_resolved: list[TuningHypothesis] = []
        layer_statuses: dict[str, str] = {}
        total_resolved = 0

        for layer_name, layer_spec in self._layers:
            params = layer_spec.get("parameters") or {}
            layer_fixed = False
            layer_deferred = False
            layer_hyps: list[TuningHypothesis] = []
            decision_rows: list[tuple[str, str, str, str, str]] = []

            for param_key, param_spec in params.items():
                candidate = catalog_index.get(param_key)
                if candidate is None:
                    continue
                if param_key in state.active_changes:
                    continue

                proposed = self._resolve_value(
                    param_key=param_key,
                    param_spec=param_spec,
                    candidate=candidate,
                    recipe_values=recipe_values,
                    kb_best=kb_best,
                    confidence_scores=confidence_scores,
                    triage_autofixes=triage_autofixes,
                    preflight_facts=preflight_facts,
                    blocked_set=blocked_set,
                )
                if proposed is None:
                    target_hint = str(
                        param_spec.get("target")
                        or param_spec.get("floor")
                        or param_spec.get("ceiling")
                        or "-"
                    )
                    decision_rows.append((
                        param_key,
                        str(candidate.current_value),
                        target_hint,
                        "graph/autofix",
                        "SKIP (at target)",
                    ))
                    continue

                if proposed.resolution_source == "llm":
                    layer_deferred = True
                    decision_rows.append((
                        param_key,
                        str(candidate.current_value),
                        "-",
                        "llm",
                        "DEFER",
                    ))
                    continue

                if not self._check_constraint(
                    param_spec, proposed.proposed_value, all_resolved
                ):
                    decision_rows.append((
                        param_key,
                        str(candidate.current_value),
                        proposed.proposed_value,
                        proposed.resolution_source,
                        "SKIP (constraint)",
                    ))
                    continue

                layer_fixed = True
                hyp = TuningHypothesis(
                    phase=TunePhase.RESOLVE,
                    parameter_key=param_key,
                    parameter_name=candidate.parameter_name,
                    domain=candidate.domain,
                    tuning_layer=candidate.tuning_layer,
                    proposed_value=proposed.proposed_value,
                    source=candidate.source,
                    apply_mode=candidate.apply_mode,
                    rationale=(
                        f"Resolver L{layer_name[:1]}: "
                        f"{proposed.resolution_source} "
                        f"({proposed.dependency_note})"
                    ),
                )
                layer_hyps.append(hyp)
                all_resolved.append(hyp)
                decision_rows.append((
                    param_key,
                    str(candidate.current_value),
                    proposed.proposed_value,
                    proposed.resolution_source,
                    "APPLY",
                ))

            if decision_rows:
                from tune.application.format_table import resolver_layer_table
                self.logger.stage_detail("tune", resolver_layer_table(layer_name, decision_rows))

            if layer_hyps:
                layer_hypotheses.append((layer_name, layer_hyps))
                total_resolved += len(layer_hyps)

            if layer_deferred:
                layer_statuses[layer_name] = LayerStatus.LLM_DEFERRED.value
            elif layer_fixed:
                layer_statuses[layer_name] = LayerStatus.FIXED.value
            else:
                layer_statuses[layer_name] = LayerStatus.OK.value

        self.logger.stage_detail(
            "tune",
            f"Resolver: {total_resolved} params across "
            f"{len(layer_hypotheses)} layers. "
            f"Statuses: {layer_statuses}",
        )
        return layer_hypotheses, layer_statuses

    def _resolve_value(
        self,
        *,
        param_key: str,
        param_spec: dict[str, Any],
        candidate: CandidateParameter,
        recipe_values: dict[str, str],
        kb_best: dict[str, str],
        confidence_scores: dict[str, tuple[int, int, float]],
        triage_autofixes: dict[str, str],
        preflight_facts: dict[str, int],
        blocked_set: set[tuple[str, str]],
    ) -> ResolvedParameter | None:
        """Try each source in resolve_from order. Return first match."""
        resolve_from = param_spec.get("resolve_from", [])
        dep_note = param_spec.get("dependency", "")

        for source in resolve_from:
            value = self._try_source(
                source=source,
                param_key=param_key,
                param_spec=param_spec,
                candidate=candidate,
                recipe_values=recipe_values,
                kb_best=kb_best,
                confidence_scores=confidence_scores,
                triage_autofixes=triage_autofixes,
                preflight_facts=preflight_facts,
            )
            if value is None:
                continue
            if value == candidate.current_value:
                return None  # Already at target.
            if (param_key, value) in blocked_set:
                continue
            return ResolvedParameter(
                parameter_key=param_key,
                layer_name="",
                layer_order=0,
                proposed_value=value,
                resolution_source=source,
                dependency_note=dep_note,
            )
        return None

    def _try_source(
        self,
        *,
        source: str,
        param_key: str,
        param_spec: dict[str, Any],
        candidate: CandidateParameter,
        recipe_values: dict[str, str],
        kb_best: dict[str, str],
        confidence_scores: dict[str, tuple[int, int, float]],
        triage_autofixes: dict[str, str],
        preflight_facts: dict[str, int],
    ) -> str | None:
        if source == "recipe":
            value = recipe_values.get(param_key)
            if value is not None and not self._recipe_value_valid(param_spec, value):
                return None  # Recipe value violates graph constraints — fall through to kb/graph.
            return value
        if source == "kb":
            conf = confidence_scores.get(param_key)
            if conf is not None and conf[2] >= 0.80:
                return kb_best.get(param_key)
            return None
        if source == "graph":
            return self._graph_value(param_spec, candidate, preflight_facts)
        if source == "autofix":
            return triage_autofixes.get(param_key)
        if source == "llm":
            return "__llm_deferred__"
        return None

    def _recipe_value_valid(self, param_spec: dict[str, Any], value: str) -> bool:
        """Return False if the recipe value violates the graph's floor/ceiling constraints.

        A stored recipe may contain pathological values (e.g. somaxconn=1 from a
        broken run). Reject any recipe value that falls outside the graph bounds so
        the resolver falls through to the graph source instead.
        """
        try:
            v = int(value)
        except (ValueError, TypeError):
            return True  # Non-integer values (enum/string) — trust the recipe.
        floor = param_spec.get("floor")
        if floor is not None and v < floor:
            return False
        ceiling = param_spec.get("ceiling")
        if ceiling is not None and v > ceiling:
            return False
        return True

    def _graph_value(
        self,
        param_spec: dict[str, Any],
        candidate: CandidateParameter,
        preflight_facts: dict[str, int],
    ) -> str | None:
        # Target value (exact).
        target = param_spec.get("target")
        if target is not None:
            return str(target)
        # Floor expression (dynamic).
        floor_expr = param_spec.get("floor_expr")
        if floor_expr is not None:
            floor_val = self._eval_expr(floor_expr, preflight_facts)
            if floor_val is not None:
                current_int = self._to_int(candidate.current_value)
                if current_int is not None and current_int >= floor_val:
                    return None  # Already above floor.
                return str(floor_val)
        # Static floor.
        floor = param_spec.get("floor")
        if floor is not None:
            current_int = self._to_int(candidate.current_value)
            if current_int is not None and current_int >= floor:
                return None
            return str(floor)
        # Ceiling.
        ceiling = param_spec.get("ceiling")
        if ceiling is not None:
            current_int = self._to_int(candidate.current_value)
            if current_int is not None and current_int <= ceiling:
                return None
            return str(ceiling)
        return None

    def _check_constraint(
        self,
        param_spec: dict[str, Any],
        proposed_value: str,
        resolved_so_far: list[TuningHypothesis],
    ) -> bool:
        """Validate cross-parameter constraints."""
        constraint = param_spec.get("constraint")
        if constraint is None:
            return True
        # Parse "MUST be <= systemd.unit.limit_nofile" style constraints.
        resolved_values = {
            h.parameter_key: h.proposed_value for h in resolved_so_far
        }
        if "<=" in constraint:
            ref_key = constraint.split("<=")[-1].strip()
            ref_value = resolved_values.get(ref_key)
            if ref_value is None:
                return True  # Can't check — assume OK.
            proposed_int = self._to_int(proposed_value)
            ref_int = self._to_int(ref_value)
            if proposed_int is not None and ref_int is not None:
                return proposed_int <= ref_int
        return True

    def _build_recipe_values(
        self, recipe_fix_sequence: list[dict[str, str]] | None
    ) -> dict[str, str]:
        if not recipe_fix_sequence:
            return {}
        return {
            step["parameter_key"]: step["value"]
            for step in recipe_fix_sequence
            if "parameter_key" in step and "value" in step
        }

    def _build_kb_best_values(
        self, context: TuneContext
    ) -> dict[str, str]:
        kb = getattr(context, "knowledge_base", None)
        artifacts = context.artifacts
        if kb is None or artifacts is None:
            return {}
        config = kb.get_prior_best_config(
            service_name=context.onboard.service_name,
            cpu_logical_cores=context.preflight.cpu.logical_cores,
            numa_nodes=context.preflight.cpu.numa_nodes,
            platform_summary=context.preflight.platform_summary,
            nic_driver=context.preflight.network.driver_name,
            exclude_run_id=artifacts.session_id,
        )
        return config or {}

    def _build_preflight_facts(
        self, context: TuneContext
    ) -> dict[str, int]:
        cpu = context.preflight.cpu
        net = context.preflight.network
        return {
            "logical_cores": cpu.logical_cores,
            "numa_nodes": cpu.numa_nodes,
            "max_queues": net.combined_queues or cpu.logical_cores,
        }

    def _build_triage_autofixes(
        self,
        context: TuneContext,
        state: TuneState,
        all_candidates: tuple[CandidateParameter, ...],
        prior_blocked_pairs: list[tuple[str, str]],
    ) -> dict[str, str]:
        """Collect all triage autofixes as a {param_key: value} dict."""
        if self.triage is None:
            return {}
        hyp_context = HypothesisContext(
            tune_context=context,
            phase=state.current_phase,
            iteration_number=0,
            candidates=all_candidates,
            deferred_candidates=(),
            history=tuple(state.history),
            active_parameter_keys=tuple(sorted(state.active_changes)),
            best_parameter_values=(),
            prior_blocked_pairs=tuple(prior_blocked_pairs),
            confidence_scores=(),
        )
        autofixes = self.triage.collect_all_autofixes(hyp_context)
        return {key: value for key, value, _reason in autofixes}

    def _eval_expr(
        self, expr: str, facts: dict[str, int]
    ) -> int | None:
        """Evaluate simple expressions like 'min(max_queues, logical_cores)'."""
        try:
            if expr.startswith("min("):
                args = expr[4:-1].split(",")
                values = [facts.get(a.strip(), 0) for a in args]
                return min(values) if values else None
            if expr.startswith("max("):
                args = expr[4:-1].split(",")
                values = [facts.get(a.strip(), 0) for a in args]
                return max(values) if values else None
            if "//" in expr:
                parts = expr.split("//")
                left = facts.get(parts[0].strip(), 0)
                right = int(parts[1].strip())
                return left // right if right else None
            # Direct fact reference.
            return facts.get(expr.strip())
        except (ValueError, KeyError, IndexError):
            return None

    @staticmethod
    def _to_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None
