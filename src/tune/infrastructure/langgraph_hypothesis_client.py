from __future__ import annotations

import json
import logging
import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import ModelCompletion, ModelUsage
from tune.infrastructure.model_config import ModelEndpointConfig

_log = logging.getLogger(__name__)


class HypothesisGraphState(TypedDict):
    # Built from HypothesisContext before invoking the graph
    service_prompt: str
    rhel_prompt: str
    full_prompt: str
    context: HypothesisContext
    # Expert recommendations appended in parallel (reducer: list concatenation)
    expert_recommendations: Annotated[list[str], operator.add]
    # Final planner output
    response: str
    usage: ModelUsage | None


@dataclass
class LangGraphHypothesisClient:
    config: ModelEndpointConfig
    prompt_builder: object  # HypothesisPromptBuilder — avoids circular import at module level
    logger: ExecutionLogger = field(default_factory=NullExecutionLogger)
    _graph: object | None = field(default=None, init=False, repr=False)

    def complete(self, context: HypothesisContext) -> ModelCompletion:
        from tune.application.hypothesis_prompt_layer import (
            format_rhel_expert_prompt,
            format_service_expert_prompt,
        )

        service_prompt = format_service_expert_prompt(context)
        rhel_prompt = format_rhel_expert_prompt(context)
        full_prompt = self.prompt_builder.build(context)  # type: ignore[union-attr]

        graph = self._get_graph()
        result = graph.invoke(
            {
                "service_prompt": service_prompt,
                "rhel_prompt": rhel_prompt,
                "full_prompt": full_prompt,
                "context": context,
                "expert_recommendations": [],
                "response": "",
                "usage": None,
            }
        )
        response = result.get("response")
        if not isinstance(response, str) or response == "":
            msg = "LangGraph hypothesis client returned an empty response."
            raise ValueError(msg)
        usage = result.get("usage")
        if usage is not None and not isinstance(usage, ModelUsage):
            msg = "LangGraph hypothesis client returned malformed usage metadata."
            raise ValueError(msg)
        return ModelCompletion(content=response, usage=usage)

    # ── Graph construction ────────────────────────────────────────────────────

    def _get_graph(self) -> object:
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> object:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as error:
            msg = "langgraph is required for LangGraphHypothesisClient."
            raise RuntimeError(msg) from error

        graph = StateGraph(HypothesisGraphState)
        graph.add_node("service_agent", self._service_agent_node)
        graph.add_node("rhel_expert", self._rhel_expert_node)
        graph.add_node("debate_planner", self._debate_planner_node)

        graph.add_edge(START, "service_agent")  # fan-out (parallel)
        graph.add_edge(START, "rhel_expert")  # fan-out (parallel)
        graph.add_edge("service_agent", "debate_planner")  # fan-in
        graph.add_edge("rhel_expert", "debate_planner")  # fan-in
        graph.add_edge("debate_planner", END)
        return graph.compile()

    # ── Agent nodes ──────────────────────────────────────────────────────────

    def _service_agent_node(self, state: HypothesisGraphState) -> dict[str, object]:
        """Service configuration expert — reasons over service/runtime candidates."""
        context = state["context"]
        prompt = state["service_prompt"]
        self._log_prompt("service_agent", prompt)
        try:
            content = self._call_llm(
                caller="service_agent",
                system=(
                    "You are the service configuration expert for HostTune. "
                    "Return strict JSON with keys: "
                    "parameter_key, proposed_value, rationale, confidence."
                ),
                prompt=prompt,
            )
        except Exception as exc:
            content = f"ERROR: service_agent failed: {exc}"
        self._log_response("service_agent", content)
        self._save_agent_artifact(context, "service_agent", prompt, content)
        return {"expert_recommendations": [content]}

    def _rhel_expert_node(self, state: HypothesisGraphState) -> dict[str, object]:
        """RHEL system tuning expert — reasons over kernel/network candidates."""
        context = state["context"]
        prompt = state["rhel_prompt"]
        self._log_prompt("rhel_expert", prompt)
        try:
            content = self._call_llm(
                caller="rhel_expert",
                system=(
                    "You are the RHEL system tuning expert for HostTune. "
                    "Return strict JSON with keys: "
                    "parameter_key, proposed_value, rationale, confidence."
                ),
                prompt=prompt,
            )
        except Exception as exc:
            content = f"ERROR: rhel_expert failed: {exc}"
        self._log_response("rhel_expert", content)
        self._save_agent_artifact(context, "rhel_expert", prompt, content)
        return {"expert_recommendations": [content]}

    def _debate_planner_node(self, state: HypothesisGraphState) -> dict[str, object]:
        """Debate planner — synthesizes expert recommendations into one final hypothesis array."""
        from tune.application.hypothesis_prompt_layer import format_debate_planner_prompt

        context = state["context"]
        recommendations = state["expert_recommendations"]
        if len(recommendations) != 2:
            _log.warning(
                "Debate planner expected 2 expert recommendations, got %d",
                len(recommendations),
            )
        planner_prompt = format_debate_planner_prompt(
            context=context,
            expert_recommendations=recommendations,
            full_prompt=state["full_prompt"],
        )
        self._log_prompt("debate_planner", planner_prompt)
        content, usage = self._call_llm_with_usage(
            caller="debate_planner",
            system=(
                "You are the tuning decision planner for HostTune. "
                "Return a JSON ARRAY of hypothesis objects, each with keys: "
                "parameter_key, proposed_value, rationale. "
                "Include both service and kernel recommendations when they are orthogonal. "
                "Return a single-element array if only one is valid."
            ),
            prompt=planner_prompt,
        )
        self._log_response("debate_planner", content)
        self._save_agent_artifact(context, "debate_planner", planner_prompt, content)
        return {"response": content, "usage": usage}

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log_prompt(self, agent: str, prompt: str) -> None:
        """Log prompt at debug level only (can be very long)."""
        if self.logger.debug_enabled():
            self.logger.stage_detail("tune", f"[{agent}] prompt ({len(prompt)} chars):")
            self.logger.stage_detail("tune", prompt)

    def _log_response(self, agent: str, response: str) -> None:
        """Log response at debug level only (can be very long)."""
        if self.logger.debug_enabled():
            self.logger.stage_detail("tune", f"[{agent}] response: {response}")

    # ── Artifact saving ───────────────────────────────────────────────────────

    def _save_agent_artifact(
        self,
        context: HypothesisContext,
        agent: str,
        prompt: str,
        response: str,
    ) -> None:
        """Save agent prompt+response to hypothesis/ subfolder for later investigation."""
        artifacts = context.tune_context.artifacts
        if artifacts is None:
            return
        iteration = context.iteration_number
        out_dir = artifacts.session_directory / "hypothesis"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"iter{iteration:03d}_{agent}.json"
        try:
            data = {
                "iteration": iteration,
                "phase": context.phase.value,
                "agent": agent,
                "prompt": prompt,
                "response": response,
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            _log.warning("Failed to save hypothesis artifact %s: %s", path, exc)

    # ── LLM call helpers ──────────────────────────────────────────────────────

    def _call_llm(self, *, caller: str, system: str, prompt: str) -> str:
        client = self._build_openai_client()
        completion = client.chat.completions.create(
            model=self.config.model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content
        if not isinstance(content, str):
            msg = f"[{caller}] OpenAI-compatible response did not include string content."
            raise ValueError(msg)
        return content

    def _call_llm_with_usage(
        self, *, caller: str, system: str, prompt: str
    ) -> tuple[str, ModelUsage | None]:
        client = self._build_openai_client()
        completion = client.chat.completions.create(
            model=self.config.model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content
        if not isinstance(content, str):
            msg = f"[{caller}] OpenAI-compatible response did not include string content."
            raise ValueError(msg)
        usage_payload = getattr(completion, "usage", None)
        usage = None
        if usage_payload is not None:
            usage = ModelUsage(
                model_name=self.config.model_name,
                input_tokens=int(getattr(usage_payload, "prompt_tokens", 0)),
                output_tokens=int(getattr(usage_payload, "completion_tokens", 0)),
                total_tokens=int(getattr(usage_payload, "total_tokens", 0)),
            )
        return content, usage

    def _build_openai_client(self) -> object:
        try:
            from openai import OpenAI
        except ImportError as error:
            msg = "openai is required for LangGraphHypothesisClient."
            raise RuntimeError(msg) from error
        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
