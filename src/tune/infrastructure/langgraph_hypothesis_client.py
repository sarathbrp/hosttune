from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import ModelCompletion, ModelUsage
from tune.infrastructure.model_config import ModelEndpointConfig

_log = logging.getLogger(__name__)


@dataclass
class LangGraphHypothesisClient:
    config: ModelEndpointConfig
    prompt_builder: object
    logger: ExecutionLogger = field(default_factory=NullExecutionLogger)

    def complete(self, context: HypothesisContext) -> ModelCompletion:
        prompt = self.prompt_builder.build(context)  # type: ignore[union-attr]
        self._log_prompt("hybrid_hypothesizer", prompt)
        content, usage = self._call_llm_with_usage(
            caller="hybrid_hypothesizer",
            system=(
                "You are the single hybrid hypothesizer for HostTune. "
                "A deterministic triage layer has already run. "
                "Return exactly one JSON object with keys: parameter_key, proposed_value, "
                "tuning_layer, apply_mode, rationale, expected_benchmark_impact, rollback_plan."
            ),
            prompt=prompt,
        )
        self._log_response("hybrid_hypothesizer", content)
        self._save_agent_artifact(context, "hybrid_hypothesizer", prompt, content)
        return ModelCompletion(content=content, usage=usage)

    def _log_prompt(self, agent: str, prompt: str) -> None:
        self.logger.stage_detail(
            "tune",
            f"{'─' * 8} {agent} PROMPT ({len(prompt)} chars) {'─' * 8}",
        )
        if self.logger.debug_enabled():
            self.logger.stage_detail("tune", prompt)

    def _log_response(self, agent: str, response: str) -> None:
        self.logger.stage_detail(
            "tune",
            f"{'─' * 8} {agent} RESPONSE {'─' * 8}\n{response}",
        )

    def _save_agent_artifact(
        self,
        context: HypothesisContext,
        agent: str,
        prompt: str,
        response: str,
    ) -> None:
        artifacts = context.tune_context.artifacts
        if artifacts is None:
            return
        iteration = context.iteration_number
        out_dir = artifacts.session_directory / "hypotheses"
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

    def _call_llm_with_usage(
        self, *, caller: str, system: str, prompt: str
    ) -> tuple[str, ModelUsage | None]:
        client = self._build_openai_client()
        try:
            completion = client.chat.completions.create(
                model=self.config.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            msg = f"[{caller}] LLM call failed: {type(exc).__name__}: {exc}"
            raise ValueError(msg) from exc
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
