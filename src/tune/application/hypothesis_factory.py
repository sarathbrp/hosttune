from __future__ import annotations

from pathlib import Path

from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from tune.application.hypothesis_generator import (
    CompressedPromptBuilder,
    HypothesisPromptBuilder,
    LlmHypothesisGenerator,
)
from tune.application.rule_based_triage import RuleBasedTriage, TriageRulesLoader
from tune.infrastructure.langgraph_hypothesis_client import LangGraphHypothesisClient
from tune.infrastructure.model_config import ModelEndpointConfigLoader


def build_langgraph_hypothesis_generator(
    env_path: Path = Path(".env"),
    triage_rules_path: Path = Path("triage-rules.yaml"),
    logger: ExecutionLogger | None = None,
    *,
    prompt_compression: bool = False,
    compiled_path: Path | None = None,
) -> LlmHypothesisGenerator:
    config = ModelEndpointConfigLoader().load(env_path)
    execution_logger = logger or NullExecutionLogger()
    ruleset = TriageRulesLoader().load(triage_rules_path)
    triage = RuleBasedTriage(ruleset=ruleset)
    prompt_builder: HypothesisPromptBuilder | CompressedPromptBuilder
    if prompt_compression:
        prompt_builder = CompressedPromptBuilder(triage=triage)
    else:
        prompt_builder = HypothesisPromptBuilder(triage=triage)
    return LlmHypothesisGenerator(
        model_client=LangGraphHypothesisClient(
            config=config,
            prompt_builder=prompt_builder,
            logger=execution_logger,
            compiled_path=compiled_path,
        ),
        triage=triage,
        logger=execution_logger,
    )
