from __future__ import annotations

from pathlib import Path

from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from tune.application.hypothesis_generator import (
    HypothesisPromptBuilder,
    LlmHypothesisGenerator,
)
from tune.infrastructure.langgraph_hypothesis_client import LangGraphHypothesisClient
from tune.infrastructure.model_config import ModelEndpointConfigLoader


def build_langgraph_hypothesis_generator(
    env_path: Path = Path(".env"),
    logger: ExecutionLogger | None = None,
) -> LlmHypothesisGenerator:
    config = ModelEndpointConfigLoader().load(env_path)
    execution_logger = logger or NullExecutionLogger()
    return LlmHypothesisGenerator(
        model_client=LangGraphHypothesisClient(
            config=config,
            prompt_builder=HypothesisPromptBuilder(),
            logger=execution_logger,
        ),
        logger=execution_logger,
    )
