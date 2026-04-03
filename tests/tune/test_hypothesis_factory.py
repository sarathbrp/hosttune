from pathlib import Path

from tune.application.hypothesis_factory import build_langgraph_hypothesis_generator


def test_hypothesis_factory_builds_llm_generator(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            (
                "GPT_OSS_BASE_URL=http://example/v1",
                "GPT_OSS_API_KEY=test-key",
                "GPT_OSS_MODEL=/models/test-model",
            )
        ),
        encoding="utf-8",
    )

    generator = build_langgraph_hypothesis_generator(env_path)

    assert generator.model_client.__class__.__name__ == "LangGraphHypothesisClient"
