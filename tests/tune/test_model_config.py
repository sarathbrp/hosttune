from pathlib import Path

import pytest

from tune.infrastructure.model_config import ModelEndpointConfigLoader


def test_model_config_loader_reads_env_file(tmp_path: Path) -> None:
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

    config = ModelEndpointConfigLoader().load(env_path)

    assert config.base_url == "http://example/v1"
    assert config.api_key == "test-key"
    assert config.model_name == "/models/test-model"


def test_model_config_loader_rejects_missing_settings(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("GPT_OSS_BASE_URL=http://example/v1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required model settings"):
        ModelEndpointConfigLoader().load(env_path)
