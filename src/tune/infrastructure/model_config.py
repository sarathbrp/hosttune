from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelEndpointConfig:
    base_url: str
    api_key: str
    model_name: str


class ModelEndpointConfigLoader:
    def load(self, env_path: Path = Path(".env")) -> ModelEndpointConfig:
        values = self._load_values(env_path)
        try:
            base_url = values["GPT_OSS_BASE_URL"]
            api_key = values["GPT_OSS_API_KEY"]
            model_name = values["GPT_OSS_MODEL"]
        except KeyError as error:
            msg = f"Missing required model setting: {error.args[0]}"
            raise ValueError(msg) from error
        return ModelEndpointConfig(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
        )

    def _load_values(self, env_path: Path) -> dict[str, str]:
        values: dict[str, str] = {
            "GPT_OSS_BASE_URL": os.environ.get("GPT_OSS_BASE_URL", ""),
            "GPT_OSS_API_KEY": os.environ.get("GPT_OSS_API_KEY", ""),
            "GPT_OSS_MODEL": os.environ.get("GPT_OSS_MODEL", ""),
        }
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped == "" or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", maxsplit=1)
                if key in values and values[key] == "":
                    values[key] = value
        missing = [key for key, value in values.items() if value == ""]
        if missing:
            msg = f"Missing required model settings: {', '.join(missing)}"
            raise ValueError(msg)
        return values
