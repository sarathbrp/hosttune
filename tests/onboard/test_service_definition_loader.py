from pathlib import Path

import pytest

from onboard.infrastructure.service_definition_loader import ServiceDefinitionLoader


def test_loader_reads_service_definition(tmp_path: Path) -> None:
    registry = tmp_path / "service-monitor"
    registry.mkdir()
    definition = registry / "nginx.yaml"
    definition.write_text("identity: {}\nhealth_check: {}\nsnapshot: {}\nrestart: {}\ntunable_surface: {}\nbenchmark_hints: {}\n", encoding="utf-8")

    data = ServiceDefinitionLoader(registry).load("nginx")

    assert "identity" in data


def test_loader_rejects_missing_definition(tmp_path: Path) -> None:
    registry = tmp_path / "service-monitor"
    registry.mkdir()

    with pytest.raises(ValueError, match="Service definition not found"):
        ServiceDefinitionLoader(registry).load("nginx")
