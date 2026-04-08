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


def test_loader_merges_legacy_perf_hierarchy_sidecar(tmp_path: Path) -> None:
    registry = tmp_path / "service-monitor"
    registry.mkdir()
    definition = registry / "nginx.yaml"
    definition.write_text(
        "identity: {}\nhealth_check: {}\nsnapshot: {}\nrestart: {}\n"
        "tunable_surface: {}\nbenchmark_hints: {}\n",
        encoding="utf-8",
    )
    sidecar = registry / "nginx-perf-hierarchy.yaml"
    sidecar.write_text(
        "version: '1.1'\n"
        "description: legacy sidecar\n"
        "groups:\n"
        "  1_cpu_parallelism:\n"
        "    description: test group\n"
        "    parameters:\n"
        "      worker_processes:\n"
        "        target_perf: auto\n",
        encoding="utf-8",
    )

    data = ServiceDefinitionLoader(registry).load("nginx")

    surface = data["tunable_surface"]
    assert isinstance(surface, dict)
    hierarchy = surface.get("performance_hierarchy")
    assert isinstance(hierarchy, dict)
    assert hierarchy["description"] == "legacy sidecar"


def test_loader_prefers_merged_perf_hierarchy_when_present(tmp_path: Path) -> None:
    registry = tmp_path / "service-monitor"
    registry.mkdir()
    definition = registry / "nginx.yaml"
    definition.write_text(
        "identity: {}\nhealth_check: {}\nsnapshot: {}\nrestart: {}\n"
        "tunable_surface:\n"
        "  performance_hierarchy:\n"
        "    version: '2.0'\n"
        "    description: merged\n"
        "benchmark_hints: {}\n",
        encoding="utf-8",
    )
    sidecar = registry / "nginx-perf-hierarchy.yaml"
    sidecar.write_text(
        "version: '1.1'\n"
        "description: sidecar\n"
        "groups: {}\n",
        encoding="utf-8",
    )

    data = ServiceDefinitionLoader(registry).load("nginx")

    surface = data["tunable_surface"]
    assert isinstance(surface, dict)
    hierarchy = surface.get("performance_hierarchy")
    assert isinstance(hierarchy, dict)
    assert hierarchy["description"] == "merged"
