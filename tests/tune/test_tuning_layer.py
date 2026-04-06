import pytest

from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.domain.tuning_layer import (
    TuningLayer,
    resolve_tuning_layer,
    tuning_layer_for_parameter_key,
)

from tests.tune.test_candidate_catalog_builder import FakeExecutor, build_tune_context


@pytest.mark.parametrize(
    ("parameter_key", "expected"),
    [
        ("sysctl.net.core.somaxconn", TuningLayer.KERNEL),
        ("network.ring.rx", TuningLayer.NETWORK),
        ("network.ring.tx", TuningLayer.NETWORK),
        ("service.directive.worker_processes", TuningLayer.SERVICE),
        ("service.directive.keepalive_timeout", TuningLayer.SERVICE),
        ("service.directive.worker_rlimit_nofile", TuningLayer.RUNTIME),
        ("runtime.prlimit.nofile_soft", TuningLayer.RUNTIME),
        ("systemd.unit.limit_nproc", TuningLayer.RUNTIME),
        ("systemd.unit.limit_nofile", TuningLayer.RUNTIME),
        ("systemd.cgroup.cpu_quota_percent", TuningLayer.RUNTIME),
    ],
)
def test_tuning_layer_for_parameter_key_maps_known_prefixes(
    parameter_key: str,
    expected: TuningLayer,
) -> None:
    assert tuning_layer_for_parameter_key(parameter_key) is expected


def test_tuning_layer_for_parameter_key_rejects_unknown_prefix() -> None:
    with pytest.raises(ValueError, match="Unknown parameter_key"):
        tuning_layer_for_parameter_key("cgroup.memory.max")


def test_resolve_tuning_layer_uses_yaml_override_when_set() -> None:
    assert resolve_tuning_layer("sysctl.net.core.somaxconn", "service") is TuningLayer.SERVICE
    assert resolve_tuning_layer("network.ring.rx", "kernel") is TuningLayer.KERNEL


def test_resolve_tuning_layer_falls_back_to_prefix_rules() -> None:
    assert resolve_tuning_layer("sysctl.net.core.somaxconn", None) is TuningLayer.KERNEL


def test_candidate_catalog_sets_tuning_layer_consistent_with_key() -> None:
    context = build_tune_context()
    candidates = CandidateCatalogBuilder().build(context, FakeExecutor())
    for candidate in candidates:
        assert candidate.tuning_layer is tuning_layer_for_parameter_key(candidate.parameter_key)
