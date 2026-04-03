from typing import cast

import pytest

from onboard.domain.models import ApplyMode, ConfigFormat, PriorityTier, ProbeType
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator


def build_valid_definition() -> dict[str, object]:
    return {
        "identity": {
            "service_name": "nginx",
            "systemd_unit_name": "nginx.service",
            "rhel_versions": ["9"],
            "service_versions": [],
            "config_paths": ["/etc/nginx/nginx.conf"],
            "config_format": "nginx",
            "working_directory": "/var/www/nginx",
            "log_paths": ["/var/log/nginx/access.log"],
        },
        "health_check": {
            "probe_type": "http",
            "target": "http://127.0.0.1/",
            "expected_status_code": 200,
            "expected_string": None,
            "expected_exit_code": None,
            "timeout_seconds": 5,
            "retries": 3,
            "warmup_seconds": 2,
        },
        "snapshot": {
            "files_to_snapshot": ["/etc/nginx/nginx.conf"],
            "runtime_state_command": "nginx -T",
            "process_state": {
                "pid_file": "/run/nginx.pid",
                "worker_process_hint": "nginx",
                "open_connections_command": "ss -tan",
            },
            "restore_sequence": ["systemctl restart nginx"],
            "snapshot_storage_location": "/var/tmp/hosttune",
        },
        "restart": {
            "reload": {"supported": True, "command": "systemctl reload nginx"},
            "restart": {
                "supported": True,
                "command": "systemctl restart nginx",
                "expected_downtime_seconds": 5,
            },
            "change_categories": {
                "service_config": "reload",
                "runtime_limits": "reload",
                "systemd_unit_limits": "restart",
                "cgroup_resource_controls": "restart",
            },
            "drain_policy": "best_effort",
            "dependency_chain": [],
            "post_restart_validation": "health_check",
        },
        "tunable_surface": {
            "allowed_directives": {
                "worker_processes": {
                    "value_type": "integer",
                    "priority_tier": "high",
                    "min_value": 1,
                    "max_value": 112,
                    "allowed_values": [],
                    "forbidden_values": ["1"],
                    "apply_mode": "reload",
                },
                "sendfile": {
                    "value_type": "enum",
                    "priority_tier": "medium",
                    "min_value": None,
                    "max_value": None,
                    "allowed_values": ["on", "off"],
                    "forbidden_values": ["off"],
                    "apply_mode": "reload",
                },
                "worker_rlimit_nofile": {
                    "value_type": "integer",
                    "priority_tier": "medium",
                    "min_value": 65535,
                    "max_value": 1048576,
                    "allowed_values": [],
                    "forbidden_values": [],
                    "apply_mode": "reload",
                },
                "keepalive_timeout": {
                    "value_type": "integer",
                    "priority_tier": "low",
                    "min_value": 1,
                    "max_value": 300,
                    "allowed_values": [],
                    "forbidden_values": [],
                    "apply_mode": "reload",
                },
            },
            "forbidden_directives": ["daemon"],
            "interdependencies": [],
            "relevant_sysctls": [
                {"name": "net.core.somaxconn", "priority_tier": "high"},
                {"name": "net.ipv4.ip_local_port_range", "priority_tier": "medium"},
            ],
            "network_ring_priority_tier": "medium",
            "runtime_limits": {
                "nofile_soft": {
                    "value_type": "integer",
                    "priority_tier": "high",
                    "min_value": 4096,
                    "max_value": 2097152,
                    "allowed_values": [],
                    "forbidden_values": [],
                    "apply_mode": "reload",
                },
            },
            "systemd_unit_limits": {
                "limit_nofile": {
                    "value_type": "integer",
                    "priority_tier": "high",
                    "min_value": 65535,
                    "max_value": 2097152,
                    "allowed_values": [],
                    "forbidden_values": [],
                    "apply_mode": "restart",
                },
                "limit_nproc": {
                    "value_type": "integer",
                    "priority_tier": "medium",
                    "min_value": 64,
                    "max_value": 655350,
                    "allowed_values": [],
                    "forbidden_values": [],
                    "apply_mode": "restart",
                },
            },
            "cgroup_resource_controls": {
                "cpu_quota_percent": {
                    "value_type": "integer",
                    "priority_tier": "medium",
                    "min_value": 10,
                    "max_value": 400,
                    "allowed_values": [],
                    "forbidden_values": [],
                    "apply_mode": "restart",
                },
                "memory_max_mib": {
                    "value_type": "integer",
                    "priority_tier": "low",
                    "min_value": 64,
                    "max_value": 65536,
                    "allowed_values": [],
                    "forbidden_values": [],
                    "apply_mode": "restart",
                },
            },
        },
        "benchmark_hints": {
            "primary_metric": "requests_per_second",
            "guardrail_metrics": ["p95_latency"],
            "expected_variance": 0.05,
            "warmup_seconds": 10,
            "interference_sources": ["access logging"],
        },
    }


def test_validator_builds_typed_service_definition() -> None:
    definition = ServiceDefinitionValidator().validate(build_valid_definition())

    assert definition.identity.config_format is ConfigFormat.NGINX
    assert definition.health_check.probe_type is ProbeType.HTTP
    assert definition.restart.change_categories["service_config"] is ApplyMode.RELOAD
    assert definition.tunable_surface.allowed_directives["worker_processes"].priority_tier.value == "high"
    assert definition.tunable_surface.allowed_directives["worker_processes"].forbidden_values == (
        "1",
    )
    assert definition.tunable_surface.network_ring_priority_tier is PriorityTier.MEDIUM
    assert [s.priority_tier for s in definition.tunable_surface.relevant_sysctls] == [
        PriorityTier.HIGH,
        PriorityTier.MEDIUM,
    ]
    assert "nofile_soft" in definition.tunable_surface.runtime_limits
    assert "limit_nofile" in definition.tunable_surface.systemd_unit_limits
    assert "limit_nproc" in definition.tunable_surface.systemd_unit_limits
    assert "cpu_quota_percent" in definition.tunable_surface.cgroup_resource_controls
    assert "memory_max_mib" in definition.tunable_surface.cgroup_resource_controls
    assert definition.restart.change_categories["runtime_limits"] is ApplyMode.RELOAD
    assert definition.restart.change_categories["systemd_unit_limits"] is ApplyMode.RESTART
    assert definition.restart.change_categories["cgroup_resource_controls"] is ApplyMode.RESTART


def test_validator_parses_sysctl_strings_as_high_tier() -> None:
    data = build_valid_definition()
    data["tunable_surface"] = {
        **cast(dict[str, object], data["tunable_surface"]),
        "relevant_sysctls": ["net.core.somaxconn"],
        "network_ring_priority_tier": "high",
    }
    definition = ServiceDefinitionValidator().validate(data)
    sysctl = definition.tunable_surface.relevant_sysctls[0]
    assert sysctl.name == "net.core.somaxconn"
    assert sysctl.priority_tier.value == "high"


def test_validator_rejects_invalid_tuning_layer_on_directive() -> None:
    data = build_valid_definition()
    surface = cast(dict[str, object], data["tunable_surface"])
    directives = cast(dict[str, object], surface["allowed_directives"])
    wp = cast(dict[str, object], directives["worker_processes"])
    directives = {**directives, "worker_processes": {**wp, "tuning_layer": "cgroup"}}
    surface = {**surface, "allowed_directives": directives}
    data = {**data, "tunable_surface": surface}
    with pytest.raises(ValueError, match="Invalid tuning_layer"):
        ServiceDefinitionValidator().validate(data)


def test_validator_parses_sysctl_tuning_layer_override() -> None:
    data = build_valid_definition()
    surface = cast(dict[str, object], data["tunable_surface"])
    surface = {
        **surface,
        "relevant_sysctls": [
            {"name": "net.core.somaxconn", "priority_tier": "high", "tuning_layer": "service"},
        ],
    }
    data = {**data, "tunable_surface": surface}
    definition = ServiceDefinitionValidator().validate(data)
    sysctl = definition.tunable_surface.relevant_sysctls[0]
    assert sysctl.name == "net.core.somaxconn"
    assert sysctl.tuning_layer == "service"


def test_validator_parses_network_ring_tuning_layer_override() -> None:
    data = build_valid_definition()
    surface = cast(dict[str, object], data["tunable_surface"])
    data = {**data, "tunable_surface": {**surface, "network_ring_tuning_layer": "runtime"}}
    definition = ServiceDefinitionValidator().validate(data)
    assert definition.tunable_surface.network_ring_tuning_layer == "runtime"


def test_validator_rejects_invalid_relevant_sysctl_entry() -> None:
    data = build_valid_definition()
    data["tunable_surface"] = {
        **cast(dict[str, object], data["tunable_surface"]),
        "relevant_sysctls": [42],
    }
    with pytest.raises(ValueError, match="relevant_sysctls"):
        ServiceDefinitionValidator().validate(data)


def test_validator_rejects_missing_required_blocks() -> None:
    data = build_valid_definition()
    data.pop("health_check")

    with pytest.raises(ValueError, match="Missing required service definition blocks"):
        ServiceDefinitionValidator().validate(data)
