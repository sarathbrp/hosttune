import pytest

from onboard.domain.models import ApplyMode, ConfigFormat, ProbeType
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
            "change_categories": {"service_config": "reload"},
            "drain_policy": "best_effort",
            "dependency_chain": [],
            "post_restart_validation": "health_check",
        },
        "tunable_surface": {
            "allowed_directives": {
                "worker_processes": {
                    "value_type": "integer",
                    "min_value": 1,
                    "max_value": 112,
                    "allowed_values": [],
                    "apply_mode": "reload",
                }
            },
            "forbidden_directives": ["daemon"],
            "interdependencies": [],
            "relevant_sysctls": ["net.core.somaxconn"],
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


def test_validator_rejects_missing_required_blocks() -> None:
    data = build_valid_definition()
    data.pop("health_check")

    with pytest.raises(ValueError, match="Missing required service definition blocks"):
        ServiceDefinitionValidator().validate(data)
