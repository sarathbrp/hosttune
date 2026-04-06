from typing import cast

import pytest

from preflight.domain.models import CommandExecutor, CommandResult
from tune.application.candidate_value_reads import (
    directive_source_path_from_nginx_dump,
    parse_directive_from_nginx_dump,
    read_network_ring_catalog_current,
    read_network_ring_catalog_current_with_source,
    read_service_directive_catalog_current,
    read_service_directive_catalog_current_with_source,
    read_sysctl_catalog_current,
    read_sysctl_catalog_current_with_source,
    sysctl_value_from_profile,
    try_read_network_ring_current,
)


def test_sysctl_value_from_profile() -> None:
    profile = (("net.core.somaxconn", "4096"), ("net.ipv4.tcp_fin_timeout", ""))
    assert sysctl_value_from_profile(profile, "net.core.somaxconn") == "4096"
    assert sysctl_value_from_profile(profile, "net.ipv4.tcp_fin_timeout") is None
    assert sysctl_value_from_profile(profile, "missing") is None


def test_read_sysctl_catalog_current_prefers_live_over_profile() -> None:
    profile = (("net.core.somaxconn", "1111"),)

    class _Ex:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command, 0, "2222", "")

    assert read_sysctl_catalog_current("net.core.somaxconn", cast(CommandExecutor, _Ex()), profile) == "2222"


def test_read_sysctl_catalog_current_falls_back_to_profile() -> None:
    profile = (("net.core.somaxconn", "1111"),)

    class _Ex:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command, 1, "", "nope")

    assert read_sysctl_catalog_current("net.core.somaxconn", cast(CommandExecutor, _Ex()), profile) == "1111"


def test_read_sysctl_catalog_current_with_source_reports_fallback() -> None:
    profile = (("net.core.somaxconn", "1111"),)

    class _Ex:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command, 1, "", "nope")

    value, source = read_sysctl_catalog_current_with_source(
        "net.core.somaxconn",
        cast(CommandExecutor, _Ex()),
        profile,
    )

    assert value == "1111"
    assert source == "preflight_sysctl_profile"


def test_read_sysctl_catalog_current_logs_live_fallback(caplog: pytest.LogCaptureFixture) -> None:
    profile = (("net.core.somaxconn", "1111"),)

    class _Ex:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command, 1, "", "permission denied")

    with caplog.at_level("WARNING"):
        value = read_sysctl_catalog_current(
            "net.core.somaxconn",
            cast(CommandExecutor, _Ex()),
            profile,
        )

    assert value == "1111"
    assert "sysctl live read failed for net.core.somaxconn" in caplog.text


def test_read_sysctl_catalog_current_no_executor_uses_profile() -> None:
    profile = (("net.core.somaxconn", "3333"),)
    assert read_sysctl_catalog_current("net.core.somaxconn", None, profile) == "3333"


def test_parse_directive_from_nginx_dump() -> None:
    dump = "# comment\nworker_processes 88;\n"
    assert parse_directive_from_nginx_dump("worker_processes", dump) == "88"
    assert parse_directive_from_nginx_dump("worker_processes", None) is None


def test_directive_source_path_from_nginx_dump_uses_last_matching_file() -> None:
    dump = (
        "# configuration file /etc/nginx/nginx.conf:\n"
        "http {\n"
        "    include /etc/nginx/conf.d/*.conf;\n"
        "}\n"
        "# configuration file /etc/nginx/conf.d/hackathon.conf:\n"
        "server {\n"
        "    limit_rate 5m;\n"
        "}\n"
    )
    assert (
        directive_source_path_from_nginx_dump("limit_rate", dump)
        == "/etc/nginx/conf.d/hackathon.conf"
    )


def test_read_service_directive_catalog_current_dump_when_no_executor() -> None:
    dump = "worker_processes 77;\n"
    assert (
        read_service_directive_catalog_current(None, "/x", "worker_processes", dump) == "77"
    )


def test_read_service_directive_catalog_current_logs_snapshot_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dump = "worker_processes 77;\n"

    class _Ex:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command, 1, "", "not found")

    with caplog.at_level("WARNING"):
        value = read_service_directive_catalog_current(
            cast(CommandExecutor, _Ex()),
            "/etc/nginx/nginx.conf",
            "worker_processes",
            dump,
        )

    assert value == "77"
    assert "directive live read fell back to runtime snapshot" in caplog.text


def test_read_service_directive_catalog_current_with_source_reports_snapshot_fallback() -> None:
    dump = "worker_processes 77;\n"

    class _Ex:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command, 1, "", "not found")

    value, source = read_service_directive_catalog_current_with_source(
        cast(CommandExecutor, _Ex()),
        "/etc/nginx/nginx.conf",
        "worker_processes",
        dump,
    )

    assert value == "77"
    assert source == "runtime_snapshot_nginx_t"


def test_read_network_ring_catalog_current_live_then_preflight() -> None:
    class _Ex:
        def run(self, command: str) -> CommandResult:
            assert "ethtool -g" in command
            return CommandResult(command, 0, "2048\n", "")

    v = read_network_ring_catalog_current("rx", "eth0", cast(CommandExecutor, _Ex()), 512)
    assert v == "2048"


def test_read_network_ring_catalog_current_falls_back_to_preflight() -> None:
    class _Ex:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command, 1, "", "")

    assert (
        read_network_ring_catalog_current("rx", "eth0", cast(CommandExecutor, _Ex()), 512) == "512"
    )


def test_read_network_ring_catalog_current_with_source_reports_preflight_fallback() -> None:
    class _Ex:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command, 1, "", "")

    value, source = read_network_ring_catalog_current_with_source(
        "rx",
        "eth0",
        cast(CommandExecutor, _Ex()),
        512,
    )
    assert value == "512"
    assert source == "preflight_network_probe"


def test_try_read_network_ring_current_returns_none_on_failure() -> None:
    class _Ex:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command, 1, "", "")

    assert try_read_network_ring_current(cast(CommandExecutor, _Ex()), "eth0", "rx") is None
