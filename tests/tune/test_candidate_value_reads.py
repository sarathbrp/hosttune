from typing import cast

from preflight.domain.models import CommandExecutor, CommandResult
from tune.application.candidate_value_reads import (
    parse_directive_from_nginx_dump,
    read_network_ring_catalog_current,
    read_service_directive_catalog_current,
    read_sysctl_catalog_current,
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


def test_read_sysctl_catalog_current_no_executor_uses_profile() -> None:
    profile = (("net.core.somaxconn", "3333"),)
    assert read_sysctl_catalog_current("net.core.somaxconn", None, profile) == "3333"


def test_parse_directive_from_nginx_dump() -> None:
    dump = "# comment\nworker_processes 88;\n"
    assert parse_directive_from_nginx_dump("worker_processes", dump) == "88"
    assert parse_directive_from_nginx_dump("worker_processes", None) is None


def test_read_service_directive_catalog_current_dump_when_no_executor() -> None:
    dump = "worker_processes 77;\n"
    assert (
        read_service_directive_catalog_current(None, "/x", "worker_processes", dump) == "77"
    )


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


def test_try_read_network_ring_current_returns_none_on_failure() -> None:
    class _Ex:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command, 1, "", "")

    assert try_read_network_ring_current(cast(CommandExecutor, _Ex()), "eth0", "rx") is None
