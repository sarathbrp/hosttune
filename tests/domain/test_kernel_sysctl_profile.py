from preflight.domain.kernel_sysctl_profile import (
    PREFLIGHT_SYSCTL_KEYS,
    contract_sysctl_names_only_extra,
    format_sysctl_profile_compact,
    merged_sysctl_profile_key_order,
    sysctl_profile_read_command,
)


def test_sysctl_profile_read_command_lists_all_keys() -> None:
    cmd = sysctl_profile_read_command()
    assert "net.core.somaxconn" in cmd
    assert "sysctl -n" in cmd
    assert "for k in " in cmd


def test_merged_sysctl_profile_key_order_appends_contract_only_keys() -> None:
    order = merged_sysctl_profile_key_order(
        ("net.core.somaxconn", "net.ipv4.ip_local_port_range", "net.core.somaxconn"),
    )
    assert order[: len(PREFLIGHT_SYSCTL_KEYS)] == PREFLIGHT_SYSCTL_KEYS
    assert order[-1] == "net.ipv4.ip_local_port_range"
    assert order.count("net.ipv4.ip_local_port_range") == 1


def test_contract_sysctl_names_only_extra_skips_base_keys() -> None:
    extra = contract_sysctl_names_only_extra(
        ("net.core.somaxconn", "net.ipv4.ip_local_port_range"),
    )
    assert extra == ("net.ipv4.ip_local_port_range",)


def test_format_sysctl_profile_compact_truncates() -> None:
    profile = (("a", "1"), ("b", "2"))
    short = format_sysctl_profile_compact(profile, max_chars=1000)
    assert short == "a=1; b=2"
    tiny = format_sysctl_profile_compact(profile, max_chars=5)
    assert "truncated" in tiny
