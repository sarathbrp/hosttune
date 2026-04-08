import pytest

from host_profile.infrastructure.host_profile_loader import HostProfileLoader


def test_loader_loads_rhel9_profile() -> None:
    profile = HostProfileLoader().load("rhel-9")
    assert profile.identity.name == "rhel-9"
    assert profile.identity.platform == "rhel"
    assert profile.tunable_surface.network_queues is not None
    assert profile.tunable_surface.cpu_governor is not None
    assert len(profile.tunable_surface.host_sysctls) > 0


def test_loader_loads_rhel9_vm_profile() -> None:
    profile = HostProfileLoader().load("rhel-9-vm")
    assert profile.identity.variant == "vm"
    assert profile.tunable_surface.network_queues is None


def test_loader_raises_on_unknown_profile() -> None:
    with pytest.raises(ValueError, match="not found"):
        HostProfileLoader().load("unknown-profile-xyz")


def test_loader_merges_legacy_perf_hierarchy_sidecar(tmp_path) -> None:
    profiles = tmp_path / "host-profiles"
    profiles.mkdir()
    (profiles / "rhel-9.yaml").write_text(
        "\n".join(
            (
                "identity:",
                "  name: rhel-9",
                "  platform: rhel",
                "  version: '9'",
                "  variant: null",
                "tunable_surface:",
                "  host_sysctls: []",
            )
        ),
        encoding="utf-8",
    )
    (profiles / "rhel-9-perf-hierarchy.yaml").write_text(
        "\n".join(
            (
                "version: '1.0'",
                "description: legacy sidecar",
                "groups:",
                "  1_systemd:",
                "    description: first",
                "    parameters:",
                "      CPUQuota:",
                "        target_perf: none",
            )
        ),
        encoding="utf-8",
    )
    profile = HostProfileLoader(profiles_dir=profiles).load("rhel-9")
    hierarchy = profile.tunable_surface.performance_hierarchy
    assert hierarchy is not None
    assert hierarchy.description == "legacy sidecar"
    assert hierarchy.groups[0].group_id == "1_systemd"


def test_loader_prefers_merged_hierarchy_when_present(tmp_path) -> None:
    profiles = tmp_path / "host-profiles"
    profiles.mkdir()
    (profiles / "rhel-9.yaml").write_text(
        "\n".join(
            (
                "identity:",
                "  name: rhel-9",
                "  platform: rhel",
                "  version: '9'",
                "  variant: null",
                "tunable_surface:",
                "  host_sysctls: []",
                "  performance_hierarchy:",
                "    version: '1.0'",
                "    description: merged",
                "    groups:",
                "      1_systemd:",
                "        description: first",
                "        parameters:",
                "          CPUQuota:",
                "            target_perf: none",
            )
        ),
        encoding="utf-8",
    )
    (profiles / "rhel-9-perf-hierarchy.yaml").write_text(
        "version: '1.0'\ndescription: sidecar\ngroups: {}\n",
        encoding="utf-8",
    )
    profile = HostProfileLoader(profiles_dir=profiles).load("rhel-9")
    hierarchy = profile.tunable_surface.performance_hierarchy
    assert hierarchy is not None
    assert hierarchy.description == "merged"
