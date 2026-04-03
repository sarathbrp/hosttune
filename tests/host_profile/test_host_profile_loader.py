import pytest

from host_profile.infrastructure.host_profile_loader import HostProfileLoader, HOST_PROFILES_DIR


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
