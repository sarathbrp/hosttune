import pytest

from host_profile.infrastructure.host_profile_validator import HostProfileValidator


def _valid_data() -> dict:
    return {
        "identity": {
            "name": "rhel-9",
            "platform": "rhel",
            "version": "9",
            "variant": None,
        },
        "tunable_surface": {
            "network_queues": {
                "min_combined": 1,
                "max_combined": 0,
                "allow_irq_affinity": True,
                "priority_tier": "high",
                "apply_mode": "reload",
            },
            "cpu_governor": {
                "allowed_governors": ["performance", "schedutil"],
                "forbidden_governors": [],
                "preferred_governor": "performance",
                "priority_tier": "high",
                "apply_mode": "reload",
            },
            "host_sysctls": [
                {
                    "name": "net.core.rmem_max",
                    "priority_tier": "medium",
                    "rationale_hint": "Max receive socket buffer",
                }
            ],
        },
    }


def test_validator_parses_valid_rhel9_profile() -> None:
    profile = HostProfileValidator().validate(_valid_data())

    assert profile.identity.name == "rhel-9"
    assert profile.identity.platform == "rhel"
    assert profile.identity.variant is None
    assert profile.tunable_surface.network_queues is not None
    assert profile.tunable_surface.network_queues.max_combined == 0
    assert profile.tunable_surface.network_queues.allow_irq_affinity is True
    assert profile.tunable_surface.cpu_governor is not None
    assert profile.tunable_surface.cpu_governor.preferred_governor == "performance"
    assert len(profile.tunable_surface.host_sysctls) == 1
    assert profile.tunable_surface.host_sysctls[0].name == "net.core.rmem_max"


def test_validator_accepts_null_network_queues_for_vm() -> None:
    data = _valid_data()
    data["tunable_surface"]["network_queues"] = None
    profile = HostProfileValidator().validate(data)
    assert profile.tunable_surface.network_queues is None


def test_validator_raises_on_missing_identity_name() -> None:
    data = _valid_data()
    del data["identity"]["name"]
    with pytest.raises(ValueError, match="non-empty string"):
        HostProfileValidator().validate(data)


def test_validator_raises_on_invalid_priority_tier() -> None:
    data = _valid_data()
    data["tunable_surface"]["network_queues"]["priority_tier"] = "critical"
    with pytest.raises(ValueError):
        HostProfileValidator().validate(data)
