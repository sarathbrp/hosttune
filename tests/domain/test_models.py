from pathlib import Path

from preflight.domain.models import EngagementPolicy, LocalTargetConfig, SshTargetConfig


def test_local_target_defaults() -> None:
    target = LocalTargetConfig()

    assert target.mode == "local"


def test_ssh_target_defaults() -> None:
    target = SshTargetConfig(host="example", user="ec2-user", private_key_path=Path("/tmp/key"))

    assert target.mode == "ssh"
    assert target.port == 22


def test_policy_fields_are_assigned() -> None:
    policy = EngagementPolicy(
        allow_reload=True,
        allow_restart=False,
        allow_reboot=False,
        rollback_required=True,
        max_iterations=5,
        benchmark_stability_threshold=0.1,
    )

    assert policy.allow_reload is True
    assert policy.max_iterations == 5
