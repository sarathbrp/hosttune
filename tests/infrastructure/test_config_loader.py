from pathlib import Path

import pytest

from preflight.infrastructure.config_loader import ConfigLoader


def test_loads_local_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
target:
  mode: local
service:
  name: nginx
policy:
  allow_reload: true
  max_iterations: 4
benchmark:
  runner:
    mode: local
  contestant_name: hosttune
  script_path: /root/hackathon-tools/benchmark.sh
  results_directory: /root/hackathon-results
  workloads:
    - homepage
""".strip(),
        encoding="utf-8",
    )

    loaded = ConfigLoader().load(config_path)

    assert loaded.target.mode == "local"
    assert loaded.policy.allow_reload is True
    assert loaded.service_name == "nginx"
    assert loaded.benchmark_config is not None
    assert loaded.benchmark_config.contestant_name == "hosttune"
    assert loaded.benchmark_config.runner_target.mode == "local"


def test_loads_ssh_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
target:
  mode: ssh
  host: 10.0.0.5
  user: ec2-user
  private_key_path: /tmp/id_rsa
  port: 2222
service:
  name: nginx
""".strip(),
        encoding="utf-8",
    )

    loaded = ConfigLoader().load(config_path)

    assert loaded.target.mode == "ssh"
    assert loaded.target.port == 2222


def test_rejects_unknown_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("target:\n  mode: serial\nservice:\n  name: nginx\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported target mode"):
        ConfigLoader().load(config_path)


def test_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration root must be a mapping"):
        ConfigLoader().load(config_path)


def test_rejects_missing_ssh_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
target:
  mode: ssh
  host: 10.0.0.5
service:
  name: nginx
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing SSH target fields"):
        ConfigLoader().load(config_path)


def test_rejects_non_mapping_benchmark_runner(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
target:
  mode: local
service:
  name: nginx
benchmark:
  contestant_name: hosttune
  script_path: /root/hackathon-tools/benchmark.sh
  results_directory: /root/hackathon-results
  workloads:
    - homepage
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="benchmark.runner must be configured"):
        ConfigLoader().load(config_path)


def test_rejects_invalid_workloads(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
target:
  mode: local
service:
  name: nginx
benchmark:
  runner:
    mode: local
  contestant_name: hosttune
  script_path: /root/hackathon-tools/benchmark.sh
  results_directory: /root/hackathon-results
  workloads: 42
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="benchmark.workloads must be a non-empty list"):
        ConfigLoader().load(config_path)


def test_rejects_missing_service_name(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("target:\n  mode: local\n", encoding="utf-8")

    with pytest.raises(ValueError, match="service.name must be a non-empty string"):
        ConfigLoader().load(config_path)
