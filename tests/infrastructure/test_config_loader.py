from pathlib import Path

import pytest

from preflight.infrastructure.config_loader import ConfigLoader


def test_loads_local_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
target:
  mode: local
policy:
  allow_reload: true
  max_iterations: 4
benchmark:
  command: "printf '123.0'"
""".strip(),
        encoding="utf-8",
    )

    loaded = ConfigLoader().load(config_path)

    assert loaded.target.mode == "local"
    assert loaded.policy.allow_reload is True
    assert loaded.benchmark_command == "printf '123.0'"


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
""".strip(),
        encoding="utf-8",
    )

    loaded = ConfigLoader().load(config_path)

    assert loaded.target.mode == "ssh"
    assert loaded.target.port == 2222


def test_rejects_unknown_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("target:\n  mode: serial\n", encoding="utf-8")

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
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing SSH target fields"):
        ConfigLoader().load(config_path)


def test_rejects_non_string_benchmark_command(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
target:
  mode: local
benchmark:
  command: 42
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="benchmark.command must be a string"):
        ConfigLoader().load(config_path)
