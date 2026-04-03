import pytest

from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator

from tests.onboard.test_service_definition_validator import build_valid_definition


def test_validator_rejects_invalid_config_format() -> None:
    data = build_valid_definition()
    data["identity"]["config_format"] = "toml"  # type: ignore[index]

    with pytest.raises(ValueError):
        ServiceDefinitionValidator().validate(data)


def test_validator_rejects_non_string_lists() -> None:
    data = build_valid_definition()
    data["identity"]["config_paths"] = [123]  # type: ignore[index]

    with pytest.raises(ValueError, match="Expected list\\[str\\]"):
        ServiceDefinitionValidator().validate(data)


def test_validator_rejects_bad_numeric_types() -> None:
    data = build_valid_definition()
    data["health_check"]["timeout_seconds"] = "5"  # type: ignore[index]

    with pytest.raises(ValueError, match="Expected integer"):
        ServiceDefinitionValidator().validate(data)


def test_validator_rejects_bad_boolean_types() -> None:
    data = build_valid_definition()
    data["restart"]["reload"]["supported"] = "yes"  # type: ignore[index]

    with pytest.raises(ValueError, match="Expected boolean"):
        ServiceDefinitionValidator().validate(data)
