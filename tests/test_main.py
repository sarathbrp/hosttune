from __future__ import annotations

import runpy

import pytest

import main
import preflight.cli


def test_main_delegates_to_step_entrypoint(monkeypatch) -> None:
    monkeypatch.setattr(main, "run_step1", lambda: 7)

    result = main.main()

    assert result == 7


def test_main_module_exits_with_step_status(monkeypatch) -> None:
    monkeypatch.setattr(preflight.cli, "main", lambda: 0)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("main", run_name="__main__")

    assert exc_info.value.code == 0
