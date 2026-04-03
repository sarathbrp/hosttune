from __future__ import annotations

from preflight.cli import main as run_step1


def main() -> int:
    return run_step1()


if __name__ == "__main__":
    raise SystemExit(main())
