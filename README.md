# Perf Agent Step 1

Step 1 bootstraps the discovery subsystem and its execution model.

## Goal

Build a typed, object-oriented foundation that can run discovery against:

- localhost
- a remote host over SSH

The output of this step is a normalized capability map and baseline benchmark contract, not tuning actions.

## Suggested Layout

```text
pyproject.toml
README.md
main.py
src/
  preflight/
    cli.py
    application/
      discovery_runner.py
    domain/
      models.py
    infrastructure/
      config_loader.py
      executors/
        base.py
        local_executor.py
        ssh_executor.py
      probes/
        base.py
    interfaces/
      console_reporter.py
tests/
  application/
  domain/
  infrastructure/
```

## Configuration

Use one typed configuration model with two target modes:

- `local`
- `ssh`

Example shape:

```yaml
target:
  mode: ssh
  host: 10.0.0.25
  port: 22
  user: ec2-user
  private_key_path: /home/me/.ssh/id_rsa
  connect_timeout_seconds: 5
```

For localhost:

```yaml
target:
  mode: local
```

## Design Notes

- `CommandExecutor` should be an interface.
- `LocalCommandExecutor` and `SshCommandExecutor` should implement the same contract.
- Each probe should own command generation for one concern area.
- `DiscoveryRunner` should orchestrate probes and return a `CapabilityMap`.
- Reporting should be a separate concern from discovery.
- Benchmark execution should be represented by a typed contract and normalized result.
- `main.py` should remain the repository-level entrypoint used to wire later steps together.

## Development Hooks

The repository uses `pre-commit` for local quality and secret scanning.

Install runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install and enable it with:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Run all hooks on demand with:

```bash
pre-commit run --all-files
```

Configured hooks include:

- `ruff`
- `ruff-format`
- `gitleaks`

## Exit Criteria

- A user can run discovery on localhost from a config file.
- A user can run discovery on a remote SSH host from a config file.
- The command output is normalized into typed discovery models and a capability map.
- All code passes `ruff`, `mypy --strict`, and `pytest` with greater than 80% coverage per production file.
