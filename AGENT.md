# Agent Engineering Contract

This repository must follow these rules for the repository and for each implementation step.

## Product Intent

The system being built here is a generic performance-tuning agent for RHEL systems.

The agent is not designed for one known issue pattern or one fixed application. It must support:

- any customer application on RHEL
- any target host reachable locally or remotely
- any benchmarkable workload, as long as a benchmark script can measure outcome

The benchmark is the ground-truth scoring function for the agent. Discovery constrains the search space, but benchmark results decide whether a change is useful.

The agent is expected to explore the target system, apply candidate tunings, rerun benchmarks as often as needed, and decide whether to keep or roll back changes based on measured impact.

Disruptive actions are part of the expected operating model when allowed by engagement policy, including:

- service reload
- service restart
- operating system reboot

This repository must therefore optimize for:

- generic runtime contracts
- environment-specific capability discovery
- benchmark-driven evaluation
- safe iterative tuning
- clear rollback and policy boundaries

## Non-Negotiable Standards

- Use strict object-oriented design. Model behavior through explicit classes, protocols, and bounded responsibilities.
- All production code must pass `mypy` in strict mode.
- All code must pass `ruff` linting and formatting checks.
- Every new production file must be covered by tests at greater than 80% line coverage.
- Prefer composition over inheritance unless inheritance is clearly simpler and bounded.
- Avoid god objects. A class should own one clear responsibility.
- Public APIs must be typed end to end.
- Side effects must be isolated behind infrastructure classes.
- Business rules must not be embedded directly in CLI entrypoints.
- Design for plugin-style extensibility. Application-specific behavior must be isolated from the generic runtime.

## Repository Shape

- The repository uses a top-level `src/` folder.
- The primary Python package lives directly under `src/`.
- Tests live under the top-level `tests/` folder.
- `pyproject.toml` lives at the repository root.
- Step boundaries should be expressed through packages, modules, and documented milestones, not wrapper directories.
- Shared abstractions should remain modular and easy to extract as the system grows.

## Quality Gates

Before considering a step complete, the implementation must provide:

- `ruff check`
- `ruff format --check`
- `mypy --strict`
- `pytest`
- coverage report proving greater than 80% line coverage for each created production file

## Architecture Rules

- Separate domain, infrastructure, orchestration, and interface concerns.
- Configuration must be represented by typed objects, not ad hoc dictionaries.
- Remote and localhost execution must share the same domain contracts.
- Shell access, SSH access, file reads, and environment inspection must be abstracted behind interfaces.
- Discovery logic must return typed capability objects, not raw command output.
- Parsers must be isolated from transport code.
- The agent must reason over normalized capability models, not over unstructured text.
- The benchmark contract must be first-class, typed, and independent from any single application.
- The runtime must evaluate changes using benchmark results, not static assumptions.
- Application-specific controls must be isolated behind explicit interfaces or plugin contracts.
- Service reload, service restart, rollback, and reboot behavior must be controlled through typed policy objects.
- Capability discovery must express actionable tuning availability, not only hardware facts.
- A step must not couple benchmark execution, discovery parsing, and tuning policy into one class.

## Step 1 Scope

Step 1 establishes the foundation only:

- typed configuration for localhost and remote hosts
- transport abstraction for local and SSH execution
- discovery command interface and typed result models
- benchmark runner interface and normalized benchmark result models
- engagement policy models for reload, restart, reboot, and rollback permissions
- capability map builder skeleton
- CLI entrypoint to run discovery
- tests for transport selection, parsing, benchmark contract behavior, and configuration validation

Step 1 must not mix in tuning logic, benchmark loops, or LLM policy decisions.
Step 1 may run a benchmark baseline, but it must not yet implement autonomous tuning iteration.

## Known architectural debt

- **Per-parameter kernel_network apply cost:** Sysctls still share one `change_categories.kernel_network` mode; **deferred vs active** is derived from that single value. Finer YAML (per knob or subcategories) remains future work.

## Implementation Bias

- Keep modules small.
- Make dependencies explicit in constructors.
- Prefer deterministic tests with fake executors over broad integration tests.
- Use dataclasses or well-scoped classes for immutable models where appropriate.
- Document assumptions at the module boundary, not inline throughout the code.
- Treat benchmark output as a typed domain object, not free-form text.
- Treat engagement permissions as explicit configuration, not hidden behavior.
