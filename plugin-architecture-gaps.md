# Plugin Architecture Gaps

## Purpose

This document captures the gap between the current modular runtime and the longer-term
product goal:

> Given a benchmarkable application on RHEL, the agent should be able to accept a
> service plugin and run the same generic diagnosis/remediation pipeline on a target
> system.

This is intentionally deferred work. The current priority remains making the new
modular architecture stable and authoritative.

---

## Product Goal

The final platform should support customer applications beyond Nginx, as long as:

- the application runs on RHEL
- a repeatable benchmark can be defined
- the application exposes a safe inspection/apply/verify surface
- the plugin defines the application-specific rules the generic runtime cannot infer

The desired shape is:

1. generic runtime owns the workflow
2. service plugin owns service-specific behavior
3. platform plugin owns RHEL/system behavior
4. benchmark and validation logic remain deterministic in Python

---

## Current State

The repo is now significantly more modular than before:

- typed runtime contracts exist
- LangGraph node topology exists
- service and platform seams exist
- reporting and memory have clearer boundaries
- the old monolith is reduced into compatibility modules

But the system is still closer to:

- adapter-extensible

than to:

- easy customer-plugin framework

---

## Main Gaps

### 1. Service adapter contract is still too low-level

Current service behavior is still centered around methods like:

- `apply_config(parameter, value)`
- `reload()`
- `get_logs()`
- `get_metrics()`

This is not enough for a generic customer-application plugin.

What is missing:

- declared benchmark support
- declared verification strength
- restart/reload semantics
- rollback capability
- safe-remediation policy
- application health model
- config discovery model

---

### 2. Benchmarking is still too Nginx-shaped

The runtime still assumes workload names and benchmark behavior that fit the hackathon
Nginx case well:

- URL-based benchmarking
- `small` / `medium` / `large`
- a `benchmark(duration, url)` style interface

That does not generalize cleanly to:

- databases
- caches
- JVM services
- queues
- RPC services
- worker processes

What is missing:

- a first-class benchmark spec
- benchmark metric definitions
- workload declarations
- pass/fail rules
- baseline comparison rules

---

### 3. Plugin onboarding is code-driven instead of contract-driven

New service onboarding currently means writing Python modules that fit internal repo
expectations.

What is missing:

- plugin manifest
- capability declaration
- onboarding checklist
- runtime validation of required hooks
- versioned plugin contract

This makes new application support harder than it should be.

---

### 4. Runtime dependencies still assume one mixed-responsibility adapter

The current runtime dependency model still centers on a single service adapter object.

That object is doing too much conceptually:

- inspection
- benchmark
- apply
- verify
- reload/restart behavior
- config ownership

Longer term, the plugin model will likely need either:

- clearer service sub-components

or:

- a richer declared service capability contract

so the adapter does not become another god object.

---

### 5. Service modules still wrap legacy Nginx behavior

The current Nginx service modules are an important migration seam, but they are not
yet a clean example of a fully independent plugin implementation.

That means the repo does not yet demonstrate the final onboarding story for a second
application.

---

### 6. Apply categories are not yet plugin-extensible enough

The apply engine currently encodes built-in categories such as:

- `resource_limits`
- `kernel`
- `network`
- `storage`
- `webserver`

This is much better than the old monolith, but the final product goal requires
service-owned categories to be extensible.

Examples of future categories:

- `database`
- `cache`
- `runtime`
- `threadpool`
- `gc`
- `connection_pool`

Right now `webserver` is still a built-in assumption.

---

### 7. Platform contract is still too thin

The RHEL/platform seam exists, but it is still narrow.

What is likely still needed for a real generic customer platform:

- system checks
- fingerprinting
- IRQ/NUMA capability detection
- resource discovery
- workload-sensitive guardrails
- rollback/restart impact hints

---

### 8. No explicit safety/capability model exists yet

This is the biggest product gap.

Before a customer plugin can be considered safe, the runtime should know things like:

- benchmark supported: yes/no
- automatic apply allowed: yes/no
- restart allowed: yes/no
- reload supported: yes/no
- rollback supported: yes/no
- verification strength: weak/strong
- maintenance window required: yes/no

Today most of that is implicit in the implementation.

That is not strong enough for generic customer onboarding.

---

### 9. Planner does not yet consume a service-declared tuning surface

The planner now returns typed plans, which is good.

But the system still lacks a canonical plugin-owned declaration of:

- supported tunables
- valid ranges
- risk levels
- runtime-safe vs restart-required settings
- recommended validation method per setting

Without that, application onboarding will still require scattered Python logic.

---

### 10. The repo is not yet “easy plugin” ready

The architecture is much better than before, but adding a second customer application
would still require understanding internal repo structure and writing multiple modules.

That means the architecture is:

- modular
- adapter-friendly

but not yet:

- easy-plugin
- declarative
- productized for broad customer onboarding

---

## What The Future Plugin Contract Likely Needs

### 1. Plugin Manifest

A plugin should declare:

- application name
- service type
- benchmark mode
- verification strength
- restart/reload policy
- safe-remediation flags
- config locations
- required binaries/commands

### 2. Service Plugin Interface

A service plugin should own:

- `preflight`
- `inspect`
- `benchmark`
- `apply`
- `verify`
- `effective_config`
- `health_checks`

### 3. Benchmark Spec

A benchmark spec should declare:

- workloads
- commands or endpoints
- duration/default settings
- primary success metric
- acceptable regression logic
- baseline comparison behavior

### 4. Tuning Surface

A tuning surface should declare:

- supported parameters
- categories
- valid ranges or enumerations
- risk levels
- whether restart is required
- whether rollback is possible

### 5. Safety Policy

A safety policy should declare:

- automatic apply allowed or not
- restart allowed or not
- rollback available or not
- when human approval is required
- what verification strength is required before benchmark comparison

---

## Recommendation

Do not fold this work into the current architecture migration yet.

The correct sequence is:

1. finish making the modular runtime authoritative
2. reduce remaining migration seams
3. then design the plugin contract explicitly
4. only after that, onboard a second real service as proof

The best proof point will not be more Nginx abstraction.
It will be adding one non-Nginx service through the new plugin contract.

---

## Bottom Line

The current codebase is now structurally capable of evolving into a generic RHEL
performance diagnosis framework.

The main remaining gap for that product goal is not more file-splitting.
It is a formal plugin contract that makes application onboarding:

- declarative
- benchmark-aware
- safety-aware
- easy to validate
- easy to extend
