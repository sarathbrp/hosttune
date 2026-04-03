# Error Handling Audit -- Unstaged Changes

## CRITICAL Issues

### 1. PrlimitApplier.current_nofile_soft -- Silent ValueError Swallowing
**File:** `src/tune/application/apply_coordinator.py` lines 290-296

```python
except ValueError:
    return None
```

This bare `except ValueError` catches errors from BOTH `read_service_pid` AND `read_nofile_soft_hard`. It silently returns `None`, which the catalog builder uses as `current_value=None`. Downstream, the `PreApplyValidator` will skip the no-op check (line 22: `if candidate.current_value is not None`), meaning a hypothesis could propose the already-active value and waste a full benchmark cycle. The caller has no idea why the value could not be read -- was the PID file missing, the PID dead, or the /proc parse broken?

**Hidden errors:** stale PID file, process died between reads, /proc/limits format change, permission denied.

**Recommendation:** Log the exception before returning None, or propagate it and let the catalog builder decide to skip the candidate entirely with a logged reason.

---

### 2. HealthValidator._validate_effective_value -- Missing prlimit Validator (False Failure)
**File:** `src/tune/application/health_validator.py` lines 194-198

The method handles `sysctl.*`, `service.directive.*`, and `network.ring.*` parameter keys, but has NO branch for the new `runtime.prlimit.*` keys. It falls through to:

```python
return ValidationCheck(
    name="effective_value",
    passed=False,
    detail="No effective value validator for applied change type.",
)
```

This means every prlimit change will be reported as validation-failed, triggering an automatic rollback (`tune_engine.py` line 220). The prlimit tuning path is dead code in production -- it will apply the change, then immediately roll it back because health validation always fails.

**User impact:** Prlimit candidates will always be rejected, wasting iteration budget, with no clear error explaining that the validator simply does not support them yet.

**Recommendation:** Add a `runtime.prlimit.*` branch that reads the actual limit via `/proc/{pid}/limits` and compares, or at minimum return `passed=True` with a detail noting the check is not yet implemented (with a loud log).

---

### 3. AttributionVerifier.verify -- System Left in Rolled-Back State on Unverified Attribution
**File:** `src/tune/application/attribution_verifier.py` lines 55-62 and `src/tune/application/tune_engine.py` lines 280-300

When `verified=False` (line 55), the change was rolled back (line 28) but is **never reapplied**. Only verified changes get reapplied (line 57). The tune_engine then reaches lines 281-300 where `attribution_verification.verified` is False, so it skips `rollback_coordinator.rollback()` (line 282 condition fails) and just logs. But the change is already gone from the target system. The `active_changes` dict is not updated to remove this key either -- the state still thinks the change is active.

**User impact:** Silent state corruption. The system believes a configuration is active when it has already been reverted. Subsequent iterations will make decisions based on stale state. The hypothesis generator will see this key in `active_parameter_keys` and avoid re-proposing it.

**Recommendation:** When attribution is unverified and the change was already rolled back by the verifier, explicitly remove it from `state.active_changes` in tune_engine.py.

---

## HIGH Issues

### 4. AttributionVerifier._calculate_average_drop -- Silent Skip on Missing Workloads
**File:** `src/tune/application/attribution_verifier.py` lines 84-93

```python
if reverted_summary is None:
    continue
...
if not drops:
    return 0.0
```

If the reverted benchmark has completely different workload names (misconfiguration, race condition), every workload is silently skipped and the method returns `0.0`. Since the verification threshold check is `average_drop > expected_variance`, a 0.0 drop means `verified=False` -- the change gets rejected with no explanation of why workloads could not be matched. The summary will say something like `average_drop=0.0000` but never mention that zero workloads were actually compared.

**Recommendation:** If `drops` is empty after the loop, raise or return a result with an explicit summary like "no comparable workloads found between accepted and reverted benchmarks."

---

### 5. tune_engine._run_iteration -- No Exception Guard Around apply_coordinator.apply
**File:** `src/tune/application/tune_engine.py` line 189

If `apply_coordinator.apply()` raises (e.g., SSH timeout, permission denied), the exception propagates out of `_run_iteration` and crashes the entire tuning loop. The iteration is never recorded. The state is left partially updated. Any previously applied changes remain on the target system with no record of what happened.

**Recommendation:** Wrap the apply call in a try/except that records a failed iteration (like the pre-apply rejection path does), logs the error, and allows the loop to continue to the next iteration.

---

### 6. PhaseController._filter_by_priority_tier -- Returns Empty Tuple Silently
**File:** `src/tune/application/phase_controller.py` line 254

When all candidates at every tier have been tried, this returns `()`. The tune_engine (line 88-90) handles empty candidates by breaking the loop:

```python
if not candidates:
    self.logger.stage_detail("tune", "No eligible candidates remain for current phase.")
    break
```

However, the log message says "current phase" but the real issue might be that the tier-filtering logic exhausted candidates while the phase still has budget remaining. This is a design ambiguity rather than a bug, but the log message is misleading -- it implies no candidates exist at all when in fact candidates exist but were filtered out by tier logic.

---

## MEDIUM Issues

### 7. resolve_tuning_layer -- Unchecked StrEnum Construction
**File:** `src/tune/domain/tuning_layer.py` line 56

```python
return TuningLayer(yaml_override)
```

If `yaml_override` contains an invalid value (typo in YAML like `"kermel"`), this will raise a raw `ValueError` with no context about which parameter or YAML field caused it. The error will surface during catalog building with no indication of which service definition file has the bad value.

**Recommendation:** Wrap in a try/except that adds the parameter_key to the error message.

### 8. AttributionVerifier.verify -- Health Check Failure After Rollback Returns verified=False Without Re-applying
**File:** `src/tune/application/attribution_verifier.py` lines 33-42

If the rollback succeeds but the baseline health check fails, it returns `verified=False` with `reverted_benchmark_result=None`. The change has been rolled back. The tune_engine will reach the `attribution_verification.verified is False` branch and skip rollback (because it checks `attribution_verification is None or attribution_verification.verified`). The system is left in rolled-back state but state.active_changes is not cleaned up.

Same underlying issue as finding #3 but triggered via a different path.
