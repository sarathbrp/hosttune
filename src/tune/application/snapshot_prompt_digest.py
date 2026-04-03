from __future__ import annotations

from snapshot.domain.models import SnapshotResult
from tune.application.benchmark_runtime_telemetry import truncate_for_prompt

# Keys aligned with onboard `process_state` / snapshot runner output (curated for LLM; no raw JSON).
_PROCESS_STATE_PROMPT_KEYS: tuple[str, ...] = (
    "pid_file",
    "worker_processes",
    "open_connections",
    "worker_process_hint",
)


def format_snapshot_digest_for_prompt(
    snapshot: SnapshotResult,
    *,
    max_runtime_chars: int = 800,
    max_paths_chars: int = 240,
    max_process_field_chars: int = 120,
    max_directory_chars: int = 96,
) -> str:
    """Short snapshot digest for LLM prompts."""
    dir_display = truncate_for_prompt(snapshot.snapshot_directory, max_directory_chars)
    paths_joined = ", ".join(snapshot.captured_paths)
    paths_display = truncate_for_prompt(paths_joined, max_paths_chars) if paths_joined else "none"

    lines = [
        f"snapshot_dir={dir_display}",
        f"captured_paths={paths_display}",
    ]
    proc_lines = _process_state_digest_lines(
        snapshot.process_state,
        max_process_field_chars=max_process_field_chars,
    )
    if proc_lines:
        lines.append("process_state (contract keys):")
        lines.extend(proc_lines)
    else:
        lines.append("process_state=(no contract key fields in snapshot result)")

    extra = _extra_process_state_key_count(snapshot.process_state)
    if extra:
        lines.append(f"process_state_other_keys={extra}")

    if snapshot.runtime_state_output and snapshot.runtime_state_output.strip():
        lines.append("runtime_state (truncated):")
        lines.append(truncate_for_prompt(snapshot.runtime_state_output, max_runtime_chars))
    else:
        lines.append("runtime_state=(not captured)")

    lines.append(f"restore_steps={len(snapshot.restore_sequence)}")
    return "\n".join(lines)


def _process_state_digest_lines(
    process_state: dict[str, str],
    *,
    max_process_field_chars: int,
) -> list[str]:
    lines: list[str] = []
    for key in _PROCESS_STATE_PROMPT_KEYS:
        if key not in process_state:
            continue
        raw = process_state[key].strip()
        if not raw:
            continue
        display = truncate_for_prompt(raw, max_process_field_chars)
        lines.append(f"  - {key}={display}")
    return lines


def _extra_process_state_key_count(process_state: dict[str, str]) -> int:
    allowed = frozenset(_PROCESS_STATE_PROMPT_KEYS)
    return sum(1 for k in process_state if k not in allowed)
