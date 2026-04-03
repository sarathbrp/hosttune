from snapshot.domain.models import SnapshotResult
from tune.application.snapshot_prompt_digest import format_snapshot_digest_for_prompt


def test_format_snapshot_digest_includes_runtime_and_key_process_state() -> None:
    snap = SnapshotResult(
        service_name="nginx",
        snapshot_directory="/tmp/snap",
        captured_paths=("/etc/nginx/nginx.conf",),
        runtime_state_output="worker_processes 4;\n",
        process_state={"pid_file": "12345\n", "worker_processes": "8\n"},
        restore_sequence=("restart",),
    )
    text = format_snapshot_digest_for_prompt(snap)
    assert "snapshot_dir=" in text
    assert "/etc/nginx/nginx.conf" in text
    assert "worker_processes" in text
    assert "pid_file=" in text
    assert "restore_steps=1" in text
    assert "process_state (contract keys):" in text


def test_format_snapshot_digest_counts_unknown_process_state_keys() -> None:
    snap = SnapshotResult(
        service_name="x",
        snapshot_directory="/d",
        captured_paths=(),
        runtime_state_output=None,
        process_state={"pid": "1", "other": "x"},
        restore_sequence=(),
    )
    text = format_snapshot_digest_for_prompt(snap)
    assert "process_state_other_keys=2" in text


def test_format_snapshot_digest_truncates_long_runtime_output() -> None:
    long_body = "x" * 5000
    snap = SnapshotResult(
        service_name="x",
        snapshot_directory="/d",
        captured_paths=(),
        runtime_state_output=long_body,
        process_state={},
        restore_sequence=(),
    )
    text = format_snapshot_digest_for_prompt(snap, max_runtime_chars=100)
    assert "truncated" in text
    assert len(text) < len(long_body)


def test_format_snapshot_digest_missing_runtime_placeholder() -> None:
    snap = SnapshotResult(
        service_name="x",
        snapshot_directory="/d",
        captured_paths=(),
        runtime_state_output=None,
        process_state={},
        restore_sequence=(),
    )
    text = format_snapshot_digest_for_prompt(snap)
    assert "runtime_state=(not captured)" in text
