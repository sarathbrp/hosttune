from __future__ import annotations

from dataclasses import dataclass

from onboard.domain.models import ServiceDefinition
from preflight.domain.models import CommandExecutor
from snapshot.domain.models import SnapshotResult


@dataclass
class SnapshotRunner:
    def run(self, service: ServiceDefinition, executor: CommandExecutor) -> SnapshotResult:
        snapshot_directory = service.snapshot.snapshot_storage_location
        executor.run(f"mkdir -p {snapshot_directory}")
        captured_paths: list[str] = []
        for path in service.snapshot.files_to_snapshot:
            executor.run(f"cp -a {path} {snapshot_directory}/")
            captured_paths.append(path)

        runtime_state_output = None
        if service.snapshot.runtime_state_command is not None:
            runtime_state_output = executor.run(service.snapshot.runtime_state_command).stdout

        process_state = self._capture_process_state(service, executor)
        restore_sequence = tuple(
            step.replace("{{ snapshot_dir }}", snapshot_directory)
            for step in service.snapshot.restore_sequence
        )
        return SnapshotResult(
            service_name=service.identity.service_name,
            snapshot_directory=snapshot_directory,
            captured_paths=tuple(captured_paths),
            runtime_state_output=runtime_state_output,
            process_state=process_state,
            restore_sequence=restore_sequence,
        )

    def _capture_process_state(
        self,
        service: ServiceDefinition,
        executor: CommandExecutor,
    ) -> dict[str, str]:
        process_state: dict[str, str] = {}
        if service.snapshot.process_state.pid_file is not None:
            pid_file = service.snapshot.process_state.pid_file
            process_state["pid_file"] = executor.run(f"cat {pid_file} 2>/dev/null || true").stdout
        if service.snapshot.process_state.open_connections_command is not None:
            process_state["open_connections"] = executor.run(
                service.snapshot.process_state.open_connections_command
            ).stdout
        if service.snapshot.process_state.worker_process_hint is not None:
            worker_name = service.snapshot.process_state.worker_process_hint
            process_state["worker_processes"] = executor.run(
                f"pgrep -fc {worker_name} || true"
            ).stdout
        return process_state
