from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandResult, NetworkInfo


@dataclass(frozen=True)
class NetworkParser:
    def parse(
        self,
        interface_name: CommandResult,
        driver_info: CommandResult,
        ring_info: CommandResult,
        queue_info: CommandResult,
    ) -> NetworkInfo:
        interface = interface_name.stdout or "unknown"
        driver_values = self._parse_key_value_lines(driver_info.stdout)
        ring_values = self._parse_ring_lines(ring_info.stdout)
        combined_queues = self._extract_combined_queues(queue_info.stdout)
        return NetworkInfo(
            interface_name=interface,
            driver_name=driver_values.get("driver", "unknown"),
            firmware_version=driver_values.get("firmware-version", "unknown"),
            rx_ring_current=self._safe_int(ring_values.get("current_rx", "0")),
            rx_ring_max=self._safe_int(ring_values.get("max_rx", "0")),
            tx_ring_current=self._safe_int(ring_values.get("current_tx", "0")),
            tx_ring_max=self._safe_int(ring_values.get("max_tx", "0")),
            combined_queues=combined_queues,
            ring_buffer_tuning_supported=self._safe_int(ring_values.get("max_rx", "0")) > 0
            or self._safe_int(ring_values.get("max_tx", "0")) > 0,
        )

    def _parse_key_value_lines(self, output: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", maxsplit=1)
            parsed[key.strip()] = value.strip()
        return parsed

    def _parse_ring_lines(self, output: str) -> dict[str, str]:
        parsed = {
            "max_rx": "0",
            "max_tx": "0",
            "current_rx": "0",
            "current_tx": "0",
        }
        section = ""
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if line == "Pre-set maximums:":
                section = "max"
                continue
            if line == "Current hardware settings:":
                section = "current"
                continue
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", maxsplit=1)
            normalized_key = key.strip().lower()
            normalized_value = value.strip()
            if normalized_key == "rx":
                parsed[f"{section}_rx"] = normalized_value
            if normalized_key == "tx":
                parsed[f"{section}_tx"] = normalized_value
        return parsed

    def _extract_combined_queues(self, output: str) -> int:
        section = ""
        for line in output.splitlines():
            normalized = line.strip()
            if normalized == "Pre-set maximums:":
                section = "max"
                continue
            if normalized == "Current hardware settings:":
                section = "current"
                continue
            if section == "current" and normalized.lower().startswith("combined:"):
                return self._safe_int(line.split(":", maxsplit=1)[1].strip())
        return 0

    def _safe_int(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return 0
