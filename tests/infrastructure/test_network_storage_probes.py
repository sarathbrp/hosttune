from preflight.domain.models import CommandResult
from preflight.infrastructure.parsers.network_parser import NetworkParser
from preflight.infrastructure.parsers.storage_parser import StorageParser
from preflight.infrastructure.probes.network_probe import NetworkProbe
from preflight.infrastructure.probes.storage_probe import StorageProbe


class FakeExecutor:
    def __init__(self, responses: dict[str, CommandResult]) -> None:
        self._responses = responses

    def run(self, command: str) -> CommandResult:
        return self._responses[command]


def test_network_probe_collects_network_info() -> None:
    probe = NetworkProbe(parser=NetworkParser())
    executor = FakeExecutor(
        {
            "ip route | awk '/default/ {print $5; exit}'": CommandResult("iface", 0, "eth0", ""),
            "ethtool -i eth0 || true": CommandResult(
                "driver", 0, "driver: ixgbe\nfirmware-version: 1.2.3", ""
            ),
            "ethtool -g eth0 || true": CommandResult(
                "ring",
                0,
                "\n".join(
                    [
                        "Pre-set maximums:",
                        "RX: 4096",
                        "TX: 4096",
                        "Current hardware settings:",
                        "RX: 512",
                        "TX: 512",
                    ]
                ),
                "",
            ),
            "ethtool -l eth0 || true": CommandResult("queue", 0, "Combined: 8", ""),
        }
    )

    network = probe.collect(executor)

    assert network.interface_name == "eth0"
    assert network.ring_buffer_tuning_supported is True


def test_storage_probe_collects_storage_info() -> None:
    probe = StorageProbe(parser=StorageParser())
    executor = FakeExecutor(
        {
            "lsblk -ndo PKNAME $(findmnt -n -o SOURCE /) 2>/dev/null | head -n 1": CommandResult(
                "device", 0, "sda", ""
            ),
            "cat /sys/block/sda/queue/rotational 2>/dev/null || true": CommandResult(
                "rot", 0, "0", ""
            ),
            "cat /sys/block/sda/queue/scheduler 2>/dev/null || true": CommandResult(
                "sched", 0, "[mq-deadline] none", ""
            ),
        }
    )

    storage = probe.collect(executor)

    assert storage.device_name == "sda"
    assert storage.scheduler_meaningful is True
