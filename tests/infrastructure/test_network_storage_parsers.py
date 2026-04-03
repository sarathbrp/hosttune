from preflight.domain.models import CommandResult
from preflight.infrastructure.parsers.network_parser import NetworkParser
from preflight.infrastructure.parsers.storage_parser import StorageParser


def test_network_parser_extracts_ring_and_queue_state() -> None:
    network = NetworkParser().parse(
        interface_name=CommandResult("iface", 0, "eth0", ""),
        driver_info=CommandResult("driver", 0, "driver: ixgbe\nfirmware-version: 1.2.3", ""),
        ring_info=CommandResult(
            "ring",
            0,
            "\n".join(
                [
                    "Pre-set maximums:",
                    "RX:             4096",
                    "TX:             4096",
                    "Current hardware settings:",
                    "RX:             512",
                    "TX:             512",
                ]
            ),
            "",
        ),
        queue_info=CommandResult(
            "queue",
            0,
            "\n".join(
                [
                    "Pre-set maximums:",
                    "Combined: 74",
                    "Current hardware settings:",
                    "Combined: 8",
                ]
            ),
            "",
        ),
    )

    assert network.driver_name == "ixgbe"
    assert network.rx_ring_max == 4096
    assert network.combined_queues == 8


def test_storage_parser_extracts_device_capabilities() -> None:
    storage = StorageParser().parse(
        device_name=CommandResult("device", 0, "nvme0n1", ""),
        rotational=CommandResult("rot", 0, "0", ""),
        scheduler=CommandResult("sched", 0, "[none] mq-deadline", ""),
    )

    assert storage.device_type == "nvme"
    assert storage.scheduler_meaningful is False


def test_storage_parser_detects_device_mapper_backing_disk() -> None:
    storage = StorageParser().parse(
        device_name=CommandResult("device", 0, "sda", ""),
        rotational=CommandResult("rot", 0, "0", ""),
        scheduler=CommandResult("sched", 0, "none [mq-deadline] kyber bfq", ""),
    )

    assert storage.device_name == "sda"
    assert storage.device_type == "ssd"


def test_storage_parser_detects_virtio_devices() -> None:
    storage = StorageParser().parse(
        device_name=CommandResult("device", 0, "vda", ""),
        rotational=CommandResult("rot", 0, "0", ""),
        scheduler=CommandResult("sched", 0, "[mq-deadline] none", ""),
    )

    assert storage.device_type == "virtio"
    assert storage.scheduler_meaningful is True


def test_storage_parser_detects_rotational_and_unknown_devices() -> None:
    rotational = StorageParser().parse(
        device_name=CommandResult("device", 0, "sdb", ""),
        rotational=CommandResult("rot", 0, "1", ""),
        scheduler=CommandResult("sched", 0, "mq-deadline [bfq]", ""),
    )
    unknown = StorageParser().parse(
        device_name=CommandResult("device", 0, "dm-0", ""),
        rotational=CommandResult("rot", 0, "x", ""),
        scheduler=CommandResult("sched", 0, "", ""),
    )

    assert rotational.device_type == "rotational"
    assert rotational.scheduler_meaningful is True
    assert unknown.device_type == "unknown"
    assert unknown.scheduler_meaningful is False
