from preflight.infrastructure.probes.base import BaseProbe


class ConcreteProbe(BaseProbe):
    @property
    def name(self) -> str:
        return "concrete"

    def collect(self, executor) -> object:  # type: ignore[no-untyped-def]
        _ = executor
        return {"ok": True}


def test_base_probe_contract_can_be_implemented() -> None:
    probe = ConcreteProbe()

    assert probe.name == "concrete"
    assert probe.collect(None) == {"ok": True}
