from __future__ import annotations

from mllm_comp.registry.catalog import (
    CapabilityKind,
    CapabilityStatus,
    active_providers,
    get_capability_spec,
)


def test_v2_algorithms_are_discoverable_without_global_activation() -> None:
    for name in ("tome", "visionzip", "pyramid_drop"):
        spec = get_capability_spec(CapabilityKind.COMPRESSOR, name)
        assert spec.status is CapabilityStatus.TARGET
        assert spec.metadata["contract_version"] == 2
        image_units = [
            item
            for item in spec.metadata["support_units"]
            if ":image:" in item["identity"]
        ]
        assert len(image_units) == 6
        statuses = {
            item["status"] for item in image_units
        }
        assert statuses == (
            {"active"}
            if name == "pyramid_drop"
            else {"declared", "active"}
        )
        assert sum(
            item["identity"].endswith(":forward")
            and item["status"] == "active"
            for item in image_units
        ) == 3
        assert spec.metadata["activation_gate"] == (
            "implemented_executor_and_valid_contract"
        )

    d2pruner = get_capability_spec(CapabilityKind.COMPRESSOR, "d2pruner")
    assert d2pruner.status is CapabilityStatus.TARGET
    assert {
        item["status"] for item in d2pruner.metadata["support_units"]
    } == {"declared"}

    providers = active_providers(CapabilityKind.COMPRESSOR)
    assert "tome" not in providers
    assert "visionzip" not in providers
    assert "pyramid_drop" not in providers
    assert "d2pruner" not in providers
