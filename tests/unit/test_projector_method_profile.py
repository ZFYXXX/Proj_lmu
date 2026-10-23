import pytest

from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.core.errors import RegistryError
from mllm_comp.core.vocabulary import Placement


def test_projector_profile_is_default_and_has_only_projector_bindings(monkeypatch):
    monkeypatch.delenv("MLLM_COMP_METHOD_PROFILE", raising=False)
    registry = builtin_multimodal_compression_contracts()
    names = {a.name for a in registry.algorithms()}
    assert {"eadp", "prunesid", "mmtok", "feature_anchor_adapter"} <= names
    assert not names & {"fastv", "pyramid_drop", "tome", "saint", "sparsevlm"}
    assert all(t.placement is Placement.PROJECTOR_LEVEL for b in registry.bindings() for t in b.targets)
    bindings = {b.name for b in registry.bindings()}
    assert all(b.conflicts <= bindings for b in registry.bindings())
    assert all(u.binding in bindings for u in registry.support.find())
    with pytest.raises(RegistryError):
        registry.binding("fastv.image.qwen25_vl")


def test_full_profile_restores_legacy_without_polluting_default(monkeypatch):
    monkeypatch.setenv("MLLM_COMP_METHOD_PROFILE", "full")
    assert builtin_multimodal_compression_contracts().algorithm("fastv")
    assert builtin_multimodal_compression_contracts(profile="projector").algorithm("prunesid")
    monkeypatch.setenv("MLLM_COMP_METHOD_PROFILE", "projector")
    with pytest.raises(RegistryError):
        builtin_multimodal_compression_contracts().algorithm("fastv")
    assert builtin_multimodal_compression_contracts(profile="full").algorithm("fastv")


def test_invalid_profile_fails_loudly():
    with pytest.raises(ValueError, match="PROFILE"):
        builtin_multimodal_compression_contracts(profile="typo")
