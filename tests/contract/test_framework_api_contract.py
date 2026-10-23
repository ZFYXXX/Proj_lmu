"""Contracts against installed lmms-eval and ms-swift APIs.

These tests skip when the optional framework is not installed. On the remote
``mllm-comp-platform`` environment they are required, not optional evidence.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


def test_installed_lmms_eval_accepts_platform_model_entrypoint() -> None:
    if importlib.util.find_spec("lmms_eval") is None:
        pytest.skip("lmms-eval is not installed in this environment")
    lmms_module = importlib.import_module("lmms_eval.api.model")
    plugin = importlib.import_module("mllm_comp.integrations.lmms_eval.plugin")
    manifests = {manifest.model_id: manifest for manifest in plugin.get_model_manifests()}

    assert issubclass(plugin.MLLMCompLMM, lmms_module.lmms)
    assert issubclass(plugin.MLLMCompChatLMM, lmms_module.lmms)
    assert set(manifests) == {
        "mllm_comp_llava_onevision",
        "mllm_comp_qwen25_vl",
        "mllm_comp_qwen3_vl",
        "qwen2_5_omni",
    }
    assert manifests["qwen2_5_omni"].simple_class_path.endswith(".MLLMCompLMM")
    assert manifests["qwen2_5_omni"].chat_class_path.endswith(".MLLMCompChatLMM")
    assert manifests["mllm_comp_llava_onevision"].simple_class_path.endswith(
        ".MLLMCompLMM"
    )
    assert manifests["mllm_comp_qwen25_vl"].simple_class_path.endswith(
        ".MLLMCompLMM"
    )


def test_installed_ms_swift_registers_native_qwen25_omni_and_sft_entrypoint() -> None:
    if importlib.util.find_spec("swift") is None:
        pytest.skip("ms-swift is not installed in this environment")
    swift_model = importlib.import_module("swift.model")
    sft_module = importlib.import_module("swift.pipelines.train.sft")

    meta = swift_model.MODEL_MAPPING["qwen2_5_omni"]

    assert callable(sft_module.sft_main)
    assert meta.model_arch.arch_name == "qwen2_5_omni"
    assert {"vision", "video", "audio"}.issubset(set(meta.tags))
    assert meta.loader.__name__ == "Qwen2_5OmniLoader"


def test_installed_frameworks_expose_qwen3_vl_contracts() -> None:
    if importlib.util.find_spec("lmms_eval") is None:
        pytest.skip("lmms-eval is not installed in this environment")
    if importlib.util.find_spec("swift") is None:
        pytest.skip("ms-swift is not installed in this environment")

    plugin = importlib.import_module("mllm_comp.integrations.lmms_eval.plugin")
    swift_model = importlib.import_module("swift.model")
    manifests = {manifest.model_id: manifest for manifest in plugin.get_model_manifests()}
    meta = swift_model.MODEL_MAPPING["qwen3_vl"]

    assert manifests["mllm_comp_qwen3_vl"].simple_class_path.endswith(
        ".MLLMCompLMM"
    )
    assert manifests["mllm_comp_qwen3_vl"].chat_class_path.endswith(
        ".MLLMCompChatLMM"
    )
    assert meta.model_arch.arch_name == "qwen3_vl"
    assert {"vision", "video"}.issubset(set(meta.tags))
    assert meta.loader.__name__ == "Qwen3VLLoader"
