from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.backbones.qwen25_omni.v2_compression import (
    apply_qwen25_omni_operation_compression,
)
from mllm_comp.backbones.qwen25_omni.implementation import Qwen25OmniModelAdapter
from mllm_comp.core.errors import InvariantError
from mllm_comp.export.operation_export import export_operation_actions
from mllm_comp.pipeline.stage import CompressionStageConfig


torch = pytest.importorskip("torch")


class _Layer(torch.nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.self_attn = torch.nn.Linear(4, 4, bias=False)
        self.self_attn.layer_idx = index
        self.mlp = torch.nn.Linear(4, 4, bias=False)

    def forward(self, value):
        return value + self.mlp(self.self_attn(value))


class _Model(torch.nn.Module):
    def __init__(self, num_layers: int = 6) -> None:
        super().__init__()
        text_config = SimpleNamespace(
            num_hidden_layers=num_layers,
            layer_types=[
                "full_attention" if index % 2 == 0 else "sliding_attention"
                for index in range(num_layers)
            ],
        )
        self.thinker = torch.nn.Module()
        self.thinker.config = SimpleNamespace(text_config=text_config)
        self.thinker.model = torch.nn.Module()
        self.thinker.model.config = text_config
        self.thinker.model.layers = torch.nn.ModuleList(
            [_Layer(index) for index in range(num_layers)]
        )
        self.thinker.model.norm = torch.nn.LayerNorm(4)
        self.thinker.lm_head = torch.nn.Linear(4, 3, bias=False)
        self.config = SimpleNamespace(
            thinker_config=SimpleNamespace(text_config=text_config)
        )

    def forward(self, value):
        for layer in self.thinker.model.layers:
            value = layer(value)
        return self.thinker.lm_head(self.thinker.model.norm(value))

    def save_pretrained(self, destination, safe_serialization=True):
        del safe_serialization
        destination = Path(destination)
        torch.save(self.state_dict(), destination / "pytorch_model.bin")
        (destination / "config.json").write_text(
            '{"num_hidden_layers": %d}'
            % self.thinker.model.config.num_hidden_layers,
            encoding="utf-8",
        )


def _stage(layers=(1, 4)) -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "spade_operation_only",
            "placement": "llm_level",
            "execution_phase": "prepare",
            "plugin": "structured_llm_layer_pruning",
            "algorithm": "structured_llm_layer_pruning",
            "variant": "structured_llm_layer_pruning.audio",
            "binding": "structured_llm_layer_pruning.audio.qwen25_omni",
            "method": "operation_compression",
            "object_family": "module_unit",
            "object_subtype": "transformer_layer",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "none",
            "operation": {
                "type": "prune",
                "target_site": "llm.layers",
                "target_scope": "llm",
                "target_kind": "layer",
                "apply_mode": "structural_rewrite",
                "shape_effect": "remove_module",
            },
            "criterion": {"selection_source": "explicit"},
            "activation_scope": {"layers": list(layers)},
        }
    )


def test_operation_applier_rewrites_reindexes_and_backpropagates() -> None:
    model = _Model()
    plan, result = apply_qwen25_omni_operation_compression(model, _stage())

    assert len(plan.operation_actions) == 2
    assert len(model.thinker.model.layers) == 4
    assert model.thinker.model.config.num_hidden_layers == 4
    assert model.thinker.model.config.layer_types == [
        "full_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ]
    assert [layer.self_attn.layer_idx for layer in model.thinker.model.layers] == [
        0,
        1,
        2,
        3,
    ]
    assert result.metadata["old_to_new_layer_mapping"] == {
        "0": 0,
        "1": None,
        "2": 1,
        "3": 2,
        "4": None,
        "5": 3,
    }
    assert result.metadata["parameter_reduction"] > 0
    assert result.metadata["state_dict_layer_indices_contiguous"] is True
    assert result.metadata["config_layer_types_reindexed"] is True
    assert model.config.mllm_comp_operations == [result.metadata]
    assert model.thinker.config.mllm_comp_operations == [result.metadata]
    value = torch.randn(2, 3, 4, requires_grad=True)
    loss = model(value).float().square().mean()
    loss.backward()
    assert value.grad is not None and value.grad.abs().sum() > 0


def test_operation_strategy_rejects_removing_all_layers() -> None:
    model = _Model(num_layers=2)
    adapter = Qwen25OmniModelAdapter()
    handles = adapter.structural_handles(model)
    plugin = __import__(
        "mllm_comp.compressors.llm_level.structured_layer_pruning",
        fromlist=["StructuredLLMLayerPruning"],
    ).StructuredLLMLayerPruning()
    stage = _stage(layers=(0, 1))
    config = stage.plugin_config()
    config["_stage_name"] = stage.name
    config["_execution_phase"] = stage.execution_phase.value
    from mllm_comp.compressors.base import CompressionContext
    from mllm_comp.core.state import ForwardState

    with pytest.raises(InvariantError, match="cannot remove every LLM layer"):
        plugin.plan(
            state=ForwardState(),
            context=CompressionContext(
                phase="eval", model=model, structural_handles=handles
            ),
            config=config,
        )


def test_operation_export_writes_hash_manifest(tmp_path: Path) -> None:
    model = _Model()
    _, result = apply_qwen25_omni_operation_compression(model, _stage())
    source = tmp_path / "source"
    source.mkdir()
    (source / "spk_dict.pt").write_bytes(b"speaker-fixture")
    manifest = export_operation_actions(
        model=model,
        output_dir=tmp_path / "exported",
        apply_result=result,
        source_artifact_dir=source,
        auxiliary_artifacts=("spk_dict.pt",),
    )

    assert manifest["operation_compression"]["result_num_hidden_layers"] == 4
    assert "pytorch_model.bin" in manifest["artifact_hashes"]
    assert "spk_dict.pt" in manifest["artifact_hashes"]
    assert manifest["auxiliary_artifacts"] == ["spk_dict.pt"]
    assert len(manifest["manifest_sha256"]) == 64
    manifest_path = Path(manifest["manifest_path"])
    assert manifest_path.is_file()
    assert manifest["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
