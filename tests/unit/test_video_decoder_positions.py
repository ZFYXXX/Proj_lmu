from types import SimpleNamespace

import pytest

from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.compressors.video.config import resolve_canonical_video_config
from mllm_comp.compressors.video.positions import video_decoder_positions
from mllm_comp.core.errors import ConfigError
from mllm_comp.run_manifest import build_run_manifest
from tests.unit.test_manifest_v2_compression_records import _config


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("algorithm,projector,target,sources,layers", [
    ("metok", "event_pool", "prefill_attention_prune", (3, 10, 19), [3, 10, 19]),
    ("prunevid", "temporal_spatial_reduce", "vtp_prefill_prune", (10,), [11]),
])
def test_video_decoder_equivalent_boundaries_and_independent_projector(backbone, algorithm, projector, target, sources, layers):
    binding = builtin_multimodal_compression_contracts().binding(f"{algorithm}.video.{backbone}")
    for operation in ({}, {"targets": {projector: {"target_site": "projector_output"}, target: {
        "target_site": "llm_decoder_layer_output_video_tokens", "layers": layers, "layer_edge": "before"}}},
        {"targets": {target: {"target_site": "llm_decoder_layer_output_video_tokens", "layers": [i - 1 for i in layers]}}}):
        assert binding.position_resolver(stage=SimpleNamespace(operation=operation), binding=binding) is binding
    for move_projector in (False, True):
        stage = SimpleNamespace(operation={"targets": {
            projector: {"target_site": "projector_input" if move_projector else "projector_output"},
            target: {"target_site": "llm_decoder_layer_post_attention_video_tokens", "layers": list(sources)},
        }})
        resolved = binding.position_resolver(stage=stage, binding=binding)
        assert resolved.metadata["position_changed"] is True
        assert resolved.dependencies == binding.dependencies
        assert [(t.name, t.order, t.required_handles) for t in resolved.targets] == [(t.name, t.order, t.required_handles) for t in binding.targets]
        assert resolved.targets[1].metadata["application_positions"] == [
            {"boundary": "post_attention_pre_ffn", "index": i} for i in sources]
        assert resolved.targets[1].metadata["score_layers"] == list(sources[:2] if algorithm == "metok" else sources)
        assert resolved.targets[0].target_site != binding.targets[0].target_site if move_projector else resolved.targets[0] is binding.targets[0]


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_video_decoder_initial_terminal_and_score_free_final_stage(backbone):
    metok = SimpleNamespace(operation={"targets": {"prefill_attention_prune": {
        "target_site": "llm_decoder_layer_output_video_tokens", "layers": [0, 2, 4], "layer_edge": "before"}}})
    # The third historical marker is not a scoring module and need not exist.
    points = video_decoder_positions(metok, (0, 2, 19), 4, algorithm="metok", backbone=backbone)
    assert [p.index for p in points] == [0, 2, 4]
    prunevid = SimpleNamespace(operation={"targets": {"vtp_prefill_prune": {
        "target_site": "llm_decoder_layer_output_video_tokens", "layers": [0], "layer_edge": "before"}}})
    assert video_decoder_positions(prunevid, (0,), 4, algorithm="prunevid", backbone=backbone)[0].index == 0
    for stage, algorithm, sources in ((metok, "metok", (0, 3, 19)), (prunevid, "prunevid", (1,))):
        with pytest.raises(ValueError, match="scoring"):
            video_decoder_positions(stage, sources, 4, algorithm=algorithm, backbone=backbone)
    with pytest.raises(ValueError, match="outside"):
        video_decoder_positions(metok, (0, 2, 19), 3, algorithm="metok", backbone=backbone)


def test_video_existing_source_layer_fields_vary_without_changing_rules():
    for algorithm, key, value in (("metok", "pruning_layers", [0, 2, 5]), ("prunevid", "vtp_selected_layer", 0)):
        canonical = resolve_canonical_video_config(algorithm)
        resolved = resolve_canonical_video_config(algorithm, {key: value})
        assert {k: v for k, v in resolved.items() if k != key} == {k: v for k, v in canonical.items() if k != key}
        invalid = ([], [0, 1], [1, 0, 2], [0, 0, 2], [False, 1, 2]) if algorithm == "metok" else (-1, True, 0.5)
        for item in invalid:
            with pytest.raises(ConfigError):
                resolve_canonical_video_config(algorithm, {key: item})
        with pytest.raises(ConfigError, match="non-canonical"):
            resolve_canonical_video_config(algorithm, {"ratio" if algorithm == "metok" else "vtp_alpha": 0.1})


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("algorithm,target,config", [
    ("metok", "prefill_attention_prune", {"pruning_layers": [0, 2, 4]}),
    ("prunevid", "vtp_prefill_prune", {"vtp_selected_layer": 0}),
])
def test_video_decoder_manifest_source_changes_keep_history_but_scope_evidence(backbone, algorithm, target, config):
    stage = {"name": "video-decoder-position", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.video",
        "binding": f"{algorithm}.video.{backbone}", "target_modalities": ["video"]}
    baseline = build_run_manifest(_config(stage, backbone_name=backbone)).compression_stages[0]
    moved = build_run_manifest(_config({**stage, "config": config}, backbone_name=backbone)).compression_stages[0]
    assert moved.metadata["canonical_evidence_applicable"] is False
    assert moved.support_status == "declared"
    assert moved.metadata["score_layers"][f"{algorithm}.video.{backbone}:{target}"] == ([0, 2] if algorithm == "metok" else [0])
    assert [{k: v for k, v in item.items() if k != "canonical_evidence_applicable"} for item in moved.evidence] == baseline.evidence
    if algorithm == "metok":
        equivalent = {**stage, "config": {"pruning_layers": [3, 10, 50]}, "operation": {"targets": {target: {
            "target_site": "llm_decoder_layer_output_video_tokens", "layers": [3, 10, 19], "layer_edge": "before"}}}}
        same = build_run_manifest(_config(equivalent, backbone_name=backbone)).compression_stages[0]
        assert same.evidence == baseline.evidence and same.support_status == baseline.support_status
