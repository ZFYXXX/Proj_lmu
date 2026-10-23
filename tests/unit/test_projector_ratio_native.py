import pytest
import torch

from tests.unit.test_projector_comparison_evaluation import tiny_state
from mllm_comp.compressors.comparison.qwen_projector import QwenProjectorComparison
from mllm_comp.compressors.comparison.anchor_adapter import QwenAnchorAdapterComparison


@pytest.mark.parametrize("method", ("cdpruner", "dppruner", "eadp", "hiprune"))
def test_ratio_budget_uses_actual_post_merger_tokens_and_records_every_layer(method):
    bundle, state, _, _ = tiny_state()
    session = QwenProjectorComparison(bundle["model"], method, {"retention_ratio": .5}, "qwen25_vl").install()
    try:
        with torch.no_grad():
            bundle["model"].generate(**state["prompt_inputs"], max_new_tokens=3, do_sample=False, pad_token_id=0)
        trace = session.runtime_metadata()["selection"]
        assert trace["source_count"] == 6
        assert trace["result_count"] == 3
        assert trace["visual_tokens_per_decoder_layer"] == [3] * 4
        assert trace["layer_averaged_retention_ratio"] == .5
    finally:
        session.uninstall()


@pytest.mark.parametrize("nonzero", (False, True))
def test_trained_mlp_native_generate_matches_mean_residual_reference(nonzero):
    from mllm_comp.analysis.common.compression_training_check import create_anchor_adapter, AnchorAdapterSpec
    from mllm_comp.analysis.common.projector_comparison import compact_prompt
    from mllm_comp.analysis.probes.visual_evidence_interaction import cached_decode
    from mllm_comp.compressors.image.fixed_visual_merge import fixed_visual_merge, FixedVisualMergeSpec
    bundle, state, _, _ = tiny_state()
    module = create_anchor_adapter(AnchorAdapterSpec(hidden_size=24, width=4)).eval()
    if nonzero:
        torch.nn.init.normal_(module.adapter[-1].weight, std=.01)
    with torch.inference_mode():
        merged = fixed_visual_merge(state["features"], grid_height=2, grid_width=3,
                                   spec=FixedVisualMergeSpec(target_count=3))
        compact = compact_prompt(state, merged.indices)
        destinations = (compact["input_ids"][0] == 10).nonzero(as_tuple=True)[0]
        compact["hidden"][0, destinations] = module({"tokens": merged.tokens})["tokens"]
        for i, members in enumerate(merged.metadata["cluster_members"]):
            source = torch.tensor([state["visual"][j] for j in members])
            compact["positions"][:, 0, destinations[i]] = state["positions"][:, 0, source].float().mean(1).round().long()
        expected = cached_decode(llm=bundle["llm"], head=bundle["model"].lm_head,
            embedding=bundle["model"].get_input_embeddings(), hidden=compact["hidden"],
            positions=compact["positions"], eos_ids=[151643], max_new=3)
        session = QwenAnchorAdapterComparison(bundle["model"], {"retention_ratio": .5}, adapter_module=module).install()
        try:
            native = bundle["model"].generate(**state["prompt_inputs"], max_new_tokens=3,
                do_sample=False, eos_token_id=151643, pad_token_id=0, repetition_penalty=1.0)
            assert native[0, state["prefix"]:].tolist() == expected["generated_token_ids"]
            assert session.runtime_metadata()["selection"]["result_count"] == 3
        finally:
            session.uninstall()


def test_ratio_rejects_conflicting_fixed_budget():
    bundle, _, _, _ = tiny_state()
    with pytest.raises(ValueError, match="not both"):
        QwenProjectorComparison(bundle["model"], "dppruner", {"retention_ratio": .25, "tokens_per_crop": 128}, "qwen25_vl")


@pytest.mark.parametrize("method", ("cdpruner", "dppruner", "eadp", "hiprune"))
def test_fixed_budget_does_not_turn_dynamic_resolution_into_a_ratio(method):
    bundle, _, _, _ = tiny_state()
    session = QwenProjectorComparison(bundle["model"], method, {"tokens_per_crop": 128}, "qwen25_vl")
    assert [session.token_budget(n) for n in (64, 128, 288, 512, 1024)] == [64, 128, 128, 128, 128]
    assert session.retention_ratio is None


def test_fixed_platform_config_contains_no_proportional_retention(tmp_path, monkeypatch):
    import json
    from scripts.eval.lmms_eval import run_projector_ratio_platform as runner
    (tmp_path / "protocol.json").write_text(json.dumps({"checkpoint": "/test/Qwen2.5-VL-7B-Instruct"}))
    monkeypatch.setattr(runner, "SOURCE", tmp_path)
    for method in runner.METHODS:
        if method == "bifocal_stage1":
            with pytest.raises(ValueError, match="no exact fixed-count"):
                runner.config(tmp_path / "run", method, "smoke", target_count=128)
            continue
        cfg = runner.config(tmp_path / "run", method, "smoke", target_count=128)
        stages = cfg["pipeline"]["stages"]
        if method == "dense":
            assert not stages
        else:
            assert stages[0]["config"]["tokens_per_crop"] == 128
            assert "retention_ratio" not in stages[0]["config"]
            assert stages[0]["name"].endswith("_k128")
