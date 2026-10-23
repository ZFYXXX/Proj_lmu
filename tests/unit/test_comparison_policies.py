import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("llava")
from tests.fixtures import native_image
from mllm_comp.compressors.comparison.operations import OperationComparison


@pytest.mark.parametrize("method", ["structural_pruning", "short_lvlm", "adallava_l", "adallava_h", "skip_vision"])
@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_calibrated_policies_train_decode_and_restore(method, tmp_path, backbone):
    torch.manual_seed(7)
    model = native_image.model(backbone, num_decoder_layers=4).train()
    decoder = native_image.decoder(model, backbone)
    hidden_size = decoder.config.hidden_size
    cfg = {"pruned_layers": [2], "original_layer_count": 4, "target_flops_retain_ratio": .75}
    if method == "skip_vision":
        cfg = {"attention_visual_retain_ratio": 1., "ffn_visual_retain_ratio": .5}
    if method == "short_lvlm":
        pairs = [{"pruned_layer": 2, "retained_layer": 3}]
        asset = tmp_path / "scp.pt"
        torch.save({"format": "short_lvlm_scp_asset_v1", "projection_pairs": pairs,
            "metadata": {"hidden_size": hidden_size}, "bases": {"3": torch.randn(hidden_size, 2) * .01}}, asset)
        cfg = {"pruned_layers": [2], "candidate_layers": [2, 3], "projection_pairs": pairs,
            "flops_reference_decoder_layers": 4, "calibration_asset_path": str(asset),
            "subspace_rank": 2, "calibrated_flops_ratio": .75, "target_flops_ratio": .75,
            "calibrated_absolute_target_error": 0}
    if method.startswith("adallava"):
        cfg = {"num_prefix_layers": 1, "target_flops_retain_ratio": .75,
            ("active_units" if method.endswith("_h") else "active_layers"): 3}
    original = decoder.layers[3].self_attn.q_proj.forward.__func__
    session = OperationComparison(model, method, cfg, backbone).install()
    try:
        inputs = native_image.inputs(torch, backbone)
        labels = torch.full_like(inputs["input_ids"], -100)
        labels[:, -1] = 3
        result = model(**inputs, labels=labels, use_cache=False)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        pixels = inputs["images"] if backbone == "llava_onevision" else inputs["pixel_values"]
        assert pixels.grad.abs().sum() > 0
        if method.startswith("adallava"):
            assert session.scheduler.mlp_head.weight.grad.abs().sum() > 0
        if method == "skip_vision":
            assert session.scheduler.score[1].weight.grad.abs().sum() > 0
        model.eval()
        with torch.no_grad():
            result = model(**inputs, use_cache=True)
            result = model(input_ids=torch.tensor([[4]]), past_key_values=result.past_key_values, use_cache=True)
            assert torch.isfinite(result.logits).all()
        if method.startswith("adallava") or method == "skip_vision":
            model.save_pretrained(tmp_path / "saved")
            saved = {k: v.clone() for k, v in session.scheduler.state_dict().items()}
            replacement = native_image.model(backbone, num_decoder_layers=4)
            reloaded = OperationComparison(replacement, method, cfg, backbone, tmp_path / "saved").install()
            try:
                for key, value in reloaded.scheduler.state_dict().items():
                    torch.testing.assert_close(value, saved[key])
            finally:
                reloaded.uninstall()
    finally:
        session.uninstall()
    assert decoder.layers[3].self_attn.q_proj.forward.__func__ is original
