import copy
import json

import pytest
import torch

from mllm_comp.analysis.common.qwen35_transfer import create_head
from mllm_comp.compressors.comparison.qwen35_image_context import (
    METHOD, create_image_head, save_image_head, load_image_head, Qwen35ImageContextProjector,
)
from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec


def test_zero_image_context_is_the_identical_visual_prior():
    torch.manual_seed(4)
    visual = create_head("visual", 32)
    image = create_image_head(32, visual=visual)
    x = torch.randn(160, 32)
    a, b = visual(x), image(x)
    assert image.spec.context_mode == "image"
    assert torch.equal(a["scores"], b["scores"]) and torch.equal(a["indices"], b["indices"])


def test_image_context_ignores_question_after_the_branch_becomes_nonzero():
    module = create_image_head(32, visual=create_head("visual", 32))
    with torch.no_grad():
        module.context_projection.weight.normal_(0, .1)
    x = torch.randn(160, 32)
    a = module(x, question=torch.randn(4, 32))
    b = module(x, question=torch.full((30, 32), float("nan")))
    assert torch.equal(a["scores"], b["scores"]) and torch.equal(a["indices"], b["indices"])


def test_continuous_topk_trains_the_image_context_branch():
    module = create_image_head(32, visual=create_head("visual", 32))
    out = module(torch.randn(160, 32), soft=True)
    assert out["tokens"].shape == (160, 32)
    assert torch.allclose(out["soft_weights"].sum(), torch.tensor(128.), atol=1e-3)
    (out["soft_weights"] * torch.linspace(-1, 1, 160)).sum().backward()
    grad = module.context_projection.weight.grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and grad.norm() > 0


def test_image_checkpoint_roundtrip_and_wrong_base_or_mode(tmp_path):
    identity = {"hidden_size": 32, "identity_sha256": "q35"}
    module = create_image_head(32, visual=create_head("visual", 32))
    folder = tmp_path / "head"
    save_image_head(module, folder, identity=identity, provenance={"step": 320})
    actual = load_image_head(folder, identity=identity, device="cpu")
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in module.state_dict().items())
    with pytest.raises(ValueError):
        load_image_head(folder, identity={**identity, "identity_sha256": "other"}, device="cpu")
    cfg = json.loads((folder / "head_config.json").read_text())
    cfg["spec"]["context_mode"] = "question"
    (folder / "head_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_image_head(folder, identity=identity, device="cpu")


class NoQuestionTokenizer:
    def encode(self, *args, **kwargs):
        raise AssertionError("The image-only selector must not tokenize questions")


@pytest.mark.parametrize("target", [3, 128])
def test_native_generation_cache_and_identity_without_reading_question(target):
    from tests.unit.test_qwen35_transfer import tiny
    model, inputs = tiny()
    before = copy.deepcopy(inputs)
    module = ContextVisualTopK(ContextVisualTopKSpec(32, target_count=target, context_mode="image"))
    call = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0, eos_token_id=2)
    with torch.no_grad():
        dense = model.generate(**inputs, **call)
    session = Qwen35ImageContextProjector(model, {"method": METHOD, "target_count": target},
        module=module, tokenizer=NoQuestionTokenizer()).install()
    try:
        with torch.no_grad():
            actual = model.generate(**inputs, **call)
        record = session.runtime_metadata()["selection"]
        assert record["method"] == METHOD and record["reads_question"] is False
        assert record["result_count"] == min(target, 6)
        assert record["decode_position_verified"] and record["observed_visual_tokens_per_decoder_layer"] == [min(target, 6)] * 4
        assert torch.equal(actual[:, :inputs["input_ids"].shape[1]], inputs["input_ids"])
        if target == 128:
            assert torch.equal(actual, dense)
    finally:
        session.uninstall()
    assert all(torch.equal(v, inputs[k]) for k, v in before.items())


def test_eval_configuration_uses_image_head_and_no_text_conditioning(tmp_path):
    from scripts.eval.lmms_eval.run_qwen35_image_context import config
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"checkpoint": "/models/Qwen3.5-9B", "revision": "pinned"}))
    folder = tmp_path / "training/image"
    folder.mkdir(parents=True)
    (folder / "latest.json").write_text(json.dumps({"step": 320, "checkpoint": "fixed320"}))
    (folder / "manifest.json").write_text(json.dumps({"status": "complete"}))
    cfg = config(tmp_path, {"identity_path": str(identity)}, "full", "chartqa", tmp_path / "run")
    stage, = cfg["pipeline"]["stages"]
    assert stage["config"]["method"] == METHOD and not stage["conditioning_modalities"]
    assert stage["config"]["target_count"] == 128 and cfg["eval"]["sample_limit"] is None
    assert cfg["eval"]["generation"]["repetition_penalty"] == 1.05
