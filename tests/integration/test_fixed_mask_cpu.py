"""Native model oracles for fixed masks, training, and layout composition."""
import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures import native_image as native
from mllm_comp.fixed_mask.runtime import install_fixed_mask, masked_attention, make_layout, load_fixed_mask, save_fixed_mask
from mllm_comp.fixed_mask.spec import normalize_spec
from mllm_comp.pipeline.stage import CompressionStageConfig


def mask_stage(backbone, spec=None):
    return CompressionStageConfig.from_dict({
        "name": "fixed", "placement": "llm_level", "execution_phase": "prepare",
        "plugin": "fixed_mask", "algorithm": "fixed_mask", "variant": "fixed_mask.image_text",
        "binding": f"fixed_mask.image_text.{backbone}", "target_modalities": ["image", "text"],
        "coupling_policy": "synchronized",
        "method": "operation_compression", "object_family": "module_unit",
        "operation": {"type": "mask", "apply_mode": "static_mask", "shape_effect": "preserve_shape"},
        "config": spec or {"t2v": {"0": [0]}, "ffn_vision": {"1": [0, 2]}},
    })


def activate(model, backbone, stages, phase="forward"):
    from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.registry.backbones import get_backbone
    from mllm_comp.backbones.base import BackboneRuntime
    runtime = BackboneRuntime(model=model, processor=None, model_path_or_id="tiny-local", metadata={})
    registry = builtin_multimodal_compression_contracts()
    for stage in stages:
        variant = registry.variant(stage.variant)
        stage.conditioning_modalities = set(variant.conditioning_modalities)
        stage.preserved_modalities = set(variant.preserved_modalities)
        stage.token_unit = variant.token_unit
        stage.media_alignment = variant.media_alignment
    return RuntimeCompressionDispatcher(contract_registry=builtin_multimodal_compression_contracts()).activate(
        config=SimpleNamespace(pipeline=SimpleNamespace(stages=stages)), facade=get_backbone(backbone),
        runtime=runtime, compression_mode="compressed", execution_phase=phase)


def test_paths_preserve_causality_and_zero_fully_blocked_rows():
    layout = make_layout(torch.tensor([[True, True, False, False]]))
    spec = normalize_spec({"v2v": {"0": [0]}, "t2v": {"0": [1]}, "t2t": {"0": [1]}})
    bias, empty = masked_attention(None, layout, spec, 0, torch.zeros(1, 4, 8), 2)
    weights = bias.softmax(-1).masked_fill(empty[..., None], 0)
    expected = torch.tensor([[[[0., 0., 0., 0.], [0., 0., 0., 0.],
                                [1/3, 1/3, 1/3, 0.], [0.25, 0.25, 0.25, 0.25]],
                               [[1., 0., 0., 0.], [0.5, 0.5, 0., 0.],
                                [0., 0., 0., 0.], [0., 0., 0., 0.]]]])
    torch.testing.assert_close(weights, expected)


@pytest.mark.parametrize("backbone", ["qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_native_noop_all_heads_ffn_and_uninstall(backbone, backend):
    torch.manual_seed(9)
    model = native.model(backbone).eval()
    model.set_attn_implementation(backend)
    args = native.inputs(torch, backbone)
    with torch.no_grad():
        expected = model(**args, use_cache=False).logits
        session = install_fixed_mask(model, {}, backbone=backbone)
        torch.testing.assert_close(model(**args, use_cache=False).logits, expected, atol=0, rtol=0)
        session.uninstall()
        decoder = native.decoder(model, backbone)
        original_names = list(dict(model.named_parameters()))
        spec = {"heads": {str(i): list(range(layer.self_attn.config.num_attention_heads)) for i, layer in enumerate(decoder.layers)},
                "ffn_all": {str(i): list(range(layer.mlp.down_proj.in_features)) for i, layer in enumerate(decoder.layers)}}
        session = install_fixed_mask(model, spec, backbone=backbone)
        # Independent oracle: with every attention head and FFN neuron removed,
        # each residual block must be the identity.
        observations = []
        handles = [layer.register_forward_hook(lambda module, inputs, output: observations.append((inputs[0], output)))
                   for layer in decoder.layers]
        try:
            model(**args, use_cache=False)
            for before, after in observations:
                if isinstance(after, tuple):
                    after = after[0]
                torch.testing.assert_close(before, after, atol=0, rtol=0)
            assert list(dict(model.named_parameters())) == original_names
        finally:
            for handle in handles:
                handle.remove()
            session.uninstall()
        torch.testing.assert_close(model(**args, use_cache=False).logits, expected, atol=0, rtol=0)


@pytest.mark.parametrize("backbone", ["qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("composition", [None, "pyramid_drop", "zoo_prune"])
def test_native_checkpoint_gradients_and_compaction(backbone, composition):
    torch.manual_seed(11)
    base = native.model(backbone, num_decoder_layers=4).train()
    if backbone == "llava_onevision" and composition == "zoo_prune":
        base.model.mm_projector = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.GELU(), torch.nn.Linear(16, 16))
    args = native.inputs(torch, backbone)
    labels = torch.full_like(args["input_ids"], -100)
    labels[0, -1] = args["input_ids"][0, -1]
    reference = None
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        stages = [mask_stage(backbone, {"t2v": {str(i): [0] for i in range(4)},
            "ffn_vision": {str(i): [0, 1] for i in range(4)}, "heads": {"0": [1]}})]
        if composition == "pyramid_drop":
            from tests.integration.test_image_pyramid_positions_cpu import _stage
            stages.append(_stage(backbone))
        if composition == "zoo_prune":
            stages.append(CompressionStageConfig.from_dict({
                "name": "zoo", "placement": "projector_level", "execution_phase": "forward",
                "plugin": "zoo_prune", "algorithm": "zoo_prune", "variant": "zoo_prune.image", "binding": f"zoo_prune.image.{backbone}",
                "target_modalities": ["image"], "config": {"retention_ratio": 0.5, "num_directions": 2, "low_rank_rank": 4}}))
        session = activate(model, backbone, stages)
        try:
            bridge = getattr(model, "mllm_compression_bridge", None)
            if bridge is not None:
                bridge.prepare_training_forward()
            output = model(**args, labels=labels, use_cache=False)
            assert torch.isfinite(output.loss)
            output.loss.backward()
            grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            assert grads and all(torch.isfinite(g).all() for g in grads.values())
            if reference is None:
                reference = (output.logits.detach(), grads)
            else:
                torch.testing.assert_close(output.logits, reference[0])
                assert grads.keys() == reference[1].keys()
                for name in grads:
                    torch.testing.assert_close(grads[name], reference[1][name], atol=2e-5, rtol=2e-4)
        finally:
            session.uninstall()


def test_full_and_lora_save_reload(tmp_path):
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(14)
    model = native.model("qwen25_vl").eval()
    args = native.inputs(torch, "qwen25_vl")
    original = copy.deepcopy(model)
    spec = {"heads": {"0": [0]}, "t2v": {"1": [1]}, "ffn_text": {"2": [1, 2]}}
    install_fixed_mask(model, spec, backbone="qwen25_vl")
    model.save_pretrained(tmp_path / "full")
    reloaded = Qwen2_5_VLForConditionalGeneration.from_pretrained(tmp_path / "full", attn_implementation="eager").eval()
    load_fixed_mask(reloaded, tmp_path / "full")
    torch.testing.assert_close(model(**args).logits, reloaded(**args).logits)
    lora = get_peft_model(model, LoraConfig(r=2, target_modules=["o_proj", "down_proj"]))
    lora.train()
    optimizer = torch.optim.SGD(lora.parameters(), lr=0.1)
    output = lora(**args, labels=args["input_ids"], use_cache=False)
    output.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in lora.named_parameters() if "lora_" in n)
    optimizer.step()
    lora.eval()
    lora.save_pretrained(tmp_path / "adapter")
    save_fixed_mask(lora, tmp_path / "adapter")
    load_fixed_mask(original, tmp_path / "adapter")
    restored = PeftModel.from_pretrained(original, tmp_path / "adapter").eval()
    torch.testing.assert_close(lora(**args).logits, restored(**args).logits)


def test_prefill_then_cached_decode():
    model = native.model("qwen25_vl").eval()
    spec = {"heads": {"0": [0, 1, 2]}, "t2v": {"1": [0]}}
    session = install_fixed_mask(model, spec, backbone="qwen25_vl")
    args = native.inputs(torch, "qwen25_vl")
    with torch.no_grad():
        prefill = model(**args, use_cache=True)
        cache = copy.deepcopy(prefill.past_key_values)
        decode_args = dict(input_ids=torch.tensor([[2]]), past_key_values=prefill.past_key_values,
                           attention_mask=torch.ones(1, 9), use_cache=True)
        masked_decode = model(**decode_args).logits
        session.uninstall()
        decode_args["past_key_values"] = cache
        torch.testing.assert_close(masked_decode, model(**decode_args).logits, atol=0, rtol=0)


@pytest.mark.parametrize("backbone", ["qwen25_vl", "llava_onevision"])
def test_native_generation_and_anyres_layout(backbone):
    if backbone == "llava_onevision":
        from tests.integration.test_projector_llava_onevision_positions_cpu import _model
        from tests.integration.test_projector_aim_vispruner_llava_onevision_cpu import _inputs
        model, args = _model().eval(), _inputs(torch)
    else:
        model, args = native.model(backbone).eval(), native.inputs(torch, backbone)
    session = install_fixed_mask(model, {"t2v": {"0": [0]}, "ffn_vision": {"1": [0]}}, backbone=backbone)
    seen = []
    def observe(module, inputs, kwargs):
        from mllm_comp.fixed_mask.runtime import KEY
        layout = kwargs.get(KEY)
        if layout is not None:
            seen.append((layout.vision.sum().item(), layout.valid.shape[-1]))
    handle = session.layers[0].register_forward_pre_hook(observe, with_kwargs=True)
    try:
        args.pop("use_cache", None)
        if backbone == "llava_onevision":
            args["inputs"] = args.pop("input_ids")
        with torch.no_grad():
            output = model.generate(**args, max_new_tokens=2, do_sample=False, use_cache=True)
        assert output.shape[-1] >= 2
        assert len(seen) == 1
        assert seen[0][0] == (1485 if backbone == "llava_onevision" else 4)
    finally:
        handle.remove()
        session.uninstall()


@pytest.mark.parametrize("backbone", ["qwen25_vl", "llava_onevision"])
def test_swift_model_boundary_save_evidence_and_cleanup(monkeypatch, tmp_path, backbone):
    # Exercise the real platform bridge with tensors; Swift orchestration is
    # represented only by its loader/save boundary, not a claimed CLI job.
    from tests.unit.test_ms_swift_training_lifecycle import _plugin
    from mllm_comp.core.vocabulary import RuntimeMode
    from mllm_comp.fixed_mask.checkpoint import patch_swift_save
    from mllm_comp.integrations.ms_swift.adapter import _collect_training_compression_evidence
    plugin = _plugin(monkeypatch)
    monkeypatch.setenv("MLLM_COMP_TRAIN_COMPRESSION_EVIDENCE_DIR", str(tmp_path))
    model = native.model(backbone).train()
    original_forward = model.forward
    stage = mask_stage(backbone)
    recorder = plugin._EvidenceRecorder(spec_hash="fixture", stage=stage, runtime_mode=RuntimeMode.DEFAULT)
    plugin._install_image_model_bridge(model, backbone_name=backbone, stage=stage, stages=[stage],
                                      runtime_mode=RuntimeMode.DEFAULT, recorder=recorder)
    args = native.inputs(torch, backbone)
    labels = torch.full_like(args["input_ids"], -100)
    labels[0, -1] = 2
    model(**args, labels=labels).loss.backward()

    class NativeSaveBoundary:
        def _save_model(self, output_dir=None, state_dict=None):
            self.model.save_pretrained(output_dir, state_dict=state_dict)

    patch_swift_save(NativeSaveBoundary)
    trainer = NativeSaveBoundary()
    trainer.model = model
    trainer._save_model(tmp_path / "checkpoint")
    evidence = _collect_training_compression_evidence(tmp_path, expected_spec_hash="fixture",
        expected_runtime_mode="default", expected_executor_family="image_fixed_mask")
    assert evidence["verified"] and evidence["checkpoint_verified"]
    plugin._close_model_bridge(model)
    assert model.forward.__func__ is original_forward.__func__
    assert not hasattr(model, "_mllm_fixed_mask")


def test_framework_loader_restores_full_and_adapter_checkpoints(monkeypatch, tmp_path):
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import get_peft_model, LoraConfig
    from mllm_comp.runtime import loader
    from mllm_comp.registry.backbones import get_backbone
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.core.serialization import to_jsonable
    base_dir, full_dir, adapter_dir = (tmp_path / name for name in ("base", "full", "adapter"))
    native.model("qwen25_vl").save_pretrained(base_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_dir, attn_implementation="eager").eval()
    stage = mask_stage("qwen25_vl")
    install_fixed_mask(model, stage.config, backbone="qwen25_vl")
    model.save_pretrained(full_dir)
    peft = get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj"]))
    with torch.no_grad():
        for name, parameter in peft.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.01)
    peft.eval().save_pretrained(adapter_dir)
    save_fixed_mask(peft, adapter_dir)
    real_facade = get_backbone("qwen25_vl")
    facade = SimpleNamespace(capabilities=real_facade.capabilities,
        processor_adapter=SimpleNamespace(load_processor=lambda *args, **kwargs: object()),
        model_adapter=SimpleNamespace(
            load_model=lambda path, **kwargs: Qwen2_5_VLForConditionalGeneration.from_pretrained(path, attn_implementation="eager").eval(),
            structural_handles=real_facade.model_adapter.structural_handles))
    monkeypatch.setattr(loader, "get_backbone", lambda name: facade)
    config = experiment_config_from_dict({"experiment": {"name": "reload"},
        "backbone": {"name": "qwen25_vl", "model_path": str(base_dir)},
        "pipeline": {"stages": [to_jsonable(stage)]}, "eval": {"compression_mode": "compressed"}})
    args = native.inputs(torch, "qwen25_vl")
    for checkpoint in (full_dir, adapter_dir):
        _, runtime = loader.load_backbone_runtime(config, checkpoint_path=str(checkpoint), device="cpu")
        try:
            if checkpoint == adapter_dir:
                torch.testing.assert_close(runtime.model(**args).logits, peft(**args).logits)
            else:
                with peft.disable_adapter():
                    torch.testing.assert_close(runtime.model(**args).logits, peft(**args).logits)
        finally:
            runtime.close()


@pytest.mark.parametrize("backbone", ["qwen25_vl", "llava_onevision"])
def test_pyramid_generate_with_fixed_masks(backbone):
    from tests.integration.test_image_pyramid_positions_cpu import _stage
    model = native.model(backbone, num_decoder_layers=4).eval()
    session = activate(model, backbone, [mask_stage(backbone), _stage(backbone, phase="generate")], phase="generate")
    args = native.inputs(torch, backbone)
    if backbone == "llava_onevision":
        args["inputs"] = args.pop("input_ids")
    try:
        with torch.no_grad():
            output = model.generate(**args, max_new_tokens=3, do_sample=False, use_cache=True)
        assert output.shape[-1] >= 3
        assert model._mllm_fixed_mask.masked_layer_calls == len(native.decoder(model, backbone).layers)
    finally:
        session.uninstall()
