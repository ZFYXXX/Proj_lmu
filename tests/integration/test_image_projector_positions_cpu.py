"""Native projector placement keeps defaults and changes the actual value path."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _stage(backbone, algorithm, position="projector_input", *, phase="forward"):
    config = {"visionzip": {"budget": 2},
        "prumerge": {"variant": "plus", "adaptive": False, "target_count": 2, "cluster_size": 2},
        "gap": {"target_count": 2}, "divprune": {"target_count": 2},
        "aim": {"target_count": 2}, "vispruner": {"target_count": 2},
        "saint": {"target_count": 2, "group_size": 1},
        "mmtok": {"target_count": 2},
        "zoo_prune": {"target_count": 2, "num_directions": 4, "low_rank_rank": 4}}[algorithm]
    return CompressionStageConfig.from_dict({
        "name": f"{algorithm}-projector-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": algorithm, "algorithm": algorithm, "variant": "saint.image.packed_feature_prune" if algorithm == "saint" else f"{algorithm}.image",
        "binding": f"{algorithm}.image.{backbone}", "target_modalities": ["image"], "config": config,
        "operation": {} if position is None else {"target_site": position},
    })


def _model(backbone):
    model = image_fixture.model(backbone)
    if backbone == "qwen25_vl":
        model.model.visual.fullatt_block_indexes = list(range(len(model.model.visual.blocks)))
    return model


def _projector(model, backbone):
    return model.model.mm_projector if backbone == "llava_onevision" else model.model.visual.merger


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["visionzip", "prumerge", "gap", "divprune", "aim", "vispruner"])
def test_native_projector_positions_and_default_alias(backbone, algorithm, model_factory=_model):
    torch = pytest.importorskip("torch")
    torch.manual_seed(371)
    base = model_factory(backbone).train()
    expected = None
    for position in (None, "projector_output", "projector_input"):
        model = copy.deepcopy(base)
        activation = image_fixture.activation(model, _stage(backbone, algorithm, position))
        counts = []
        handle = _projector(model, backbone).register_forward_pre_hook(
            lambda module, args: counts.append(args[0].numel() // args[0].shape[-1]))
        activation.install()
        torch.manual_seed(372)
        raw = image_fixture.inputs(torch, backbone)
        pixels = raw["images" if backbone == "llava_onevision" else "pixel_values"]
        pixels.requires_grad_(True)
        labels = torch.full_like(raw["input_ids"], -100)
        labels[0, -1] = raw["input_ids"][0, -1]
        try:
            output = model(**raw, labels=labels, use_cache=False)
            output.loss.backward()
            assert model.lm_head.weight.grad is not None and torch.isfinite(model.lm_head.weight.grad).all()
            assert pixels.grad is not None and pixels.grad.abs().sum() > 0 and torch.isfinite(pixels.grad).all()
            gradients = {name: parameter.grad.detach().clone() for name, parameter in _projector(model, backbone).named_parameters() if parameter.grad is not None}
            assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
            assert any(value.abs().sum() > 0 for value in gradients.values())
            if expected is None:
                expected = (output.logits.detach(), gradients, counts[:])
            elif position == "projector_output":
                torch.testing.assert_close(output.logits, expected[0], rtol=0, atol=0)
                assert gradients.keys() == expected[1].keys()
                for name, gradient in gradients.items():
                    torch.testing.assert_close(gradient, expected[1][name], rtol=0, atol=0)
                assert counts == expected[2]
            else:
                assert min(counts) < min(expected[2]), (counts, expected[2])
                records = activation.runtime_metadata()["layout_delta_records"]
                assert records and any("pre_" in str(record.get("site", record.get("target_site", "")))
                    or "pre_" in str(record.get("result_layout", {}).get("target_site", "")) for record in records)
        finally:
            handle.remove()
            activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_projector_source_and_values_on_different_gpus(backbone, algorithms=("prumerge", "gap"), stage_factory=_stage, model_factory=_model):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires the authorized two-GPU validation job.")
    from accelerate import dispatch_model
    torch.manual_seed(374)
    model = model_factory(backbone).train()
    device_map = {name: "cuda:1" for name, _ in model.named_children() if name != "model"}
    if backbone == "llava_onevision":
        device_map.update({f"model.{name}": "cuda:0" if name == "vision_tower" else "cuda:1"
            for name, _ in model.model.named_children()})
    else:
        device_map.update({f"model.{name}": "cuda:1" for name, _ in model.model.named_children() if name != "visual"})
        device_map.update({f"model.visual.{name}": "cuda:1" if name in {"merger", "deepstack_merger_list"} else "cuda:0"
            for name, _ in model.model.visual.named_children()})
    dispatch_model(model, device_map=device_map)
    # The same external cues serve values on different devices. Qwen3 also
    # uses different physical positions for its main and remaining branches.
    for algorithm in algorithms:
        stage = stage_factory(backbone, algorithm)
        if backbone == "qwen3_vl" and algorithm != "tome":
            target = {"prumerge": "deepstack_adaptive_merge", "gap": "deepstack_grounding_aware_prune",
                "vispruner": "deepstack_image_features", "saint": "deepstack_image_features",
                "mmtok": "deepstack_image_features", "zoo_prune": "deepstack_image_features",
                "nuwa": "stage1_deepstack_aggregation", "restore": "shared_deepstack_distinctive_anchor_merge"}[algorithm]
            stage.operation = {"targets": {target: {"target_site": "projector_input"}}}
        activation = image_fixture.activation(model, stage)
        activation.install()
        raw = {name: value.detach().to("cuda:1" if backbone == "llava_onevision" and name != "images" else "cuda:0") if torch.is_tensor(value) else value
            for name, value in image_fixture.inputs(torch, backbone).items()}
        pixels = raw["images" if backbone == "llava_onevision" else "pixel_values"]
        pixels.requires_grad_(True)
        labels = torch.full_like(raw["input_ids"], -100)
        labels[0, -1] = raw["input_ids"][0, -1]
        try:
            output = model(**raw, labels=labels, use_cache=False)
            output.loss.backward()
            assert torch.isfinite(output.loss)
            assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in _projector(model, backbone).parameters())
        finally:
            activation.uninstall()
            model.zero_grad(set_to_none=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_vispruner_projector_source_and_values_on_different_gpus(backbone):
    test_projector_source_and_values_on_different_gpus(backbone, algorithms=("vispruner",))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "llava_onevision"])
def test_saint_projector_default_alias_and_actual_values(backbone):
    test_native_projector_positions_and_default_alias(backbone, "saint")


@pytest.mark.parametrize("backbone", ["qwen3_vl", "llava_onevision"])
def test_saint_projector_source_and_values_on_different_gpus(backbone):
    test_projector_source_and_values_on_different_gpus(backbone, algorithms=("saint",))


def test_saint_encoder_compact_values_to_projector_on_other_gpu():
    from tests.integration.test_saint_qwen25_vl_positions_cpu import _stage as encoder_stage
    test_projector_source_and_values_on_different_gpus("qwen25_vl", algorithms=("saint",),
        stage_factory=lambda backbone, algorithm: encoder_stage((0, 2), mid=True))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_tome_output_source_and_values_on_different_gpus(backbone):
    from tests.integration.test_projector_tome_qwen3_vl_positions_cpu import _stage as tome_stage

    for position in ("projector_input", "projector_output"):
        test_projector_source_and_values_on_different_gpus(backbone, algorithms=("tome",),
            stage_factory=lambda base, algorithm: tome_stage(position, backbone=base))


def _mmtok_zoo_model(backbone):
    model = _model(backbone)
    if backbone == "llava_onevision":
        torch = pytest.importorskip("torch")
        linear = model.model.mm_projector
        model.model.mm_projector = torch.nn.Sequential(
            linear, torch.nn.GELU(), torch.nn.Linear(linear.out_features, linear.out_features))
    return model


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["mmtok", "zoo_prune"])
def test_mmtok_zoo_projector_default_alias_and_actual_values(backbone, algorithm):
    test_native_projector_positions_and_default_alias(backbone, algorithm, model_factory=_mmtok_zoo_model)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_mmtok_zoo_projector_sources_and_values_on_different_gpus(backbone):
    test_projector_source_and_values_on_different_gpus(backbone, algorithms=("mmtok", "zoo_prune"), model_factory=_mmtok_zoo_model)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("position", ["projector_input", "projector_output"])
def test_zoo_full_proxy_uses_offload_execution_device(backbone, position):
    torch = pytest.importorskip("torch")
    from accelerate import cpu_offload
    torch.manual_seed(379)
    base = _mmtok_zoo_model(backbone).eval()
    raw = image_fixture.inputs(torch, backbone)
    expected = None
    for offloaded in (False, True):
        model = copy.deepcopy(base)
        stage = _stage(backbone, "zoo_prune", position)
        stage.config.update(projector_proxy="full")
        projector = _projector(model, backbone)
        if offloaded:
            cpu_offload(projector, execution_device=torch.device("cpu"))
            assert all(parameter.device.type == "meta" for parameter in projector.parameters())
        activation = image_fixture.activation(model, stage)
        activation.install()
        try:
            with torch.no_grad():
                output = model(**raw, use_cache=False)
            if expected is None:
                expected = output.logits
            else:
                torch.testing.assert_close(output.logits, expected, rtol=0, atol=0)
                assert all(parameter.device.type == "meta" for parameter in projector.parameters())
        finally:
            activation.uninstall()


def test_zoo_full_proxy_accepts_integer_zero_offload_device():
    torch = pytest.importorskip("torch")
    from accelerate import cpu_offload
    if not torch.cuda.is_available():
        pytest.skip("Requires the authorized GPU validation job.")
    model = _mmtok_zoo_model("llava_onevision").eval()
    projector = _projector(model, "llava_onevision")
    cpu_offload(projector, execution_device=0)
    stage = _stage("llava_onevision", "zoo_prune")
    stage.config.update(projector_proxy="full")
    activation = image_fixture.activation(model, stage)
    activation.install()
    try:
        with torch.no_grad():
            output = model(**image_fixture.inputs(torch, "llava_onevision"), use_cache=False)
        assert torch.isfinite(output.logits).all()
        assert all(parameter.device.type == "meta" for parameter in projector.parameters())
    finally:
        activation.uninstall()
