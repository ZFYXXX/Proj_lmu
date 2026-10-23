"""Portable selection budgets and checkpoint/config contracts."""
import base64
import json
from types import SimpleNamespace

import pytest

from mllm_comp.fixed_mask.generate import generate_spec
from mllm_comp.fixed_mask.spec import normalize_spec
from mllm_comp.fixed_mask.checkpoint import validate_checkpoint


def artifact():
    return {"payload": {
        "scores_by_family": {"normalized_taylor": {task: {
            f"layer:{layer}/path:t2v/head:{i}": value
            for layer in (0, 1) for i, value in enumerate((3., 1., 4., 2.))}
            for task in ("a", "b")}},
        "score_vectors_by_family": {"normalized_taylor": {task: {
            f"layer:{layer}/ffn_neuron/group:image": [3., 1., 4., 2.]
            for layer in (0, 1)} for task in ("a", "b")}},
    }}


@pytest.mark.parametrize("allocation", ["layer", "global"])
def test_taylor_and_random_share_budget_and_candidate_universe(allocation):
    kwargs = dict(ratios={"t2v": .5, "ffn_vision": .5}, allocation=allocation)
    taylor = generate_spec(artifact(), **kwargs)
    random = generate_spec(artifact(), selection="random", **kwargs)
    assert taylor["t2v"] == taylor["ffn_vision"] == {"0": [1, 3], "1": [1, 3]}
    assert taylor["provenance"]["budgets"] == random["provenance"]["budgets"]
    assert random == generate_spec(artifact(), selection="random", **kwargs)
    assert random != generate_spec(artifact(), selection="random", seed=123, **kwargs)


def test_per_layer_budget_differs_from_global_budget():
    value = artifact()
    for scores in value["payload"]["scores_by_family"]["normalized_taylor"].values():
        for key in scores:
            if key.startswith("layer:1"):
                scores[key] += 100
    global_mask = generate_spec(value, {"t2v": .5}, allocation="global")
    layer_mask = generate_spec(value, {"t2v": .5}, allocation="layer")
    assert global_mask["t2v"] == {"0": [0, 1, 2, 3]}
    assert layer_mask["t2v"] == {"0": [1, 3], "1": [1, 3]}


def test_missing_and_nonfinite_measurements_are_not_silent_masks():
    value = artifact()
    value["payload"]["scores_by_family"]["normalized_taylor"]["b"].pop("layer:0/path:t2v/head:0")
    with pytest.raises(ValueError, match="candidate coverage"):
        generate_spec(value, {"t2v": .5})
    value = artifact()
    value["payload"]["score_vectors_by_family"]["normalized_taylor"]["a"]["layer:0/ffn_neuron/group:image"][0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        generate_spec(value, {"ffn_vision": .5})


@pytest.mark.parametrize("bad", [{"heads": {"0": [-1]}}, {"t2v": {"0": [1, 1]}},
    {"ffn_text": {"-1": [0]}}, {"mask_scope": "all"}, {"learned_gate": {}}, {"heads": {"0": [True]}}])
def test_invalid_masks_rejected(bad):
    with pytest.raises(ValueError):
        normalize_spec(bad)


def test_checkpoint_requires_matching_mask(tmp_path):
    spec = {"t2v": {"0": [1]}}
    (tmp_path / "fixed_mask.json").write_text(json.dumps({"backbone": "qwen25_vl", "spec": spec}))
    stages = [SimpleNamespace(enabled=True, algorithm="fixed_mask", config=spec)]
    assert validate_checkpoint(tmp_path, "qwen25_vl", stages)
    with pytest.raises(ValueError, match="requires"):
        validate_checkpoint(tmp_path, "qwen25_vl", stages, compression_mode="none")
    with pytest.raises(ValueError, match="differ"):
        validate_checkpoint(tmp_path, "llava_onevision", stages)
    stages[0].config = {"t2v": {"0": [2]}}
    with pytest.raises(ValueError, match="differ"):
        validate_checkpoint(tmp_path, "qwen25_vl", stages)


@pytest.mark.parametrize("backbone", ["qwen25_vl", "llava_onevision"])
def test_documented_configs_resolve_and_swift_payload_roundtrips(backbone):
    from pathlib import Path
    import yaml
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.pipeline.definition import resolve_compression_definition
    from mllm_comp.integrations.ms_swift.adapter import _training_compression_spec
    from mllm_comp.core.errors import BackendIntegrationError
    source = Path(__file__).parents[2] / "configs/fixed_mask" / f"{backbone}.yaml"
    config = experiment_config_from_dict(yaml.safe_load(source.read_text(encoding="utf-8")))
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="eval")
    assert len(definition.stages) == 1
    encoded, info = _training_compression_spec(config)
    payload = json.loads(base64.urlsafe_b64decode(encoded))
    assert payload["schema_version"] == 5 and payload["stages"][0]["algorithm"] == "fixed_mask"
    assert info["enabled"]
    config.train.args["packing"] = True
    with pytest.raises(BackendIntegrationError, match="packing"):
        _training_compression_spec(config)
