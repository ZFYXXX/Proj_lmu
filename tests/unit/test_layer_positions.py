"""Public application coordinates preserve rule schedules and boundary aliases."""

import pytest

from mllm_comp.compressors.layer_positions import LayerPosition, resolve_layer_positions


def test_before_layer_alias_and_initial_input():
    defaults = [0, 2]
    before = resolve_layer_positions(
        {"layers": [1, 3], "layer_edge": "before"}, defaults,
        boundary="post_layer", num_layers=4,
    )
    assert before == resolve_layer_positions({}, defaults, boundary="post_layer", num_layers=4)
    assert defaults == [0, 2]
    assert resolve_layer_positions(
        {"layers": [0], "layer_edge": "before"}, [2], boundary="post_layer", num_layers=4,
    ) == (LayerPosition("inter_layer", 0),)


@pytest.mark.parametrize("operation", [
    {"layers": [-1]}, {"layers": [True]}, {"layers": []},
    {"layers": [2, 1]}, {"layers": [1, 1]}, {"layers": [4]},
    {"layers": [1], "layer_edge": "unknown"},
])
def test_reject_invalid_application_coordinates(operation):
    with pytest.raises(ValueError):
        resolve_layer_positions(operation, [0], boundary="post_layer", num_layers=4)


def test_inner_boundary_cannot_be_relabelled_as_layer_entry():
    with pytest.raises(ValueError, match="full-layer"):
        resolve_layer_positions(
            {"layers": [0], "layer_edge": "before"}, [0],
            boundary="post_attention_pre_ffn", num_layers=4,
        )


def test_initial_and_final_boundary_in_one_schedule():
    assert resolve_layer_positions(
        {"layers": [0, 1], "layer_edge": "before"}, (), boundary="post_layer", num_layers=1,
    ) == (LayerPosition("inter_layer", 0), LayerPosition("inter_layer", 1))
    with pytest.raises(ValueError):
        resolve_layer_positions({"layers": [2], "layer_edge": "before"}, (), boundary="post_layer", num_layers=1)
