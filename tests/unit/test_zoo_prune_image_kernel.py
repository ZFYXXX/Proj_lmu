from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.zoo_prune import (
    ZOOProjectorProxyCache,
    ZOOPruneSpec,
    build_zoo_projector_proxy,
    estimate_zoo_sensitivity,
    merge_zoo_prune_executor_config,
    sample_zoo_directions,
    sensitivity_aware_diverse_indices,
    zoo_prune_image_tokens,
)


def test_centered_sensitivity_and_sads_are_exact_and_source_ordered() -> None:
    inputs = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.5, 4.0]])
    directions = torch.eye(2)
    spec = ZOOPruneSpec(
        target_count=2,
        num_directions=2,
        step_size=0.01,
        projector_proxy="full",
        direction_batch_size=1,
    )
    sensitivity = estimate_zoo_sensitivity(
        inputs,
        lambda value: value.square(),
        spec=spec,
        directions=directions,
    )
    assert torch.allclose(
        sensitivity,
        torch.tensor([3.0, 4.0, 4.5]),
        atol=1e-4,
        rtol=0,
    )

    projected = torch.tensor(
        [[1.0, 0.0], [0.99, 0.10], [0.0, 1.0], [-1.0, 0.0]]
    )
    source_order, selection_order = sensitivity_aware_diverse_indices(
        projected,
        torch.tensor([4.0, 3.0, 2.0, 1.0]),
        2,
    )
    assert selection_order.tolist() == [0, 2]
    assert source_order.tolist() == [0, 2]


def test_low_rank_proxy_is_lazy_local_and_layout_delta_is_exact() -> None:
    first = torch.nn.Linear(3, 3)
    activation = torch.nn.GELU()
    second = torch.nn.Linear(3, 2)
    layers = (first, activation, second)
    projector_inputs = torch.tensor(
        [
            [0.2, -0.5, 1.0],
            [0.1, 0.4, -0.3],
            [1.2, 0.7, -0.2],
            [-0.8, 0.9, 0.5],
        ]
    )
    projected = second(activation(first(projector_inputs))).detach()
    spec = ZOOPruneSpec(
        target_count=2,
        num_directions=4,
        low_rank_rank=3,
        direction_batch_size=2,
    )
    cache = ZOOProjectorProxyCache()

    rng_before = torch.random.get_rng_state().clone()
    directions = sample_zoo_directions(
        4,
        3,
        device=projector_inputs.device,
        dtype=projector_inputs.dtype,
        seed=0,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.allclose(directions.norm(dim=-1), torch.ones(4), atol=1e-6)

    output = zoo_prune_image_tokens(
        projected,
        projector_inputs,
        layers,
        spec=spec,
        proxy_cache=cache,
        source_layout={"partition": "image"},
        result_layout={"position_policy": "sparse"},
    )
    assert output.tokens.shape == (2, 2)
    assert output.indices.tolist() == sorted(output.indices.tolist())
    assert output.layout_delta.source_count == 4
    assert output.layout_delta.result_count == 2
    assert output.layout_delta.kept_indices == tuple(output.indices.tolist())
    assert output.metadata["low_rank_solver_used"] == "exact+exact"
    assert output.metadata["token_order"] == "source_index"
    assert torch.isfinite(output.sensitivity).all()


@pytest.mark.parametrize("inference_parameters", [False, True])
def test_low_rank_proxy_tracks_projector_updates(inference_parameters: bool) -> None:
    with torch.inference_mode(inference_parameters):
        projector = torch.nn.Linear(3, 3)
        with torch.no_grad():
            projector.weight.copy_(torch.diag(torch.tensor([3.0, 2.0, 1.0])))
            projector.bias.zero_()
    inputs = torch.eye(3)
    layers = (projector,)
    spec = ZOOPruneSpec(low_rank_rank=2)
    cache = ZOOProjectorProxyCache()
    first = cache.resolve(layers, spec)
    before = first(inputs).clone()
    if not inference_parameters:
        assert cache.resolve(layers, spec) is first
        optimizer = torch.optim.SGD(projector.parameters(), lr=0.1)
        projector(inputs).square().sum().backward()
        optimizer.step()
    else:
        with torch.inference_mode():
            projector.weight.mul_(2.0)
            projector.bias.add_(0.5)

    updated = cache.resolve(layers, spec)
    fresh = build_zoo_projector_proxy(layers, spec=spec)
    torch.testing.assert_close(updated(inputs), fresh(inputs))
    assert not torch.allclose(updated(inputs), before)
    if not inference_parameters:
        assert cache.resolve(layers, spec) is updated


def test_randomized_proxy_preserves_cpu_and_all_cuda_rng_states() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices to observe sibling RNG state")
    first = torch.nn.Linear(3, 3)
    second = torch.nn.Linear(3, 2)
    inputs = torch.tensor(
        [[0.2, -0.5, 1.0], [0.1, 0.4, -0.3], [1.2, 0.7, -0.2]]
    )
    projected = second(first(inputs)).detach()
    spec = ZOOPruneSpec(
        target_count=2,
        num_directions=2,
        low_rank_rank=2,
        low_rank_solver="randomized",
        direction_batch_size=1,
    )
    torch.random.default_generator.manual_seed(101)
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            torch.cuda.manual_seed(200 + device_index)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = tuple(
        torch.cuda.get_rng_state(device_index).clone()
        for device_index in range(torch.cuda.device_count())
    )

    zoo_prune_image_tokens(
        projected,
        inputs,
        (first, second),
        spec=spec,
        proxy_cache=ZOOProjectorProxyCache(),
    )

    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert all(
        torch.equal(torch.cuda.get_rng_state(device_index), expected)
        for device_index, expected in enumerate(cuda_before)
    )


def test_config_rejects_unknown_keys_and_ambiguous_or_non_numeric_budget() -> None:
    stage = SimpleNamespace(
        operation={},
        criterion={},
        schedule={},
        config={"retention_rato": 0.1},
    )
    with pytest.raises(ValueError, match="retention_rato"):
        merge_zoo_prune_executor_config(stage=stage, binding_config=None)
    with pytest.raises(ValueError, match="only one explicit budget"):
        merge_zoo_prune_executor_config(
            stage=SimpleNamespace(
                operation={},
                criterion={},
                schedule={},
                config={"retention_ratio": 0.1, "target_count": 8},
            ),
            binding_config=None,
        )
    with pytest.raises(TypeError, match="real number"):
        ZOOPruneSpec(retention_ratio=True)
