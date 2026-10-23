"""Backbone-neutral SparseVLM v1 visual-token sparsification.

The fixed fuxian snapshot reproduces the public ``official_v1`` path: text
raters are selected once from prompt embeddings, decoder layers 2/6/15 use a
fixed schedule scaled from 576 visual tokens, and low-score visual tokens are
partly recycled with deterministic density-peak average clustering.  Provider
bridges own serialized image spans, position semantics, masks, and cache
legalization; this module owns the shared selection and merge mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

from mllm_comp.core.layout import LayoutDelta, OrderingPolicy

from .decoder_config import require_finite_real
from ._rules import score_rule, selection_rule, label_rule, rule_metadata, replay_rule_groups
from .decoder_lifecycle import layout_merge_groups


@dataclass(frozen=True)
class SparseVLMSpec:
    variant: str = "official_v1"
    prune_layers: tuple[int, ...] = (2, 6, 15)
    schedule_keep_tokens: tuple[int, ...] = (300, 200, 110)
    schedule_base_visual_tokens: int = 576
    scale_schedule_to_input: bool = True
    recycle_ratio: float = 0.3
    cluster_divisor: int = 10
    position_mode: str = "contiguous"
    single_image_only: bool = True
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.variant != "official_v1":
            raise ValueError("variant must be 'official_v1'.")
        layers = _integer_tuple(self.prune_layers, "prune_layers", minimum=0)
        schedule = _integer_tuple(
            self.schedule_keep_tokens,
            "schedule_keep_tokens",
            minimum=1,
        )
        if not layers:
            raise ValueError("prune_layers must not be empty.")
        if layers != tuple(sorted(set(layers))):
            raise ValueError("prune_layers must be sorted and unique.")
        if len(layers) != len(schedule):
            raise ValueError(
                "prune_layers and schedule_keep_tokens must have equal length."
            )
        if (
            isinstance(self.schedule_base_visual_tokens, bool)
            or not isinstance(self.schedule_base_visual_tokens, int)
            or self.schedule_base_visual_tokens < 2
        ):
            raise ValueError("schedule_base_visual_tokens must be at least two.")
        if not isinstance(self.scale_schedule_to_input, bool):
            raise TypeError("scale_schedule_to_input must be a boolean.")
        recycle_ratio = require_finite_real(self.recycle_ratio, "recycle_ratio")
        if not 0.0 < recycle_ratio <= 1.0:
            raise ValueError("recycle_ratio must be in (0, 1].")
        if (
            isinstance(self.cluster_divisor, bool)
            or not isinstance(self.cluster_divisor, int)
            or self.cluster_divisor < 1
        ):
            raise ValueError("cluster_divisor must be a positive integer.")
        if self.position_mode != "contiguous":
            raise ValueError("official_v1 requires position_mode='contiguous'.")
        if self.single_image_only is not True:
            raise ValueError("official_v1 native adaptation is single-image only.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer.")
        object.__setattr__(self, "prune_layers", layers)
        object.__setattr__(self, "schedule_keep_tokens", schedule)
        object.__setattr__(self, "recycle_ratio", recycle_ratio)

    def stage_index(self, layer_index: int) -> int:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer.")
        try:
            return self.prune_layers.index(layer_index)
        except ValueError as exc:
            raise ValueError(
                f"layer {layer_index} is not a configured SparseVLM boundary."
            ) from exc

    def stage_target(self, layer_index: int, initial_visual_tokens: int) -> int:
        if (
            isinstance(initial_visual_tokens, bool)
            or not isinstance(initial_visual_tokens, int)
            or initial_visual_tokens < 1
        ):
            raise ValueError("initial_visual_tokens must be a positive integer.")
        target = self.schedule_keep_tokens[self.stage_index(layer_index)]
        if self.scale_schedule_to_input:
            target = round(
                target * initial_visual_tokens / self.schedule_base_visual_tokens
            )
        return max(1, int(target))


@dataclass(frozen=True)
class SparseVLMRaterSelection:
    rater_indices: Any
    candidate_indices: Any
    candidate_scores: Any


@dataclass(frozen=True)
class SparseVLMResult:
    hidden_states: Any
    representative_indices: Any
    selected_image_indices: Any
    recycled_image_indices: Any
    dropped_image_indices: Any
    result_image_indices: Any
    image_scores: Any
    cluster_assignments: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)


def select_sparsevlm_text_raters(
    input_embeddings: Any,
    *,
    image_indices: Any,
    text_candidate_indices: Any,
    rules: Mapping[str, Any] | None = None,
) -> SparseVLMRaterSelection:
    """Select the public v1 prompt raters with strict-mean thresholding."""

    torch = _torch()
    if (
        not isinstance(input_embeddings, torch.Tensor)
        or input_embeddings.ndim != 2
        or not input_embeddings.is_floating_point()
    ):
        raise ValueError("input_embeddings must be a floating rank-2 tensor.")
    length = int(input_embeddings.shape[0])
    images = _index_tensor(
        image_indices,
        device=input_embeddings.device,
        name="image_indices",
    )
    candidates = _index_tensor(
        text_candidate_indices,
        device=input_embeddings.device,
        name="text_candidate_indices",
    )
    _validate_source_indices(images, length, "image_indices")
    _validate_source_indices(candidates, length, "text_candidate_indices")
    if int(images.numel()) < 1:
        raise ValueError("SparseVLM requires at least one image token.")
    if int(candidates.numel()) < 1:
        raise ValueError("SparseVLM requires at least one text-rater candidate.")
    if set(images.tolist()).intersection(candidates.tolist()):
        raise ValueError("text-rater candidates may not overlap image tokens.")
    if int(candidates[0].item()) <= int(images[-1].item()):
        raise ValueError("SparseVLM raters must come from post-image text.")
    visual = input_embeddings.index_select(0, images).float()
    text = input_embeddings.index_select(0, candidates).float()
    relation = visual @ text.transpose(0, 1)
    candidate_scores = relation.softmax(dim=1).mean(dim=0)
    candidate_scores = score_rule(rules, "rater_scorer", {
        "input_embeddings": input_embeddings, "image_indices": images,
        "candidate_indices": candidates, "relation": relation,
    }, candidate_scores)
    selected_local = (
        (candidate_scores > candidate_scores.mean()).nonzero(as_tuple=False).flatten()
    )
    if int(selected_local.numel()) == 0:
        selected_local = candidate_scores.argmax().view(1)
    custom = selection_rule(rules, "rater_selector", {"input_embeddings": input_embeddings,
        "image_indices": images, "candidate_indices": candidates, "scores": candidate_scores,
        "keep_count": int(selected_local.numel())}, count=int(candidates.numel()),
        keep_count=int(selected_local.numel()), device=candidates.device)
    if custom is not None:
        selected_local = custom
    raters = candidates.index_select(0, selected_local).sort().values
    return SparseVLMRaterSelection(
        rater_indices=raters,
        candidate_indices=candidates,
        candidate_scores=candidate_scores,
    )


def remap_sparsevlm_attention_scores(scores: Any, delta: LayoutDelta, source_image_indices: Any) -> Any:
    """Carry captured Attention column mass through an earlier select/merge."""
    torch = _torch()
    targets = [delta.old_to_new[int(source)] for source in source_image_indices.tolist()]
    targets = torch.tensor([target if target is not None else -1 for target in targets],
        dtype=torch.long, device=scores.device)
    retained = targets >= 0
    result_targets, inverse = torch.unique(targets[retained], sorted=True, return_inverse=True)
    return scores.new_zeros(result_targets.numel()).index_add(0, inverse, scores[retained])


def sparsevlm_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    image_attention_scores: Any,
    layer_index: int,
    initial_visual_tokens: int,
    spec: SparseVLMSpec | None = None,
    protected_indices: Iterable[int] = (),
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> SparseVLMResult:
    """Apply one public-v1 SparseVLM select/recycle/merge stage."""

    torch = _torch()
    active_spec = spec or SparseVLMSpec()
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 2
        or not hidden_states.is_floating_point()
    ):
        raise ValueError("hidden_states must be a floating rank-2 tensor.")
    sequence_length = int(hidden_states.shape[0])
    images = _index_tensor(
        image_indices,
        device=hidden_states.device,
        name="image_indices",
    )
    _validate_source_indices(images, sequence_length, "image_indices")
    image_count = int(images.numel())
    if image_count < 1:
        raise ValueError("SparseVLM requires at least one image token.")
    expected_span = torch.arange(
        int(images[0].item()),
        int(images[-1].item()) + 1,
        dtype=torch.long,
        device=images.device,
    )
    if not torch.equal(images, expected_span):
        raise ValueError("SparseVLM requires one contiguous serialized image span.")
    scores = _score_tensor(
        image_attention_scores,
        count=image_count,
        device=hidden_states.device,
    )
    protected = tuple(sorted(set(int(index) for index in protected_indices)))
    if any(index < 0 or index >= sequence_length for index in protected):
        raise ValueError("protected_indices contains an out-of-range index.")
    image_lookup = {value: index for index, value in enumerate(images.tolist())}
    protected_image_local = tuple(
        image_lookup[index] for index in protected if index in image_lookup
    )
    target = active_spec.stage_target(layer_index, initial_visual_tokens)
    if target >= image_count or len(protected_image_local) >= image_count:
        return _identity_result(
            hidden_states,
            images,
            scores,
            protected=protected,
            layer_index=layer_index,
            target=target,
            media_unit=media_unit,
            layout=source_layout or {},
            result_layout=result_layout,
        )
    scores = score_rule(rules, "scorer", {"hidden_states": hidden_states,
        "image_indices": images, "image_attention_scores": scores,
        "layer_index": layer_index, "keep_count": target}, scores)
    direct_keep = min(
        image_count - 1,
        max(target, len(protected_image_local), 1),
    )
    selected_mask = torch.zeros(
        image_count,
        dtype=torch.bool,
        device=hidden_states.device,
    )
    if protected_image_local:
        selected_mask[
            torch.tensor(
                protected_image_local,
                dtype=torch.long,
                device=hidden_states.device,
            )
        ] = True
    remaining = direct_keep - int(selected_mask.sum().item())
    if remaining:
        eligible_scores = scores.masked_fill(selected_mask, float("-inf"))
        selected_mask[_stable_descending_indices(eligible_scores)[:remaining]] = True
    custom = selection_rule(rules, "selector", {"hidden_states": hidden_states,
        "image_indices": images, "scores": scores, "keep_count": direct_keep,
        "protected_indices": protected_image_local, "layer_index": layer_index},
        count=image_count, keep_count=direct_keep, device=images.device, protected=protected_image_local)
    if custom is not None:
        selected_mask.zero_()
        selected_mask[custom] = True
    selected_local = selected_mask.nonzero(as_tuple=False).flatten()
    dropped_local = (~selected_mask).nonzero(as_tuple=False).flatten()
    recycle_count = min(
        int(dropped_local.numel()),
        int(int(dropped_local.numel()) * active_spec.recycle_ratio) + 1,
    )
    recycled_order = _stable_descending_indices(scores.index_select(0, dropped_local))
    recycled_local = dropped_local.index_select(0, recycled_order[:recycle_count])
    custom = selection_rule(rules, "recycle_selector", {"hidden_states": hidden_states,
        "image_indices": images, "candidate_indices": dropped_local, "scores": scores,
        "keep_count": recycle_count, "layer_index": layer_index},
        count=image_count, keep_count=recycle_count, device=images.device)
    if custom is not None:
        if not set(custom.tolist()).issubset(set(dropped_local.tolist())):
            raise ValueError("recycle_selector must select only dropped image indices.")
        recycled_local = custom
    cluster_count = min(
        recycle_count,
        int(recycle_count / active_spec.cluster_divisor) + 1,
    )
    recycled_source = images.index_select(0, recycled_local)
    merged_hidden, assignments, centers = density_peak_average(
        hidden_states.index_select(0, recycled_source),
        cluster_count,
        random_seed=active_spec.random_seed + active_spec.stage_index(layer_index),
        rules=rules,
    )
    selected_source = images.index_select(0, selected_local)
    visual_start = int(images[0].item())
    visual_end = int(images[-1].item()) + 1
    output = torch.cat(
        (
            hidden_states[:visual_start],
            hidden_states.index_select(0, selected_source),
            merged_hidden,
            hidden_states[visual_end:],
        ),
        dim=0,
    )
    old_to_new: list[int | None] = [None] * sequence_length
    for source in range(visual_start):
        old_to_new[source] = source
    for offset, source in enumerate(selected_source.tolist()):
        old_to_new[source] = visual_start + offset
    cluster_start = visual_start + int(selected_source.numel())
    for local, source in enumerate(recycled_source.tolist()):
        old_to_new[source] = cluster_start + int(assignments[local].item())
    suffix_start = cluster_start + cluster_count
    for source in range(visual_end, sequence_length):
        old_to_new[source] = suffix_start + source - visual_end
    non_image = tuple(
        index
        for index in range(sequence_length)
        if not visual_start <= index < visual_end
    )
    layout_protected = tuple(sorted(set(non_image) | set(protected)))
    delta = LayoutDelta.from_mapping(
        source_count=sequence_length,
        old_to_new=old_to_new,
        result_count=int(output.shape[0]),
        media_unit=media_unit,
        protected_indices=layout_protected,
        source_layout=source_layout or {},
        result_layout=result_layout or {},
        ordering_policy=OrderingPolicy.EXPLICIT,
    )
    representatives = _layout_representatives(delta, device=hidden_states.device)
    recycled_set = set(recycled_source.tolist())
    dropped_source = torch.tensor(
        [
            source
            for source in images.tolist()
            if source not in set(selected_source.tolist())
            and source not in recycled_set
        ],
        dtype=torch.long,
        device=hidden_states.device,
    )
    result_images = torch.arange(
        visual_start,
        cluster_start + cluster_count,
        dtype=torch.long,
        device=hidden_states.device,
    )
    result = SparseVLMResult(
        hidden_states=output,
        representative_indices=representatives,
        selected_image_indices=selected_source,
        recycled_image_indices=recycled_source,
        dropped_image_indices=dropped_source,
        result_image_indices=result_images,
        image_scores=scores,
        cluster_assignments=assignments,
        layout_delta=delta,
        metadata={
            **rule_metadata(rules),
            "algorithm": "sparsevlm",
            "variant": "official_v1",
            "layer_index": layer_index,
            "schedule_target": target,
            "source_tokens": sequence_length,
            "result_tokens": int(output.shape[0]),
            "source_image_tokens": image_count,
            "selected_visual_tokens": int(selected_source.numel()),
            "recycled_visual_tokens": recycle_count,
            "merged_visual_tokens": cluster_count,
            "result_image_tokens": int(result_images.numel()),
            "cluster_centers": tuple(int(value) for value in centers.tolist()),
            "representative_indices": tuple(
                int(value) for value in representatives.tolist()
            ),
            "compression_applied": not delta.is_identity,
        },
    )

    if rules and "aggregator" in rules:
        result = replace(result, hidden_states=replay_rule_groups(
            hidden_states, layout_merge_groups(delta), rules=rules))
    return result


def density_peak_average(
    tokens: Any,
    cluster_count: int,
    *,
    random_seed: int,
    rules: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Return deterministic public-v1 density-peak average clusters."""

    torch = _torch()
    functional = _functional()
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 2
        or not tokens.is_floating_point()
    ):
        raise ValueError("tokens must be a floating rank-2 tensor.")
    token_count, hidden_size = (int(value) for value in tokens.shape)
    if token_count < 1:
        raise ValueError("SparseVLM cannot cluster an empty token set.")
    if (
        isinstance(cluster_count, bool)
        or not isinstance(cluster_count, int)
        or cluster_count < 1
    ):
        raise ValueError("cluster_count must be a positive integer.")
    cluster_count = min(token_count, cluster_count)
    distances = torch.cdist(tokens.float(), tokens.float(), p=2) / hidden_size**0.5
    nearest = torch.topk(
        distances,
        k=cluster_count,
        dim=-1,
        largest=False,
    ).values
    density = (-(nearest**2).mean(dim=-1)).exp()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    noise = torch.rand(density.shape, generator=generator, dtype=torch.float32)
    density = density + noise.to(device=density.device, dtype=density.dtype) * 1e-6
    higher_density = density[None, :] > density[:, None]
    max_distance = distances.max().reshape(1, 1)
    nearest_higher = (
        torch.where(higher_density, distances, max_distance).min(dim=-1).values
    )
    center_scores = density * nearest_higher
    center_scores = score_rule(rules, "cluster_scorer", {"tokens": tokens,
        "distances": distances, "density": density, "keep_count": cluster_count}, center_scores)
    centers = _stable_descending_indices(center_scores)[:cluster_count]
    custom = selection_rule(rules, "cluster_selector", {"tokens": tokens,
        "scores": center_scores, "keep_count": cluster_count}, count=token_count,
        keep_count=cluster_count, device=tokens.device)
    if custom is not None:
        centers = custom
    assignments = distances.index_select(0, centers).argmin(dim=0)
    assignments[centers] = torch.arange(
        cluster_count,
        dtype=torch.long,
        device=tokens.device,
    )
    assignments = label_rule(rules, "grouper", {"tokens": tokens, "distances": distances,
        "centers": centers, "group_count": cluster_count, "default_labels": assignments}, assignments)
    if (bool(((assignments < 0) | (assignments >= cluster_count)).any())
            or not torch.equal(assignments[centers], torch.arange(cluster_count, device=tokens.device))):
        raise ValueError("SparseVLM grouper must assign each source to a valid group and retain centers in their own groups.")
    one_hot = functional.one_hot(assignments, num_classes=cluster_count).to(
        tokens.dtype
    )
    counts = one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
    merged = one_hot.transpose(0, 1) @ tokens / counts
    return merged, assignments, centers


def _identity_result(
    hidden_states: Any,
    images: Any,
    scores: Any,
    *,
    protected: tuple[int, ...],
    layer_index: int,
    target: int,
    media_unit: str,
    layout: Mapping[str, Any],
    result_layout: Mapping[str, Any] | None = None,
) -> SparseVLMResult:
    torch = _torch()
    length = int(hidden_states.shape[0])
    delta = LayoutDelta.identity(
        length,
        media_unit=media_unit,
        protected_indices=tuple(sorted(set(range(length)) | set(protected))),
        layout=layout,
    )
    if result_layout is not None:
        delta = replace(delta, result_layout=result_layout)
    indices = torch.arange(length, dtype=torch.long, device=hidden_states.device)
    empty = torch.empty(0, dtype=torch.long, device=hidden_states.device)
    return SparseVLMResult(
        hidden_states=hidden_states,
        representative_indices=indices,
        selected_image_indices=images,
        recycled_image_indices=empty,
        dropped_image_indices=empty,
        result_image_indices=images,
        image_scores=scores,
        cluster_assignments=empty,
        layout_delta=delta,
        metadata={
            "algorithm": "sparsevlm",
            "variant": "official_v1",
            "layer_index": layer_index,
            "schedule_target": target,
            "source_tokens": length,
            "result_tokens": length,
            "source_image_tokens": int(images.numel()),
            "result_image_tokens": int(images.numel()),
            "compression_applied": False,
        },
    )


def _layout_representatives(delta: LayoutDelta, *, device: Any) -> Any:
    torch = _torch()
    groups: list[list[int]] = [[] for _ in range(delta.result_count)]
    for source in range(delta.source_count):
        result = delta.old_to_new[source]
        if result is not None:
            groups[result].append(source)
    return torch.tensor(
        [group[0] for group in groups],
        dtype=torch.long,
        device=device,
    )


def _integer_tuple(value: Any, name: str, *, minimum: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers.")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers.") from exc
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise TypeError(f"{name} must contain only integers.")
    if any(item < minimum for item in result):
        raise ValueError(f"{name} values must be at least {minimum}.")
    return result


def _index_tensor(value: Any, *, device: Any, name: str) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.long)
    else:
        result = torch.tensor(tuple(value), dtype=torch.long, device=device)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return result


def _validate_source_indices(value: Any, upper: int, name: str) -> None:
    items = value.tolist()
    if items != sorted(set(items)):
        raise ValueError(f"{name} must be unique and in source order.")
    if any(index < 0 or index >= upper for index in items):
        raise ValueError(f"{name} contains an out-of-range index.")


def _score_tensor(value: Any, *, count: int, device: Any) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("image_attention_scores must be a torch tensor.")
    result = value.to(device=device, dtype=torch.float32)
    if result.ndim != 1 or int(result.numel()) != count:
        raise ValueError(
            "image_attention_scores must contain one value per image token."
        )
    if not bool(torch.isfinite(result).all().item()) or bool((result < 0).any().item()):
        raise ValueError("image_attention_scores must be finite and non-negative.")
    return result


def _stable_descending_indices(values: Any) -> Any:
    torch = _torch()
    return torch.argsort(values, descending=True, stable=True)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("SparseVLM execution requires PyTorch.") from exc
    return torch


def _functional() -> Any:
    try:
        from torch.nn import functional
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("SparseVLM execution requires PyTorch.") from exc
    return functional


__all__ = [
    "SparseVLMRaterSelection",
    "SparseVLMResult",
    "SparseVLMSpec",
    "density_peak_average",
    "select_sparsevlm_text_raters",
    "sparsevlm_image_tokens",
]
