"""Numerical audit helpers for opt-in real-checkpoint image compression tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Any


def _max_error(actual: Any, expected: Any) -> float:
    if int(actual.numel()) == 0:
        return 0.0
    return float((actual.detach().float() - expected.detach().float()).abs().max().item())


def _assert_close(actual: Any, expected: Any, *, operation: str) -> float:
    torch = __import__("torch")
    error = _max_error(actual, expected)
    if actual.dtype == torch.bfloat16:
        atol, rtol = 8e-2, 3e-2
    elif actual.dtype == torch.float16:
        atol, rtol = 3e-2, 2e-2
    else:
        atol, rtol = 2e-5, 2e-5
    scale = float(expected.detach().float().abs().max().item())
    assert error <= atol + rtol * scale, (
        f"{operation} numerical error {error} exceeds "
        f"{atol + rtol * scale}."
    )
    return error


def _partition_is_exact(groups: Any, source_count: int) -> bool:
    flattened = [int(index) for group in groups for index in group]
    return sorted(flattened) == list(range(source_count)) and len(flattened) == len(
        set(flattened)
    )


def _tome_audit(hidden_states: Any, keys: Any, kwargs: dict[str, Any], result: Any) -> dict[str, Any]:
    torch = __import__("torch")
    is_batched = hidden_states.ndim == 3
    values = hidden_states if is_batched else hidden_states.unsqueeze(0)
    batch_size, source_count = int(values.shape[0]), int(values.shape[1])
    output = result.hidden_states if is_batched else result.hidden_states.unsqueeze(0)
    output_sizes = result.token_sizes if is_batched else result.token_sizes.unsqueeze(0)

    supplied_sizes = kwargs.get("token_sizes")
    if supplied_sizes is None:
        sizes = torch.ones(
            (batch_size, source_count, 1),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        sizes = supplied_sizes
        if not torch.is_tensor(sizes):
            sizes = torch.as_tensor(sizes, dtype=values.dtype, device=values.device)
        if not is_batched:
            sizes = sizes.unsqueeze(0)
        if sizes.ndim == 2:
            sizes = sizes.unsqueeze(-1)

    if is_batched:
        metric = keys.detach().float().mean(dim=2) if keys.ndim == 4 else keys.detach().float()
    else:
        metric = (
            keys.detach().float().mean(dim=1).unsqueeze(0)
            if keys.ndim == 3
            else keys.detach().float().unsqueeze(0)
        )
    metric = torch.nn.functional.normalize(metric, p=2.0, dim=-1, eps=float(kwargs.get("eps", 1e-6)))

    protected = tuple(sorted(int(item) for item in kwargs.get("protected_indices", ())))
    protected_set = set(protected)
    eligible_a = tuple(index for index in range(0, source_count, 2) if index not in protected_set)
    eligible_b = tuple(index for index in range(1, source_count, 2) if index not in protected_set)
    effective_r = int(result.metadata["r_effective"])
    value_error = 0.0
    size_error = 0.0
    margins: list[float] = []
    best_scores_by_batch = None
    best_destinations_by_batch = None
    ranked_slots_by_batch = None
    if effective_r:
        a_tensor = torch.tensor(eligible_a, dtype=torch.long, device=metric.device)
        b_tensor = torch.tensor(eligible_b, dtype=torch.long, device=metric.device)
        similarities = metric.index_select(1, a_tensor) @ metric.index_select(
            1, b_tensor
        ).transpose(-1, -2)
        best_scores_by_batch, best_destinations_by_batch = similarities.max(dim=-1)
        ranked_slots_by_batch = torch.argsort(
            best_scores_by_batch,
            dim=-1,
            descending=True,
            stable=True,
        )

    for batch_index, groups in enumerate(result.source_groups):
        assert _partition_is_exact(groups, source_count)
        for result_index, group in enumerate(groups):
            indices = torch.tensor(group, dtype=torch.long, device=values.device)
            group_sizes = sizes[batch_index].index_select(0, indices)
            expected_value = (
                values[batch_index].index_select(0, indices) * group_sizes
            ).sum(dim=0) / group_sizes.sum(dim=0)
            expected_size = group_sizes.sum(dim=0)
            value_error = max(
                value_error,
                _assert_close(
                    output[batch_index, result_index],
                    expected_value,
                    operation="ToMe size-weighted merge",
                ),
            )
            size_error = max(
                size_error,
                _assert_close(
                    output_sizes[batch_index, result_index],
                    expected_size,
                    operation="ToMe token-mass sum",
                ),
            )
        _assert_close(
            output_sizes[batch_index].sum(),
            sizes[batch_index].sum(),
            operation="ToMe total token-mass conservation",
        )

        if effective_r:
            assert best_scores_by_batch is not None
            assert best_destinations_by_batch is not None
            assert ranked_slots_by_batch is not None
            best_scores = best_scores_by_batch[batch_index]
            best_destinations = best_destinations_by_batch[batch_index]
            ranked_slots = ranked_slots_by_batch[batch_index].tolist()
            expected_sources = {eligible_a[slot] for slot in ranked_slots[:effective_r]}
            source_to_group = {
                int(source): group
                for group in groups
                for source in group
            }
            actual_sources = {
                source
                for source in eligible_a
                if len(source_to_group[source]) > 1
                and any(item in set(eligible_b) for item in source_to_group[source])
            }
            assert actual_sources == expected_sources, (
                f"ToMe selected sources differ in batch {batch_index}: "
                f"actual={sorted(actual_sources)}, expected={sorted(expected_sources)}, "
                f"groups={[group for group in groups if len(group) > 1]}"
            )
            for source in expected_sources:
                source_slot = eligible_a.index(source)
                destination = eligible_b[int(best_destinations[source_slot].item())]
                assert destination in source_to_group[source]
            selected_scores = [
                float(best_scores[eligible_a.index(source)].item())
                for source in expected_sources
            ]
            unselected_scores = [
                float(best_scores[slot].item())
                for slot in range(len(eligible_a))
                if eligible_a[slot] not in expected_sources
            ]
            if unselected_scores:
                margin = min(selected_scores) - max(unselected_scores)
                assert margin >= -1e-6
                margins.append(margin)

    return {
        "source_count": source_count,
        "result_count": int(output.shape[1]),
        "merged_token_count": source_count - int(output.shape[1]),
        "batch_size": batch_size,
        "source_partition_exact": True,
        "token_mass_conserved": True,
        "maximum_value_error": value_error,
        "maximum_mass_error": size_error,
        "selection_criterion": "highest_bipartite_cosine_redundancy",
        "selection_matches_reference": True,
        "minimum_selected_similarity_margin": min(margins) if margins else None,
        "removed_information_preserved_by_merge": True,
    }


def _visionzip_audit(
    tokens: Any,
    attention: Any,
    keys: Any,
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    torch = __import__("torch")
    source_count = int(tokens.shape[0])
    groups = result.source_groups
    assert _partition_is_exact(groups, source_count)
    representatives = tuple(int(index) for index in result.indices.tolist())
    assert len(representatives) == len(groups)
    contextual = tuple(int(item) for item in result.metadata["contextual_indices"])
    value_error = 0.0
    for result_index, group in enumerate(groups):
        target = representatives[result_index]
        assert target in group
        expected = tokens[target]
        if target in contextual:
            assigned = tuple(index for index in group if index != target)
            if assigned:
                indices = torch.tensor(assigned, dtype=torch.long, device=tokens.device)
                expected = expected + tokens.index_select(0, indices).mean(dim=0)
        else:
            assert group == (target,)
        value_error = max(
            value_error,
            _assert_close(
                result.tokens[result_index],
                expected,
                operation="VisionZip target plus assigned mean",
            ),
        )

    attention_value = attention
    if attention_value.ndim == 4:
        attention_value = attention_value[0]
    if attention_value.ndim == 3:
        attention_value = attention_value.detach().float().mean(dim=0)
    else:
        attention_value = attention_value.detach().float()
    mode = result.metadata["attention_mode"]
    if mode == "cls":
        cls_index = int(kwargs["cls_index"])
        importance = attention_value[cls_index]
    else:
        importance = attention_value.mean(dim=-2)

    protected = tuple(int(item) for item in result.metadata["protected_indices"])
    candidates = tuple(index for index in range(source_count) if index not in set(protected))
    dominant = tuple(int(item) for item in result.metadata["dominant_indices"])
    expected_dominant = tuple(
        sorted(candidates, key=lambda index: (-float(importance[index].item()), index))[
            : len(dominant)
        ]
    )
    assert dominant == expected_dominant
    non_dominant = tuple(index for index in candidates if index not in set(dominant))
    if contextual:
        stride = len(non_dominant) // len(contextual)
        assert contextual == non_dominant[::stride][:len(contextual)], (
            "VisionZip contextual targets must follow prefix-stride sampling."
        )

        key_value = keys
        if key_value.ndim == 4:
            key_value = key_value[0]
        if key_value.ndim == 3:
            key_value = key_value.mean(dim=0)
        key_value = torch.nn.functional.normalize(
            key_value.detach().float(), p=2.0, dim=-1
        )
        group_by_source = {
            int(source): group
            for group in groups
            for source in group
        }
        targets = torch.tensor(contextual, dtype=torch.long, device=key_value.device)
        sources = tuple(
            source for source in non_dominant if source not in set(contextual)
        )
        source_tensor = torch.tensor(
            sources,
            dtype=torch.long,
            device=key_value.device,
        )
        similarities = key_value.index_select(0, source_tensor) @ key_value.index_select(
            0, targets
        ).T
        assignments = similarities.argmax(dim=-1).tolist()
        for source, assignment in zip(sources, assignments):
            destination = contextual[int(assignment)]
            assert destination in group_by_source[source], (
                f"VisionZip source {source} was assigned outside its most-similar "
                f"context target {destination}."
            )

    non_dominant_scores = [float(importance[index].item()) for index in non_dominant]
    dominant_scores = [float(importance[index].item()) for index in dominant]
    importance_margin = (
        min(dominant_scores) - max(non_dominant_scores)
        if dominant_scores and non_dominant_scores
        else None
    )
    if importance_margin is not None:
        assert importance_margin >= -1e-6
    return {
        "source_count": source_count,
        "result_count": int(result.tokens.shape[0]),
        "source_partition_exact": True,
        "maximum_value_error": value_error,
        "dominant_tokens_match_top_attention": True,
        "minimum_dominant_importance_margin": importance_margin,
        "context_targets_match_uniform_sampling": True,
        "context_assignments_match_normalized_key_similarity": True,
        "removed_information_preserved_by_merge": True,
    }


def _pyramid_reference_scores(kwargs: dict[str, Any], *, device: Any) -> Any:
    """Independently evaluate native Q/K, visible-key softmax and head means."""
    torch = __import__("torch")
    image_count = int(kwargs["image_count"])

    def canonical_keys(value: Any, *, expected_tokens: int | None) -> Any:
        keys = value.to(device=device)
        if keys.ndim == 2:
            keys = keys.unsqueeze(1)
        elif keys.ndim == 3 and expected_tokens is not None:
            if int(keys.shape[0]) != expected_tokens:
                assert int(keys.shape[1]) == expected_tokens
                keys = keys.transpose(0, 1)
        else:
            assert keys.ndim == 3
        if expected_tokens is not None:
            assert int(keys.shape[0]) == expected_tokens
        return keys

    image_keys = canonical_keys(kwargs["image_keys"], expected_tokens=image_count)
    attention_keys_value = kwargs.get("attention_keys")
    if attention_keys_value is None:
        attention_keys = image_keys
        image_key_indices = None
    else:
        attention_keys = canonical_keys(attention_keys_value, expected_tokens=None)
        image_key_indices = torch.as_tensor(kwargs["image_key_indices"], dtype=torch.long, device=device)
        assert int(image_key_indices.numel()) == image_count

    query = kwargs["instruction_query"].to(device=device)
    if query.ndim == 1:
        query = query.reshape(1, 1, -1)
    elif query.ndim == 2:
        query = query.unsqueeze(0)
    else:
        assert query.ndim == 3
    if int(query.shape[1]) == 1 and int(attention_keys.shape[1]) > 1:
        query = query.expand(-1, attention_keys.shape[1], -1)
    if int(attention_keys.shape[1]) == 1 and int(query.shape[1]) > 1:
        attention_keys = attention_keys.expand(-1, query.shape[1], -1)
    assert tuple(query.shape[1:]) == tuple(attention_keys.shape[1:])

    reference_logits = torch.einsum(
        "qhd,thd->tqh", query, attention_keys
    ) * (float(query.shape[-1]) ** -0.5)
    attention_key_mask = kwargs.get("attention_key_mask")
    if attention_key_mask is not None:
        visible = torch.as_tensor(
            attention_key_mask,
            dtype=torch.bool,
            device=device,
        )
        if visible.ndim == 1:
            visible = visible[:, None, None]
        else:
            assert tuple(visible.shape) == (
                int(query.shape[0]),
                int(attention_keys.shape[0]),
            )
            visible = visible.transpose(0, 1).unsqueeze(-1)
        reference_logits = reference_logits.masked_fill(~visible, float("-inf"))
    reference_probabilities = torch.softmax(
        reference_logits, dim=0, dtype=torch.float32
    ).to(dtype=reference_logits.dtype)
    if image_key_indices is not None:
        reference_probabilities = reference_probabilities.index_select(
            0, image_key_indices
        )
    reference_scores = reference_probabilities.mean(dim=(-1, -2))
    return reference_scores


def _pyramid_attention_audit(kwargs: dict[str, Any], scores: Any) -> dict[str, Any]:
    reference_scores = _pyramid_reference_scores(kwargs, device=scores.device)
    score_error = _assert_close(scores, reference_scores, operation="PyramidDrop visible-attention score")
    query = kwargs["instruction_query"]
    return {
        "source_image_count": int(reference_scores.numel()),
        "instruction_query_count": int(query.shape[0]) if query.ndim == 3 else 1,
        "attention_scores_match_reference": True,
        "maximum_attention_score_error": score_error,
    }


def _pyramid_audit(hidden_states: Any, kwargs: dict[str, Any], result: Any) -> dict[str, Any]:
    """Check selection against its input scores, independently of result metadata."""
    import math

    torch = __import__("torch")
    image_indices = torch.as_tensor(kwargs["image_indices"], dtype=torch.long, device=result.image_scores.device)
    image_count = int(image_indices.numel())
    reference_scores = kwargs.get("image_attention_scores")
    if reference_scores is None:
        reference_scores = _pyramid_reference_scores(
            {**kwargs, "image_count": image_count, "image_key_indices": image_indices},
            device=result.image_scores.device,
        )
    else:
        reference_scores = reference_scores.to(device=result.image_scores.device)
    score_error = _assert_close(result.image_scores, reference_scores, operation="PyramidDrop selection input scores")
    protected = set(int(item) for item in kwargs.get("protected_indices", ()))
    protected_slots = {
        slot for slot, sequence_index in enumerate(image_indices.tolist()) if sequence_index in protected
    }
    spec = kwargs["spec"]
    keep_count = min(image_count, max(
        image_count - math.ceil((1 - spec.keep_ratio) * image_count),
        spec.minimum_keep_tokens, len(protected_slots),
    ))
    kept_image = set(int(item) for item in result.kept_image_indices.tolist())
    assert len(kept_image) == keep_count
    ranked_slots = sorted(range(image_count), key=lambda slot: (-float(reference_scores[slot].item()), slot))
    expected_slots = protected_slots | set(
        [slot for slot in ranked_slots if slot not in protected_slots][:keep_count - len(protected_slots)]
    )
    expected_kept = {int(image_indices[slot].item()) for slot in expected_slots}
    assert kept_image == expected_kept
    if spec.restore_source_order:
        assert result.kept_image_indices.tolist() == sorted(expected_kept)

    source_count = int(hidden_states.shape[0])
    image_set = set(int(item) for item in image_indices.tolist())
    expected_dropped = image_set - expected_kept
    expected_sequence = [index for index in range(source_count) if index not in expected_dropped]
    assert result.kept_indices.tolist() == expected_sequence
    assert result.dropped_indices.tolist() == sorted(expected_dropped)
    value_error = _assert_close(
        result.hidden_states,
        hidden_states.index_select(0, torch.tensor(expected_sequence, dtype=torch.long, device=hidden_states.device)),
        operation="PyramidDrop selected values",
    )
    expected_mapping = {source: target for target, source in enumerate(expected_sequence)}
    delta = result.layout_delta
    assert delta.source_count == source_count and delta.result_count == len(expected_sequence)
    assert delta.kept_indices == tuple(expected_sequence)
    assert delta.dropped_indices == tuple(sorted(expected_dropped))
    assert dict(delta.old_to_new) == {source: expected_mapping.get(source) for source in range(source_count)}
    assert not delta.merge_groups
    selected_unprotected_scores = [float(reference_scores[slot].item()) for slot in expected_slots if slot not in protected_slots]
    dropped_scores = [float(reference_scores[slot].item()) for slot in range(image_count) if int(image_indices[slot].item()) in expected_dropped]
    score_margin = min(selected_unprotected_scores) - max(dropped_scores) if selected_unprotected_scores and dropped_scores else None
    if score_margin is not None:
        assert score_margin >= -1e-6
    return {
        "source_count": source_count,
        "result_count": int(result.hidden_states.shape[0]),
        "source_image_count": image_count,
        "result_image_count": int(result.kept_image_indices.numel()),
        "all_non_image_tokens_preserved": True,
        "selection_input_scores_preserved": True,
        "maximum_input_score_error": score_error,
        "kept_tokens_match_top_input_attention": True,
        "minimum_kept_score_margin": score_margin,
        "maximum_value_error": value_error,
        "layout_matches_reference_selection": True,
        "dropped_tokens_are_not_merged": True,
    }


@dataclass
class SemanticAuditRecorder:
    """Audit scoring and selection separately; retain only JSON-safe summaries."""

    tome: list[dict[str, Any]] = field(default_factory=list)
    visionzip: list[dict[str, Any]] = field(default_factory=list)
    pyramid_drop: list[dict[str, Any]] = field(default_factory=list)
    pyramid_attention: list[dict[str, Any]] = field(default_factory=list)

    def install(self, monkeypatch: Any, adapter_module: Any) -> None:
        original_tome = adapter_module.tome_merge_image
        original_visionzip = adapter_module.visionzip_image
        from mllm_comp.compressors.image import decoder_selection

        original_pyramid = decoder_selection.pyramid_drop
        original_attention = decoder_selection.pyramid_attention_scores

        @wraps(original_tome)
        def audited_tome(hidden_states: Any, keys: Any, **kwargs: Any) -> Any:
            result = original_tome(hidden_states, keys, **kwargs)
            self.tome.append(_tome_audit(hidden_states, keys, kwargs, result))
            return result

        @wraps(original_visionzip)
        def audited_visionzip(
            tokens: Any, attention: Any, keys: Any, **kwargs: Any
        ) -> Any:
            result = original_visionzip(tokens, attention, keys, **kwargs)
            self.visionzip.append(
                _visionzip_audit(tokens, attention, keys, kwargs, result)
            )
            return result

        @wraps(original_attention)
        def audited_attention(**kwargs: Any) -> Any:
            scores = original_attention(**kwargs)
            self.pyramid_attention.append(_pyramid_attention_audit(kwargs, scores))
            return scores

        @wraps(original_pyramid)
        def audited_pyramid(hidden_states: Any, **kwargs: Any) -> Any:
            result = original_pyramid(hidden_states, **kwargs)
            if kwargs.get("image_attention_scores") is None:
                self.pyramid_attention.append(_pyramid_attention_audit(
                    {**kwargs, "image_count": int(result.image_scores.numel()),
                     "image_key_indices": kwargs["image_indices"]}, result.image_scores,
                ))
            self.pyramid_drop.append(_pyramid_audit(hidden_states, kwargs, result))
            return result

        monkeypatch.setattr(adapter_module, "tome_merge_image", audited_tome)
        monkeypatch.setattr(adapter_module, "visionzip_image", audited_visionzip)
        monkeypatch.setattr(decoder_selection, "pyramid_attention_scores", audited_attention)
        monkeypatch.setattr(decoder_selection, "pyramid_drop", audited_pyramid)

    def mark(self) -> tuple[int, int, int, int]:
        return len(self.tome), len(self.visionzip), len(self.pyramid_drop), len(self.pyramid_attention)

    def since(self, mark: tuple[int, int, int, int]) -> dict[str, Any]:
        if self.pyramid_drop[mark[2] :]:
            assert self.pyramid_attention[mark[3] :], "PyramidDrop selection ran without a native Q/K score audit"
        return {
            "tome": self.tome[mark[0] :],
            "visionzip": self.visionzip[mark[1] :],
            "pyramid_drop": self.pyramid_drop[mark[2] :],
            "pyramid_attention": self.pyramid_attention[mark[3] :],
        }


def _is_subsequence(source: Any, target: Any) -> bool:
    source_columns = source.reshape(-1, source.shape[-1]).T
    target_columns = target.reshape(-1, target.shape[-1]).T
    cursor = 0
    for target_column in target_columns:
        while cursor < int(source_columns.shape[0]) and not bool(
            (source_columns[cursor] == target_column).all().item()
        ):
            cursor += 1
        if cursor == int(source_columns.shape[0]):
            return False
        cursor += 1
    return True


@dataclass
class DecoderPositionAuditRecorder:
    """Capture decoder cache positions and RoPE inputs without changing execution."""

    language_model: Any
    mode: str
    layer_events: list[dict[str, Any]] = field(default_factory=list)
    rotary_positions: list[Any] = field(default_factory=list)
    _handles: list[Any] = field(default_factory=list)

    def install(self) -> None:
        assert self.mode in {"text_1d", "mrope"}
        for layer_index, layer in enumerate(self.language_model.layers):
            def layer_hook(
                _module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                hidden = args[0] if args else kwargs.get("hidden_states")
                cache_position = kwargs.get("cache_position")
                cache_contiguous = None
                if cache_position is not None:
                    torch = __import__("torch")
                    expected = torch.arange(
                        int(cache_position.numel()),
                        dtype=cache_position.dtype,
                        device=cache_position.device,
                    )
                    cache_contiguous = bool(
                        torch.equal(cache_position.reshape(-1), expected)
                    )
                self.layer_events.append(
                    {
                        "layer_index": index,
                        "sequence_length": int(hidden.shape[-2]),
                        "cache_position_length": (
                            None
                            if cache_position is None
                            else int(cache_position.numel())
                        ),
                        "cache_position_contiguous": cache_contiguous,
                    }
                )

            self._handles.append(
                layer.register_forward_pre_hook(layer_hook, with_kwargs=True)
            )

        def rotary_hook(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            positions = args[1] if len(args) > 1 else kwargs.get("position_ids")
            hidden = args[0] if args else kwargs.get("x")
            if positions is None or hidden is None:
                return
            assert int(positions.shape[-1]) == int(hidden.shape[-2])
            self.rotary_positions.append(positions.detach().long().cpu())

        self._handles.append(
            self.language_model.rotary_emb.register_forward_pre_hook(
                rotary_hook,
                with_kwargs=True,
            )
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def mark(self) -> tuple[int, int]:
        return len(self.layer_events), len(self.rotary_positions)

    def first_positions_since(self, mark: tuple[int, int]) -> Any:
        positions = self.rotary_positions[mark[1] :]
        assert positions, "decoder rotary position audit captured no position ids"
        return positions[0]

    def summary_since(
        self,
        mark: tuple[int, int],
        *,
        baseline_positions: Any | None = None,
    ) -> dict[str, Any]:
        torch = __import__("torch")
        layers = self.layer_events[mark[0] :]
        positions = self.rotary_positions[mark[1] :]
        assert layers and positions
        cache_checks = [
            item["cache_position_contiguous"]
            for item in layers
            if item["cache_position_contiguous"] is not None
        ]
        assert all(cache_checks)
        chain_subsequence = all(
            _is_subsequence(previous, current)
            for previous, current in zip(positions, positions[1:])
        )
        assert chain_subsequence
        baseline_subsequence = (
            None
            if baseline_positions is None
            else _is_subsequence(baseline_positions, positions[0])
        )
        if baseline_subsequence is not None:
            assert baseline_subsequence

        text_contiguous = None
        if self.mode == "text_1d":
            text_contiguous = True
            for value in positions:
                expected = torch.arange(value.shape[-1], dtype=value.dtype)
                flattened = value.reshape(-1, value.shape[-1])
                text_contiguous = text_contiguous and all(
                    torch.equal(row, expected) for row in flattened
                )
            assert text_contiguous
        else:
            assert all(value.ndim == 3 and int(value.shape[0]) in (3, 4) for value in positions)

        return {
            "mode": self.mode,
            "rotary_rebuild_count": len(positions),
            "decoder_layer_call_count": len(layers),
            "sequence_lengths": sorted(
                {int(item["sequence_length"]) for item in layers}, reverse=True
            ),
            "position_lengths": [int(value.shape[-1]) for value in positions],
            "cache_positions_reindexed_contiguously": all(cache_checks),
            "rotary_positions_follow_compaction_subsequence": chain_subsequence,
            "first_rotary_positions_are_baseline_subsequence": baseline_subsequence,
            "text_position_ids_contiguous": text_contiguous,
        }
