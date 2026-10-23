"""Qwen3.5 decoder-boundary FastV adapter.

The first implementation deliberately runs without KV cache. Qwen3.5 uses a
hybrid linear/full-attention cache, so compacting only part of that cache would
silently change the method. The adapter applies one source-ordered selection at
the configured decoder boundary and compacts all subsequent layer inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mllm_comp.compressors.image.fastv import (
    FastVSpec,
    fastv_image_tokens,
    fastv_scorer_query_indices,
)
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.layout import layout_delta_record


class Qwen3_5FastVActivation:
    algorithm = "fastv"
    binding_name = "fastv.image.qwen3_5"

    def __init__(
        self,
        *,
        model: Any,
        runtime: Any = None,
        facade: Any = None,
        stage: Any = None,
        binding: Any,
        binding_config: Any = None,
        execution_phase: Any = None,
        contract_projection: Any = None,
    ) -> None:
        del runtime, facade, execution_phase, contract_projection
        if getattr(binding, "name", None) != self.binding_name:
            raise BackendIntegrationError(
                f"Qwen3.5 FastV received a foreign binding: {getattr(binding, 'name', None)!r}."
            )
        config: dict[str, Any] = {}
        for source in (getattr(stage, "config", None), getattr(binding_config, "config", None)):
            if isinstance(source, Mapping):
                config.update(source)
        allowed = {"prune_layer", "prune_ratio", "score_source", "min_visual_tokens", "random_seed"}
        unknown = set(config) - allowed
        if unknown:
            raise BackendIntegrationError(f"Unknown Qwen3.5 FastV config keys: {sorted(unknown)}.")
        self.spec = FastVSpec(**config)
        if self.spec.prune_layer < 1:
            raise BackendIntegrationError("Qwen3.5 FastV requires prune_layer >= 1 for attention-guided selection.")
        self.model = model
        self._installed = False
        self._original_generate = model.generate
        self._had_generate = "generate" in getattr(model, "__dict__", {})
        self._handles: list[Any] = []
        self._attention_module: Any = None
        self._attention_forward: Any = None
        self._request_records: list[dict[str, Any]] = []
        self._layout_records: list[dict[str, Any]] = []
        self._rollback_errors: list[str] = []
        self._uninstall_errors: list[str] = []

    def install(self) -> None:
        if self._installed:
            return
        import types

        owner = self

        def generate_hook(_model: Any, *args: Any, **kwargs: Any) -> Any:
            call_kwargs = dict(kwargs)
            if args and "input_ids" not in call_kwargs:
                call_kwargs["input_ids"] = args[0]
            input_ids = call_kwargs.get("input_ids")
            if input_ids is None or call_kwargs.get("pixel_values") is None:
                return owner._original_generate(*args, **kwargs)
            # lmms-eval may pass a GenerationConfig whose use_cache=True
            # overrides the YAML scalar. Override per call without mutating the
            # model's persistent generation_config.
            call_kwargs["use_cache"] = False
            return owner._generate_with_decoder_selection(args, kwargs, call_kwargs)

        self.model.generate = types.MethodType(generate_hook, self.model)
        self._installed = True

    def _generate_with_decoder_selection(
        self,
        args: tuple[Any, ...],
        original_kwargs: Mapping[str, Any],
        call_kwargs: dict[str, Any],
    ) -> Any:
        import torch

        input_ids = call_kwargs["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise BackendIntegrationError("Qwen3.5 FastV smoke adapter requires batch size one.")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise BackendIntegrationError("Qwen3.5 config has no image_token_id.")
        image_indices = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
        if image_indices.numel() == 0:
            return self._original_generate(*args, **dict(original_kwargs))

        text_model = getattr(getattr(self.model, "model", None), "language_model", None)
        layers = getattr(text_model, "layers", None)
        if not layers:
            raise BackendIntegrationError("Qwen3.5 language model exposes no decoder layers.")
        boundary = self.spec.prune_layer
        score_layer = boundary - 1
        if boundary >= len(layers):
            raise BackendIntegrationError(
                f"FastV prune_layer={boundary} exceeds Qwen3.5 depth {len(layers)}."
            )
        scorer_block = layers[score_layer]
        attention = getattr(scorer_block, "self_attn", None)
        if getattr(scorer_block, "block_type", None) != "full_attention" or attention is None:
            raise BackendIntegrationError(
                f"FastV score layer {score_layer} is not a full-attention layer; Qwen3.5 layer_types must be respected."
            )

        attention_mask = call_kwargs.get("attention_mask")
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[0]
        query_indices = fastv_scorer_query_indices(
            torch.arange(input_ids.shape[1], device=input_ids.device).eq(int(image_token_id)),
            score_source=self.spec.score_source,
            attention_mask=attention_mask,
        )
        state: dict[str, Any] = {
            "image_indices": image_indices,
            "query_indices": query_indices,
            "attention_key_mask": attention_mask,
            "scorer_query": None,
            "attention_keys": None,
            "base_keep_indices": None,
            "source_length": int(input_ids.shape[1]),
            "record": None,
        }

        import types

        original_prepare = self.model.prepare_inputs_for_generation

        def prepare_hook(
            _model: Any,
            input_ids: Any,
            next_sequence_length: Any = None,
            past_key_values: Any = None,
            attention_mask: Any = None,
            inputs_embeds: Any = None,
            is_first_iteration: Any = False,
            pixel_values: Any = None,
            image_grid_thw: Any = None,
            mm_token_type_ids: Any = None,
            pixel_values_videos: Any = None,
            video_grid_thw: Any = None,
            **prepare_kwargs: Any,
        ) -> Any:
            # Transformers slices to the newest token even without a cache.
            # Recompute the full prefix so each decoder pass still sees the
            # full image span and FastV can rank with the current last token.
            prepare_kwargs["use_cache"] = False
            if input_ids is not None and input_ids.ndim == 2:
                current_images = torch.nonzero(
                    input_ids[0] == int(image_token_id), as_tuple=False
                ).flatten()
                if current_images.numel():
                    state["image_indices"] = current_images
                    state["attention_key_mask"] = (
                        attention_mask[0]
                        if attention_mask is not None and attention_mask.ndim == 2
                        else None
                    )
                    state["query_indices"] = fastv_scorer_query_indices(
                        input_ids[0].eq(int(image_token_id)),
                        score_source=self.spec.score_source,
                        attention_mask=attention_mask,
                    )
                    state["source_length"] = int(input_ids.shape[1])
                    state["scorer_query"] = None
                    state["attention_keys"] = None
                    state["base_keep_indices"] = None
            return original_prepare(
                input_ids,
                None,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                is_first_iteration=is_first_iteration,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                **prepare_kwargs,
            )

        self.model.prepare_inputs_for_generation = types.MethodType(
            prepare_hook, self.model
        )
        previous_rope = getattr(self.model.model, "rope_deltas", None)
        try:
            self._install_request_hooks(text_model, layers, score_layer, boundary, state)
            invoke_kwargs = dict(original_kwargs)
            invoke_kwargs["use_cache"] = False
            output = self._original_generate(*args, **invoke_kwargs)
        finally:
            self._remove_request_hooks()
            self.model.prepare_inputs_for_generation = original_prepare
            self.model.model.rope_deltas = previous_rope

        if state["record"] is None:
            raise BackendIntegrationError("FastV decoder boundary did not execute during generation.")
        record = state["record"]
        record["generated"] = True
        record["kv_cache"] = False
        self._request_records.append(record)
        return output

    def _install_request_hooks(
        self, text_model: Any, layers: Any, score_layer: int, boundary: int, state: dict[str, Any]
    ) -> None:
        import types
        import torch
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        attention = layers[score_layer].self_attn
        original_forward = attention.forward
        self._attention_module = attention
        self._attention_forward = original_forward
        owner = self

        def capture_forward(module: Any, hidden_states: Any, position_embeddings: Any, *args: Any, **kwargs: Any) -> Any:
            if state["scorer_query"] is None:
                batch, sequence, _ = hidden_states.shape
                head_dim = int(module.head_dim)
                query, _gate = torch.chunk(
                    module.q_proj(hidden_states).view(batch, sequence, -1, head_dim * 2), 2, dim=-1
                )
                query = module.q_norm(query).transpose(1, 2)
                key = module.k_norm(
                    module.k_proj(hidden_states).view(batch, sequence, -1, head_dim)
                ).transpose(1, 2)
                cos, sin = position_embeddings
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
                repeat = query.shape[1] // key.shape[1]
                if repeat > 1:
                    key = key.repeat_interleave(repeat, dim=1)
                qidx = state["query_indices"].to(query.device)
                if int(qidx.max().item()) >= sequence:
                    raise BackendIntegrationError("FastV scorer query index exceeds decoder sequence length.")
                state["scorer_query"] = query[0, :, qidx[0], :].detach()
                state["attention_keys"] = key[0].transpose(0, 1).detach()
            return original_forward(hidden_states, position_embeddings, *args, **kwargs)

        attention.forward = types.MethodType(capture_forward, attention)

        def prune_pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise BackendIntegrationError("FastV expected decoder states shaped [1, sequence, hidden].")
            sequence = int(hidden.shape[1])
            if state["base_keep_indices"] is None:
                if state["scorer_query"] is None or state["attention_keys"] is None:
                    raise BackendIntegrationError("FastV attention Q/K was not captured before its prune boundary.")
                result = fastv_image_tokens(
                    hidden[0],
                    image_indices=state["image_indices"].to(hidden.device),
                    scorer_query=state["scorer_query"],
                    attention_keys=state["attention_keys"],
                    attention_key_mask=state["attention_key_mask"],
                    spec=self.spec,
                    media_unit="llm_token",
                    source_layout={"backbone": "qwen3_5", "score_layer": score_layer},
                    result_layout={"backbone": "qwen3_5", "prune_layer": boundary},
                )
                state["base_keep_indices"] = result.kept_indices.detach()
                state["source_length"] = sequence
                self._layout_records.append(layout_delta_record(result.layout_delta))
                state["record"] = {
                    "algorithm": "fastv",
                    "score_layer": score_layer,
                    "prune_layer": boundary,
                    "source_tokens": sequence,
                    "result_tokens": int(result.hidden_states.shape[0]),
                    "image_tokens_before": int(state["image_indices"].numel()),
                    "image_tokens_after": int(result.kept_image_indices.numel()),
                    "score_source": self.spec.score_source,
                    "cache_policy": "disabled_no_cache_recompute",
                    "attention_layer_type": "full_attention",
                }
                hidden = result.hidden_states.unsqueeze(0)
                keep = result.kept_indices
            else:
                keep = state["base_keep_indices"].to(hidden.device)
                if sequence == int(state["source_length"]):
                    hidden = hidden.index_select(1, keep)
                elif sequence != int(keep.numel()):
                    raise BackendIntegrationError(
                        "FastV decoder sequence length is neither the source nor the compact layout."
                    )

            def select_sequence(value: Any, dims: tuple[int, ...]) -> Any:
                if not torch.is_tensor(value):
                    return value
                result_value = value
                for dim in dims:
                    result_value = result_value.index_select(dim, keep.to(result_value.device))
                return result_value

            mutable_args = list(args)
            if mutable_args:
                mutable_args[0] = hidden
            else:
                kwargs["hidden_states"] = hidden
            pe = kwargs.get("position_embeddings")
            if isinstance(pe, tuple):
                kwargs["position_embeddings"] = tuple(
                    select_sequence(value, (1,)) for value in pe
                )
            mask = kwargs.get("attention_mask")
            if torch.is_tensor(mask):
                if mask.ndim == 4:
                    kwargs["attention_mask"] = select_sequence(mask, (-2, -1))
                elif mask.ndim == 2:
                    kwargs["attention_mask"] = select_sequence(mask, (-1,))
            pos = kwargs.get("position_ids")
            if torch.is_tensor(pos):
                kwargs["position_ids"] = select_sequence(pos, (-1,))
            cache_position = kwargs.get("cache_position")
            if torch.is_tensor(cache_position) and cache_position.ndim == 1 and cache_position.numel() == sequence:
                kwargs["cache_position"] = select_sequence(cache_position, (0,))
            return tuple(mutable_args), kwargs

        for layer_index in range(boundary, len(layers)):
            self._handles.append(
                layers[layer_index].register_forward_pre_hook(
                    prune_pre_hook, with_kwargs=True
                )
            )

    def _remove_request_hooks(self) -> None:
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except Exception as exc:  # noqa: BLE001
                self._rollback_errors.append(repr(exc))
        self._handles.clear()
        if self._attention_module is not None and self._attention_forward is not None:
            try:
                self._attention_module.forward = self._attention_forward
            except Exception as exc:  # noqa: BLE001
                self._rollback_errors.append(repr(exc))
        self._attention_module = None
        self._attention_forward = None

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._remove_request_hooks()
        if self._had_generate:
            self.model.generate = self._original_generate
        else:
            delattr(self.model, "generate")
        self._installed = False

    def clear_diagnostic_records(self) -> None:
        self._request_records.clear()
        self._layout_records.clear()
        self._rollback_errors.clear()
        self._uninstall_errors.clear()

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backbone": "qwen3_5",
            "algorithm": self.algorithm,
            "variant": "fastv.image",
            "binding": self.binding_name,
            "target_site": "decoder_prune_boundary",
            "prune_layer": self.spec.prune_layer,
            "score_layer": self.spec.prune_layer - 1,
            "prune_ratio": self.spec.prune_ratio,
            "score_source": self.spec.score_source,
            "cache_policy": "use_cache_false_only",
            "installed": self._installed,
            "generation_calls": len(self._request_records),
            "layout_delta_records_count": len(self._layout_records),
            "layout_delta_records": list(self._layout_records),
            "requests": list(self._request_records),
            "rollback_errors": list(self._rollback_errors),
            "uninstall_errors": list(self._uninstall_errors),
            "video_support": False,
        }


class Qwen3_5PyramidDropActivation(Qwen3_5FastVActivation):
    """Multi-stage PyramidDrop over Qwen3.5's hybrid decoder, without KV cache."""

    algorithm = "pyramid_drop"
    binding_name = "pyramid_drop.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from mllm_comp.compressors.image.pyramid_drop import PyramidDropSpec

        binding = kwargs.get("binding")
        if getattr(binding, "name", None) != self.binding_name:
            raise BackendIntegrationError(
                f"Qwen3.5 PyramidDrop received a foreign binding: {getattr(binding, 'name', None)!r}."
            )
        self.model = kwargs["model"]
        merged: dict[str, Any] = {}
        stage = kwargs.get("stage")
        binding_config = kwargs.get("binding_config")
        for source in (getattr(stage, "config", None), getattr(binding_config, "config", None)):
            if isinstance(source, Mapping):
                merged.update(source)
        allowed = {"keep_ratio", "minimum_keep_tokens", "num_stages", "restore_source_order", "layer_indices"}
        unknown = set(merged) - allowed
        if unknown:
            raise BackendIntegrationError(f"Unknown Qwen3.5 PyramidDrop config keys: {sorted(unknown)}.")
        self.spec = PyramidDropSpec(
            keep_ratio=float(merged.get("keep_ratio", 0.5)),
            minimum_keep_tokens=int(merged.get("minimum_keep_tokens", 2)),
            num_stages=int(merged.get("num_stages", 4)),
            restore_source_order=bool(merged.get("restore_source_order", True)),
        )
        self.config = merged
        self._installed = False
        self._original_generate = self.model.generate
        self._had_generate = "generate" in getattr(self.model, "__dict__", {})
        self._handles: list[Any] = []
        self._attention_modules: list[Any] = []
        self._attention_forwards: list[Any] = []
        self._request_records: list[dict[str, Any]] = []
        self._layout_records: list[dict[str, Any]] = []
        self._rollback_errors: list[str] = []
        self._uninstall_errors: list[str] = []

    def _generate_with_decoder_selection(
        self,
        args: tuple[Any, ...],
        original_kwargs: Mapping[str, Any],
        call_kwargs: dict[str, Any],
    ) -> Any:
        import torch
        import types
        from mllm_comp.compressors.image.pyramid_drop import (
            pyramid_attention_key_visibility,
            pyramid_attention_scores,
            pyramid_instruction_indices,
        )

        input_ids = call_kwargs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise BackendIntegrationError("Qwen3.5 PyramidDrop requires one tokenized prompt and batch size one.")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise BackendIntegrationError("Qwen3.5 config has no image_token_id.")
        image_indices = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
        if image_indices.numel() == 0:
            return self._original_generate(*args, **dict(original_kwargs))

        text_model = self.model.model.language_model
        layers = text_model.layers
        text_config = self.model.config.text_config
        layer_types = tuple(text_config.layer_types)
        depth = len(layers)
        configured = self.config.get("layer_indices")
        if configured is None:
            boundaries = tuple(
                (depth * stage) // self.spec.num_stages - 1
                for stage in range(1, self.spec.num_stages)
            )
        else:
            boundaries = tuple(int(index) for index in configured)
        if not boundaries or tuple(sorted(set(boundaries))) != boundaries:
            raise BackendIntegrationError("PyramidDrop layer_indices must be strictly increasing and non-empty.")
        if any(index < 0 or index >= depth - 1 for index in boundaries):
            raise BackendIntegrationError("PyramidDrop boundaries must leave a later decoder layer.")
        score_layers = boundaries
        boundary_layers = tuple(index + 1 for index in boundaries)
        if any(layer_types[index] != "full_attention" for index in score_layers):
            raise BackendIntegrationError(
                "Qwen3.5 PyramidDrop score layers must coincide with full-attention blocks; "
                f"got {[(index, layer_types[index]) for index in score_layers]} for the configured schedule."
            )

        attention_mask = call_kwargs.get("attention_mask")
        source_mask = attention_mask[0] if attention_mask is not None and attention_mask.ndim == 2 else None
        state: dict[str, Any] = {
            "source_length": int(input_ids.shape[1]),
            "source_attention_mask": source_mask,
            "current_indices": torch.arange(input_ids.shape[1], device=input_ids.device),
            "image_indices": image_indices,
            "scores": {},
            "stage_records": [],
            "layout_records_start": len(self._layout_records),
        }

        original_prepare = self.model.prepare_inputs_for_generation

        def prepare_hook(
            _model: Any,
            input_ids: Any,
            next_sequence_length: Any = None,
            past_key_values: Any = None,
            attention_mask: Any = None,
            inputs_embeds: Any = None,
            is_first_iteration: Any = False,
            pixel_values: Any = None,
            image_grid_thw: Any = None,
            mm_token_type_ids: Any = None,
            pixel_values_videos: Any = None,
            video_grid_thw: Any = None,
            **prepare_kwargs: Any,
        ) -> Any:
            prepare_kwargs["use_cache"] = False
            if input_ids is not None and input_ids.ndim == 2:
                current_images = torch.nonzero(
                    input_ids[0] == int(image_token_id), as_tuple=False
                ).flatten()
                if current_images.numel():
                    state["source_length"] = int(input_ids.shape[1])
                    state["source_attention_mask"] = (
                        attention_mask[0]
                        if attention_mask is not None and attention_mask.ndim == 2
                        else None
                    )
                    state["current_indices"] = torch.arange(
                        input_ids.shape[1], device=input_ids.device
                    )
                    state["image_indices"] = current_images
                    state["scores"] = {}
                    state["stage_records"] = []
            return original_prepare(
                input_ids,
                None,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                is_first_iteration=is_first_iteration,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                **prepare_kwargs,
            )

        self.model.prepare_inputs_for_generation = types.MethodType(prepare_hook, self.model)
        previous_rope = getattr(self.model.model, "rope_deltas", None)
        try:
            self._install_pyramid_hooks(
                layers, score_layers, boundary_layers, state,
                pyramid_instruction_indices, pyramid_attention_key_visibility,
                pyramid_attention_scores,
            )
            invoke_kwargs = dict(original_kwargs)
            invoke_kwargs["use_cache"] = False
            output = self._original_generate(*args, **invoke_kwargs)
        finally:
            self._remove_request_hooks()
            self.model.prepare_inputs_for_generation = original_prepare
            self.model.model.rope_deltas = previous_rope

        if not state["stage_records"]:
            raise BackendIntegrationError("PyramidDrop decoder stages did not run during generation.")
        self._request_records.append(
            {
                "algorithm": self.algorithm,
                "generated": True,
                "kv_cache": False,
                "stage_records": list(state["stage_records"]),
                "layer_indices": list(boundaries),
                "source_length": state["source_length"],
            }
        )
        return output

    def _install_pyramid_hooks(
        self,
        layers: Any,
        score_layers: tuple[int, ...],
        boundary_layers: tuple[int, ...],
        state: dict[str, Any],
        instruction_indices_fn: Any,
        visibility_fn: Any,
        score_fn: Any,
    ) -> None:
        import types
        import torch
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        from mllm_comp.compressors.image.pyramid_drop import pyramid_drop

        owner = self
        for layer_index in score_layers:
            attention = layers[layer_index].self_attn
            original_forward = attention.forward
            self._attention_modules.append(attention)
            self._attention_forwards.append(original_forward)

            def capture_forward(
                module: Any,
                hidden_states: Any,
                position_embeddings: Any,
                *args: Any,
                __layer_index: int = layer_index,
                __original_forward: Any = original_forward,
                **kwargs: Any,
            ) -> Any:
                if __layer_index not in state["scores"]:
                    batch, sequence, _ = hidden_states.shape
                    head_dim = int(module.head_dim)
                    query, _gate = torch.chunk(
                        module.q_proj(hidden_states).view(batch, sequence, -1, head_dim * 2), 2, dim=-1
                    )
                    query = module.q_norm(query).transpose(1, 2)
                    key = module.k_norm(
                        module.k_proj(hidden_states).view(batch, sequence, -1, head_dim)
                    ).transpose(1, 2)
                    cos, sin = position_embeddings
                    query, key = apply_rotary_pos_emb(query, key, cos, sin)
                    repeat = query.shape[1] // key.shape[1]
                    if repeat > 1:
                        key = key.repeat_interleave(repeat, dim=1)
                    current_indices = state["current_indices"].to(hidden_states.device)
                    image_positions = state["image_indices"].to(hidden_states.device)
                    padding = state["source_attention_mask"]
                    if padding is not None:
                        padding = padding.to(hidden_states.device).index_select(0, current_indices)
                    image_mask = torch.zeros(sequence, dtype=torch.bool, device=hidden_states.device)
                    image_mask[image_positions] = True
                    queries = instruction_indices_fn(image_mask, attention_mask=padding)
                    visibility = visibility_fn(
                        padding,
                        instruction_indices=queries,
                        key_count=sequence,
                    )
                    query_values = query[0, :, queries, :].permute(1, 0, 2).contiguous()
                    key_values = key[0].transpose(0, 1).contiguous()
                    scores = score_fn(
                        instruction_query=query_values,
                        image_keys=key_values.index_select(0, image_positions),
                        image_count=int(image_positions.numel()),
                        device=hidden_states.device,
                        attention_keys=key_values,
                        attention_key_mask=visibility,
                        image_key_indices=image_positions,
                    )
                    state["scores"][__layer_index] = scores.detach()
                return __original_forward(hidden_states, position_embeddings, *args, **kwargs)

            attention.forward = types.MethodType(capture_forward, attention)

        first_boundary = boundary_layers[0]

        def boundary_hook_factory(layer_index: int):
            def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise BackendIntegrationError("PyramidDrop expected decoder states shaped [1, sequence, hidden].")
                sequence = int(hidden.shape[1])
                if layer_index in boundary_layers:
                    score_layer = layer_index - 1
                    image_indices = state["image_indices"].to(hidden.device)
                    scores = state["scores"].pop(score_layer, None)
                    if scores is None:
                        raise BackendIntegrationError(
                            f"PyramidDrop score layer {score_layer} did not produce attention scores."
                        )
                    result = pyramid_drop(
                        hidden[0],
                        image_indices=image_indices,
                        image_attention_scores=scores,
                        spec=self.spec,
                        media_unit="llm_token",
                        source_layout={"backbone": "qwen3_5", "completed_layer": score_layer},
                        result_layout={"backbone": "qwen3_5", "stage_boundary": layer_index},
                    )
                    old_current = state["current_indices"].to(hidden.device)
                    state["current_indices"] = old_current.index_select(0, result.kept_indices)
                    state["image_indices"] = torch.nonzero(
                        torch.isin(result.kept_indices, result.kept_image_indices),
                        as_tuple=False,
                    ).flatten()
                    hidden = result.hidden_states.unsqueeze(0)
                    owner._layout_records.append(layout_delta_record(result.layout_delta))
                    state["stage_records"].append(
                        {
                            "score_layer": score_layer,
                            "prune_boundary": layer_index,
                            "source_sequence_tokens": sequence,
                            "result_sequence_tokens": int(hidden.shape[1]),
                            "image_tokens_after": int(state["image_indices"].numel()),
                            "keep_ratio": self.spec.keep_ratio,
                        }
                    )
                current_indices = state["current_indices"].to(hidden.device)
                current_length = int(current_indices.numel())
                if int(hidden.shape[1]) == int(state["source_length"]):
                    if current_length != int(state["source_length"]):
                        hidden = hidden.index_select(1, current_indices)
                elif int(hidden.shape[1]) != current_length:
                    raise BackendIntegrationError(
                        "PyramidDrop decoder sequence is neither source nor current compact length."
                    )

                def select_axis(value: Any, dim: int) -> Any:
                    if not torch.is_tensor(value):
                        return value
                    length = int(value.shape[dim])
                    if length == int(state["source_length"]) and current_length != int(state["source_length"]):
                        return value.index_select(dim, current_indices.to(value.device))
                    if length not in {current_length, int(state["source_length"])}:
                        raise BackendIntegrationError(
                            f"PyramidDrop sequence field has unsupported length {length}; "
                            f"expected {current_length} or {state['source_length']}."
                        )
                    return value

                mutable_args = list(args)
                if mutable_args:
                    mutable_args[0] = hidden
                else:
                    kwargs["hidden_states"] = hidden
                pe = kwargs.get("position_embeddings")
                if isinstance(pe, tuple):
                    kwargs["position_embeddings"] = tuple(select_axis(value, 1) for value in pe)
                mask = kwargs.get("attention_mask")
                if torch.is_tensor(mask):
                    if mask.ndim == 4:
                        mask = select_axis(mask, -2)
                        mask = select_axis(mask, -1)
                    elif mask.ndim == 2:
                        mask = select_axis(mask, -1)
                    kwargs["attention_mask"] = mask
                pos = kwargs.get("position_ids")
                if torch.is_tensor(pos):
                    kwargs["position_ids"] = select_axis(pos, -1)
                cache_position = kwargs.get("cache_position")
                if torch.is_tensor(cache_position) and cache_position.ndim == 1:
                    kwargs["cache_position"] = select_axis(cache_position, 0)
                return tuple(mutable_args), kwargs
            return pre_hook

        for layer_index in range(first_boundary, len(layers)):
            self._handles.append(
                layers[layer_index].register_forward_pre_hook(
                    boundary_hook_factory(layer_index), with_kwargs=True
                )
            )

    def _remove_request_hooks(self) -> None:
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except Exception as exc:  # noqa: BLE001
                self._rollback_errors.append(repr(exc))
        self._handles.clear()
        for module, original in zip(self._attention_modules, self._attention_forwards):
            try:
                module.forward = original
            except Exception as exc:  # noqa: BLE001
                self._rollback_errors.append(repr(exc))
        self._attention_modules.clear()
        self._attention_forwards.clear()

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backbone": "qwen3_5",
            "algorithm": self.algorithm,
            "variant": "pyramid_drop.image",
            "binding": self.binding_name,
            "target_site": "decoder_stage_boundaries",
            "keep_ratio_per_stage": self.spec.keep_ratio,
            "num_stages": self.spec.num_stages,
            "cache_policy": "use_cache_false_only",
            "installed": self._installed,
            "generation_calls": len(self._request_records),
            "layout_delta_records_count": len(self._layout_records),
            "layout_delta_records": list(self._layout_records),
            "requests": list(self._request_records),
            "rollback_errors": list(self._rollback_errors),
            "uninstall_errors": list(self._uninstall_errors),
            "video_support": False,
        }


class Qwen3_5DARTActivation(Qwen3_5FastVActivation):
    """Single-boundary DART using Qwen3.5 rotary keys and decoder-normalized states."""

    algorithm = "dart"
    binding_name = "dart.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from dataclasses import fields
        from mllm_comp.compressors.image.dart import DARTSpec

        binding = kwargs.get("binding")
        if getattr(binding, "name", None) != self.binding_name:
            raise BackendIntegrationError(
                f"Qwen3.5 DART received a foreign binding: {getattr(binding, 'name', None)!r}."
            )
        self.model = kwargs["model"]
        config: dict[str, Any] = {}
        for source in (getattr(kwargs.get("stage"), "config", None), getattr(kwargs.get("binding_config"), "config", None)):
            if isinstance(source, Mapping):
                config.update(source)
        allowed = {field.name for field in fields(DARTSpec)}
        unknown = set(config) - allowed
        if unknown:
            raise BackendIntegrationError(f"Unknown Qwen3.5 DART config keys: {sorted(unknown)}.")
        self.spec = DARTSpec(**config)
        self._installed = False
        self._original_generate = self.model.generate
        self._had_generate = "generate" in getattr(self.model, "__dict__", {})
        self._handles: list[Any] = []
        self._attention_module: Any = None
        self._attention_forward: Any = None
        self._request_records: list[dict[str, Any]] = []
        self._layout_records: list[dict[str, Any]] = []
        self._rollback_errors: list[str] = []
        self._uninstall_errors: list[str] = []

    def _generate_with_decoder_selection(
        self,
        args: tuple[Any, ...],
        original_kwargs: Mapping[str, Any],
        call_kwargs: dict[str, Any],
    ) -> Any:
        import torch
        import types

        input_ids = call_kwargs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise BackendIntegrationError("Qwen3.5 DART requires tokenized input and batch size one.")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise BackendIntegrationError("Qwen3.5 config has no image_token_id.")
        image_indices = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
        if image_indices.numel() == 0:
            return self._original_generate(*args, **dict(original_kwargs))
        text_model = self.model.model.language_model
        layers = text_model.layers
        boundary = self.spec.prune_after_layer
        score_layer = boundary - 1
        layer_types = tuple(self.model.config.text_config.layer_types)
        if boundary >= len(layers) or layer_types[score_layer] != "full_attention":
            raise BackendIntegrationError(
                f"DART requires a preceding full-attention score layer; layer {score_layer} is "
                f"{layer_types[score_layer] if score_layer < len(layer_types) else 'out of range'}."
            )
        attention_mask = call_kwargs.get("attention_mask")
        state: dict[str, Any] = {
            "source_length": int(input_ids.shape[1]),
            "source_attention_mask": attention_mask[0] if attention_mask is not None and attention_mask.ndim == 2 else None,
            "image_indices": image_indices,
            "text_candidates": None,
            "key_states": None,
            "base_keep_indices": None,
            "record": None,
        }
        original_prepare = self.model.prepare_inputs_for_generation

        def prepare_hook(
            _model: Any,
            input_ids: Any,
            next_sequence_length: Any = None,
            past_key_values: Any = None,
            attention_mask: Any = None,
            inputs_embeds: Any = None,
            is_first_iteration: Any = False,
            pixel_values: Any = None,
            image_grid_thw: Any = None,
            mm_token_type_ids: Any = None,
            pixel_values_videos: Any = None,
            video_grid_thw: Any = None,
            **prepare_kwargs: Any,
        ) -> Any:
            prepare_kwargs["use_cache"] = False
            if input_ids is not None and input_ids.ndim == 2:
                current_images = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
                if current_images.numel():
                    mask_row = attention_mask[0] if attention_mask is not None and attention_mask.ndim == 2 else None
                    image_mask = input_ids[0].eq(int(image_token_id))
                    state.update(
                        source_length=int(input_ids.shape[1]),
                        source_attention_mask=mask_row,
                        image_indices=current_images,
                        text_candidates=dart_post_image_text_indices(image_mask, attention_mask=mask_row),
                        key_states=None,
                        base_keep_indices=None,
                    )
            return original_prepare(
                input_ids,
                None,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                is_first_iteration=is_first_iteration,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                **prepare_kwargs,
            )

        from mllm_comp.compressors.image.dart import dart_post_image_text_indices

        self.model.prepare_inputs_for_generation = types.MethodType(prepare_hook, self.model)
        previous_rope = getattr(self.model.model, "rope_deltas", None)
        try:
            self._install_dart_hooks(text_model, layers, score_layer, boundary, state)
            invoke_kwargs = dict(original_kwargs)
            invoke_kwargs["use_cache"] = False
            output = self._original_generate(*args, **invoke_kwargs)
        finally:
            self._remove_request_hooks()
            self.model.prepare_inputs_for_generation = original_prepare
            self.model.model.rope_deltas = previous_rope
        if state["record"] is None:
            raise BackendIntegrationError("DART decoder boundary did not execute during generation.")
        self._request_records.append({**state["record"], "generated": True, "kv_cache": False})
        return output

    def _install_dart_hooks(
        self, text_model: Any, layers: Any, score_layer: int, boundary: int, state: dict[str, Any]
    ) -> None:
        import types
        import torch
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        from mllm_comp.compressors.image.dart import dart_image_tokens

        attention = layers[score_layer].self_attn
        original_forward = attention.forward
        self._attention_module, self._attention_forward = attention, original_forward

        def capture_keys(module: Any, hidden_states: Any, position_embeddings: Any, *args: Any, **kwargs: Any) -> Any:
            if state["key_states"] is None:
                batch, sequence, _ = hidden_states.shape
                head_dim = int(module.head_dim)
                query, _gate = torch.chunk(
                    module.q_proj(hidden_states).view(batch, sequence, -1, head_dim * 2), 2, dim=-1
                )
                query = module.q_norm(query).transpose(1, 2)
                key = module.k_norm(
                    module.k_proj(hidden_states).view(batch, sequence, -1, head_dim)
                ).transpose(1, 2)
                cos, sin = position_embeddings
                _query, key = apply_rotary_pos_emb(query, key, cos, sin)
                state["key_states"] = key[0].transpose(0, 1).detach()
            return original_forward(hidden_states, position_embeddings, *args, **kwargs)

        attention.forward = types.MethodType(capture_keys, attention)

        def prune_pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise BackendIntegrationError("DART expected decoder states shaped [1, sequence, hidden].")
            sequence = int(hidden.shape[1])
            if state["base_keep_indices"] is None:
                if state["key_states"] is None or state["text_candidates"] is None:
                    raise BackendIntegrationError("DART key/prompt pivot state was not captured before pruning.")
                result = dart_image_tokens(
                    hidden[0],
                    text_model.norm(hidden[0]),
                    state["key_states"],
                    image_indices=state["image_indices"].to(hidden.device),
                    text_candidate_indices=state["text_candidates"].to(hidden.device),
                    spec=self.spec,
                    media_unit="llm_token",
                    source_layout={"backbone": "qwen3_5", "score_layer": score_layer},
                    result_layout={"backbone": "qwen3_5", "prune_layer": boundary},
                )
                state["base_keep_indices"] = result.kept_indices.detach()
                state["source_length"] = sequence
                self._layout_records.append(layout_delta_record(result.layout_delta))
                state["record"] = {
                    "algorithm": "dart",
                    "score_layer": score_layer,
                    "prune_layer": boundary,
                    "source_tokens": sequence,
                    "result_tokens": int(result.hidden_states.shape[0]),
                    "image_tokens_before": int(state["image_indices"].numel()),
                    "image_tokens_after": int(result.kept_image_indices.numel()),
                    "image_pivots": int(result.image_pivot_indices.numel()),
                    "text_pivots": int(result.text_pivot_indices.numel()),
                    "cache_policy": "disabled_no_cache_recompute",
                }
                hidden = result.hidden_states.unsqueeze(0)
                keep = result.kept_indices
            else:
                keep = state["base_keep_indices"].to(hidden.device)
                if sequence == int(state["source_length"]):
                    hidden = hidden.index_select(1, keep)
                elif sequence != int(keep.numel()):
                    raise BackendIntegrationError("DART decoder sequence is neither source nor compact length.")

            def select_sequence(value: Any, dims: tuple[int, ...]) -> Any:
                if not torch.is_tensor(value):
                    return value
                result_value = value
                for dim in dims:
                    length = int(result_value.shape[dim])
                    if length == int(state["source_length"]) and int(keep.numel()) != int(state["source_length"]):
                        result_value = result_value.index_select(dim, keep.to(result_value.device))
                    elif length not in {int(state["source_length"]), int(keep.numel())}:
                        raise BackendIntegrationError(
                            f"DART sequence field has unsupported length {length}."
                        )
                return result_value

            mutable_args = list(args)
            if mutable_args:
                mutable_args[0] = hidden
            else:
                kwargs["hidden_states"] = hidden
            pe = kwargs.get("position_embeddings")
            if isinstance(pe, tuple):
                kwargs["position_embeddings"] = tuple(select_sequence(value, (1,)) for value in pe)
            mask = kwargs.get("attention_mask")
            if torch.is_tensor(mask):
                if mask.ndim == 4:
                    kwargs["attention_mask"] = select_sequence(mask, (-2, -1))
                elif mask.ndim == 2:
                    kwargs["attention_mask"] = select_sequence(mask, (-1,))
            pos = kwargs.get("position_ids")
            if torch.is_tensor(pos):
                kwargs["position_ids"] = select_sequence(pos, (-1,))
            cache_position = kwargs.get("cache_position")
            if torch.is_tensor(cache_position) and cache_position.ndim == 1 and cache_position.numel() == sequence:
                kwargs["cache_position"] = select_sequence(cache_position, (0,))
            return tuple(mutable_args), kwargs

        for layer_index in range(boundary, len(layers)):
            self._handles.append(
                layers[layer_index].register_forward_pre_hook(
                    prune_pre_hook, with_kwargs=True
                )
            )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backbone": "qwen3_5",
            "algorithm": self.algorithm,
            "variant": "dart.image",
            "binding": self.binding_name,
            "target_site": "decoder_prune_boundary",
            "prune_after_layer": self.spec.prune_after_layer,
            "score_layer": self.spec.prune_after_layer - 1,
            "retain_ratio": self.spec.retain_ratio,
            "cache_policy": "use_cache_false_only",
            "installed": self._installed,
            "generation_calls": len(self._request_records),
            "layout_delta_records_count": len(self._layout_records),
            "layout_delta_records": list(self._layout_records),
            "requests": list(self._request_records),
            "rollback_errors": list(self._rollback_errors),
            "uninstall_errors": list(self._uninstall_errors),
            "video_support": False,
        }


class Qwen3_5IVCPruneActivation(Qwen3_5FastVActivation):
    """Single-boundary IVC-Prune using native M-RoPE and decoder value states."""

    algorithm = "ivc_prune"
    binding_name = "ivc_prune.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from dataclasses import fields
        from mllm_comp.compressors.image.ivc_prune import IVCPruneSpec

        binding = kwargs.get("binding")
        if getattr(binding, "name", None) != self.binding_name:
            raise BackendIntegrationError(
                f"Qwen3.5 IVC-Prune received a foreign binding: {getattr(binding, 'name', None)!r}."
            )
        self.model = kwargs["model"]
        config: dict[str, Any] = {}
        for source in (getattr(kwargs.get("stage"), "config", None), getattr(kwargs.get("binding_config"), "config", None)):
            if isinstance(source, Mapping):
                config.update(source)
        allowed = {field.name for field in fields(IVCPruneSpec)}
        unknown = set(config) - allowed
        if unknown:
            raise BackendIntegrationError(f"Unknown Qwen3.5 IVC-Prune config keys: {sorted(unknown)}.")
        self.spec = IVCPruneSpec(**config)
        self._installed = False
        self._original_generate = self.model.generate
        self._had_generate = "generate" in getattr(self.model, "__dict__", {})
        self._handles: list[Any] = []
        self._attention_module = None
        self._attention_forward = None
        self._request_records: list[dict[str, Any]] = []
        self._layout_records: list[dict[str, Any]] = []
        self._rollback_errors: list[str] = []
        self._uninstall_errors: list[str] = []

    def _generate_with_decoder_selection(
        self,
        args: tuple[Any, ...],
        original_kwargs: Mapping[str, Any],
        call_kwargs: dict[str, Any],
    ) -> Any:
        import torch
        import types
        from mllm_comp.compressors.image.ivc_prune import ivc_post_image_text_indices

        input_ids = call_kwargs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise BackendIntegrationError("Qwen3.5 IVC-Prune requires tokenized input and batch size one.")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise BackendIntegrationError("Qwen3.5 config has no image_token_id.")
        image_indices = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
        if image_indices.numel() == 0:
            return self._original_generate(*args, **dict(original_kwargs))
        text_model = self.model.model.language_model
        layers = text_model.layers
        boundary = self.spec.prune_layer
        score_layer = boundary - 1
        layer_types = tuple(self.model.config.text_config.layer_types)
        if boundary >= len(layers) or layer_types[score_layer] != "full_attention":
            raise BackendIntegrationError(
                f"IVC-Prune requires full-attention value states before boundary {boundary}; "
                f"layer {score_layer} is {layer_types[score_layer] if score_layer < len(layer_types) else 'out of range'}."
            )
        attention_mask = call_kwargs.get("attention_mask")
        state: dict[str, Any] = {
            "source_length": int(input_ids.shape[1]),
            "source_attention_mask": attention_mask[0] if attention_mask is not None and attention_mask.ndim == 2 else None,
            "image_indices": image_indices,
            "text_queries": None,
            "base_keep_indices": None,
            "record": None,
        }

        original_prepare = self.model.prepare_inputs_for_generation

        def prepare_hook(
            _model: Any,
            input_ids: Any,
            next_sequence_length: Any = None,
            past_key_values: Any = None,
            attention_mask: Any = None,
            inputs_embeds: Any = None,
            is_first_iteration: Any = False,
            pixel_values: Any = None,
            image_grid_thw: Any = None,
            mm_token_type_ids: Any = None,
            pixel_values_videos: Any = None,
            video_grid_thw: Any = None,
            **prepare_kwargs: Any,
        ) -> Any:
            prepare_kwargs["use_cache"] = False
            if input_ids is not None and input_ids.ndim == 2:
                current_images = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
                if current_images.numel():
                    mask_row = attention_mask[0] if attention_mask is not None and attention_mask.ndim == 2 else None
                    image_mask = input_ids[0].eq(int(image_token_id))
                    state.update(
                        source_length=int(input_ids.shape[1]),
                        source_attention_mask=mask_row,
                        image_indices=current_images,
                        text_queries=ivc_post_image_text_indices(
                            image_mask, attention_mask=mask_row,
                            skip=self.spec.post_image_text_skip,
                        ),
                        base_keep_indices=None,
                    )
            return original_prepare(
                input_ids, None,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                is_first_iteration=is_first_iteration,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                **prepare_kwargs,
            )

        self.model.prepare_inputs_for_generation = types.MethodType(prepare_hook, self.model)
        previous_rope = getattr(self.model.model, "rope_deltas", None)
        try:
            self._install_ivc_hooks(layers, boundary, state)
            invoke_kwargs = dict(original_kwargs)
            invoke_kwargs["use_cache"] = False
            output = self._original_generate(*args, **invoke_kwargs)
        finally:
            self._remove_request_hooks()
            self.model.prepare_inputs_for_generation = original_prepare
            self.model.model.rope_deltas = previous_rope
        if state["record"] is None:
            raise BackendIntegrationError("IVC-Prune decoder boundary did not execute during generation.")
        self._request_records.append({**state["record"], "generated": True, "kv_cache": False})
        return output

    def _install_ivc_hooks(self, layers: Any, boundary: int, state: dict[str, Any]) -> None:
        import torch
        from mllm_comp.compressors.image.ivc_prune import (
            IVCPruneSpec,
            ivc_coordinate_scores,
            ivc_prune_image_tokens,
            project_ivc_value_states,
        )
        if not isinstance(self.spec, IVCPruneSpec):
            raise BackendIntegrationError("IVC-Prune spec was not initialized.")

        def prune_pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise BackendIntegrationError("IVC-Prune expected decoder states shaped [1, sequence, hidden].")
            sequence = int(hidden.shape[1])
            if state["base_keep_indices"] is None:
                positions = kwargs.get("position_embeddings")
                if not isinstance(positions, tuple) or len(positions) != 2:
                    raise BackendIntegrationError("IVC-Prune requires native decoder RoPE cosine/sine tensors.")
                image_indices = state["image_indices"].to(hidden.device)
                cosine, sine = ivc_coordinate_scores(
                    positions,
                    image_indices=image_indices,
                    sequence_length=sequence,
                )
                score_layer = boundary - 1
                values = project_ivc_value_states(hidden[0], layers[score_layer])
                result = ivc_prune_image_tokens(
                    hidden[0],
                    image_indices=image_indices,
                    cosine_scores=cosine,
                    sine_scores=sine,
                    value_states=values,
                    text_query_indices=state["text_queries"].to(hidden.device),
                    before_layer=boundary,
                    spec=self.spec,
                    media_unit="llm_token",
                    source_layout={"backbone": "qwen3_5", "score_layer": score_layer},
                    result_layout={"backbone": "qwen3_5", "prune_layer": boundary},
                )
                state["base_keep_indices"] = result.kept_indices.detach()
                state["source_length"] = sequence
                self._layout_records.append(layout_delta_record(result.layout_delta))
                state["record"] = {
                    "algorithm": "ivc_prune",
                    "score_layer": score_layer,
                    "prune_layer": boundary,
                    "source_tokens": sequence,
                    "result_tokens": int(result.hidden_states.shape[0]),
                    "image_tokens_before": int(state["image_indices"].numel()),
                    "image_tokens_after": int(result.kept_image_indices.numel()),
                    "text_query_tokens": int(state["text_queries"].numel()),
                    "cache_policy": "disabled_no_cache_recompute",
                }
                hidden = result.hidden_states.unsqueeze(0)
                keep = result.kept_indices
            else:
                keep = state["base_keep_indices"].to(hidden.device)
                if sequence == int(state["source_length"]):
                    hidden = hidden.index_select(1, keep)
                elif sequence != int(keep.numel()):
                    raise BackendIntegrationError("IVC-Prune decoder sequence is neither source nor compact length.")

            def select_sequence(value: Any, dims: tuple[int, ...]) -> Any:
                if not torch.is_tensor(value):
                    return value
                result_value = value
                for dim in dims:
                    length = int(result_value.shape[dim])
                    if length == int(state["source_length"]) and int(keep.numel()) != int(state["source_length"]):
                        result_value = result_value.index_select(dim, keep.to(result_value.device))
                    elif length not in {int(state["source_length"]), int(keep.numel())}:
                        raise BackendIntegrationError(f"IVC-Prune sequence field has unsupported length {length}.")
                return result_value

            mutable_args = list(args)
            if mutable_args:
                mutable_args[0] = hidden
            else:
                kwargs["hidden_states"] = hidden
            pe = kwargs.get("position_embeddings")
            if isinstance(pe, tuple):
                kwargs["position_embeddings"] = tuple(select_sequence(value, (1,)) for value in pe)
            mask = kwargs.get("attention_mask")
            if torch.is_tensor(mask):
                if mask.ndim == 4:
                    kwargs["attention_mask"] = select_sequence(select_sequence(mask, (-2,)), (-1,))
                elif mask.ndim == 2:
                    kwargs["attention_mask"] = select_sequence(mask, (-1,))
            pos = kwargs.get("position_ids")
            if torch.is_tensor(pos):
                kwargs["position_ids"] = select_sequence(pos, (-1,))
            cache_position = kwargs.get("cache_position")
            if torch.is_tensor(cache_position) and cache_position.ndim == 1 and cache_position.numel() == sequence:
                kwargs["cache_position"] = select_sequence(cache_position, (0,))
            return tuple(mutable_args), kwargs

        for layer_index in range(boundary, len(layers)):
            self._handles.append(
                layers[layer_index].register_forward_pre_hook(prune_pre_hook, with_kwargs=True)
            )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backbone": "qwen3_5",
            "algorithm": self.algorithm,
            "variant": "ivc_prune.image",
            "binding": self.binding_name,
            "target_site": "decoder_prune_boundary",
            "prune_layer": self.spec.prune_layer,
            "score_layer": self.spec.prune_layer - 1,
            "total_retain_ratio": self.spec.total_retain_ratio,
            "cache_policy": "use_cache_false_only",
            "installed": self._installed,
            "generation_calls": len(self._request_records),
            "layout_delta_records_count": len(self._layout_records),
            "layout_delta_records": list(self._layout_records),
            "requests": list(self._request_records),
            "rollback_errors": list(self._rollback_errors),
            "uninstall_errors": list(self._uninstall_errors),
            "video_support": False,
        }


class _Qwen3_5FeatherQOnlyProjection:
    """Expose the query half of Qwen3.5/3.8's packed Q-and-gate projection."""

    def __init__(self, projection: Any) -> None:
        self._projection = projection

    def __call__(self, hidden_states: Any) -> Any:
        query_and_gate = self._projection(hidden_states)
        if query_and_gate.shape[-1] % 2:
            raise BackendIntegrationError(
                "Qwen3.5 FEATHER expected an even-width packed query/gate projection."
            )
        query, _gate = query_and_gate.chunk(2, dim=-1)
        return query


class _Qwen3_5FeatherAttentionView:
    """Adapter-only view; keep the shared FEATHER scorer model-agnostic."""

    def __init__(self, attention: Any) -> None:
        self._attention = attention
        self.q_proj = _Qwen3_5FeatherQOnlyProjection(attention.q_proj)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._attention, name)


class _Qwen3_5FeatherLayerView:
    def __init__(self, layer: Any) -> None:
        self.input_layernorm = layer.input_layernorm
        self.self_attn = _Qwen3_5FeatherAttentionView(layer.self_attn)


class Qwen3_5FEATHERActivation(Qwen3_5FastVActivation):
    """Two-stage FEATHER using no-RoPE Q/K from Qwen3.5 full-attention layers."""

    algorithm = "feather"
    binding_name = "feather.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from dataclasses import fields
        from mllm_comp.compressors.image.feather import FEATHERSpec

        binding = kwargs.get("binding")
        if getattr(binding, "name", None) != self.binding_name:
            raise BackendIntegrationError(
                f"Qwen3.5 FEATHER received a foreign binding: {getattr(binding, 'name', None)!r}."
            )
        self.model = kwargs["model"]
        config: dict[str, Any] = {}
        for source in (getattr(kwargs.get("stage"), "config", None), getattr(kwargs.get("binding_config"), "config", None)):
            if isinstance(source, Mapping):
                config.update(source)
        if "stage_layers" in config:
            config["stage_layers"] = tuple(int(value) for value in config["stage_layers"])
        allowed = {field.name for field in fields(FEATHERSpec)}
        unknown = set(config) - allowed
        if unknown:
            raise BackendIntegrationError(f"Unknown Qwen3.5 FEATHER config keys: {sorted(unknown)}.")
        self.spec = FEATHERSpec(**config)
        self._installed = False
        self._original_generate = self.model.generate
        self._had_generate = "generate" in getattr(self.model, "__dict__", {})
        self._handles: list[Any] = []
        self._attention_module = None
        self._attention_forward = None
        self._request_records: list[dict[str, Any]] = []
        self._layout_records: list[dict[str, Any]] = []
        self._rollback_errors: list[str] = []
        self._uninstall_errors: list[str] = []

    def _generate_with_decoder_selection(
        self,
        args: tuple[Any, ...],
        original_kwargs: Mapping[str, Any],
        call_kwargs: dict[str, Any],
    ) -> Any:
        import torch
        import types
        from mllm_comp.compressors.image.fastv import fastv_scorer_query_indices
        from mllm_comp.compressors.image.feather import feather_no_rope_scores

        input_ids = call_kwargs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise BackendIntegrationError("Qwen3.5 FEATHER requires tokenized input and batch size one.")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise BackendIntegrationError("Qwen3.5 config has no image_token_id.")
        image_indices = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
        if image_indices.numel() == 0:
            return self._original_generate(*args, **dict(original_kwargs))
        text_model = self.model.model.language_model
        layers = text_model.layers
        layer_types = tuple(self.model.config.text_config.layer_types)
        score_layers = tuple(layer - 1 for layer in self.spec.stage_layers)
        if any(layer < 0 or layer >= len(layers) for layer in score_layers):
            raise BackendIntegrationError("FEATHER stage_layers exceed Qwen3.5 decoder depth.")
        if any(layer_types[layer] != "full_attention" for layer in score_layers):
            raise BackendIntegrationError(
                f"FEATHER scoring layers must be full-attention blocks: "
                f"{[(layer, layer_types[layer]) for layer in score_layers]}."
            )
        attention_mask = call_kwargs.get("attention_mask")
        state: dict[str, Any] = {
            "source_length": int(input_ids.shape[1]),
            "source_attention_mask": attention_mask[0] if attention_mask is not None and attention_mask.ndim == 2 else None,
            "current_indices": torch.arange(input_ids.shape[1], device=input_ids.device),
            "image_indices": image_indices,
            "initial_visual_tokens": int(image_indices.numel()),
            "scores": {},
            "stage_records": [],
        }
        original_prepare = self.model.prepare_inputs_for_generation

        def prepare_hook(
            _model: Any,
            input_ids: Any,
            next_sequence_length: Any = None,
            past_key_values: Any = None,
            attention_mask: Any = None,
            inputs_embeds: Any = None,
            is_first_iteration: Any = False,
            pixel_values: Any = None,
            image_grid_thw: Any = None,
            mm_token_type_ids: Any = None,
            pixel_values_videos: Any = None,
            video_grid_thw: Any = None,
            **prepare_kwargs: Any,
        ) -> Any:
            prepare_kwargs["use_cache"] = False
            if input_ids is not None and input_ids.ndim == 2:
                current_images = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
                if current_images.numel():
                    state.update(
                        source_length=int(input_ids.shape[1]),
                        source_attention_mask=(attention_mask[0] if attention_mask is not None and attention_mask.ndim == 2 else None),
                        current_indices=torch.arange(input_ids.shape[1], device=input_ids.device),
                        image_indices=current_images,
                        initial_visual_tokens=int(current_images.numel()),
                        scores={},
                        stage_records=[],
                    )
            return original_prepare(
                input_ids, None,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                is_first_iteration=is_first_iteration,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                **prepare_kwargs,
            )

        self.model.prepare_inputs_for_generation = types.MethodType(prepare_hook, self.model)
        previous_rope = getattr(self.model.model, "rope_deltas", None)
        try:
            self._install_feather_hooks(layers, state, fastv_scorer_query_indices, feather_no_rope_scores)
            invoke_kwargs = dict(original_kwargs)
            invoke_kwargs["use_cache"] = False
            output = self._original_generate(*args, **invoke_kwargs)
        finally:
            self._remove_request_hooks()
            self.model.prepare_inputs_for_generation = original_prepare
            self.model.model.rope_deltas = previous_rope
        if not state["stage_records"]:
            raise BackendIntegrationError("FEATHER stages did not execute during generation.")
        self._request_records.append(
            {
                "algorithm": self.algorithm,
                "generated": True,
                "kv_cache": False,
                "stage_layers": list(self.spec.stage_layers),
                "stages": list(state["stage_records"]),
                "initial_visual_tokens": int(state["initial_visual_tokens"]),
            }
        )
        return output

    def _install_feather_hooks(
        self, layers: Any, state: dict[str, Any], query_indices_fn: Any, score_fn: Any
    ) -> None:
        import torch
        from mllm_comp.compressors.image.feather import feather_image_tokens

        stage_layers = tuple(self.spec.stage_layers)
        score_layers = tuple(layer - 1 for layer in stage_layers)

        for score_layer in score_layers:
            def output_hook(_module: Any, _args: tuple[Any, ...], output: Any, *, __score_layer: int = score_layer) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise BackendIntegrationError("FEATHER scorer hidden states must have shape [1, sequence, hidden].")
                current_indices = state["current_indices"].to(hidden.device)
                image_indices = state["image_indices"].to(hidden.device)
                padding = state["source_attention_mask"]
                if padding is not None:
                    padding = padding.to(hidden.device).index_select(0, current_indices)
                image_mask = torch.zeros(int(hidden.shape[1]), dtype=torch.bool, device=hidden.device)
                image_mask[image_indices] = True
                queries = query_indices_fn(image_mask, score_source="last_token", attention_mask=padding)
                scores = score_fn(
                    hidden[0], _Qwen3_5FeatherLayerView(layers[__score_layer]),
                    image_indices=image_indices,
                    query_indices=queries,
                )
                state["scores"][__score_layer] = scores.detach()

            self._handles.append(layers[score_layer].register_forward_hook(output_hook))

        first_boundary = stage_layers[0]

        def boundary_hook_factory(layer_index: int):
            def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise BackendIntegrationError("FEATHER expected decoder states shaped [1, sequence, hidden].")
                sequence = int(hidden.shape[1])
                if layer_index in stage_layers:
                    stage_index = stage_layers.index(layer_index)
                    score_layer = layer_index - 1
                    scores = state["scores"].pop(score_layer, None)
                    if scores is None:
                        raise BackendIntegrationError(f"FEATHER score layer {score_layer} did not produce salience scores.")
                    result = feather_image_tokens(
                        hidden[0],
                        image_indices=state["image_indices"].to(hidden.device),
                        image_attention_scores=scores,
                        before_layer=layer_index,
                        initial_visual_tokens=int(state["initial_visual_tokens"]),
                        spec=self.spec,
                        media_unit="llm_token",
                        source_layout={"backbone": "qwen3_5", "score_layer": score_layer},
                        result_layout={"backbone": "qwen3_5", "stage": stage_index, "boundary": layer_index},
                    )
                    old_current = state["current_indices"].to(hidden.device)
                    state["current_indices"] = old_current.index_select(0, result.kept_indices)
                    state["image_indices"] = torch.nonzero(
                        torch.isin(result.kept_indices, result.kept_image_indices), as_tuple=False
                    ).flatten()
                    hidden = result.hidden_states.unsqueeze(0)
                    self._layout_records.append(layout_delta_record(result.layout_delta))
                    state["stage_records"].append(
                        {
                            "score_layer": score_layer,
                            "before_layer": layer_index,
                            "source_sequence_tokens": sequence,
                            "result_sequence_tokens": int(hidden.shape[1]),
                            "image_tokens_after": int(state["image_indices"].numel()),
                            "target_mode": result.metadata.get("target_mode"),
                        }
                    )

                current_indices = state["current_indices"].to(hidden.device)
                current_length = int(current_indices.numel())
                if int(hidden.shape[1]) == int(state["source_length"]):
                    if current_length != int(state["source_length"]):
                        hidden = hidden.index_select(1, current_indices)
                elif int(hidden.shape[1]) != current_length:
                    raise BackendIntegrationError("FEATHER decoder sequence is neither source nor current compact length.")

                def select_axis(value: Any, dim: int) -> Any:
                    if not torch.is_tensor(value):
                        return value
                    length = int(value.shape[dim])
                    if length == int(state["source_length"]) and current_length != int(state["source_length"]):
                        return value.index_select(dim, current_indices.to(value.device))
                    if length not in {current_length, int(state["source_length"])}:
                        raise BackendIntegrationError(f"FEATHER sequence field has unsupported length {length}.")
                    return value

                mutable_args = list(args)
                if mutable_args:
                    mutable_args[0] = hidden
                else:
                    kwargs["hidden_states"] = hidden
                pe = kwargs.get("position_embeddings")
                if isinstance(pe, tuple):
                    kwargs["position_embeddings"] = tuple(select_axis(value, 1) for value in pe)
                mask = kwargs.get("attention_mask")
                if torch.is_tensor(mask):
                    if mask.ndim == 4:
                        mask = select_axis(mask, -2)
                        mask = select_axis(mask, -1)
                    elif mask.ndim == 2:
                        mask = select_axis(mask, -1)
                    kwargs["attention_mask"] = mask
                pos = kwargs.get("position_ids")
                if torch.is_tensor(pos):
                    kwargs["position_ids"] = select_axis(pos, -1)
                cache_position = kwargs.get("cache_position")
                if torch.is_tensor(cache_position) and cache_position.ndim == 1:
                    kwargs["cache_position"] = select_axis(cache_position, 0)
                return tuple(mutable_args), kwargs
            return pre_hook

        for layer_index in range(first_boundary, len(layers)):
            self._handles.append(layers[layer_index].register_forward_pre_hook(boundary_hook_factory(layer_index), with_kwargs=True))

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backbone": "qwen3_5",
            "algorithm": self.algorithm,
            "variant": "feather.image",
            "binding": self.binding_name,
            "target_site": "decoder_stage_boundaries",
            "stage_layers": list(self.spec.stage_layers),
            "reduction_ratio": self.spec.reduction_ratio,
            "cache_policy": "use_cache_false_only",
            "installed": self._installed,
            "generation_calls": len(self._request_records),
            "layout_delta_records_count": len(self._layout_records),
            "layout_delta_records": list(self._layout_records),
            "requests": list(self._request_records),
            "rollback_errors": list(self._rollback_errors),
            "uninstall_errors": list(self._uninstall_errors),
            "video_support": False,
        }


class Qwen3_5StagewiseDecoderActivation(Qwen3_5FastVActivation):
    """Shared no-cache decoder-stage bridge for BTP/HalfV-style selectors."""

    @staticmethod
    def _load_spec(kwargs: Mapping[str, Any], spec_type: Any, spec_fields: Any) -> Any:
        config: dict[str, Any] = {}
        for source in (
            getattr(kwargs.get("stage"), "config", None),
            getattr(kwargs.get("binding_config"), "config", None),
        ):
            if isinstance(source, Mapping):
                config.update(source)
        allowed = {field.name for field in spec_fields}
        unknown = set(config) - allowed
        if unknown:
            raise BackendIntegrationError(
                f"Unknown Qwen3.5 {getattr(spec_type, '__name__', 'decoder')} config keys: {sorted(unknown)}."
            )
        return spec_type(**config)

    @staticmethod
    def _qk(module: Any, hidden_states: Any, position_embeddings: Any, *, apply_rope: bool) -> tuple[Any, Any]:
        import torch
        batch, sequence, _ = hidden_states.shape
        head_dim = int(module.head_dim)
        query, _gate = torch.chunk(
            module.q_proj(hidden_states).view(batch, sequence, -1, head_dim * 2),
            2,
            dim=-1,
        )
        query = module.q_norm(query).transpose(1, 2)
        keys = module.k_norm(
            module.k_proj(hidden_states).view(batch, sequence, -1, head_dim)
        ).transpose(1, 2)
        if apply_rope:
            if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
                raise BackendIntegrationError("Qwen3.5 stagewise scorer requires native rotary embeddings.")
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
            cos, sin = position_embeddings
            query, keys = apply_rotary_pos_emb(query, keys, cos, sin)
        repeat = query.shape[1] // keys.shape[1]
        if repeat > 1:
            keys = keys.repeat_interleave(repeat, dim=1)
        return query, keys

    def __init__(self, *, model: Any, binding: Any, runtime: Any = None) -> None:
        if getattr(binding, "name", None) != self.binding_name:
            raise BackendIntegrationError(
                f"Qwen3.5 {self.algorithm} received a foreign binding: {getattr(binding, 'name', None)!r}."
            )
        self.model = model
        self.runtime = runtime
        self._installed = False
        self._original_generate = model.generate
        self._had_generate = "generate" in getattr(model, "__dict__", {})
        self._handles: list[Any] = []
        self._attention_module = None
        self._attention_forward = None
        self._request_records: list[dict[str, Any]] = []
        self._layout_records: list[dict[str, Any]] = []
        self._rollback_errors: list[str] = []
        self._uninstall_errors: list[str] = []

    def _stage_plan(self, depth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        raise NotImplementedError

    def _capture_stage_scores(
        self, stage_index: int, module: Any, hidden_states: Any,
        position_embeddings: Any, state: dict[str, Any],
    ) -> Any:
        raise NotImplementedError

    def _apply_stage(
        self, stage_index: int, hidden_states: Any, image_indices: Any,
        scores: Any, state: dict[str, Any], boundary_layer: int,
    ) -> Any:
        raise NotImplementedError

    def _generate_with_decoder_selection(
        self,
        args: tuple[Any, ...],
        original_kwargs: Mapping[str, Any],
        call_kwargs: dict[str, Any],
    ) -> Any:
        import torch
        import types

        input_ids = call_kwargs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise BackendIntegrationError(f"Qwen3.5 {self.algorithm} requires batch size one.")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise BackendIntegrationError("Qwen3.5 config has no image_token_id.")
        initial_images = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
        if initial_images.numel() == 0:
            return self._original_generate(*args, **dict(original_kwargs))
        text_model = self.model.model.language_model
        layers = text_model.layers
        layer_types = tuple(self.model.config.text_config.layer_types)
        boundary_layers, score_layers = self._stage_plan(len(layers))
        if not boundary_layers or len(score_layers) > len(boundary_layers):
            raise BackendIntegrationError(f"Qwen3.5 {self.algorithm} has an invalid stage schedule.")
        if any(layer < 0 or layer >= len(layers) for layer in boundary_layers):
            raise BackendIntegrationError(f"Qwen3.5 {self.algorithm} boundary exceeds decoder depth.")
        if any(layer < 0 or layer >= len(layers) or layer_types[layer] != "full_attention" for layer in score_layers):
            raise BackendIntegrationError(
                f"Qwen3.5 {self.algorithm} score layers must be full attention: "
                f"{[(layer, layer_types[layer] if 0 <= layer < len(layer_types) else 'out of range') for layer in score_layers]}."
            )
        if any(layer_types[layer] != "full_attention" for layer in score_layers):
            raise BackendIntegrationError(
                f"Qwen3.5 {self.algorithm} score layers must be full attention: "
                f"{[(layer, layer_types[layer]) for layer in score_layers]}."
            )
        attention_mask = call_kwargs.get("attention_mask")
        state: dict[str, Any] = {
            "source_length": int(input_ids.shape[1]),
            "source_attention_mask": attention_mask[0] if attention_mask is not None and attention_mask.ndim == 2 else None,
            "current_indices": torch.arange(input_ids.shape[1], device=input_ids.device),
            "image_indices": initial_images,
            "initial_visual_tokens": int(initial_images.numel()),
            "scores": {},
            "stage_records": [],
        }
        original_prepare = self.model.prepare_inputs_for_generation

        def prepare_hook(
            _model: Any,
            input_ids: Any,
            next_sequence_length: Any = None,
            past_key_values: Any = None,
            attention_mask: Any = None,
            inputs_embeds: Any = None,
            is_first_iteration: Any = False,
            pixel_values: Any = None,
            image_grid_thw: Any = None,
            mm_token_type_ids: Any = None,
            pixel_values_videos: Any = None,
            video_grid_thw: Any = None,
            **prepare_kwargs: Any,
        ) -> Any:
            prepare_kwargs["use_cache"] = False
            if input_ids is not None and input_ids.ndim == 2:
                images = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
                if images.numel():
                    state.update(
                        source_length=int(input_ids.shape[1]),
                        source_attention_mask=(attention_mask[0] if attention_mask is not None and attention_mask.ndim == 2 else None),
                        current_indices=torch.arange(input_ids.shape[1], device=input_ids.device),
                        image_indices=images,
                        initial_visual_tokens=int(images.numel()),
                        scores={},
                        stage_records=[],
                    )
            return original_prepare(
                input_ids, None,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                is_first_iteration=is_first_iteration,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                **prepare_kwargs,
            )

        self.model.prepare_inputs_for_generation = types.MethodType(prepare_hook, self.model)
        previous_rope = getattr(self.model.model, "rope_deltas", None)
        try:
            self._install_stagewise_hooks(layers, boundary_layers, score_layers, state)
            invoke_kwargs = dict(original_kwargs)
            invoke_kwargs["use_cache"] = False
            output = self._original_generate(*args, **invoke_kwargs)
        finally:
            self._remove_request_hooks()
            self.model.prepare_inputs_for_generation = original_prepare
            self.model.model.rope_deltas = previous_rope
        if not state["stage_records"]:
            raise BackendIntegrationError(f"Qwen3.5 {self.algorithm} stages did not execute.")
        self._request_records.append(
            {
                "algorithm": self.algorithm,
                "generated": True,
                "kv_cache": False,
                "stage_records": list(state["stage_records"]),
                "boundary_layers": list(boundary_layers),
                "score_layers": list(score_layers),
                "initial_visual_tokens": int(state["initial_visual_tokens"]),
            }
        )
        return output

    def _install_stagewise_hooks(
        self, layers: Any, boundary_layers: tuple[int, ...],
        score_layers: tuple[int, ...], state: dict[str, Any],
    ) -> None:
        import torch

        for stage_index, score_layer in enumerate(score_layers):
            attention = getattr(layers[score_layer], "self_attn", None)
            if attention is None:
                raise BackendIntegrationError(f"{self.algorithm} scorer layer {score_layer} has no self_attn.")

            def score_pre_hook(
                module: Any, args: tuple[Any, ...], kwargs: dict[str, Any],
                *, __stage: int = stage_index,
            ) -> None:
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise BackendIntegrationError(f"{self.algorithm} score input must have shape [1, sequence, hidden].")
                state["scores"][__stage] = self._capture_stage_scores(
                    __stage, module, hidden, kwargs.get("position_embeddings"), state
                )

            self._handles.append(attention.register_forward_pre_hook(score_pre_hook, with_kwargs=True))

        first_boundary = boundary_layers[0]

        def boundary_hook_factory(layer_index: int):
            def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise BackendIntegrationError(f"{self.algorithm} expected decoder states shaped [1, sequence, hidden].")
                sequence = int(hidden.shape[1])
                if layer_index in boundary_layers:
                    stage_index = boundary_layers.index(layer_index)
                    scores = state["scores"].pop(stage_index, None)
                    result = self._apply_stage(
                        stage_index, hidden[0], state["image_indices"].to(hidden.device),
                        scores, state, layer_index,
                    )
                    keep = result.kept_indices
                    old_current = state["current_indices"].to(hidden.device)
                    state["current_indices"] = old_current.index_select(0, keep)
                    state["image_indices"] = torch.nonzero(
                        torch.isin(keep, result.kept_image_indices), as_tuple=False
                    ).flatten()
                    hidden = result.hidden_states.unsqueeze(0)
                    self._layout_records.append(layout_delta_record(result.layout_delta))
                    state["stage_records"].append(
                        {
                            "stage_index": stage_index,
                            "score_layer": (
                                score_layers[stage_index]
                                if stage_index < len(score_layers)
                                else None
                            ),
                            "boundary_layer": layer_index,
                            "source_sequence_tokens": sequence,
                            "result_sequence_tokens": int(hidden.shape[1]),
                            "image_tokens_after": int(state["image_indices"].numel()),
                            "stage_metadata": dict(result.metadata),
                        }
                    )
                current_indices = state["current_indices"].to(hidden.device)
                current_length = int(current_indices.numel())
                if int(hidden.shape[1]) == int(state["source_length"]):
                    if current_length != int(state["source_length"]):
                        hidden = hidden.index_select(1, current_indices)
                elif int(hidden.shape[1]) != current_length:
                    raise BackendIntegrationError(f"{self.algorithm} decoder sequence is neither source nor compact length.")

                def select_axis(value: Any, dim: int) -> Any:
                    if not torch.is_tensor(value):
                        return value
                    length = int(value.shape[dim])
                    if length == int(state["source_length"]) and current_length != int(state["source_length"]):
                        return value.index_select(dim, current_indices.to(value.device))
                    if length not in {current_length, int(state["source_length"])}:
                        raise BackendIntegrationError(f"{self.algorithm} sequence field has unsupported length {length}.")
                    return value

                mutable_args = list(args)
                if mutable_args:
                    mutable_args[0] = hidden
                else:
                    kwargs["hidden_states"] = hidden
                pe = kwargs.get("position_embeddings")
                if isinstance(pe, tuple):
                    kwargs["position_embeddings"] = tuple(select_axis(value, 1) for value in pe)
                mask = kwargs.get("attention_mask")
                if torch.is_tensor(mask):
                    if mask.ndim == 4:
                        mask = select_axis(mask, -2)
                        mask = select_axis(mask, -1)
                    elif mask.ndim == 2:
                        mask = select_axis(mask, -1)
                    kwargs["attention_mask"] = mask
                pos = kwargs.get("position_ids")
                if torch.is_tensor(pos):
                    kwargs["position_ids"] = select_axis(pos, -1)
                cache_position = kwargs.get("cache_position")
                if torch.is_tensor(cache_position) and cache_position.ndim == 1:
                    kwargs["cache_position"] = select_axis(cache_position, 0)
                return tuple(mutable_args), kwargs
            return pre_hook

        for layer_index in range(first_boundary, len(layers)):
            self._handles.append(
                layers[layer_index].register_forward_pre_hook(
                    boundary_hook_factory(layer_index), with_kwargs=True
                )
            )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backbone": "qwen3_5",
            "algorithm": self.algorithm,
            "variant": f"{self.algorithm}.image",
            "binding": self.binding_name,
            "target_site": "decoder_stage_boundaries",
            "cache_policy": "use_cache_false_only",
            "installed": self._installed,
            "generation_calls": len(self._request_records),
            "layout_delta_records_count": len(self._layout_records),
            "layout_delta_records": list(self._layout_records),
            "requests": list(self._request_records),
            "rollback_errors": list(self._rollback_errors),
            "uninstall_errors": list(self._uninstall_errors),
            "video_support": False,
        }


class Qwen3_5NuwaDecoderActivation(Qwen3_5StagewiseDecoderActivation):
    """Nüwa Stage 2 on its native post-layer-16 full-attention query."""

    algorithm = "nuwa"
    binding_name = "nuwa.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from mllm_comp.compressors.image.nuwa import NUWA_CONFIG_KEYS, NuwaSpec

        config: dict[str, Any] = {}
        for source in (
            getattr(kwargs.get("stage"), "config", None),
            getattr(kwargs.get("binding_config"), "config", None),
        ):
            if isinstance(source, Mapping):
                config.update({key: value for key, value in source.items() if key in NUWA_CONFIG_KEYS})
        self.spec = NuwaSpec(**config)
        super().__init__(model=kwargs["model"], binding=kwargs.get("binding"))

    def _stage_plan(self, depth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        self.spec.validate_decoder_depth(depth)
        boundary = self.spec.stage2_after_layer
        score_layer = self.spec.stage2_score_layer
        if score_layer is not None:
            layer_types = tuple(self.model.config.text_config.layer_types)
            if layer_types[score_layer] != "full_attention":
                raise BackendIntegrationError(
                    "Qwen3.8 Nüwa's released last-text-attention score requires "
                    f"full attention at layer {score_layer}; found {layer_types[score_layer]}."
                )
        return (boundary,), (() if score_layer is None else (score_layer,))

    def _capture_stage_scores(
        self, stage_index: int, module: Any, hidden_states: Any,
        position_embeddings: Any, state: dict[str, Any],
    ) -> Any:
        import torch
        from mllm_comp.compressors.image.fastv import fastv_scorer_query_indices
        from mllm_comp.compressors.image.pyramid_drop import (
            pyramid_attention_key_visibility,
            pyramid_attention_scores,
        )

        query, keys = self._qk(module, hidden_states, position_embeddings, apply_rope=True)
        current = state["current_indices"].to(hidden_states.device)
        image = state["image_indices"].to(hidden_states.device)
        padding = state["source_attention_mask"]
        if padding is not None:
            padding = padding.to(hidden_states.device).index_select(0, current)
        image_mask = torch.zeros(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
        image_mask[image] = True
        queries = fastv_scorer_query_indices(
            image_mask, score_source="last_token", attention_mask=padding
        )
        visibility = pyramid_attention_key_visibility(
            padding, instruction_indices=queries, key_count=int(hidden_states.shape[1])
        )
        query_values = query[0, :, queries, :].permute(1, 0, 2).contiguous()
        key_values = keys[0].transpose(0, 1).contiguous()
        return pyramid_attention_scores(
            instruction_query=query_values,
            image_keys=key_values.index_select(0, image),
            image_count=int(image.numel()),
            attention_keys=key_values,
            attention_key_mask=visibility,
            image_key_indices=image,
            device=hidden_states.device,
        )

    def _apply_stage(
        self, stage_index: int, hidden_states: Any, image_indices: Any,
        scores: Any, state: dict[str, Any], boundary_layer: int,
    ) -> Any:
        from mllm_comp.compressors.image.nuwa import nuwa_stage2_image_tokens

        if self.spec.stage2_score_layer is not None and scores is None:
            raise BackendIntegrationError("Qwen3.8 Nüwa Stage 2 did not capture its attention scores.")
        result = nuwa_stage2_image_tokens(
            hidden_states,
            image_indices=image_indices,
            image_attention_scores=scores,
            spec=self.spec,
            media_unit="llm_token",
            source_layout={"backbone": "qwen3_5", "stage": "stage2",
                           "score_layer": self.spec.stage2_score_layer},
            result_layout={"backbone": "qwen3_5", "stage": "stage2",
                           "boundary_layer": boundary_layer,
                           "position_policy": "source_representative_rpme"},
        )
        return result


class Qwen3_5SparseVLMActivation(Qwen3_5StagewiseDecoderActivation):
    """SparseVLM with its original raters/recycling kernel at Qwen full-attn layers."""

    algorithm = "sparsevlm"
    binding_name = "sparsevlm.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from dataclasses import fields
        from mllm_comp.compressors.image.sparsevlm import SparseVLMSpec

        self.spec = self._load_spec(kwargs, SparseVLMSpec, fields(SparseVLMSpec))
        self.score_layers = tuple(self.spec.prune_layers)
        self.boundary_layers = tuple(layer + 1 for layer in self.score_layers)
        self._rater_source_indices: Any = None
        self._initial_visual_tokens: int | None = None
        super().__init__(model=kwargs["model"], binding=kwargs.get("binding"))

    def _generate_with_decoder_selection(
        self, args: tuple[Any, ...], original_kwargs: Mapping[str, Any],
        call_kwargs: dict[str, Any],
    ) -> Any:
        import torch
        from mllm_comp.compressors.image.sparsevlm import select_sparsevlm_text_raters

        input_ids = call_kwargs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise BackendIntegrationError("Qwen3.8 SparseVLM requires batch size one input_ids.")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise BackendIntegrationError("Qwen3.8 config has no image_token_id.")
        image_indices = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
        if image_indices.numel() < 1:
            self._rater_source_indices = None
            self._initial_visual_tokens = None
            return super()._generate_with_decoder_selection(args, original_kwargs, call_kwargs)
        attention_mask = call_kwargs.get("attention_mask")
        candidates = torch.arange(int(image_indices[-1]) + 1, int(input_ids.shape[1]),
                                  device=input_ids.device)
        if attention_mask is not None and attention_mask.ndim == 2:
            visible = attention_mask[0].to(input_ids.device).bool().index_select(0, candidates)
            candidates = candidates[visible]
        if candidates.numel() < 1:
            raise BackendIntegrationError("Qwen3.8 SparseVLM found no post-image text-rater candidates.")
        embeddings = self.model.get_input_embeddings()(input_ids[0])
        selected = select_sparsevlm_text_raters(
            embeddings,
            image_indices=image_indices,
            text_candidate_indices=candidates,
        )
        self._rater_source_indices = selected.rater_indices.detach()
        self._initial_visual_tokens = int(image_indices.numel())
        try:
            return super()._generate_with_decoder_selection(args, original_kwargs, call_kwargs)
        finally:
            self._rater_source_indices = None
            self._initial_visual_tokens = None

    def _stage_plan(self, depth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not self.score_layers or self.score_layers[-1] + 1 >= depth:
            raise BackendIntegrationError("Qwen3.8 SparseVLM schedule exceeds decoder depth.")
        layer_types = tuple(self.model.config.text_config.layer_types)
        invalid = [(layer, layer_types[layer]) for layer in self.score_layers
                   if layer < 0 or layer >= depth or layer_types[layer] != "full_attention"]
        if invalid:
            raise BackendIntegrationError(
                "Qwen3.8 SparseVLM scoring layers must be full attention; "
                f"unsupported layer types: {invalid}. Select a Qwen3.8-specific "
                "prune_layers schedule while keeping the SparseVLM kernel unchanged."
            )
        return self.boundary_layers, self.score_layers

    def _capture_stage_scores(
        self, stage_index: int, module: Any, hidden_states: Any,
        position_embeddings: Any, state: dict[str, Any],
    ) -> Any:
        import torch
        from mllm_comp.compressors.image.pyramid_drop import (
            pyramid_attention_key_visibility,
            pyramid_attention_scores,
        )

        if self._rater_source_indices is None:
            raise BackendIntegrationError("SparseVLM prompt raters were not initialized.")
        query, keys = self._qk(module, hidden_states, position_embeddings, apply_rope=True)
        current = state["current_indices"].to(hidden_states.device)
        image = state["image_indices"].to(hidden_states.device)
        source_raters = self._rater_source_indices.to(current.device)
        local_raters = []
        for source in source_raters:
            found = torch.nonzero(current == source, as_tuple=False).flatten()
            if found.numel() != 1:
                raise BackendIntegrationError("SparseVLM text-rater identity was lost during sequence compaction.")
            local_raters.append(found[0])
        raters = torch.stack(local_raters).to(hidden_states.device)
        padding = state["source_attention_mask"]
        if padding is not None:
            padding = padding.to(hidden_states.device).index_select(0, current)
        visibility = pyramid_attention_key_visibility(
            padding, instruction_indices=raters, key_count=int(hidden_states.shape[1])
        )
        query_values = query[0, :, raters, :].permute(1, 0, 2).contiguous()
        key_values = keys[0].transpose(0, 1).contiguous()
        return pyramid_attention_scores(
            instruction_query=query_values,
            image_keys=key_values.index_select(0, image),
            image_count=int(image.numel()),
            attention_keys=key_values,
            attention_key_mask=visibility,
            image_key_indices=image,
            device=hidden_states.device,
        )

    def _apply_stage(
        self, stage_index: int, hidden_states: Any, image_indices: Any,
        scores: Any, state: dict[str, Any], boundary_layer: int,
    ) -> Any:
        from types import SimpleNamespace
        from mllm_comp.compressors.image.sparsevlm import sparsevlm_image_tokens

        if scores is None or self._initial_visual_tokens is None:
            raise BackendIntegrationError("SparseVLM attention scores were not captured before pruning.")
        source_layer = self.score_layers[stage_index]
        result = sparsevlm_image_tokens(
            hidden_states,
            image_indices=image_indices,
            image_attention_scores=scores,
            layer_index=source_layer,
            initial_visual_tokens=self._initial_visual_tokens,
            spec=self.spec,
            media_unit="llm_token",
            source_layout={"backbone": "qwen3_5", "score_layer": source_layer},
            result_layout={"backbone": "qwen3_5", "boundary_layer": boundary_layer},
        )
        representative_indices = result.representative_indices
        image_representatives = representative_indices.index_select(
            0, result.result_image_indices
        )
        # Stagewise sequence repair expects source representatives for every
        # output slot and source representatives for the resulting image span.
        return SimpleNamespace(
            hidden_states=result.hidden_states,
            kept_indices=representative_indices,
            kept_image_indices=image_representatives,
            layout_delta=result.layout_delta,
            metadata=dict(result.metadata),
        )


class Qwen3_5FinePrunerActivation(Qwen3_5StagewiseDecoderActivation):
    """FinePruner with IAC stages and ARP scores on native full-attn blocks."""

    algorithm = "finepruner"
    binding_name = "finepruner.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from dataclasses import fields
        from mllm_comp.compressors.image.finepruner import FinePrunerSpec

        self.spec = self._load_spec(kwargs, FinePrunerSpec, fields(FinePrunerSpec))
        self.score_layers = tuple(layer - 1 for layer in self.spec.arp_target_after_layers)
        self.boundary_layers = (*self.spec.iac_after_layers, self.spec.arp_apply_after_layer)
        super().__init__(model=kwargs["model"], binding=kwargs.get("binding"))

    def _stage_plan(self, depth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        self.spec.validate_decoder_depth(depth)
        if not self.boundary_layers or self.boundary_layers[-1] >= depth:
            raise BackendIntegrationError("Qwen3.8 FinePruner schedule exceeds decoder depth.")
        layer_types = tuple(self.model.config.text_config.layer_types)
        invalid = [(layer, layer_types[layer]) for layer in self.score_layers
                   if layer < 0 or layer >= depth or layer_types[layer] != "full_attention"]
        if invalid:
            raise BackendIntegrationError(
                "Qwen3.8 FinePruner ARP sources must be full-attention layers; "
                f"unsupported layer types: {invalid}."
            )
        return tuple(self.boundary_layers), self.score_layers

    def _capture_stage_scores(
        self, stage_index: int, module: Any, hidden_states: Any,
        position_embeddings: Any, state: dict[str, Any],
    ) -> Any:
        import torch
        from mllm_comp.compressors.image.fastv import fastv_scorer_query_indices
        from mllm_comp.compressors.image.finepruner import finepruner_attention_head_slice
        from mllm_comp.compressors.image.pyramid_drop import pyramid_attention_key_visibility

        if stage_index == 0:
            state["arp_attentions"] = []
        query, keys = self._qk(module, hidden_states, position_embeddings, apply_rope=True)
        current = state["current_indices"].to(hidden_states.device)
        images = state["image_indices"].to(hidden_states.device)
        padding = state["source_attention_mask"]
        if padding is not None:
            padding = padding.to(hidden_states.device).index_select(0, current)
        image_mask = torch.zeros(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
        image_mask[images] = True
        queries = fastv_scorer_query_indices(
            image_mask, score_source="last_token", attention_mask=padding
        )
        visibility = pyramid_attention_key_visibility(
            padding, instruction_indices=queries, key_count=int(hidden_states.shape[1])
        )
        query_values = query[0, :, queries, :].permute(1, 0, 2).contiguous()
        key_values = keys[0].transpose(0, 1).contiguous()
        scores = finepruner_attention_head_slice(
            query=query_values,
            attention_keys=key_values,
            image_key_indices=images,
            attention_key_mask=visibility,
        )
        state["arp_attentions"].append(scores)
        return scores

    def _apply_stage(
        self, stage_index: int, hidden_states: Any, image_indices: Any,
        scores: Any, state: dict[str, Any], boundary_layer: int,
    ) -> Any:
        from mllm_comp.compressors.image.finepruner import (
            finepruner_arp_image_tokens,
            finepruner_iac_image_tokens,
        )

        common = {
            "image_indices": image_indices,
            "initial_visual_tokens": int(state["initial_visual_tokens"]),
            "spec": self.spec,
            "media_unit": "llm_token",
            "source_layout": {"backbone": "qwen3_5", "stage_index": stage_index},
            "result_layout": {"backbone": "qwen3_5", "boundary_layer": boundary_layer},
        }
        if stage_index < len(self.spec.iac_after_layers):
            return finepruner_iac_image_tokens(
                hidden_states,
                completed_layers=self.spec.iac_after_layers[stage_index],
                **common,
            )
        attentions = state.get("arp_attentions", [])
        if len(attentions) != len(self.score_layers):
            raise BackendIntegrationError(
                "Qwen3.8 FinePruner ARP did not capture every configured full-attention source."
            )
        result = finepruner_arp_image_tokens(
            hidden_states,
            attentions,
            completed_layers=self.spec.arp_apply_after_layer,
            **common,
        )
        state["arp_attentions"] = []
        return result


class Qwen3_5TransPruneActivation(Qwen3_5StagewiseDecoderActivation):
    """TransPrune with the original TTV/IGA kernels and full-attn IGA layers."""

    algorithm = "transprune"
    binding_name = "transprune.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from dataclasses import fields
        from mllm_comp.compressors.image.transprune import TransPruneSpec

        self.spec = self._load_spec(kwargs, TransPruneSpec, fields(TransPruneSpec))
        self.score_layers = ()
        self.boundary_layers = tuple(self.spec.prune_after_layers)
        self.transition_layers = tuple(self.spec.accumulating_layer_indices)
        super().__init__(model=kwargs["model"], binding=kwargs.get("binding"))

    def _stage_plan(self, depth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        self.spec.validate_decoder_depth(depth)
        boundaries = tuple(self.spec.prune_after_layers)
        if not boundaries or boundaries[-1] >= depth:
            raise BackendIntegrationError("Qwen3.8 TransPrune schedule exceeds decoder depth.")
        layer_types = tuple(self.model.config.text_config.layer_types)
        bad = [(layer, layer_types[layer]) for layer in boundaries
               if layer < 0 or layer >= depth or layer_types[layer] != "full_attention"]
        if bad:
            raise BackendIntegrationError(
                "Qwen3.8 TransPrune IGA query layers must be full attention; "
                f"unsupported layers: {bad}."
            )
        return boundaries, ()

    def _capture_stage_scores(
        self, stage_index: int, module: Any, hidden_states: Any,
        position_embeddings: Any, state: dict[str, Any],
    ) -> Any:
        import torch
        from mllm_comp.compressors.image.dart import dart_post_image_text_indices
        from mllm_comp.compressors.image.pyramid_drop import pyramid_attention_key_visibility
        from mllm_comp.compressors.image.transprune import transprune_instruction_guided_attention

        projection = getattr(module, "q_proj", None)
        scoring_device = getattr(getattr(projection, "weight", None), "device", hidden_states.device)
        scoring_hidden = hidden_states.to(scoring_device)
        if isinstance(position_embeddings, tuple):
            compacted = []
            current_indices = state["current_indices"].to(scoring_device)
            source_length = int(state["source_length"])
            for value in position_embeddings:
                if not torch.is_tensor(value):
                    compacted.append(value)
                    continue
                value = value.to(scoring_device)
                if value.ndim >= 2 and int(value.shape[-2]) == source_length and int(current_indices.numel()) != source_length:
                    value = value.index_select(-2, current_indices)
                if value.ndim >= 2 and int(value.shape[-2]) != int(scoring_hidden.shape[1]):
                    raise BackendIntegrationError(
                        "TransPrune RoPE sequence axis does not match the compacted hidden-state layout."
                    )
                compacted.append(value)
            position_embeddings = tuple(compacted)
        query, keys = self._qk(module, scoring_hidden, position_embeddings, apply_rope=True)
        current = state["current_indices"].to(scoring_device)
        images = state["image_indices"].to(scoring_device)
        padding = state["source_attention_mask"]
        if padding is not None:
            padding = padding.to(scoring_device).index_select(0, current)
        image_mask = torch.zeros(scoring_hidden.shape[1], dtype=torch.bool, device=scoring_device)
        image_mask[images] = True
        queries = dart_post_image_text_indices(image_mask, attention_mask=padding, labels=None)
        if not queries.numel():
            raise BackendIntegrationError("Qwen3.8 TransPrune requires post-image text queries.")
        visibility = pyramid_attention_key_visibility(
            padding, instruction_indices=queries, key_count=int(scoring_hidden.shape[1])
        )
        query_values = query[0, :, queries, :].permute(1, 0, 2).contiguous()
        key_values = keys[0].transpose(0, 1).contiguous()
        return transprune_instruction_guided_attention(
            instruction_queries=query_values,
            image_keys=key_values.index_select(0, images),
            attention_keys=key_values,
            image_indices=images,
            attention_key_mask=visibility,
        )

    def _install_stagewise_hooks(
        self, layers: Any, boundary_layers: tuple[int, ...],
        score_layers: tuple[int, ...], state: dict[str, Any],
    ) -> None:
        import torch
        from mllm_comp.compressors.image.transprune import transprune_layer_transition_variation

        # At each selected boundary the original TransPrune uses the next
        # block's post-image query. Capture its Q/K before that block is pruned.
        for stage_index, layer_index in enumerate(boundary_layers):
            layer = layers[layer_index]
            attention = getattr(layer, "self_attn", None)
            if getattr(layer, "block_type", None) != "full_attention" or attention is None:
                raise BackendIntegrationError(
                    f"Qwen3.8 TransPrune IGA boundary {layer_index} is not full attention."
                )

            def iga_pre_hook(
                _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any],
                *, __stage: int = stage_index, __attention: Any = attention,
            ) -> None:
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None:
                    raise BackendIntegrationError("TransPrune IGA layer received no hidden states.")
                state["scores"][__stage] = self._capture_stage_scores(
                    __stage, __attention, hidden, kwargs.get("position_embeddings"), state
                )

            self._handles.append(layer.register_forward_pre_hook(iga_pre_hook, with_kwargs=True))

        for layer_index in self.transition_layers:
            if layer_index >= len(layers):
                raise BackendIntegrationError("Qwen3.8 TransPrune TTV source exceeds decoder depth.")
            layer = layers[layer_index]
            mixer = getattr(layer, "self_attn", None)
            if mixer is None:
                mixer = getattr(layer, "linear_attn", None)
            required = (getattr(layer, "input_layernorm", None), mixer,
                        getattr(layer, "post_attention_layernorm", None),
                        getattr(layer, "mlp", None))
            if any(module is None for module in required):
                raise BackendIntegrationError(
                    f"Qwen3.8 TransPrune cannot observe all four transitions at layer {layer_index}."
                )
            values: dict[str, Any] = {}

            def save(field: str, target: dict[str, Any]):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                    tensor = output[0] if isinstance(output, (tuple, list)) else output
                    if not torch.is_tensor(tensor) or tensor.ndim != 3:
                        raise BackendIntegrationError("TransPrune module transition must be [batch, sequence, hidden].")
                    target[field] = tensor.detach()
                return hook

            self._handles.append(required[0].register_forward_hook(save("attention_before", values)))
            self._handles.append(mixer.register_forward_hook(save("attention_after", values)))
            self._handles.append(required[2].register_forward_hook(save("ffn_before", values)))
            self._handles.append(required[3].register_forward_hook(save("ffn_after", values)))

            def layer_done_hook(_module: Any, _inputs: tuple[Any, ...], _output: Any,
                                *, __layer: int = layer_index, __values: dict[str, Any] = values) -> None:
                if __layer == self.transition_layers[0]:
                    state["ttv_history"] = []
                fields = ("attention_before", "attention_after", "ffn_before", "ffn_after")
                if any(name not in __values for name in fields):
                    missing = [name for name in fields if name not in __values]
                    raise BackendIntegrationError(
                        f"TransPrune TTV capture incomplete at layer {__layer}; missing {missing}."
                    )
                images = state["image_indices"]
                state.setdefault("ttv_history", []).append(
                    transprune_layer_transition_variation(
                        __values["attention_before"], __values["attention_after"],
                        __values["ffn_before"], __values["ffn_after"],
                        image_indices=images,
                        attention_image_indices=images,
                        direction_rule=self.spec.direction_rule,
                        softmax_scope=self.spec.ttv_softmax_scope,
                    )
                )
                __values.clear()

            self._handles.append(layer.register_forward_hook(layer_done_hook))

        super()._install_stagewise_hooks(layers, boundary_layers, score_layers, state)

    def _apply_stage(
        self, stage_index: int, hidden_states: Any, image_indices: Any,
        scores: Any, state: dict[str, Any], boundary_layer: int,
    ) -> Any:
        from mllm_comp.compressors.image.transprune import (
            select_transprune_ttv_history,
            transprune_image_tokens,
        )

        history = state.get("ttv_history", [])
        if not history or scores is None:
            raise BackendIntegrationError("TransPrune requires accumulated TTV and next-layer IGA before pruning.")
        source_boundary = self.spec.boundary_after_layer_indices[stage_index]
        result = transprune_image_tokens(
            hidden_states,
            image_indices=image_indices,
            boundary_after_layer=source_boundary,
            ttv_history=history,
            image_attention_scores=scores,
            initial_visual_tokens=int(state["initial_visual_tokens"]),
            spec=self.spec,
            media_unit="llm_token",
            source_layout={"backbone": "qwen3_5", "stage": stage_index,
                           "boundary_after_layer": source_boundary},
            result_layout={"backbone": "qwen3_5", "score_layer": boundary_layer,
                           "position_policy": "contiguous"},
        )
        state["ttv_history"] = select_transprune_ttv_history(
            history, result.selected_visual_indices
        )
        return result


class Qwen3_5BTPActivation(Qwen3_5StagewiseDecoderActivation):
    algorithm = "btp"
    binding_name = "btp.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from dataclasses import fields
        from mllm_comp.compressors.image.btp import BalancedTokenPruningSpec
        self.spec = self._load_spec(kwargs, BalancedTokenPruningSpec, fields(BalancedTokenPruningSpec))
        self.schedule = self.spec.boundary_after_layer_indices
        self.score_layers = self.schedule[:-1]
        self.boundary_layers = tuple(layer + 1 for layer in self.schedule)
        super().__init__(model=kwargs["model"], binding=kwargs.get("binding"))

    def _stage_plan(self, depth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if self.boundary_layers[-1] >= depth:
            raise BackendIntegrationError("BTP configured final stage exceeds Qwen3.5 decoder depth.")
        return self.boundary_layers, self.score_layers

    def _capture_stage_scores(self, stage_index: int, module: Any, hidden_states: Any, position_embeddings: Any, state: dict[str, Any]) -> Any:
        import torch
        from mllm_comp.compressors.image.fastv import fastv_scorer_query_indices
        from mllm_comp.compressors.image.pyramid_drop import pyramid_attention_key_visibility, pyramid_attention_scores
        query, keys = self._qk(module, hidden_states, position_embeddings, apply_rope=True)
        current = state["current_indices"].to(hidden_states.device)
        image = state["image_indices"].to(hidden_states.device)
        padding = state["source_attention_mask"]
        if padding is not None:
            padding = padding.to(hidden_states.device).index_select(0, current)
        image_mask = torch.zeros(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
        image_mask[image] = True
        queries = fastv_scorer_query_indices(image_mask, score_source="last_token", attention_mask=padding)
        visibility = pyramid_attention_key_visibility(padding, instruction_indices=queries, key_count=int(hidden_states.shape[1]))
        query_values = query[0, :, queries, :].permute(1, 0, 2).contiguous()
        key_values = keys[0].transpose(0, 1).contiguous()
        return pyramid_attention_scores(
            instruction_query=query_values,
            image_keys=key_values.index_select(0, image),
            image_count=int(image.numel()),
            device=hidden_states.device,
            attention_keys=key_values,
            attention_key_mask=visibility,
            image_key_indices=image,
        )

    def _apply_stage(self, stage_index: int, hidden_states: Any, image_indices: Any, scores: Any, state: dict[str, Any], boundary_layer: int) -> Any:
        from mllm_comp.compressors.image.btp import balanced_token_pruning_image_tokens
        if stage_index < len(self.schedule) - 1 and scores is None:
            raise BackendIntegrationError(f"BTP score for stage {stage_index} is missing.")
        return balanced_token_pruning_image_tokens(
            hidden_states,
            image_indices=image_indices,
            boundary_after_layer=self.schedule[stage_index],
            image_attention_scores=scores,
            spec=self.spec,
            media_unit="llm_token",
            source_layout={"backbone": "qwen3_5", "stage": stage_index},
            result_layout={"backbone": "qwen3_5", "stage": stage_index, "boundary": boundary_layer},
        )


class Qwen3_5HalfVActivation(Qwen3_5StagewiseDecoderActivation):
    algorithm = "halfv"
    binding_name = "halfv.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from dataclasses import fields
        from mllm_comp.compressors.image.halfv import HalfVSpec
        self.spec = self._load_spec(kwargs, HalfVSpec, fields(HalfVSpec))
        self.schedule = self.spec.boundary_after_layer_indices
        self.score_layers = self.schedule
        self.boundary_layers = tuple(layer + 1 for layer in self.schedule)
        super().__init__(model=kwargs["model"], binding=kwargs.get("binding"))

    def _stage_plan(self, depth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if self.boundary_layers[-1] >= depth:
            raise BackendIntegrationError("HalfV SSR boundary exceeds Qwen3.5 decoder depth.")
        return self.boundary_layers, self.score_layers

    def _capture_stage_scores(self, stage_index: int, module: Any, hidden_states: Any, position_embeddings: Any, state: dict[str, Any]) -> Any:
        import torch
        from mllm_comp.compressors.image.fastv import fastv_scorer_query_indices
        from mllm_comp.compressors.image.halfv import halfv_last_text_attention_scores
        query, keys = self._qk(module, hidden_states, position_embeddings, apply_rope=(stage_index != 0))
        current = state["current_indices"].to(hidden_states.device)
        image = state["image_indices"].to(hidden_states.device)
        padding = state["source_attention_mask"]
        if padding is not None:
            padding = padding.to(hidden_states.device).index_select(0, current)
        image_mask = torch.zeros(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
        image_mask[image] = True
        queries = fastv_scorer_query_indices(image_mask, score_source="last_token", attention_mask=padding)
        query_value = query[0, :, queries[-1], :].unsqueeze(0)
        image_keys = keys[0].transpose(0, 1).index_select(0, image)
        return halfv_last_text_attention_scores(query_value, image_keys)

    def _apply_stage(self, stage_index: int, hidden_states: Any, image_indices: Any, scores: Any, state: dict[str, Any], boundary_layer: int) -> Any:
        from mllm_comp.compressors.image.halfv import halfv_image_tokens
        if scores is None:
            raise BackendIntegrationError(f"HalfV score for stage {stage_index} is missing.")
        selection_hidden = (
            self.model.model.language_model.norm(hidden_states)
            if stage_index == 0 else None
        )
        return halfv_image_tokens(
            hidden_states,
            image_indices=image_indices,
            boundary_after_layer=self.schedule[stage_index],
            image_attention_scores=scores,
            initial_visual_tokens=int(state["initial_visual_tokens"]),
            selection_hidden_states=selection_hidden,
            spec=self.spec,
            media_unit="llm_token",
            source_layout={"backbone": "qwen3_5", "stage": stage_index},
            result_layout={"backbone": "qwen3_5", "stage": stage_index, "boundary": boundary_layer},
        )


class Qwen3_5SGLActivation(Qwen3_5StagewiseDecoderActivation):
    """SGL guide-attention selection with Qwen3.8's no-cache layout repair."""

    algorithm = "sgl"
    binding_name = "sgl.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from dataclasses import fields
        from mllm_comp.compressors.image.sgl import SGLGuideController, SGLSpec

        self.spec = self._load_spec(kwargs, SGLSpec, fields(SGLSpec))
        self.score_layers: tuple[int, ...] = ()
        self.boundary_layers: tuple[int, ...] = ()
        self.runtime = kwargs.get("runtime")
        if self.runtime is None:
            raise BackendIntegrationError("Qwen3.8 SGL requires the active target runtime.")
        self.guide = SGLGuideController(
            backbone_name="qwen3_vl",
            target_runtime=self.runtime,
            spec=self.spec,
        )
        super().__init__(
            model=kwargs["model"],
            binding=kwargs.get("binding"),
            runtime=self.runtime,
        )

    def _stage_plan(self, depth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self.spec.prune_layer(depth),), ()

    def _capture_stage_scores(
        self, stage_index: int, module: Any, hidden_states: Any,
        position_embeddings: Any, state: dict[str, Any],
    ) -> Any:
        raise BackendIntegrationError("SGL scores must come from its aligned guide model.")

    def _apply_stage(
        self, stage_index: int, hidden_states: Any, image_indices: Any,
        scores: Any, state: dict[str, Any], boundary_layer: int,
    ) -> Any:
        from mllm_comp.compressors.image.sgl import sgl_image_tokens

        if scores is not None:
            raise BackendIntegrationError("Qwen3.8 SGL unexpectedly received target-model scores.")
        guide_scores = state.get("sgl_guide_scores")
        if guide_scores is None:
            guide_scores = self.guide.consume_scores(
                target_visual_tokens=int(image_indices.numel()),
                device=hidden_states.device,
            )
            state["sgl_guide_scores"] = guide_scores.detach()
        else:
            guide_scores = guide_scores.to(device=hidden_states.device)
        return sgl_image_tokens(
            hidden_states,
            image_indices=image_indices,
            image_scores=guide_scores,
            spec=self.spec,
            media_unit="llm_token",
            source_layout={"backbone": "qwen3_5", "guide": self.spec.guide_model_path},
            result_layout={"backbone": "qwen3_5", "boundary_layer": boundary_layer},
        )

    def install(self) -> None:
        if self._installed:
            return
        super().install()
        try:
            self.guide.attach()
        except BaseException:
            super().uninstall()
            raise

    def uninstall(self) -> None:
        try:
            self.guide.detach()
        finally:
            super().uninstall()

    def runtime_metadata(self) -> dict[str, Any]:
        return {**super().runtime_metadata(), **self.guide.runtime_metadata()}


class Qwen3_5D2PrunerActivation(Qwen3_5StagewiseDecoderActivation):
    """D²Pruner selection using its checkpoint/layer-calibrated spatial prior."""

    algorithm = "d2pruner"
    binding_name = "d2pruner.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        from dataclasses import fields
        from mllm_comp.compressors.image.d2pruner import D2PrunerSpec
        from mllm_comp.compressors.image.decoder_selection import D2PrunerDecoderMethod

        self.spec = self._load_spec(kwargs, D2PrunerSpec, fields(D2PrunerSpec))
        self.runtime = kwargs.get("runtime")
        if self.runtime is None:
            raise BackendIntegrationError("Qwen3.8 D²Pruner requires the active target runtime.")
        self.method = D2PrunerDecoderMethod(
            stage=kwargs.get("stage"),
            config={field.name: getattr(self.spec, field.name) for field in fields(D2PrunerSpec)},
            rules=None,
            backbone="qwen3_5",
        )
        try:
            self.method.prepare(self.runtime)
        except (TypeError, ValueError, OSError) as exc:
            raise BackendIntegrationError(f"Qwen3.8 D²Pruner prior validation failed: {exc}") from exc
        self.score_layers = (self.spec.prune_after_layer - 1,)
        self.boundary_layers = (self.spec.prune_after_layer,)
        super().__init__(
            model=kwargs["model"],
            binding=kwargs.get("binding"),
            runtime=self.runtime,
        )

    def _stage_plan(self, depth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        boundary, score_layer = self.boundary_layers[0], self.score_layers[0]
        layer_types = tuple(self.model.config.text_config.layer_types)
        if boundary >= depth or score_layer < 0:
            raise BackendIntegrationError("Qwen3.8 D²Pruner boundary exceeds decoder depth.")
        if layer_types[score_layer] != "full_attention":
            raise BackendIntegrationError(
                "Qwen3.8 D²Pruner scorer must be a full-attention block; "
                f"layer {score_layer} is {layer_types[score_layer]!r}."
            )
        return self.boundary_layers, self.score_layers

    def _capture_stage_scores(
        self, stage_index: int, module: Any, hidden_states: Any,
        position_embeddings: Any, state: dict[str, Any],
    ) -> Any:
        import torch
        from mllm_comp.compressors.image.fastv import fastv_scorer_query_indices
        from mllm_comp.compressors.image.pyramid_drop import (
            pyramid_attention_key_visibility,
            pyramid_attention_scores,
        )

        query, keys = self._qk(module, hidden_states, position_embeddings, apply_rope=True)
        current = state["current_indices"].to(hidden_states.device)
        images = state["image_indices"].to(hidden_states.device)
        padding = state["source_attention_mask"]
        if padding is not None:
            padding = padding.to(hidden_states.device).index_select(0, current)
        image_mask = torch.zeros(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
        image_mask[images] = True
        queries = fastv_scorer_query_indices(
            image_mask, score_source="last_token", attention_mask=padding
        )
        visibility = pyramid_attention_key_visibility(
            padding, instruction_indices=queries, key_count=int(hidden_states.shape[1])
        )
        key_values = keys[0].transpose(0, 1).contiguous()
        return pyramid_attention_scores(
            rules=None,
            instruction_query=query[0, :, queries, :].permute(1, 0, 2).contiguous(),
            image_keys=key_values.index_select(0, images),
            image_count=int(images.numel()),
            device=hidden_states.device,
            attention_keys=key_values,
            attention_key_mask=visibility,
            image_key_indices=images,
        )

    def _apply_stage(
        self, stage_index: int, hidden_states: Any, image_indices: Any,
        scores: Any, state: dict[str, Any], boundary_layer: int,
    ) -> Any:
        if scores is None:
            raise BackendIntegrationError("Qwen3.8 D²Pruner did not capture scorer attention.")
        return self.method.select(
            hidden_states,
            image_indices=image_indices,
            image_attention_scores=scores,
            media_unit="llm_token",
            source_layout={"backbone": "qwen3_5", "score_layer": self.score_layers[0]},
            result_layout={"backbone": "qwen3_5", "boundary_layer": boundary_layer},
        )

    def runtime_metadata(self) -> dict[str, Any]:
        metadata = super().runtime_metadata()
        prior, prior_metadata = self.method.prior or (None, {})
        metadata.update(
            {
                "bias_prior_path": self.spec.bias_prior_path,
                "bias_prior_metadata": dict(prior_metadata),
                "bias_prior_loaded": prior is not None,
            }
        )
        return metadata
