"""Native video prefill/forward pruning shared by METok and PruneVid.

Methods own scoring, selection and request-local decision history. These
loops own native masks, positions, cache layout, DeepStack and checkpoint
replay. Incremental decode uses the original model and video bridge.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from mllm_comp.backbones._decoder_boundary import decoder_post_attention_boundary
from mllm_comp.compressors.image.pyramid_drop import pyramid_attention_key_visibility
from mllm_comp.compressors.layer_positions import LayerPosition
from mllm_comp.core.errors import BackendIntegrationError, InvariantError


class VideoDecoderPruningExecutor:
    """Shared loops for installed executors with a per-request pruning call.

    Subclasses construct that call and publish their existing cache/layout
    records. Installation and signal-provider ownership stay in the executor.
    """

    def _decoder_forward_replacement(
        self,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        def replacement(_decoder: Any, *args: Any, **kwargs: Any) -> Any:
            active = self._active.get()
            if active is None:
                return original(*args, **kwargs)
            try:
                if self.backbone == "llava_onevision":
                    return self._run_llava_decoder(original, args, kwargs, active)
                if self.backbone == "qwen25_vl":
                    return self._run_qwen25_decoder(original, args, kwargs, active)
                return self._run_qwen3_decoder(original, args, kwargs, active)
            finally:
                self._active.set(None)

        return replacement

    def _run_llava_decoder(
        self,
        original: Callable[..., Any],
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        active: Any,
    ) -> Any:
        torch = _torch()
        from mllm_comp.backbones.llava_onevision.provider_bridge import (
            _bind_decoder_call,
            _decoder_masks,
            _decoder_query_keys,
            _new_dynamic_cache,
            _resolve_components,
            _transformers_decoder_helpers,
        )
        from mllm_comp.backbones.llava_onevision.video_compression import (
            plan_llava_onevision_video_splice,
        )

        core = _resolve_components(self.model).core
        call = _bind_decoder_call(inspect.signature(original), tuple(args), kwargs)
        input_ids = call.pop("input_ids")
        inputs_embeds = call.pop("inputs_embeds")
        padding_mask = call.pop("attention_mask")
        position_ids = call.pop("position_ids")
        past_key_values = call.pop("past_key_values")
        use_cache = call.pop("use_cache")
        cache_position = call.pop("cache_position")
        if call.pop("output_attentions", None) not in (None, False) or call.pop(
            "output_hidden_states", None
        ) not in (None, False):
            raise BackendIntegrationError(
                f"LLaVA {self.decoder_label} cannot expose decoder histories"
            )
        if call.pop("return_dict", None) is False:
            raise BackendIntegrationError(
                f"LLaVA {self.decoder_label} requires return_dict=True"
            )
        inputs_embeds = _resolve_inputs(
            core, input_ids, inputs_embeds, self.decoder_label
        )
        _require_batch_one(inputs_embeds, "LLaVA", self.decoder_label)
        sequence_length = int(inputs_embeds.shape[1])
        use_cache = bool(
            getattr(core.config, "use_cache", False) if use_cache is None else use_cache
        )
        if use_cache and past_key_values is None:
            past_key_values = _new_dynamic_cache(core.config)
        if padding_mask is None:
            padding_mask = torch.ones(
                1,
                sequence_length,
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        if tuple(padding_mask.shape) != (1, sequence_length):
            raise BackendIntegrationError(
                f"LLaVA {self.decoder_label} requires a rank-2 batch-one padding mask"
            )
        if cache_position is None:
            cache_position = torch.arange(sequence_length, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if tuple(position_ids.shape) != (1, sequence_length):
            raise BackendIntegrationError(
                f"LLaVA {self.decoder_label} requires rank-2 native text positions"
            )

        layers = tuple(core.layers)
        source_sequence_length = (
            sequence_length
            + active.projector_result.source_count
            - active.projector_result.result_count
        )
        splice_plan = plan_llava_onevision_video_splice(
            active.request,
            active.projector_result,
            sequence_length=source_sequence_length,
        )
        visual_positions = tuple(
            splice_plan.result_visual_span[0] + index
            for index, item in enumerate(splice_plan.packed_items)
            if item[1] == "patch"
        )
        if len(visual_positions) != active.projector_result.result_count:
            raise InvariantError(f"LLaVA {self.decoder_label} visual positions changed")
        visual_region = tuple(range(*splice_plan.result_visual_span))

        (
            BaseModelOutputWithPast,
            create_causal_mask,
            create_sliding_window_causal_mask,
        ) = _transformers_decoder_helpers()
        hidden_states = inputs_embeds
        masks = _decoder_masks(
            core,
            hidden_states,
            padding_mask,
            position_ids,
            cache_position,
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
        position_embeddings = core.rotary_emb(hidden_states, position_ids)
        pruning = self._new_decoder_call(
            active.projector_result, sequence_length, len(layers)
        )
        positions = pruning.positions

        def apply_stage(stage_index, point):
            nonlocal hidden_states, padding_mask, position_ids, visual_positions, sequence_length, cache_position, position_embeddings, visual_region, masks
            result = pruning.select(stage_index, hidden_states, visual_positions)
            cache_record = self._repair_decoder_cache(
                past_key_values,
                result,
                use_cache=use_cache,
                position=point,
            )
            keep = _index_tensor(result.kept_sequence_indices, hidden_states)
            old_visual = visual_positions
            old_region = visual_region
            hidden_states = hidden_states.index_select(1, keep)
            padding_mask = padding_mask.index_select(
                1,
                keep.to(padding_mask.device),
            )
            if pruning.llava_position_policy == "contiguous":
                position_ids = torch.arange(
                    hidden_states.shape[1],
                    device=position_ids.device,
                    dtype=position_ids.dtype,
                ).unsqueeze(0)
            else:
                position_ids = position_ids.index_select(
                    -1, keep.to(position_ids.device)
                )
            visual_positions = _mapped_positions(result, old_visual)
            visual_region = _mapped_positions(result, old_region)
            sequence_length = result.sequence_layout_delta.result_count
            cache_position = torch.arange(
                sequence_length,
                device=hidden_states.device,
            )
            masks = _decoder_masks(
                core,
                hidden_states,
                padding_mask,
                position_ids,
                cache_position,
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
            position_embeddings = core.rotary_emb(hidden_states, position_ids)
            self._publish_decoder_pruning(result, cache_record, point)
            return keep

        for layer_index in range(len(layers) + 1):
            point = LayerPosition("inter_layer", layer_index)
            layer = layers[layer_index] if layer_index < len(layers) else None
            for stage_index, score_layer in enumerate(pruning.score_layers):
                if score_layer == layer_index:
                    query_positions = _post_visual_queries(
                        padding_mask,
                        visual_end=visual_region[-1] + 1,
                        sequence_length=sequence_length,
                        operation=self.decoder_label,
                    )
                    layer_mask = masks[
                        getattr(layer, "attention_type", "full_attention")
                    ]
                    pruning.source_scores[stage_index] = _native_pruning_scores(
                        scorer=pruning.score,
                        qk_function=_decoder_query_keys,
                        decoder_layer=layer,
                        hidden_states=hidden_states,
                        visual_positions=visual_positions,
                        query_positions=query_positions,
                        position_embeddings=position_embeddings,
                        attention_mask=layer_mask,
                        padding_mask=padding_mask,
                    )
                if positions[stage_index] == point:
                    apply_stage(stage_index, point)
            if layer is None:
                break
            mid = LayerPosition("post_attention_pre_ffn", layer_index)

            def apply_mid(residual, point=mid):
                nonlocal hidden_states
                hidden_states = residual
                keeps = []
                for stage_index in range(len(positions)):
                    if positions[stage_index] == point:
                        keeps.append(apply_stage(stage_index, point))

                def replay(value):
                    for keep in keeps:
                        value = value.index_select(1, keep.to(value.device))
                    return value

                return hidden_states, replay

            layer_mask = masks[getattr(layer, "attention_type", "full_attention")]
            run_layer = (
                decoder_post_attention_boundary
                if mid in positions
                else lambda layer, values, **kw: layer(values, **kw)
            )
            output = run_layer(
                layer,
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **(
                    {"apply_compression": apply_mid, "tuple_output": True}
                    if mid in positions
                    else {}
                ),
                **call,
            )
            hidden_states = output[0] if isinstance(output, tuple) else output
        if pruning.completed != len(positions):
            raise InvariantError(
                f"LLaVA {self.decoder_label} missed a pruning boundary"
            )
        hidden_states = core.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _run_qwen25_decoder(
        self,
        original: Callable[..., Any],
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        active: Any,
    ) -> Any:
        torch = _torch()
        from mllm_comp.backbones.qwen25_vl.provider_bridge import (
            _bind_language_call,
            _decoder_query_keys,
            _decoder_state,
            _effective_bool,
            _new_dynamic_cache,
            _transformers_decoder_helpers,
        )

        language_model = self.model.model.language_model
        call = _bind_language_call(original, args, kwargs)
        input_ids = call.pop("input_ids")
        inputs_embeds = call.pop("inputs_embeds")
        padding_mask = call.pop("attention_mask")
        position_ids = call.pop("position_ids")
        past_key_values = call.pop("past_key_values")
        use_cache = _effective_bool(
            call.pop("use_cache"),
            getattr(language_model.config, "use_cache", False),
        )
        output_attentions = _effective_bool(
            call.pop("output_attentions"),
            getattr(language_model.config, "output_attentions", False),
        )
        output_hidden_states = _effective_bool(
            call.pop("output_hidden_states"),
            getattr(language_model.config, "output_hidden_states", False),
        )
        return_dict = _effective_bool(
            call.pop("return_dict"),
            getattr(language_model.config, "use_return_dict", True),
        )
        cache_position = call.pop("cache_position")
        if output_attentions or output_hidden_states or not return_dict:
            raise BackendIntegrationError(
                f"Qwen2.5-VL {self.decoder_label} requires return_dict without histories"
            )
        if isinstance(padding_mask, Mapping):
            raise BackendIntegrationError(
                f"Qwen2.5-VL {self.decoder_label} requires a rank-2 padding mask"
            )
        inputs_embeds = _resolve_inputs(
            language_model, input_ids, inputs_embeds, self.decoder_label
        )
        _require_batch_one(inputs_embeds, "Qwen2.5-VL", self.decoder_label)
        sequence_length = int(inputs_embeds.shape[1])
        if use_cache and past_key_values is None:
            past_key_values = _new_dynamic_cache(language_model.config)
        if cache_position is None:
            cache_position = torch.arange(sequence_length, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, 1, -1)
        if position_ids.ndim != 3 or int(position_ids.shape[0]) not in (3, 4):
            raise BackendIntegrationError(
                f"Qwen2.5-VL {self.decoder_label} requires native M-RoPE positions"
            )
        prompt = self._bridge._prompt_splice.get()
        if prompt is None:
            raise InvariantError(
                f"Qwen2.5-VL {self.decoder_label} prompt state is missing"
            )
        visual_positions = prompt.plan.result_placeholder_positions
        if len(visual_positions) != active.projector_result.result_count:
            raise InvariantError(
                f"Qwen2.5-VL {self.decoder_label} visual positions changed"
            )
        layers = tuple(language_model.layers)
        (
            BaseModelOutputWithPast,
            create_causal_mask,
            create_sliding_window_causal_mask,
        ) = _transformers_decoder_helpers()
        hidden_states = inputs_embeds
        masks, text_positions, rotary_positions = _decoder_state(
            language_model,
            hidden_states,
            padding_mask,
            position_ids,
            cache_position,
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
        position_embeddings = language_model.rotary_emb(
            hidden_states,
            rotary_positions,
        )
        pruning = self._new_decoder_call(
            active.projector_result, sequence_length, len(layers)
        )
        positions = pruning.positions

        def apply_stage(stage_index, point):
            nonlocal hidden_states, padding_mask, position_ids, visual_positions, sequence_length, cache_position, position_embeddings, masks, text_positions, rotary_positions
            result = pruning.select(stage_index, hidden_states, visual_positions)
            cache_record = self._repair_decoder_cache(
                past_key_values,
                result,
                use_cache=use_cache,
                position=point,
            )
            keep = _index_tensor(result.kept_sequence_indices, hidden_states)
            old_visual = visual_positions
            hidden_states = hidden_states.index_select(1, keep)
            if padding_mask is not None:
                padding_mask = padding_mask.index_select(
                    -1,
                    keep.to(padding_mask.device),
                )
            position_ids = position_ids.index_select(
                -1,
                keep.to(position_ids.device),
            )
            visual_positions = _mapped_positions(result, old_visual)
            sequence_length = result.sequence_layout_delta.result_count
            cache_position = torch.arange(
                sequence_length,
                device=hidden_states.device,
            )
            masks, text_positions, rotary_positions = _decoder_state(
                language_model,
                hidden_states,
                padding_mask,
                position_ids,
                cache_position,
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
            position_embeddings = language_model.rotary_emb(
                hidden_states,
                rotary_positions,
            )
            self._publish_decoder_pruning(result, cache_record, point)
            return keep

        for layer_index in range(len(layers) + 1):
            point = LayerPosition("inter_layer", layer_index)
            layer = layers[layer_index] if layer_index < len(layers) else None
            for stage_index, score_layer in enumerate(pruning.score_layers):
                if score_layer == layer_index:
                    query_positions = _post_visual_queries(
                        padding_mask,
                        visual_end=visual_positions[-1] + 1,
                        sequence_length=sequence_length,
                        operation=self.decoder_label,
                    )
                    layer_mask = masks[
                        getattr(layer, "attention_type", "full_attention")
                    ]
                    pruning.source_scores[stage_index] = _native_pruning_scores(
                        scorer=pruning.score,
                        qk_function=_decoder_query_keys,
                        decoder_layer=layer,
                        hidden_states=hidden_states,
                        visual_positions=visual_positions,
                        query_positions=query_positions,
                        position_embeddings=position_embeddings,
                        attention_mask=layer_mask,
                        padding_mask=padding_mask,
                    )
                if positions[stage_index] == point:
                    apply_stage(stage_index, point)
            if layer is None:
                break
            mid = LayerPosition("post_attention_pre_ffn", layer_index)

            def apply_mid(residual, point=mid):
                nonlocal hidden_states
                hidden_states = residual
                keeps = []
                for stage_index in range(len(positions)):
                    if positions[stage_index] == point:
                        keeps.append(apply_stage(stage_index, point))

                def replay(value):
                    for keep in keeps:
                        value = value.index_select(1, keep.to(value.device))
                    return value

                return hidden_states, replay

            layer_mask = masks[getattr(layer, "attention_type", "full_attention")]
            run_layer = (
                decoder_post_attention_boundary
                if mid in positions
                else lambda layer, values, **kw: layer(values, **kw)
            )
            output = run_layer(
                layer,
                hidden_states,
                attention_mask=layer_mask,
                position_ids=text_positions,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **({"apply_compression": apply_mid} if mid in positions else {}),
                **call,
            )
            hidden_states = output[0] if isinstance(output, tuple) else output
        if pruning.completed != len(positions):
            raise InvariantError(
                f"Qwen2.5-VL {self.decoder_label} missed a pruning boundary"
            )
        hidden_states = language_model.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def _run_qwen3_decoder(
        self,
        original: Callable[..., Any],
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        active: Any,
    ) -> Any:
        torch = _torch()
        from mllm_comp.backbones.qwen3_vl.provider_bridge import (
            _attention_key_visibility,
            _bind_language_call,
            _decoder_query_keys,
            _new_dynamic_cache,
            _transformers_decoder_helpers,
        )

        language_model = self.model.model.language_model
        call = _bind_language_call(original, args, kwargs)
        input_ids = call.pop("input_ids")
        inputs_embeds = call.pop("inputs_embeds")
        padding_mask = call.pop("attention_mask")
        position_ids = call.pop("position_ids")
        past_key_values = call.pop("past_key_values")
        use_cache_value = call.pop("use_cache")
        use_cache = bool(
            getattr(language_model.config, "use_cache", False)
            if use_cache_value is None
            else use_cache_value
        )
        cache_position = call.pop("cache_position")
        visual_pos_masks = call.pop("visual_pos_masks")
        deepstack = call.pop("deepstack_visual_embeds")
        deepstack = None if deepstack is None else list(deepstack)
        for name in ("output_attentions", "output_hidden_states"):
            if call.pop(name, None) not in (None, False):
                raise BackendIntegrationError(
                    f"Qwen3-VL {self.decoder_label} cannot expose decoder histories"
                )
        if call.pop("return_dict", None) is False:
            raise BackendIntegrationError(
                f"Qwen3-VL {self.decoder_label} requires return_dict=True"
            )
        inputs_embeds = _resolve_inputs(
            language_model, input_ids, inputs_embeds, self.decoder_label
        )
        _require_batch_one(inputs_embeds, "Qwen3-VL", self.decoder_label)
        sequence_length = int(inputs_embeds.shape[1])
        if use_cache and past_key_values is None:
            past_key_values = _new_dynamic_cache(language_model.config)
        if cache_position is None:
            cache_position = torch.arange(sequence_length, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, 1, -1)
        if position_ids.ndim != 3 or int(position_ids.shape[0]) not in (3, 4):
            raise BackendIntegrationError(
                f"Qwen3-VL {self.decoder_label} requires native M-RoPE positions"
            )
        if visual_pos_masks is None or tuple(visual_pos_masks.shape) != (
            1,
            sequence_length,
        ):
            raise BackendIntegrationError(
                f"Qwen3-VL {self.decoder_label} requires exact visual_pos_masks"
            )
        visual_positions = tuple(
            int(value)
            for value in visual_pos_masks[0]
            .bool()
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        if len(visual_positions) != active.projector_result.result_count:
            raise InvariantError(
                f"Qwen3-VL {self.decoder_label} visual positions changed"
            )
        if int(position_ids.shape[0]) == 4:
            text_positions = position_ids[0]
            rotary_positions = position_ids[1:]
        else:
            text_positions = position_ids[0]
            rotary_positions = position_ids
        BaseModelOutputWithPast, create_causal_mask = _transformers_decoder_helpers()
        hidden_states = inputs_embeds
        attention_mask = create_causal_mask(
            config=language_model.config,
            input_embeds=hidden_states,
            attention_mask=padding_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_positions,
        )
        position_embeddings = language_model.rotary_emb(
            hidden_states,
            rotary_positions,
        )
        layers = tuple(language_model.layers)
        pruning = self._new_decoder_call(
            active.projector_result, sequence_length, len(layers)
        )
        positions = pruning.positions
        pending_branch_keeps = []

        def apply_stage(stage_index, point):
            nonlocal hidden_states, padding_mask, position_ids, visual_positions, sequence_length, cache_position, position_embeddings, visual_pos_masks, text_positions, rotary_positions, attention_mask
            result = pruning.select(stage_index, hidden_states, visual_positions)
            cache_record = self._repair_decoder_cache(
                past_key_values,
                result,
                use_cache=use_cache,
                position=point,
            )
            keep = _index_tensor(result.kept_sequence_indices, hidden_states)
            old_visual = visual_positions
            hidden_states = hidden_states.index_select(1, keep)
            if padding_mask is not None:
                padding_mask = padding_mask.index_select(
                    -1,
                    keep.to(padding_mask.device),
                )
            position_ids = position_ids.index_select(
                -1,
                keep.to(position_ids.device),
            )
            visual_pos_masks = visual_pos_masks.index_select(
                -1,
                keep.to(visual_pos_masks.device),
            )
            pending_branch_keeps.append((point.index, result.kept_visual_indices))
            visual_positions = _mapped_positions(result, old_visual)
            sequence_length = result.sequence_layout_delta.result_count
            cache_position = torch.arange(
                sequence_length,
                device=hidden_states.device,
            )
            if int(position_ids.shape[0]) == 4:
                text_positions = position_ids[0]
                rotary_positions = position_ids[1:]
            else:
                text_positions = position_ids[0]
                rotary_positions = position_ids
            attention_mask = create_causal_mask(
                config=language_model.config,
                input_embeds=hidden_states,
                attention_mask=padding_mask,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=text_positions,
            )
            position_embeddings = language_model.rotary_emb(
                hidden_states,
                rotary_positions,
            )
            self._publish_decoder_pruning(
                result,
                cache_record,
                point,
                deepstack_branch_count=len(deepstack or ()),
            )
            return keep

        def gather_pending_branches():
            # Run outside the boundary's no-grad decision callback.
            if deepstack is not None:
                for first_branch, kept in pending_branch_keeps:
                    for branch_index in range(first_branch, len(deepstack)):
                        branch = deepstack[branch_index]
                        deepstack[branch_index] = branch.index_select(
                            0, _index_tensor(kept, branch)
                        )
            pending_branch_keeps.clear()

        for layer_index in range(len(layers) + 1):
            point = LayerPosition("inter_layer", layer_index)
            layer = layers[layer_index] if layer_index < len(layers) else None
            for stage_index, score_layer in enumerate(pruning.score_layers):
                if score_layer == layer_index:
                    query_positions = _post_visual_queries(
                        padding_mask,
                        visual_end=visual_positions[-1] + 1,
                        sequence_length=sequence_length,
                        operation=self.decoder_label,
                    )
                    pruning.source_scores[stage_index] = _native_pruning_scores(
                        scorer=pruning.score,
                        qk_function=_decoder_query_keys,
                        decoder_layer=layer,
                        hidden_states=hidden_states,
                        visual_positions=visual_positions,
                        query_positions=query_positions,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        visibility_function=_attention_key_visibility,
                    )
                if positions[stage_index] == point:
                    apply_stage(stage_index, point)
            gather_pending_branches()
            if layer is None:
                break
            mid = LayerPosition("post_attention_pre_ffn", layer_index)

            def apply_mid(residual, point=mid):
                nonlocal hidden_states
                hidden_states = residual
                keeps = []
                for stage_index in range(len(positions)):
                    if positions[stage_index] == point:
                        keeps.append(apply_stage(stage_index, point))

                def replay(value):
                    for keep in keeps:
                        value = value.index_select(1, keep.to(value.device))
                    return value

                return hidden_states, replay

            run_layer = (
                decoder_post_attention_boundary
                if mid in positions
                else lambda layer, values, **kw: layer(values, **kw)
            )
            hidden_states = run_layer(
                layer,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_positions,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **({"apply_compression": apply_mid} if mid in positions else {}),
                **call,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            gather_pending_branches()
            if deepstack is not None and layer_index < len(deepstack):
                hidden_states = language_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack[layer_index],
                )
        if pruning.completed != len(positions):
            raise InvariantError(
                f"Qwen3-VL {self.decoder_label} missed a pruning boundary"
            )
        hidden_states = language_model.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def _native_pruning_scores(
    *,
    qk_function: Callable[..., tuple[Any, Any, Any]],
    decoder_layer: Any,
    hidden_states: Any,
    visual_positions: tuple[int, ...],
    query_positions: tuple[int, ...],
    position_embeddings: tuple[Any, Any],
    attention_mask: Any,
    padding_mask: Any,
    visibility_function: Callable[..., Any] = pyramid_attention_key_visibility,
    scorer: Callable[..., Any],
) -> Any:
    torch = _torch()
    visual = torch.tensor(
        visual_positions,
        dtype=torch.long,
        device=hidden_states.device,
    )
    queries = torch.tensor(
        query_positions,
        dtype=torch.long,
        device=hidden_states.device,
    )
    with torch.no_grad():
        query, _visual_keys, attention_keys = qk_function(
            decoder_layer,
            hidden_states,
            instruction_indices=queries,
            image_indices=visual,
            position_embeddings=position_embeddings,
        )
        visibility = visibility_function(
            attention_mask,
            instruction_indices=queries,
            key_count=int(attention_keys.shape[0]),
            padding_mask=padding_mask,
        )
        return scorer(
            hidden_states,
            query,
            attention_keys,
            visual_positions=visual_positions,
            attention_key_mask=visibility,
        )


def _mapped_positions(
    result: Any,
    positions: Iterable[int],
) -> tuple[int, ...]:
    mapped = tuple(
        result.sequence_layout_delta.old_to_new[position] for position in positions
    )
    return tuple(int(value) for value in mapped if value is not None)


def _post_visual_queries(
    padding_mask: Any,
    *,
    visual_end: int,
    sequence_length: int,
    operation: str,
) -> tuple[int, ...]:
    if not 0 < visual_end <= sequence_length:
        raise InvariantError(f"{operation} visual span is outside the decoder sequence")
    if padding_mask is None:
        result = tuple(range(visual_end, sequence_length))
    else:
        if (
            getattr(padding_mask, "ndim", None) != 2
            or int(padding_mask.shape[0]) != 1
            or int(padding_mask.shape[1]) != sequence_length
        ):
            raise BackendIntegrationError(
                f"{operation} requires a rank-2 batch-one padding mask"
            )
        result = tuple(
            index
            for index in range(visual_end, sequence_length)
            if bool(padding_mask[0, index].item())
        )
    if not result:
        raise BackendIntegrationError(
            f"{operation} requires post-visual text query tokens"
        )
    return result


def _resolve_inputs(
    owner: Any, input_ids: Any, inputs_embeds: Any, operation: str
) -> Any:
    if (input_ids is None) == (inputs_embeds is None):
        raise BackendIntegrationError(
            f"{operation} decoder requires exactly one of input_ids or inputs_embeds"
        )
    return owner.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds


def _require_batch_one(value: Any, backbone: str, operation: str) -> None:
    if getattr(value, "ndim", None) != 3 or int(value.shape[0]) != 1:
        raise BackendIntegrationError(
            f"{backbone} {operation} decoder requires batch size one"
        )


def _index_tensor(values: Sequence[int], reference: Any) -> Any:
    torch = _torch()
    return torch.tensor(values, dtype=torch.long, device=reference.device)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "Video decoder pruning provider execution requires the existing PyTorch environment"
        ) from exc
    return torch
