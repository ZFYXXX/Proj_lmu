"""Native post-merger selection with aligned MRoPE, labels and DeepStack taps."""
from contextvars import ContextVar

from .operations import OperationComparison, has_cache
from mllm_comp.fixed_mask.runtime import _call, make_layout


class QwenProjectorComparison(OperationComparison):
    def __init__(self, model, method, config, backbone):
        config = dict(config)
        self.retention_ratio = config.pop("retention_ratio", None)
        if self.retention_ratio is not None:
            if "tokens_per_crop" in config or isinstance(self.retention_ratio, bool):
                raise ValueError("Specify retention_ratio or tokens_per_crop, not both")
            self.retention_ratio = float(self.retention_ratio)
            if not 0 < self.retention_ratio <= 1:
                raise ValueError("retention_ratio must be in (0, 1]")
        if method == "hiprune":
            config.setdefault("object_layer", max(1, round(9 * len(model.model.visual.blocks) / 24)))
        if method == "eadp":
            # Qwen release scripts use beta=2; preserve explicitly supplied beta.
            config.setdefault("beta", 2.0)
        super().__init__(model, method, config, backbone)
        self.is_projector = True
        self.source = ContextVar(f"comparison_vision_{id(self)}", default=None)

    def token_budget(self, source_count):
        if self.retention_ratio is None:
            return min(self.config["tokens_per_crop"], source_count)
        return min(source_count, max(1, int(round(source_count * self.retention_ratio))))

    def install_layer(self, layer, index):
        super().install_layer(layer, index)
        original = layer.forward
        def forward(_layer, *args, **kwargs):
            from .operations import KEY
            context = kwargs.get(KEY)
            if context is not None and not context[1]:
                state = context[0]
                counts = state["runtime"].setdefault("observed_visual_tokens_by_layer", [None] * len(self.layers))
                counts[index] = int(state["layout"].vision[0].sum())
            return original(*args, **kwargs)
        self.patch(layer, "forward", forward)

    def install_qwen_input(self):
        super().install_qwen_input()
        original = self.model.prepare_inputs_for_generation

        def prepare(_model, *args, **kwargs):
            import torch
            prepared = original(*args, **kwargs)
            if has_cache(prepared.get("past_key_values")):
                if self.request is None:
                    raise ValueError("Projector generation cache has no owning request")
                ids = prepared["input_ids"]
                offset = self.request["next_position"]
                physical = torch.arange(offset, offset + ids.shape[-1], device=ids.device)
                # GenerationMixin advances its input sequence at the original
                # prompt length. The decoder KV cache has already been pruned.
                # Use that compact cache offset with the adjusted rope_deltas;
                # otherwise decode is shifted by the number of deleted tokens.
                rotary = (physical[None] + self.model.model.rope_deltas).unsqueeze(0).expand(3, -1, -1)
                old_positions = prepared.get("position_ids")
                if old_positions is not None and old_positions.ndim == 3 and old_positions.shape[0] == 4:
                    text = physical.view(1, 1, -1).expand(1, ids.shape[0], -1)
                    rotary = torch.cat((text, rotary), dim=0)
                prepared["position_ids"] = rotary
                prepared["cache_position"] = physical
                record = getattr(self, "last_selection", None)
                if record is not None and record.get("first_decode_position") is None:
                    first = rotary[-3:, :, 0]
                    expected = record["expected_first_decode_position"]
                    verified = bool((first == expected).all())
                    record.update(first_decode_position=first.reshape(-1).tolist(),
                        decode_position_verified=verified)
                    if not verified:
                        raise ValueError(f"{self.method} first decode M-RoPE does not follow compact prefill")
            return prepared

        self.patch(self.model, "prepare_inputs_for_generation", prepare)

    def prepare_source_inputs(self, call):
        import torch
        grid = call.get("image_grid_thw")
        if grid is None or len(grid) != 1 or int(grid[0, 0]) != 1:
            raise ValueError("Qwen projector comparisons require one image grid")
        state = {"grid": grid, "attention": {}}
        if self.method in {"cdpruner", "eadp"}:
            from .llava import instruction_ids
            ids = call["input_ids"].clone()
            config = self.model.config
            for value in (config.image_token_id, config.vision_start_token_id, config.vision_end_token_id):
                ids[ids == value] = -200
            selected = instruction_ids(ids, call.get("attention_mask"), call.get("labels"))
            state["text"] = self.decoder.embed_tokens(selected)
        self.source.set(state)

    def install_vision_capture(self):
        if self.method not in {"hiprune", "dop"}:
            return
        blocks = self.model.model.visual.blocks
        if not -len(blocks) <= self.config["vision_layer"] < len(blocks):
            raise ValueError("HiPrune deep attention layer exceeds this vision encoder's depth")
        indices = {self.config["vision_layer"] % len(blocks)}
        if self.method == "hiprune":
            indices.add(self.config["object_layer"] - 1)
        if any(index >= len(blocks) for index in indices):
            raise ValueError("HiPrune layer policy exceeds this vision encoder's depth")
        for index in indices:
            self.install_attention_capture(blocks[index].attn, index)
        if self.method == "dop":
            # The selected native vision layer supplies similarity features;
            # group its spatial patches in the same order as the merger.
            block = blocks[self.config["vision_layer"] % len(blocks)]
            original = block.forward
            def capture(_block, *args, **kwargs):
                output = original(*args, **kwargs)
                visual = self.model.model.visual
                features = output.detach().reshape(-1, visual.spatial_merge_size ** 2 * output.shape[-1])
                if self.backbone == "qwen25_vl":
                    order, _ = visual.get_window_index(self.source.get()["grid"])
                    features = features.index_select(0, order.argsort().to(features.device))
                self.source.get()["features"] = features
                return output
            self.patch(block, "forward", capture)

    def install_attention_capture(self, attention, index):
        original = attention.forward
        def forward(module, *args, **kwargs):
            import torch
            from mllm_comp.backbones._vision_attention import apply_vision_rotary, streaming_vision_attention
            call = _call(original, args, kwargs)
            hidden = call["hidden_states"]
            with torch.no_grad():
                qkv = module.qkv(hidden).reshape(len(hidden), 3, module.num_heads, -1)
                q, k = apply_vision_rotary(qkv[:, 0], qkv[:, 1], call["position_embeddings"])
                counts = (call["cu_seqlens"][1:] - call["cu_seqlens"][:-1]).tolist()
                received = []
                for query, key, value in zip(q.split(counts), k.split(counts), qkv[:, 2].split(counts)):
                    _, scores = streaming_vision_attention(query, key, value, scaling=module.scaling,
                        token_sizes=None, capture_mean_received=True)
                    received.append(scores * len(query) / len(hidden))
                scores = torch.cat(received)
                visual = self.model.model.visual
                scores = scores.reshape(-1, visual.spatial_merge_size ** 2).mean(-1)
                if self.backbone == "qwen25_vl":
                    window_order, _ = visual.get_window_index(self.source.get()["grid"])
                    scores = scores.index_select(0, window_order.argsort().to(scores.device))
                self.source.get()["attention"][index] = scores[None]
            return original(*args, **kwargs)
        self.patch(attention, "forward", forward)

    def prepare_qwen_embeddings(self, call, hidden):
        import torch
        from mllm_comp.backbones.image_training import compact_ignored_labels
        layout = self.incoming.get()
        source = self.source.get()
        visual_ids = layout.vision[0].nonzero(as_tuple=True)[0]
        features = hidden[0].index_select(0, visual_ids)
        merge = self.model.model.visual.spatial_merge_size
        height, width = (int(x) // merge for x in source["grid"][0, 1:])
        if height * width != len(features):
            raise ValueError("Qwen image placeholders and post-merger grid disagree")
        config = {**self.config, "tokens_per_crop": self.token_budget(len(features))}
        if self.method in {"cdpruner", "dppruner"}:
            from .cdpruner import select_cdpruner_tokens
            _, selected, stats = select_cdpruner_tokens(features[None], source.get("text"), cfg=config)
        elif self.method == "eadp":
            from .eadp import select_eadp_indices
            selected, stats = select_eadp_indices(features[None], source["text"],
                patches_per_side=width, grid_height=height, cfg=config)
        elif self.method == "prunesid":
            from .prunesid import select_prunesid_indices
            selected, stats = select_prunesid_indices(features[None], cfg=config)
        elif self.method == "dop":
            from mllm_comp.compressors.image.vispruner import select_vispruner_indices
            depth = len(self.model.model.visual.blocks)
            selected = [select_vispruner_indices(source["features"],
                source["attention"][self.config["vision_layer"] % depth][0],
                config["tokens_per_crop"],
                important_ratio=self.config["important_ratio"], removal_step=self.config["removal_step"]).sort().values]
            stats = {"token_reducer": "vispruner", "attention_source": "native_vision_mean_received"}
        else:
            from .hiprune import select_hiprune_indices
            attention = source["attention"]
            depth = len(self.model.model.visual.blocks)
            selected, stats = select_hiprune_indices(attention[self.config["object_layer"] - 1],
                attention[self.config["vision_layer"] % depth], tokens_per_crop=config["tokens_per_crop"],
                alpha=self.config["alpha"], patches_per_side=width, grid_height=height)
        local = selected[0].to(hidden.device)
        return self.compact_selection(call, hidden, local, stats)

    def compact_selection(self, call, hidden, local, stats):
        import torch
        from mllm_comp.backbones.image_training import compact_ignored_labels
        layout = self.incoming.get()
        source = self.source.get()
        visual_ids = layout.vision[0].nonzero(as_tuple=True)[0]
        merge = self.model.model.visual.spatial_merge_size
        height, width = (int(x) // merge for x in source["grid"][0, 1:])
        count = len(visual_ids)
        if len(local) != self.token_budget(count) or len(local.unique()) != len(local):
            raise ValueError("Projector selection has a duplicate or wrong per-image budget")
        keep = ~layout.vision[0].clone()
        keep[visual_ids.index_select(0, local)] = True
        indices = keep.nonzero(as_tuple=True)[0]
        if self.method == "prunesid":
            # The official K<=150 path feeds semantic-group order into the
            # causal decoder. A Boolean keep mask would silently sort it away.
            if not torch.equal(visual_ids, torch.arange(int(visual_ids[0]), int(visual_ids[-1]) + 1, device=visual_ids.device)):
                raise ValueError("PruneSID requires one contiguous image placeholder span")
            indices = torch.cat((indices[indices < visual_ids[0]],
                visual_ids.index_select(0, local), indices[indices > visual_ids[-1]]))
        labels = self.labels.get()
        if labels is not None:
            self.labels.set(compact_ignored_labels(labels, indices, source_length=hidden.shape[1], operation=self.method))
        positions = call["position_ids"].index_select(-1, indices)
        rotary = positions[1:] if positions.shape[0] == 4 else positions
        physical = torch.arange(len(indices), device=positions.device).view(1, 1, -1)
        call["position_ids"] = torch.cat((physical, rotary), dim=0)
        call["inputs_embeds"] = hidden.index_select(1, indices)
        call["input_ids"] = None
        updated = layout.select(indices)
        self.incoming.set(updated)
        call["attention_mask"] = updated.valid
        if "visual_pos_masks" in call:
            call["visual_pos_masks"] = updated.vision
        if call.get("deepstack_visual_embeds") is not None:
            call["deepstack_visual_embeds"] = [branch.index_select(0, local.to(branch.device)) for branch in call["deepstack_visual_embeds"]]
        self.model.model.rope_deltas = self.model.model.rope_deltas + hidden.shape[1] - len(indices)
        self.last_selection = {"source_count": count, "result_count": len(local),
            "expected_first_decode_position": int(rotary.max()) + 1,
            "first_decode_position": None, "decode_position_verified": None,
            "requested_retention_ratio": self.retention_ratio, "actual_retention_ratio": len(local) / count,
            "visual_tokens_per_decoder_layer": [len(local)] * len(self.layers),
            "layer_averaged_retention_ratio": len(local) / count,
            "native_grid": [height, width], "selected_indices": local.tolist(), **stats}
        self.last_selection["gathered_visual_indices"] = [int(i) for i in indices[updated.vision[0]] - visual_ids[0]]
        if self.last_selection["gathered_visual_indices"] != local.tolist():
            raise ValueError("Projector gather changed selector output order")
        return call["inputs_embeds"]

    def runtime_metadata(self):
        if self.request is not None and hasattr(self, "last_selection"):
            counts = self.request["runtime"].get("observed_visual_tokens_by_layer")
            if counts is not None:
                self.last_selection["observed_visual_tokens_per_decoder_layer"] = list(counts)
                if all(value is not None for value in counts):
                    self.last_selection["layer_averaged_retention_ratio"] = sum(counts) / (len(counts) * self.last_selection["source_count"])
        return {**super().runtime_metadata(), "selection": getattr(self, "last_selection", None),
            "budget_policy": {"retention_ratio": self.retention_ratio, "fixed_tokens": None if self.retention_ratio is not None else self.config["tokens_per_crop"]},
            "layout_delta_records": [self.last_selection] if hasattr(self, "last_selection") else [],
            "adaptation": "native_rectangular_post_merger_grid", "position_policy": "sparse_mrope"}

    def clear_diagnostic_records(self):
        super().clear_diagnostic_records()
        if hasattr(self, "last_selection"):
            del self.last_selection
