"""Qwen2.5-Omni runtime support for encoder-level audio token merging.

This adapter targets the verified Transformers 4.57.3 audio tower. It keeps
the provider modules and parameters intact while replacing the audio encoder
loop during an active request so that sequence lengths can change between
layers. Selected layers merge after the MHSA residual and before the FFN,
matching A-ToMe's insertion point.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from itertools import accumulate
from time import perf_counter
from types import MethodType, SimpleNamespace
from typing import Any

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.layer_positions import LayerPosition, resolve_layer_positions
from mllm_comp.core.errors import InvariantError
from mllm_comp.core.plan import CompressionPlan
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import ExecutionPhase, Placement

from .audio_compression import (
    _compressed_audio_rope_index,
    _materialize_report_value,
    _module_execution_device,
    _register_strategy_module,
    _refresh_strategy_module,
    _require_verified_transformers,
    _run_audio_preflight,
    _torch,
    legalize_qwen25_omni_audio_tokens,
)
@dataclass
class _ActiveEncoderRequest:
    original_counts: list[int]
    keep_counts: list[int | None]
    feature_lengths: list[int]
    initial_chunk_lengths: list[list[int]]
    planned_final_chunk_lengths: list[list[int] | None]
    initial_encoder_counts: list[int]
    planned_final_encoder_counts: list[int | None]
    sample_indices: list[int]
    audio_unit_indices: list[int]
    audio_forward_calls: int = 0
    layer_reports: list[dict[str, Any]] = field(default_factory=list)
    score_seconds: float = 0.0
    merge_seconds: float = 0.0

    @property
    def identity(self) -> bool:
        return self.original_counts == self.keep_counts


def mean_received_attention_scores(
    attention: Any,
    hidden_states: Any,
    chunk_lengths: list[int],
) -> Any:
    """Return per-token attention received, averaged over queries and heads."""

    torch = _torch()
    if sum(chunk_lengths) != int(hidden_states.shape[0]):
        raise InvariantError(
            "Attention-pruning chunk lengths do not match hidden states."
        )
    score_chunks: list[Any] = []
    offset = 0
    with torch.no_grad():
        for length in chunk_lengths:
            chunk = hidden_states[offset : offset + length]
            query = attention.q_proj(chunk).reshape(
                length,
                attention.num_heads,
                -1,
            ).transpose(0, 1)
            key = attention.k_proj(chunk).reshape(
                length,
                attention.num_heads,
                -1,
            ).transpose(0, 1)
            weights = torch.matmul(query, key.transpose(-2, -1)) * attention.scaling
            weights = torch.softmax(weights, dim=-1, dtype=torch.float32)
            scores = weights.sum(dim=-2).mean(dim=0)
            if not bool(torch.isfinite(scores).all().item()):
                raise InvariantError(
                    "Encoder attention importance contains NaN or Inf."
                )
            score_chunks.append(scores)
            offset += length
    return torch.cat(score_chunks, dim=0)


class Qwen25OmniEncoderCompressionController:
    """Install one registered layerwise audio-encoder compression strategy."""

    def __init__(
        self,
        model: Any,
        stage: Any,
        *,
        profile_enabled: bool = False,
        debug_enabled: bool = False,
        execution: Any | None = None,
    ) -> None:
        _require_verified_transformers(model)
        if execution is None:
            from .v2_compression import resolve_qwen25_omni_stage_execution

            execution = resolve_qwen25_omni_stage_execution(stage)
        if execution.executor_family != "encoder_token":
            raise InvariantError(
                f"{stage.plugin} does not use the encoder-token executor."
            )
        self.model = model
        self.stage = stage
        self.profile_enabled = bool(profile_enabled)
        self.debug_enabled = bool(debug_enabled)
        self.execution = execution
        self.plugin = execution.create_plugin()
        self.plugin_config = stage.plugin_config()
        self.plugin_config["_stage_name"] = stage.name
        self.plugin_config["_execution_phase"] = stage.execution_phase.value
        self.plugin_config, self.spec = execution.prepare_config(self.plugin_config)
        self.strategy = execution.create_strategy(self.spec, operation=stage.operation, stage=stage)
        self._requires_layer_input = bool(getattr(self.strategy, "requires_layer_input", True))
        self.thinker = getattr(model, "thinker", model)
        self.audio_tower = getattr(self.thinker, "audio_tower", None)
        self.layers = list(getattr(self.audio_tower, "layers", ()))
        if self.thinker is None or self.audio_tower is None or not self.layers:
            raise InvariantError(
                "Qwen2.5-Omni encoder compression could not resolve "
                "thinker.audio_tower.layers."
            )
        self.active_layers = self.strategy.active_layer_indices(len(self.layers))
        # Native encoder execution is chronological, including legacy explicit
        # layer lists whose configuration order was not sorted.
        source_layers = tuple(sorted(self.active_layers))
        self.application_positions = resolve_layer_positions(
            stage.operation, source_layers, boundary=execution.compression_boundary,
            num_layers=len(self.layers),
        )
        if len(self.application_positions) != len(source_layers):
            raise InvariantError("operation.layers must preserve the encoder rule's number of stages.")
        self._position_by_rule_layer = dict(zip(source_layers, self.application_positions))
        self._rule_layer_by_position = dict(zip(self.application_positions, source_layers))
        score_layer_index = getattr(self.strategy, "score_layer_index", None)
        self._score_layer_by_rule_layer = {
            rule: score_layer_index(rule, len(self.layers)) if self._requires_layer_input and callable(score_layer_index) else rule
            for rule in source_layers
        }
        for rule, position in self._position_by_rule_layer.items():
            source = self._score_layer_by_rule_layer[rule]
            if self._requires_layer_input and source > position.index:
                raise InvariantError(
                    f"Encoder application {position.as_dict()} precedes scoring layer {source}'s input."
                )
        _register_strategy_module(self)
        self._active_request: _ActiveEncoderRequest | None = None
        self._layout_observer: Any | None = None
        self._last_report: dict[str, Any] | None = None
        self._installed = False

    @property
    def last_report(self) -> dict[str, Any] | None:
        if self._last_report is not None:
            self._last_report = _materialize_report_value(self._last_report)
        return self._last_report

    @last_report.setter
    def last_report(self, value: dict[str, Any] | None) -> None:
        self._last_report = value

    def install(self) -> None:
        if self._installed:
            return
        self._original_audio_forward = self.audio_tower.forward
        self._original_output_lengths = self.audio_tower._get_feat_extract_output_lengths
        self._original_get_rope_index = self.thinker.get_rope_index
        self.audio_tower._mllm_encoder_compression = self
        self.audio_tower.forward = MethodType(
            self._compressed_audio_forward,
            self.audio_tower,
        )
        self.audio_tower._get_feat_extract_output_lengths = MethodType(
            self._compressed_output_lengths,
            self.audio_tower,
        )
        self.thinker.get_rope_index = MethodType(
            self._compressed_rope_index,
            self.thinker,
        )
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.audio_tower.forward = self._original_audio_forward
        self.audio_tower._get_feat_extract_output_lengths = self._original_output_lengths
        self.thinker.get_rope_index = self._original_get_rope_index
        del self.audio_tower._mllm_encoder_compression
        self._installed = False

    def prepare_state(self, state: ForwardState) -> tuple[ForwardState, CompressionPlan]:
        adapt_plan = getattr(self.strategy, "adapt_plan", None)
        plan = self.plugin.plan(
            state=state,
            context=CompressionContext(
                phase=(
                    "train"
                    if self.stage.execution_phase is ExecutionPhase.FORWARD
                    else "eval"
                ),
                model=self.model,
                structural_handles={"audio_encoder": self.audio_tower},
                extra={"defer_token_budget": callable(adapt_plan)},
            ),
            config=self.plugin_config,
        )
        if callable(adapt_plan):
            plan = adapt_plan(plan, state=state, audio_tower=self.audio_tower)
        if any(action.metadata["keep_count"] is None for action in plan.token_actions):
            # The same installed encoder loop realizes natural budgets before
            # placeholder legalization; no separate preview implementation.
            with self.request(plan) as realized:
                _run_audio_preflight(self.audio_tower, state)
            actions = [replace(action, metadata={
                **action.metadata, "keep_count": realized.keep_counts[i],
                "final_chunk_lengths": realized.planned_final_chunk_lengths[i],
                "final_encoder_count": realized.planned_final_encoder_counts[i],
            }) for i, action in enumerate(plan.token_actions)]
            plan = replace(plan, token_actions=actions, this_stage_metadata=replace(plan.this_stage_metadata, stats={
                **plan.this_stage_metadata.stats, "planned_audio_tokens": sum(realized.keep_counts),
                "planned_encoder_tokens": sum(realized.planned_final_encoder_counts),
                "preflight_audio_encoder_pass": True,
            }))
        if plan.legalization_policy is None:
            raise InvariantError(
                f"{self.plugin.name} plan omitted its legalization policy."
            )
        legalized = legalize_qwen25_omni_audio_tokens(
            state,
            token_actions=plan.token_actions,
            policy=plan.legalization_policy,
            stage_name=plan.stage_name,
            expected_target_site=self.execution.target_site,
        )
        return legalized, plan

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **self.strategy.runtime_metadata(self.active_layers),
            "target_site": self.execution.target_site,
            "compression_boundary": self.execution.compression_boundary,
            "score_layers_zero_based": sorted(set(self._score_layer_by_rule_layer.values())) if self._requires_layer_input else [],
            "application_positions": [p.as_dict() for p in self.application_positions],
        }

    @contextmanager
    def request(self, plan: CompressionPlan) -> Any:
        if self._active_request is not None:
            raise InvariantError(
                "Nested Qwen2.5-Omni encoder compression requests are unsupported."
            )
        active = _ActiveEncoderRequest(
            original_counts=[
                int(action.metadata["original_count"])
                for action in plan.token_actions
            ],
            keep_counts=[
                None if action.metadata["keep_count"] is None else int(action.metadata["keep_count"])
                for action in plan.token_actions
            ],
            feature_lengths=[
                int(action.metadata["feature_length"])
                for action in plan.token_actions
            ],
            initial_chunk_lengths=[
                [int(value) for value in action.metadata["initial_chunk_lengths"]]
                for action in plan.token_actions
            ],
            planned_final_chunk_lengths=[
                None if action.metadata["final_chunk_lengths"] is None else [int(value) for value in action.metadata["final_chunk_lengths"]]
                for action in plan.token_actions
            ],
            initial_encoder_counts=[
                int(action.metadata["initial_encoder_count"])
                for action in plan.token_actions
            ],
            planned_final_encoder_counts=[
                None if action.metadata["final_encoder_count"] is None else int(action.metadata["final_encoder_count"])
                for action in plan.token_actions
            ],
            sample_indices=[
                int(action.metadata["sample_index"])
                for action in plan.token_actions
            ],
            audio_unit_indices=[
                int(action.metadata["audio_unit_index"])
                for action in plan.token_actions
            ],
        )
        self._active_request = active
        self.last_report = None
        try:
            yield active
            if active.original_counts and active.audio_forward_calls < 1:
                raise InvariantError(
                    f"{self.plugin.name} planned audio units but the encoder was not called."
                )
            self.last_report = {
                "stage_name": plan.stage_name,
                "plugin": self.plugin.name,
                "execution_phase": plan.execution_phase.value,
                "placement": Placement.ENCODER_LEVEL.value,
                **self.runtime_metadata(),
                "original_audio_tokens": sum(active.original_counts),
                "kept_audio_tokens": sum(active.keep_counts),
                "initial_encoder_tokens": sum(active.initial_encoder_counts),
                "final_encoder_tokens": sum(active.planned_final_encoder_counts),
                "audio_forward_calls": active.audio_forward_calls,
                "score_seconds": (
                    active.score_seconds if self.profile_enabled else None
                ),
                "merge_seconds": (
                    active.merge_seconds if self.profile_enabled else None
                ),
                "compression_seconds": (
                    active.merge_seconds if self.profile_enabled else None
                ),
                "layers": (
                    list(active.layer_reports)
                    if self.debug_enabled or self.profile_enabled
                    else None
                ),
            }
        finally:
            self._active_request = None

    def _compressed_output_lengths(
        self,
        audio_tower: Any,
        input_lengths: Any,
    ) -> Any:
        del audio_tower
        audio_feat_lengths, audio_output_lengths = self._original_output_lengths(
            input_lengths
        )
        active = self._active_request
        if active is None or not active.original_counts:
            return audio_feat_lengths, audio_output_lengths
        feature_lengths = [int(value) for value in input_lengths.tolist()]
        if feature_lengths != active.feature_lengths:
            raise InvariantError(
                "Encoder compression feature lengths do not match the plan: "
                f"runtime={feature_lengths}, planned={active.feature_lengths}"
            )
        originals = [int(value) for value in audio_output_lengths.tolist()]
        if originals != active.original_counts:
            raise InvariantError(
                "Encoder compression placeholders do not match uncompressed output "
                f"output lengths: spans={active.original_counts}, encoder={originals}."
            )
        if None in active.keep_counts:
            # Preflight only needs the CNN lengths; its pooled count is not yet known.
            return audio_feat_lengths, audio_output_lengths
        compressed_lengths = audio_output_lengths.new_tensor(active.keep_counts)
        return audio_feat_lengths, compressed_lengths

    def _compressed_rope_index(
        self,
        thinker: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del thinker
        active = self._active_request
        if active is None or active.identity or not active.keep_counts:
            return self._original_get_rope_index(*args, **kwargs)
        return _compressed_audio_rope_index(
            self._original_get_rope_index, active.keep_counts, *args, **kwargs
        )

    def _normalized_score_input(self, rule_layer: int, hidden: Any) -> Any:
        norm = self.layers[self._score_layer_by_rule_layer[rule_layer]].self_attn_layer_norm
        with _torch().no_grad():
            return norm(hidden.to(_module_execution_device(norm)))

    def _compress_boundary(
        self, rule_layer: int, hidden: Any, source_input: Any,
        chunk_lengths: list[int], *, telemetry: _ActiveEncoderRequest | None,
        replay_plan: Any = None,
    ) -> tuple[Any, list[int], Any]:
        _refresh_strategy_module(self)
        output_lengths = self._compressed_chunk_lengths(rule_layer, chunk_lengths)
        if replay_plan is not None:
            lengths, decision = replay_plan
            return self.strategy.replay(hidden, decision), list(lengths), decision
        started = perf_counter() if telemetry is not None and self.profile_enabled else None
        probe = None
        if self._requires_layer_input:
            if source_input is None:
                source_input = self._normalized_score_input(rule_layer, hidden)
            layer = self.layers[self._score_layer_by_rule_layer[rule_layer]]
            score_device = _module_execution_device(layer.self_attn.k_proj)
            with _torch().no_grad():
                probe = self.strategy.probe(layer, source_input.to(score_device), chunk_lengths).to(hidden.device)
        collect_details = telemetry is not None and (self.debug_enabled or self.profile_enabled)
        output, actual_lengths, reports, plan = self.strategy.compress(
            hidden, probe, chunk_lengths, layer_index=rule_layer,
            active_layers=self.active_layers, collect_details=collect_details,
        )
        if None not in output_lengths and actual_lengths != output_lengths:
            raise InvariantError(f"Encoder strategy produced {actual_lengths}, expected {output_lengths}.")
        output_lengths = actual_lengths
        if started is not None:
            telemetry.merge_seconds += perf_counter() - started
        if collect_details:
            position = self._position_by_rule_layer[rule_layer]
            layer_index = position.index if position.boundary != "inter_layer" else (
                position.index - 1 if position.index else None
            )
            telemetry.layer_reports.append({
                "layer_index": layer_index,
                "score_layer_index": self._score_layer_by_rule_layer[rule_layer] if self._requires_layer_input else None,
                "application_position": position.as_dict(),
                "input_tokens": sum(chunk_lengths), "output_tokens": sum(output_lengths),
                "removed_tokens": sum(chunk_lengths) - sum(output_lengths),
                "input_chunk_lengths": list(chunk_lengths),
                "output_chunk_lengths": list(output_lengths), "chunks": reports,
            })
        return output, output_lengths, plan

    def _run_compressed_layer(
        self, layer: Any, layer_index: int, hidden_states: Any,
        cu_seqlens: Any, chunk_lengths: list[int], *, rule_layer: int,
        source_input: Any, replay_plan: Any,
        telemetry: _ActiveEncoderRequest | None, attention_mask: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Any, list[int], Any]:
        runtime_boundaries = [int(value) for value in cu_seqlens.tolist()]
        expected_boundaries = [0, *accumulate(chunk_lengths)]
        if runtime_boundaries != expected_boundaries:
            raise InvariantError(
                f"Encoder compression layer {layer_index} received stale cu_seqlens: "
                f"runtime={runtime_boundaries}, expected={expected_boundaries}"
            )
        if self._requires_layer_input and replay_plan is None and source_input is None and self._score_layer_by_rule_layer[rule_layer] == layer_index:
            source_input = self._normalized_score_input(rule_layer, hidden_states)
        if self.execution.compression_boundary == "post_layer":
            boundary_states = layer(
                hidden_states, cu_seqlens=cu_seqlens,
                attention_mask=attention_mask, **kwargs,
            )[0]
        else:
            hidden_states = hidden_states.to(_module_execution_device(layer.self_attn_layer_norm))
            normalized = layer.self_attn_layer_norm(hidden_states)
            boundary_states = hidden_states + layer.self_attn(
                hidden_states=normalized, cu_seqlens=cu_seqlens,
                attention_mask=attention_mask, **kwargs,
            )
        output, output_lengths, plan = self._compress_boundary(
            rule_layer, boundary_states, source_input, chunk_lengths,
            telemetry=telemetry, replay_plan=replay_plan,
        )
        if self.execution.compression_boundary == "post_attention_pre_ffn":
            output = output + layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(output))))
            torch = _torch()
            if output.dtype == torch.float16:
                clamp_value = torch.finfo(output.dtype).max - 1000
                output = torch.clamp(output, min=-clamp_value, max=clamp_value)
        return output, output_lengths, plan

    def _compressed_chunk_lengths(
        self,
        layer_index: int,
        chunk_lengths: list[int],
    ) -> list[int | None]:
        return self.strategy.output_lengths(
            layer_index,
            chunk_lengths,
            self.active_layers,
        )

    def _compression_enabled(self, layer_index: int) -> bool:
        return self.strategy.enabled(layer_index, self.active_layers)

    def _compressed_audio_forward(
        self,
        audio_tower: Any,
        input_features: Any,
        feature_lens: Any | None = None,
        aftercnn_lens: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        active = self._active_request
        if active is None or not active.original_counts:
            return self._original_audio_forward(
                input_features,
                feature_lens=feature_lens,
                aftercnn_lens=aftercnn_lens,
                **kwargs,
            )
        if feature_lens is None or aftercnn_lens is None:
            raise InvariantError(
                "Encoder compression requires feature_lens and aftercnn_lens."
            )
        torch = _torch()
        feature_values = [int(value) for value in feature_lens.tolist()]
        if feature_values != active.feature_lengths:
            raise InvariantError(
                "Encoder compression feature lengths differ from the plan."
            )

        chunk_num = torch.ceil(feature_lens / (audio_tower.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [audio_tower.n_window * 2] * int(chunk_num.sum().item()),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = torch.nn.functional.pad(
            chunk_num,
            (1, 0),
            value=-1,
        ).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (audio_tower.n_window * 2)
        chunk_lengths = torch.where(
            chunk_lengths == 0,
            audio_tower.n_window * 2,
            chunk_lengths,
        )
        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = (
            audio_tower.padded_and_mask_function(
                chunk_list,
                chunk_lengths,
                padding_value=0,
                padding_side="right",
            )
        )
        padded_embed = torch.nn.functional.gelu(
            audio_tower.conv1(padded_feature)
        ) * padded_mask
        padded_embed = torch.nn.functional.gelu(
            audio_tower.conv2(padded_embed)
        ).transpose(1, 2)
        padded_embed = padded_embed + audio_tower.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        current_chunk_lengths = [
            int(value) for value in padded_mask_after_cnn.sum(1).tolist()
        ]
        planned_initial_chunks = [
            length for chunks in active.initial_chunk_lengths for length in chunks
        ]
        if current_chunk_lengths != planned_initial_chunks:
            raise InvariantError(
                "Encoder compression runtime chunk lengths do not match the plan: "
                f"runtime={current_chunk_lengths}, planned={planned_initial_chunks}"
            )
        original_aftercnn = [int(value) for value in aftercnn_lens.tolist()]
        if original_aftercnn != active.initial_encoder_counts:
            raise InvariantError(
                "Encoder compression aftercnn lengths do not match the plan: "
                f"runtime={original_aftercnn}, planned={active.initial_encoder_counts}"
            )

        use_gradient_checkpointing = bool(
            getattr(audio_tower, "gradient_checkpointing", False)
            and getattr(audio_tower, "training", False)
            and torch.is_grad_enabled()
        )
        checkpoint = getattr(audio_tower, "_gradient_checkpointing_func", None)
        if use_gradient_checkpointing and not callable(checkpoint):
            raise InvariantError(
                "Encoder compression found gradient_checkpointing=True without a "
                "provider checkpoint function."
            )

        pending = {i for i in self.active_layers if self._compression_enabled(i)}
        source_inputs: dict[int, Any] = {}
        initial_rule = self._rule_layer_by_position.get(LayerPosition("inter_layer", 0))
        if initial_rule in pending:
            hidden_states, current_chunk_lengths, initial_plan = self._compress_boundary(
                initial_rule, hidden_states, None, current_chunk_lengths, telemetry=active,
            )
            if self._layout_observer is not None:
                self._layout_observer(self.strategy, initial_plan, current_chunk_lengths,
                                      [len(chunks) for chunks in active.initial_chunk_lengths])
            pending.remove(initial_rule)

        for layer_index, encoder_layer in enumerate(audio_tower.layers):
            if self._requires_layer_input:
                for rule in pending:
                    application = self._position_by_rule_layer[rule]
                    if self._score_layer_by_rule_layer[rule] == layer_index and application.index > layer_index + (application.boundary == "inter_layer"):
                        source_inputs[rule] = self._normalized_score_input(rule, hidden_states)
            position = (
                LayerPosition("inter_layer", layer_index + 1)
                if self.execution.compression_boundary == "post_layer"
                else LayerPosition("post_attention_pre_ffn", layer_index)
            )
            rule_layer = self._rule_layer_by_position.get(position)
            boundaries = [0, *accumulate(current_chunk_lengths)]
            cu_seqlens = torch.tensor(
                boundaries,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            attention_mask = audio_tower._prepare_attention_mask(
                hidden_states,
                cu_seqlens,
            )
            input_chunk_lengths = list(current_chunk_lengths)
            if rule_layer in pending:
                expected_output_lengths = self._compressed_chunk_lengths(
                    rule_layer,
                    input_chunk_lengths,
                )
                telemetry_pending = [True]
                source_box = [source_inputs.pop(rule_layer, None)]
                plan_box = [None]

                def compressed_layer_forward(
                    states: Any,
                    *,
                    _layer: Any = encoder_layer,
                    _layer_index: int = layer_index,
                    _rule_layer: int = rule_layer,
                    _source_box: list[Any] = source_box,
                    _plan_box: list[Any] = plan_box,
                    _cu_seqlens: Any = cu_seqlens,
                    _chunk_lengths: tuple[int, ...] = tuple(input_chunk_lengths),
                    _expected_lengths: tuple[int | None, ...] = tuple(expected_output_lengths),
                    _attention_mask: Any = attention_mask,
                    _kwargs: dict[str, Any] = dict(kwargs),
                    _active: _ActiveEncoderRequest = active,
                    _telemetry_pending: list[bool] = telemetry_pending,
                ) -> Any:
                    telemetry = _active if _telemetry_pending[0] else None
                    _telemetry_pending[0] = False
                    output, output_lengths, replay_plan = self._run_compressed_layer(
                        _layer,
                        _layer_index,
                        states,
                        _cu_seqlens,
                        list(_chunk_lengths),
                        rule_layer=_rule_layer,
                        source_input=_source_box[0],
                        replay_plan=_plan_box[0],
                        telemetry=telemetry,
                        attention_mask=_attention_mask,
                        **_kwargs,
                    )
                    # The checkpoint owns realized layout alongside the rule's
                    # opaque decision, including content-dependent chunk sizes.
                    _plan_box[0] = (tuple(output_lengths), replay_plan)
                    _source_box[0] = None
                    if None not in _expected_lengths and output_lengths != list(_expected_lengths):
                        raise InvariantError(
                            f"Encoder compression layer {_layer_index} produced "
                            "non-deterministic "
                            f"lengths {output_lengths}; expected {list(_expected_lengths)}."
                        )
                    return output

                if use_gradient_checkpointing:
                    _refresh_strategy_module(self)
                    if (not hidden_states.requires_grad and self.strategy_runtime_module is not None
                            and any(p.requires_grad for p in self.strategy_runtime_module.parameters())):
                        # Reentrant checkpoint needs a tracked input even when
                        # the frozen encoder only trains the captured HTA module.
                        hidden_states.requires_grad_(True)
                    hidden_states = checkpoint(
                        compressed_layer_forward,
                        hidden_states,
                    )
                else:
                    hidden_states = compressed_layer_forward(hidden_states)
                current_chunk_lengths = list(plan_box[0][0])
                if self._layout_observer is not None:
                    self._layout_observer(self.strategy, plan_box[0][1], current_chunk_lengths,
                                          [len(chunks) for chunks in active.initial_chunk_lengths])
                pending.remove(rule_layer)
                with torch.no_grad():
                    for source_layer, values in source_inputs.items():
                        source_inputs[source_layer] = self.strategy.replay(values, plan_box[0][1]).detach()
            else:
                def original_layer_forward(
                    states: Any,
                    *,
                    _layer: Any = encoder_layer,
                    _cu_seqlens: Any = cu_seqlens,
                    _attention_mask: Any = attention_mask,
                    _kwargs: dict[str, Any] = dict(kwargs),
                ) -> Any:
                    return _layer(
                        states,
                        cu_seqlens=_cu_seqlens,
                        attention_mask=_attention_mask,
                        **_kwargs,
                    )[0]

                if use_gradient_checkpointing:
                    hidden_states = checkpoint(
                        original_layer_forward,
                        hidden_states,
                    )
                else:
                    hidden_states = original_layer_forward(hidden_states)
            if sum(current_chunk_lengths) != int(hidden_states.shape[0]):
                raise InvariantError(
                    f"Encoder compression layer {layer_index} output length does not match "
                    "its updated chunk lengths."
                )

        chunk_counts = [int(value) for value in chunk_num.tolist()]
        final_encoder_lengths: list[int] = []
        offset = 0
        for count in chunk_counts:
            final_encoder_lengths.append(
                sum(current_chunk_lengths[offset : offset + count])
            )
            offset += count
        if offset != len(current_chunk_lengths):
            raise InvariantError(
                "Encoder compression did not consume every runtime audio chunk."
            )
        if None in active.planned_final_encoder_counts:
            offset = 0
            for index, count in enumerate(chunk_counts):
                active.planned_final_chunk_lengths[index] = list(current_chunk_lengths[offset:offset + count])
                offset += count
            active.planned_final_encoder_counts = list(final_encoder_lengths)
        if any(length < 2 for length in final_encoder_lengths):
            raise InvariantError("Encoder compression leaves fewer than two tokens for an audio unit's pooler.")
        if final_encoder_lengths != active.planned_final_encoder_counts:
            raise InvariantError(
                "Encoder compression final lengths do not match the deterministic plan: "
                f"runtime={final_encoder_lengths}, "
                f"planned={active.planned_final_encoder_counts}"
            )

        hidden_states_list = hidden_states.split(final_encoder_lengths, dim=0)
        token_audio_list: list[Any] = []
        actual_output_lengths: list[int] = []
        for each_audio_states in hidden_states_list:
            each_audio_states = audio_tower.avg_pooler(
                each_audio_states.transpose(0, 1)
            ).transpose_(0, 1)
            each_audio_states = audio_tower.ln_post(each_audio_states)
            # Validate this encoder stage before a downstream projector hook
            # applies its own budget to the same audio unit.
            actual_output_lengths.append(int(each_audio_states.shape[0]))
            each_audio_states = audio_tower.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        if None in active.keep_counts:
            active.keep_counts = list(actual_output_lengths)
        if actual_output_lengths != active.keep_counts:
            raise InvariantError(
                "Encoder compression post-pool lengths do not match LLM placeholders: "
                f"runtime={actual_output_lengths}, planned={active.keep_counts}"
            )
        token_audio = torch.cat(token_audio_list, dim=0)
        active.audio_forward_calls += 1
        return _base_model_output(token_audio)


def _base_model_output(last_hidden_state: Any) -> Any:
    """Construct the provider output lazily so core unit tests need no Transformers."""

    try:
        from transformers.modeling_outputs import BaseModelOutput
    except ImportError:
        return SimpleNamespace(last_hidden_state=last_hidden_state)
    return BaseModelOutput(last_hidden_state=last_hidden_state)


__all__ = [
    "Qwen25OmniEncoderCompressionController",
]
