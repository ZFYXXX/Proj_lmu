"""Multimodal-v2 LLM-level token compression for Qwen2.5-Omni.

During training forward and inference prefill, a strategy shortens only audio
tokens at the configured layer boundaries. Following layers receive a rebuilt
mask, original absolute M-RoPE embeddings and cache metadata matching their
physical KV length. Checkpoint recomputation replays the saved decision;
incremental decode never selects tokens again.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, fields, is_dataclass, replace
from inspect import signature
from time import perf_counter
from types import MethodType
from typing import Any, Mapping

from mllm_comp.backbones._decoder_boundary import decoder_post_attention_forward
from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.layer_positions import LayerPosition, resolve_layer_positions
from mllm_comp.core.errors import InvariantError
from mllm_comp.core.metadata import StageMetadata
from mllm_comp.core.plan import CompressionPlan
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import ExecutionPhase, Placement

from .audio_compression import (
    _materialize_report_value,
    _module_execution_device,
    _register_strategy_module,
    _refresh_strategy_module,
    _require_verified_transformers,
    _torch,
)
@dataclass
class _PreparedLLMContext:
    attention_mask: Any
    labels: Any | None
    semantic_rows: list[list[str]]
    input_width: int


@dataclass(frozen=True)
class _MergeReplayUnit:
    sample_index: int
    slots: tuple[int, ...]
    output_positions: tuple[int, ...]
    replay_plan: Any


@dataclass
class _ActiveLLMRequest:
    plan: CompressionPlan
    original_attention_mask: Any
    original_labels: Any | None
    semantic_rows: list[list[str]]
    input_width: int
    current_positions: Any
    instruction_by_original: Any
    audio_units: list[dict[str, Any]]
    layer_entry_positions: dict[int, Any] = field(default_factory=dict)
    layer_input_positions: dict[int, Any] = field(default_factory=dict)
    layer_exit_positions: dict[int, Any] = field(default_factory=dict)
    score_inputs: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    compression_positions: dict[int, Any] = field(default_factory=dict)
    merge_replay_units: dict[
        int, tuple[_MergeReplayUnit, ...]
    ] = field(default_factory=dict)
    stage_reports: list[dict[str, Any]] = field(default_factory=list)
    prefill_complete: bool = False
    prefill_forward_calls: int = 0
    decode_forward_calls: int = 0
    score_seconds: float = 0.0
    compression_seconds: float = 0.0
    initial_hidden_states: Any | None = None

    @property
    def original_counts(self) -> list[int]:
        return [int(item["original_count"]) for item in self.audio_units]

    @property
    def current_counts(self) -> list[int]:
        result: list[int] = []
        for unit in self.audio_units:
            row = self.current_positions[int(unit["sample_index"])]
            result.append(
                int(
                    (
                        (row >= int(unit["span_start"]))
                        & (row < int(unit["span_end"]))
                    ).sum().item()
                )
            )
        return result


class Qwen25OmniLLMTokenCompressionController:
    """Shared Qwen executor for all registered LLM-token strategies."""

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
        if execution.executor_family != "llm_token":
            raise InvariantError(
                f"{stage.plugin} does not use the LLM-token executor."
            )
        self.model = model
        self.stage = stage
        self.execution = execution
        self.profile_enabled = bool(profile_enabled)
        self.debug_enabled = bool(debug_enabled)
        self.plugin_config = stage.plugin_config()
        self.plugin_config["_stage_name"] = stage.name
        self.plugin_config["_execution_phase"] = stage.execution_phase.value
        self.plugin = execution.create_plugin()
        self.plugin_config, self.spec = execution.prepare_config(self.plugin_config)
        self.strategy = execution.create_strategy(self.spec, operation=stage.operation, stage=stage)
        self._requires_layer_input = bool(getattr(self.strategy, "requires_layer_input", True))
        score_parameters = signature(self.strategy.score_audio_tokens).parameters if self._requires_layer_input else {}
        self._score_accepts_initial_hidden_states = (
            "initial_hidden_states" in score_parameters
        )
        self._score_accepts_layer_index = "layer_index" in score_parameters
        self._output_length_accepts_layer_index = (
            "layer_index" in signature(self.strategy.output_length).parameters
        )
        self._merge_audio_tokens = getattr(
            self.strategy, "merge_audio_tokens", None
        )
        self._merge_accepts_layer_index = callable(self._merge_audio_tokens) and "layer_index" in signature(self._merge_audio_tokens).parameters
        self._replay_merge_audio_tokens = getattr(
            self.strategy, "replay_merge_audio_tokens", None
        )
        self._score_with_grad = bool(
            getattr(self.strategy, "score_with_grad", True)
        )
        self._compression_boundary = execution.compression_boundary
        self._requires_instruction_queries = bool(
            getattr(self.strategy, "requires_instruction_queries", True)
        )
        if self._compression_boundary not in {
            "post_layer",
            "post_attention_pre_ffn",
        }:
            raise InvariantError(
                f"Unsupported LLM-token compression boundary "
                f"{self._compression_boundary!r}."
            )
        self.thinker = getattr(model, "thinker", model)
        self.llm = getattr(self.thinker, "model", None)
        self.layers = list(getattr(self.llm, "layers", ()))
        if self.thinker is None or self.llm is None or not self.layers:
            raise InvariantError(
                "LLM-token compression requires thinker.model.layers."
            )
        self._encoder_rule = execution.binding.canonical_executor_family == "encoder_token"
        source_depth = len(self.thinker.audio_tower.layers) if self._encoder_rule else len(self.layers)
        self.stage_layers = self.strategy.stage_layer_indices(source_depth)
        self.application_positions = resolve_layer_positions(
            stage.operation, self.stage_layers,
            boundary=self._compression_boundary, num_layers=len(self.layers),
        )
        if len(self.application_positions) != len(self.stage_layers):
            raise InvariantError(
                "operation.layers must preserve the rule's number of compression stages; "
                "change its original schedule to change stage count."
            )
        self._position_by_rule_layer = dict(zip(self.stage_layers, self.application_positions))
        self._rule_layer_by_position = dict(zip(self.application_positions, self.stage_layers))
        for source, position in self._position_by_rule_layer.items():
            if (source > position.index and self._requires_layer_input
                    and not self._score_accepts_initial_hidden_states):
                raise InvariantError(
                    f"Compression at {position.as_dict()} precedes scoring layer {source}'s "
                    "input. Choose an available scoring layer in the original rule schedule."
                )
        _register_strategy_module(self)
        self._original_layer_forwards = [layer.forward for layer in self.layers]
        self._original_thinker_forward = self.thinker.forward
        self._learned_input_hook = None
        self.encoder_references = {}
        encoder_offset = getattr(self.strategy, "encoder_attention_layer_offset", None)
        if encoder_offset is not None:
            from .audio_attention_reference import AudioAttentionReference

            self.encoder_references[None] = AudioAttentionReference(
                self.thinker.audio_tower, encoder_offset, profile_enabled=self.profile_enabled,
                distinct_importance=bool(getattr(self.strategy, "uses_distinct_merge_prune_importance", False)),
            )
        if self._encoder_rule:
            from .audio_attention_reference import AudioAttentionReference
            depth = len(self.thinker.audio_tower.layers)
            self.encoder_references = {
                source: AudioAttentionReference(self.thinker.audio_tower, source - depth, profile_enabled=self.profile_enabled,
                                                probe_kind=self.strategy.encoder_reference_kind)
                for source in self.stage_layers
            }
        self._active_request: _ActiveLLMRequest | None = None
        self._replay_request: _ActiveLLMRequest | None = None
        self._prepared_context: _PreparedLLMContext | None = None
        self._last_report: dict[str, Any] | None = None
        self._last_token_layouts: dict[int, list[list[str]]] | None = None
        self._installed = False

    @property
    def last_report(self) -> dict[str, Any] | None:
        if self._last_report is not None:
            self._last_report = _materialize_report_value(self._last_report)
        return self._last_report

    @property
    def last_token_layouts(self) -> dict[int, list[list[str]]] | None:
        return self._last_token_layouts

    def runtime_metadata(self) -> dict[str, Any]:
        strategy_metadata = self.strategy.runtime_metadata(self.stage_layers)
        if None in self.encoder_references:
            strategy_metadata["encoder_score_layer_zero_based"] = self.encoder_references[None].layer_index
        if self._compression_boundary != "post_layer":
            strategy_metadata.pop("compression_boundary_after_layer_zero_based", None)
        if self.application_positions != resolve_layer_positions(
            {}, self.stage_layers, boundary=self._compression_boundary
        ):
            for name in ("stage_layers_zero_based", "stage_boundaries_one_based",
                         "compression_boundary_after_layer_zero_based", "pruning_layer_zero_based"):
                strategy_metadata.pop(name, None)
        return {
            **strategy_metadata,
            "target_site": self.execution.target_site,
            "compression_boundary": self._compression_boundary,
            "score_layers_zero_based": list(self.stage_layers) if self._requires_layer_input else [],
            "application_positions": [p.as_dict() for p in self.application_positions],
        }

    def install(self) -> None:
        if self._installed:
            return
        for reference in self.encoder_references.values():
            reference.install()
        if self.strategy_runtime_module is not None:
            self._learned_input_hook = self.llm.register_forward_pre_hook(self._track_learned_rule_input, with_kwargs=True)
        for layer_index, layer in enumerate(self.layers):
            original = self._original_layer_forwards[layer_index]

            def compressed_layer_forward(
                layer_self: Any,
                *args: Any,
                _layer_index: int = layer_index,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                del layer_self
                return self._layer_forward(
                    _layer_index,
                    _original,
                    args,
                    kwargs,
                )

            layer.forward = MethodType(compressed_layer_forward, layer)

        original_thinker = self._original_thinker_forward

        def compressed_thinker_forward(
            thinker_self: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return self._thinker_forward(thinker_self, original_thinker, args, kwargs)

        self.thinker.forward = MethodType(compressed_thinker_forward, self.thinker)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for reference in reversed(tuple(self.encoder_references.values())):
            reference.uninstall()
        if self._learned_input_hook is not None:
            self._learned_input_hook.remove()
            self._learned_input_hook = None
        for layer, original in zip(self.layers, self._original_layer_forwards):
            layer.forward = original
        self.thinker.forward = self._original_thinker_forward
        self._installed = False

    def _track_learned_rule_input(self, module: Any, args: Any, kwargs: Any) -> None:
        if (self._active_request is None or not self._active_request.audio_units or not module.training
                or not getattr(module, "gradient_checkpointing", False) or not _torch().is_grad_enabled()):
            return
        _refresh_strategy_module(self)
        hidden = kwargs.get("inputs_embeds", args[4] if len(args) > 4 else None)
        if (hidden is not None and not hidden.requires_grad
                and any(p.requires_grad for p in self.strategy_runtime_module.parameters())):
            # Establish the gradient path before native layer checkpoints. The
            # backbone parameters and raw input features remain frozen.
            hidden.requires_grad_(True)

    def prepare_state(self, state: ForwardState) -> tuple[ForwardState, CompressionPlan]:
        plan = self.plugin.plan(
            state=state,
            context=CompressionContext(
                phase=(
                    "train"
                    if self.stage.execution_phase is ExecutionPhase.FORWARD
                    else "eval"
                ),
                model=self.model,
                structural_handles={"llm": self.llm, **({"audio_encoder": self.thinker.audio_tower} if self._encoder_rule else {})},
                extra={"runtime_lifecycle": "prefill_then_decode",
                       "defer_token_budget": callable(getattr(self.strategy, "adapt_plan", None))},
            ),
            config=self.plugin_config,
        )
        adapt_plan = getattr(self.strategy, "adapt_plan", None)
        if callable(adapt_plan):
            plan = adapt_plan(plan)
        if plan.legalization_policy is None or (
            plan.legalization_policy.position_policy != "preserve_absolute"
        ):
            raise InvariantError(
                "LLM-token compression requires position_policy='preserve_absolute'."
            )
        input_shape = getattr(state.input_ids, "shape", None)
        mask_shape = getattr(state.attention_mask, "shape", None)
        if (
            input_shape is None
            or len(input_shape) != 2
            or mask_shape is None
            or tuple(mask_shape) != tuple(input_shape)
        ):
            raise InvariantError(
                "LLM-token compression requires rank-2 input_ids and attention_mask."
            )
        semantic_rows = state.modality_metadata.get(
            "semantic_token_labels_by_sample"
        )
        semantic_rows_valid = (
            isinstance(semantic_rows, list)
            and len(semantic_rows) == int(input_shape[0])
            and all(
                isinstance(row, list) and len(row) == int(input_shape[1])
                for row in semantic_rows
            )
        )
        if self._requires_instruction_queries and not semantic_rows_valid:
            raise InvariantError(
                "LLM-token compression requires per-token semantic labels so system, "
                "special and answer tokens cannot become attention queries."
            )
        if not semantic_rows_valid:
            semantic_rows = [
                ["other"] * int(input_shape[1]) for _ in range(int(input_shape[0]))
            ]
        if not isinstance(semantic_rows, list) or not all(
            isinstance(row, list) for row in semantic_rows
        ):
            raise InvariantError("LLM-token semantic labels could not be normalized.")
        self._prepared_context = _PreparedLLMContext(
            attention_mask=state.attention_mask,
            labels=state.labels,
            semantic_rows=[[str(item) for item in row] for row in semantic_rows],
            input_width=int(input_shape[1]),
        )
        metadata = dict(state.modality_metadata)
        planned_original = sum(
            int(action.metadata["original_count"]) for action in plan.token_actions
        )
        planned_counts = [action.metadata["keep_count"] for action in plan.token_actions]
        planned_final = None if None in planned_counts else sum(int(n) for n in planned_counts)
        compression_state = {
            "enabled": True,
            "stage_name": plan.stage_name,
            "plugin": self.execution.name,
            "placement": Placement.LLM_LEVEL.value,
            "target_site": self.execution.target_site,
            "lifecycle": "prefill_only",
            "prefill_compression": True,
            "decode_compression": False,
            "position_policy": "preserve_absolute",
            "original_audio_tokens": planned_original,
            "kept_audio_tokens": planned_final,
            "selection_deferred_to_llm_prefill": True,
        }
        metadata["audio_compression"] = compression_state
        history = dict(state.stage_history)
        history[plan.stage_name] = StageMetadata(
            stage_name=plan.stage_name,
            stats=dict(compression_state),
        )
        return replace(state, modality_metadata=metadata, stage_history=history), plan

    @contextmanager
    def request(self, plan: CompressionPlan) -> Any:
        if self._active_request is not None:
            raise InvariantError("Nested LLM-token compression requests are unsupported.")
        prepared = self._prepared_context
        self._prepared_context = None
        if prepared is None:
            raise InvariantError("prepare_state() must be called before request().")
        active = self._build_active_request(plan, prepared)
        reset_runtime_state = getattr(self.strategy, "reset_runtime_state", None)
        if callable(reset_runtime_state):
            reset_runtime_state()
        self._active_request = active
        self._last_report = None
        self._last_token_layouts = None
        try:
            for reference in self.encoder_references.values():
                reference.start(enabled=bool(active.audio_units))
            yield active
            active.score_seconds += sum(reference.score_seconds for reference in self.encoder_references.values())
            if active.audio_units and active.prefill_forward_calls != 1:
                raise InvariantError(
                    "LLM-token compression expected exactly one prefill; got "
                    f"{active.prefill_forward_calls}."
                )
            self._last_token_layouts = _semantic_token_layouts_by_layer_entry(active)
            self._last_report = {
                "stage_name": plan.stage_name,
                "plugin": self.execution.name,
                "execution_phase": plan.execution_phase.value,
                "placement": Placement.LLM_LEVEL.value,
                **self.runtime_metadata(),
                "original_audio_tokens": sum(active.original_counts),
                "kept_audio_tokens": sum(active.current_counts),
                "prefill_forward_calls": active.prefill_forward_calls,
                "decode_forward_calls": active.decode_forward_calls,
                "decode_bypassed": active.decode_forward_calls > 0,
                "selection_repeated_during_decode": False,
                "score_seconds": (
                    active.score_seconds if self.profile_enabled else None
                ),
                "compression_seconds": (
                    active.compression_seconds if self.profile_enabled else None
                ),
                "stages": list(active.stage_reports),
            }
        finally:
            for reference in self.encoder_references.values():
                reference.clear()
            # HeadRouter's released integration probes the original input
            # embeddings with layer M-1 projections.  Never retain that tensor
            # in the replay request after the discrete layout is fixed.
            active.initial_hidden_states = None
            active.score_inputs.clear()
            if bool(getattr(self.llm, "training", False)):
                # Gradient-checkpoint recomputation happens after forward/request
                # teardown.  Integer layouts and detached merge decisions may be
                # retained, but hidden activations are never kept alive.
                self._replay_request = active
            self._active_request = None

    def _build_active_request(
        self,
        plan: CompressionPlan,
        prepared: _PreparedLLMContext,
    ) -> _ActiveLLMRequest:
        torch = _torch()
        attention_mask = prepared.attention_mask
        batch_size = int(attention_mask.shape[0])
        positions = torch.arange(
            prepared.input_width,
            dtype=torch.long,
            device=attention_mask.device,
        ).unsqueeze(0).expand(batch_size, -1).clone()
        instruction = torch.zeros(
            (batch_size, prepared.input_width),
            dtype=torch.bool,
            device=attention_mask.device,
        )
        for sample_index, row in enumerate(prepared.semantic_rows):
            instruction[sample_index] = torch.tensor(
                [item == "instruction" for item in row],
                dtype=torch.bool,
                device=attention_mask.device,
            )
        if prepared.labels is not None:
            instruction &= prepared.labels.to(device=attention_mask.device).eq(-100)
        units = [
            {
                "sample_index": int(action.metadata["sample_index"]),
                "audio_unit_index": int(action.metadata["audio_unit_index"]),
                "span_start": int(action.metadata["span_start"]),
                "span_end": int(action.metadata["span_end"]),
                "original_count": int(action.metadata["original_count"]),
            }
            for action in plan.token_actions
        ]
        audio_samples = sorted({int(item["sample_index"]) for item in units})
        missing_instruction = [
            sample_index
            for sample_index in audio_samples
            if not bool(instruction[sample_index].any().item())
        ]
        if self._requires_instruction_queries and missing_instruction:
            raise InvariantError(
                "LLM-token attention scoring found no instruction query tokens "
                f"for audio samples {missing_instruction}; refusing to fall back "
                "to system, special or answer tokens."
            )
        return _ActiveLLMRequest(
            plan=plan,
            original_attention_mask=attention_mask,
            original_labels=prepared.labels,
            semantic_rows=prepared.semantic_rows,
            input_width=prepared.input_width,
            current_positions=positions,
            instruction_by_original=instruction,
            audio_units=units,
        )

    def _layer_forward(
        self,
        layer_index: int,
        original_forward: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        hidden_states, positional = _hidden_from_call(args, kwargs)
        active = self._active_request
        if active is None:
            replay = self._replay_request
            if (
                replay is None
                or not bool(getattr(self.llm, "training", False))
                or layer_index not in replay.layer_entry_positions
            ):
                return original_forward(*args, **kwargs)
            return self._replay_layer(
                replay,
                layer_index,
                original_forward,
                args,
                kwargs,
                hidden_states,
                positional,
            )

        _move_request_layout(active, hidden_states.device)
        q_length = int(hidden_states.shape[1])
        if active.prefill_complete or q_length == 1:
            if layer_index == 0:
                active.decode_forward_calls += 1
            return self._decode_layer(
                active,
                layer_index,
                original_forward,
                args,
                kwargs,
            )

        entry = active.current_positions
        if q_length != int(entry.shape[1]):
            raise InvariantError(
                f"LLM prefill layer {layer_index} hidden width {q_length} does not "
                f"match tracked layout {int(entry.shape[1])}."
            )
        if layer_index == 0:
            active.prefill_forward_calls += 1
            active.initial_hidden_states = hidden_states.detach()
        active.layer_input_positions[layer_index] = entry.detach().clone()
        self._capture_score_input(active, layer_index, hidden_states, entry)
        start_rule = self._rule_layer_by_position.get(LayerPosition("inter_layer", 0))
        if layer_index == 0 and start_rule is not None and active.audio_units:
            # At the stack entrance, native RoPE already has the original layout.
            embeddings = kwargs.get("position_embeddings")
            hidden_states, entry = self._apply_compression(
                active, start_rule, hidden_states, entry, embeddings
            )
            args, kwargs = _replace_hidden_call(args, kwargs, hidden_states, positional)
            active.current_positions = entry
        # Native attention/cache input can differ from the checkpoint call input
        # only at the stack's initial boundary.
        active.layer_entry_positions[layer_index] = entry.detach().clone()
        call_kwargs = self._prefill_kwargs(active, entry, kwargs)
        rule_layer = self._rule_layer_at_output(layer_index)
        should_compress = rule_layer is not None and bool(active.audio_units)
        if should_compress:
            def compress(hidden):
                result, exit_positions = self._apply_compression(
                    active, rule_layer, hidden, entry, call_kwargs.get("position_embeddings"),
                )
                active.current_positions = exit_positions
                return result

            output = self._run_compressed_layer(
                layer_index, original_forward, args, call_kwargs, compress,
            )
        else:
            output = original_forward(*args, **call_kwargs)
        active.layer_exit_positions[layer_index] = (
            active.current_positions.detach().clone()
        )
        if layer_index == len(self.layers) - 1:
            active.prefill_complete = True
        return output

    def _run_compressed_layer(
        self,
        layer_index: int,
        original_forward: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        compress: Any,
    ) -> Any:
        """Use the same native boundary for the first call and checkpoint replay."""

        if self._compression_boundary == "post_layer":
            output = original_forward(*args, **kwargs)
            return (compress(output[0]), *output[1:])

        layer = self.layers[layer_index]
        native_forward = getattr(layer, "_old_forward", original_forward)

        def boundary_forward(layer_self, *layer_args, **layer_kwargs):
            bound = signature(native_forward).bind(*layer_args, **layer_kwargs)
            values = dict(bound.arguments)
            values.update(values.pop("kwargs", {}))
            return decoder_post_attention_forward(
                layer_self, **values, apply_compression=compress, tuple_output=True,
            )

        if not hasattr(layer, "_hf_hook"):
            return boundary_forward(layer, *args, **kwargs)
        # Keep Accelerate's full-layer input/output and offload hooks around the
        # native value path. The outer Omni loop already owns checkpointing.
        patch = MethodPatch.apply(
            layer, "_old_forward", boundary_forward, native_forward, label="audio LLM boundary",
        )
        try:
            return original_forward(*args, **kwargs)
        finally:
            patch.restore()

    def _replay_layer(
        self,
        replay: _ActiveLLMRequest,
        layer_index: int,
        original_forward: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        hidden_states: Any,
        positional: bool,
    ) -> Any:
        input_positions = replay.layer_input_positions[layer_index]
        if int(hidden_states.shape[1]) != int(input_positions.shape[1]):
            return original_forward(*args, **kwargs)
        entry = replay.layer_entry_positions[layer_index]
        start_rule = self._rule_layer_by_position.get(LayerPosition("inter_layer", 0))
        if layer_index == 0 and start_rule is not None and replay.audio_units:
            hidden_states = self._replay_compression(
                replay, start_rule, hidden_states, input_positions
            )
            args, kwargs = _replace_hidden_call(args, kwargs, hidden_states, positional)
        call_kwargs = self._prefill_kwargs(replay, entry, kwargs)
        rule_layer = self._rule_layer_at_output(layer_index)
        should_compress = rule_layer is not None and bool(replay.audio_units)
        if should_compress:
            return self._run_compressed_layer(
                layer_index, original_forward, args, call_kwargs,
                lambda hidden: self._replay_compression(replay, rule_layer, hidden, entry),
            )
        return original_forward(*args, **call_kwargs)

    def _rule_layer_at_output(self, layer_index: int) -> int | None:
        position = (
            LayerPosition("inter_layer", layer_index + 1)
            if self._compression_boundary == "post_layer"
            else LayerPosition("post_attention_pre_ffn", layer_index)
        )
        return self._rule_layer_by_position.get(position)

    def _capture_score_input(
        self, active: _ActiveLLMRequest, layer_index: int, hidden: Any, positions: Any
    ) -> None:
        if (not self._requires_layer_input or not active.audio_units or layer_index not in self.stage_layers
                or layer_index in active.compression_positions
                or layer_index in active.score_inputs):
            return
        with _torch().no_grad():
            layer = self.layers[layer_index]
            normalized = layer.input_layernorm(hidden.to(_module_execution_device(layer.input_layernorm)))
        active.score_inputs[layer_index] = (normalized, positions.detach().clone())

    def _apply_compression(
        self, active: _ActiveLLMRequest, rule_layer: int,
        hidden: Any, positions: Any, embeddings: Any,
    ) -> tuple[Any, Any]:
        _refresh_strategy_module(self)
        positions = positions.to(hidden.device)
        _move_request_layout(active, hidden.device)
        # At boundary k the input to layer k already exists even though its
        # native forward has not started. This is the before/after alias case.
        normalized = None
        if self._requires_layer_input:
            if rule_layer not in active.score_inputs:
                self._capture_score_input(active, rule_layer, hidden, positions)
            normalized, source_positions = active.score_inputs.pop(rule_layer)
            normalized = _gather_by_absolute_layout(
                normalized, source_positions.to(normalized.device), positions.to(normalized.device)
            )
        result, compact_positions = self._compress_boundary(
            active, rule_layer, hidden, normalized, embeddings, positions
        )
        active.compression_positions[rule_layer] = compact_positions.detach().clone()
        return result, compact_positions

    def _replay_compression(
        self, replay: _ActiveLLMRequest, rule_layer: int, hidden: Any, positions: Any
    ) -> Any:
        positions = positions.to(hidden.device)
        if callable(self._merge_audio_tokens):
            return self._replay_merged_hidden(replay, rule_layer, hidden, positions)
        return _gather_by_absolute_layout(
            hidden, positions, replay.compression_positions[rule_layer].to(hidden.device)
        )

    def _position_report(self, rule_layer: int) -> dict[str, Any]:
        position = self._position_by_rule_layer[rule_layer]
        return {
            "layer_index_zero_based": (
                position.index - 1 if position.boundary == "inter_layer" and position.index
                else position.index if position.boundary != "inter_layer" else None
            ),
            "application_position": position.as_dict(),
            "score_layer_index_zero_based": rule_layer if self._requires_layer_input else None,
            **({"encoder_score_layer_zero_based": rule_layer} if self._encoder_rule else {}),
            **({"boundary_one_based": position.index} if position.index and
               position.boundary == "inter_layer" else {}),
        }

    def _replay_merged_hidden(
        self,
        replay: _ActiveLLMRequest,
        layer_index: int,
        output_hidden: Any,
        entry_positions: Any,
    ) -> Any:
        replay_units = replay.merge_replay_units.get(layer_index)
        if replay_units is None or not callable(self._replay_merge_audio_tokens):
            raise InvariantError(
                "Gradient-checkpoint replay requires a saved merge "
                "plan and replay_merge_audio_tokens()."
            )

        valid_mask = _valid_layout_mask(replay, entry_positions)
        replacements: dict[int, list[dict[str, Any]]] = {}
        for unit in replay_units:
            slots = _torch().tensor(
                unit.slots,
                dtype=_torch().long,
                device=output_hidden.device,
            )
            merged = self._replay_merge_audio_tokens(
                hidden_states=output_hidden[unit.sample_index, slots],
                replay_plan=unit.replay_plan,
            )
            if int(merged.shape[0]) != len(unit.output_positions):
                raise InvariantError(
                    "Post-attention merge replay produced a different token count."
                )
            replacements.setdefault(unit.sample_index, []).append(
                {
                    "slots": slots,
                    "hidden_states": merged,
                    "positions": entry_positions.new_tensor(unit.output_positions),
                }
            )

        compact_hidden, replayed_positions = _replace_audio_units(
            output_hidden,
            entry_positions,
            valid_mask,
            replacements,
        )
        expected_positions = replay.compression_positions[layer_index].to(
            device=replayed_positions.device
        )
        if not _torch().equal(replayed_positions, expected_positions):
            raise InvariantError(
                "Post-attention merge replay changed the saved absolute layout."
            )

        return compact_hidden

    def _prefill_kwargs(
        self,
        active: _ActiveLLMRequest,
        positions: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        call_kwargs = dict(kwargs)
        _validate_supported_cache(call_kwargs.get("past_key_values"))
        position_embeddings = call_kwargs.get("position_embeddings")
        compact_embeddings = _gather_sequence_value(
            position_embeddings,
            positions,
            original_width=active.input_width,
            pad_value=0,
        )
        if position_embeddings is not None:
            call_kwargs["position_embeddings"] = compact_embeddings
        position_ids = call_kwargs.get("position_ids")
        if position_ids is not None:
            call_kwargs["position_ids"] = _gather_sequence_value(
                position_ids,
                positions,
                original_width=active.input_width,
                pad_value=0,
            )
        attention_mask = call_kwargs.get("attention_mask")
        if attention_mask is not None:
            call_kwargs["attention_mask"] = _compact_prefill_attention_mask(
                attention_mask,
                positions,
            )
        cache_position = call_kwargs.get("cache_position")
        if cache_position is not None and int(cache_position.numel()) != int(
            positions.shape[1]
        ):
            call_kwargs["cache_position"] = _torch().arange(
                int(positions.shape[1]),
                device=cache_position.device,
                dtype=cache_position.dtype,
            )
        return call_kwargs

    def _decode_layer(
        self,
        active: _ActiveLLMRequest,
        layer_index: int,
        original_forward: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        call_kwargs = dict(kwargs)
        _validate_supported_cache(call_kwargs.get("past_key_values"))
        prompt_positions = active.layer_entry_positions.get(layer_index)
        attention_mask = call_kwargs.get("attention_mask")
        if prompt_positions is not None and attention_mask is not None:
            call_kwargs["attention_mask"] = _compact_decode_attention_mask(
                attention_mask,
                prompt_positions,
                original_width=active.input_width,
            )
        return original_forward(*args, **call_kwargs)

    def _compress_boundary(
        self,
        active: _ActiveLLMRequest,
        layer_index: int,
        output_hidden: Any,
        normalized_hidden: Any,
        position_embeddings: Any | None,
        positions: Any,
    ) -> tuple[Any, Any]:
        torch = _torch()
        valid_mask = _valid_layout_mask(active, positions)
        instruction_mask = _layout_mask_from_original(
            active.instruction_by_original,
            positions,
        )
        audio_mask = torch.zeros_like(valid_mask)
        for unit in active.audio_units:
            sample = int(unit["sample_index"])
            audio_mask[sample] |= (
                (positions[sample] >= int(unit["span_start"]))
                & (positions[sample] < int(unit["span_end"]))
            )
        scores = None
        reference = self.encoder_references.get(layer_index, self.encoder_references.get(None))
        if reference is not None:
            scores = reference.scores_for(active, positions).to(output_hidden.device)
        if self._requires_layer_input:
            score_started = perf_counter() if self.profile_enabled else None
            score_arguments = {
                "attention_module": self.layers[layer_index].self_attn,
                "normalized_hidden_states": normalized_hidden,
                "position_embeddings": position_embeddings,
                "absolute_positions": positions,
                "instruction_mask": instruction_mask,
                "valid_mask": valid_mask,
                "audio_mask": audio_mask,
            }
            if self._score_accepts_initial_hidden_states:
                score_arguments["initial_hidden_states"] = _gather_by_absolute_layout(
                    active.initial_hidden_states.to(output_hidden.device),
                    active.layer_input_positions[0].to(output_hidden.device), positions,
                )
            if self._score_accepts_layer_index:
                score_arguments["layer_index"] = layer_index
            score_device = _module_execution_device(self.layers[layer_index].self_attn.q_proj)
            score_arguments = {name: _score_value_to_device(value, score_device)
                               for name, value in score_arguments.items()}
            score_context = nullcontext() if self._score_with_grad else torch.no_grad()
            with score_context:
                scores = self.strategy.score_audio_tokens(**score_arguments)
            scores = _score_value_to_device(scores, output_hidden.device)
            if score_started is not None:
                active.score_seconds += perf_counter() - score_started

        compression_started = perf_counter() if self.profile_enabled else None
        if callable(self._merge_audio_tokens):
            compact_hidden, exit_positions = self._merge_boundary(
                active,
                layer_index,
                output_hidden,
                positions,
                valid_mask,
                scores,
            )
            if compression_started is not None:
                active.compression_seconds += perf_counter() - compression_started
            return compact_hidden, exit_positions

        keep_masks = valid_mask & ~audio_mask
        unit_reports: list[dict[str, Any]] = []
        for unit in active.audio_units:
            sample = int(unit["sample_index"])
            slots = torch.nonzero(
                (
                    (positions[sample] >= int(unit["span_start"]))
                    & (positions[sample] < int(unit["span_end"]))
                    & valid_mask[sample]
                ),
                as_tuple=False,
            ).flatten()
            current_count = int(slots.numel())
            if current_count == 0:
                continue
            keep_count = self._strategy_output_length(current_count, layer_index)
            relative = self.strategy.select_indices(scores[sample, slots], keep_count)
            selected_slots = slots.index_select(0, relative.to(device=slots.device))
            keep_masks[sample, selected_slots] = True
            kept_absolute = positions[sample, selected_slots]
            unit_reports.append(
                {
                    "sample_index": sample,
                    "audio_unit_index": int(unit["audio_unit_index"]),
                    "input_tokens": current_count,
                    "output_tokens": keep_count,
                    "kept_absolute_sequence_positions": kept_absolute.detach(),
                    "indices_chronological": (
                        True
                        if int(kept_absolute.numel()) < 2
                        else (kept_absolute[1:] >= kept_absolute[:-1]).all().detach()
                    ),
                    "query_source": "instruction",
                    "answer_query_leakage": False,
                }
            )
        compact_hidden, exit_positions = _compact_hidden_and_positions(
            output_hidden,
            positions,
            keep_masks,
        )
        if compression_started is not None:
            active.compression_seconds += perf_counter() - compression_started
        active.stage_reports.append(
            {
                **self._position_report(layer_index),
                "input_sequence_width": int(positions.shape[1]),
                "output_sequence_width": int(exit_positions.shape[1]),
                "input_audio_tokens": sum(item["input_tokens"] for item in unit_reports),
                "output_audio_tokens": sum(item["output_tokens"] for item in unit_reports),
                "position_policy": "preserve_absolute",
                "units": unit_reports if self.debug_enabled else None,
            }
        )
        return compact_hidden, exit_positions

    def _strategy_output_length(
        self, current_count: int, layer_index: int
    ) -> int | None:
        if self._output_length_accepts_layer_index:
            count = self.strategy.output_length(current_count, layer_index=layer_index)
        else:
            count = self.strategy.output_length(current_count)
        return None if count is None else int(count)

    def _merge_boundary(
        self,
        active: _ActiveLLMRequest,
        layer_index: int,
        output_hidden: Any,
        positions: Any,
        valid_mask: Any,
        scores: Any,
    ) -> tuple[Any, Any]:
        torch = _torch()
        replacements: dict[int, list[dict[str, Any]]] = {}
        unit_reports: list[dict[str, Any]] = []
        save_replay_plan = bool(getattr(self.llm, "training", False))
        replay_units: list[_MergeReplayUnit] = []
        for unit in active.audio_units:
            sample = int(unit["sample_index"])
            slots = torch.nonzero(
                (
                    (positions[sample] >= int(unit["span_start"]))
                    & (positions[sample] < int(unit["span_end"]))
                    & valid_mask[sample]
                ),
                as_tuple=False,
            ).flatten()
            current_count = int(slots.numel())
            if current_count == 0:
                continue
            output_count = self._strategy_output_length(
                current_count, layer_index
            )
            merged, representatives, details = self._merge_audio_tokens(
                hidden_states=output_hidden[sample, slots],
                scores=scores,
                sample_index=sample,
                audio_unit_index=int(unit["audio_unit_index"]),
                slots=slots,
                output_length=output_count,
                **({"layer_index": layer_index} if self._merge_accepts_layer_index else {}),
            )
            if output_count is None:
                output_count = int(merged.shape[0])
            details = dict(details)
            checkpoint_replay_plan = details.pop(
                "_checkpoint_replay_plan", None
            )
            representatives = representatives.to(
                device=slots.device, dtype=torch.long
            )
            kept_absolute = positions[sample, slots.index_select(0, representatives)]
            for reference in self.encoder_references.values():
                reference.map_llm_keys(active, positions, sample=sample, slots=slots,
                                       kept_absolute=kept_absolute, replay=self._replay_merge_audio_tokens,
                                       replay_plan=checkpoint_replay_plan)
            if save_replay_plan:
                if (
                    checkpoint_replay_plan is None
                    or not callable(self._replay_merge_audio_tokens)
                ):
                    raise InvariantError(
                        "Training requires a detached merge replay plan."
                    )
                replay_units.append(
                    _MergeReplayUnit(
                        sample_index=sample,
                        slots=tuple(
                            int(item)
                            for item in slots.detach().cpu().tolist()
                        ),
                        output_positions=tuple(
                            int(item)
                            for item in kept_absolute.detach().cpu().tolist()
                        ),
                        replay_plan=checkpoint_replay_plan,
                    )
                )
            replacements.setdefault(sample, []).append(
                {
                    "slots": slots,
                    "hidden_states": merged,
                    "positions": kept_absolute,
                }
            )
            unit_reports.append(
                {
                    "sample_index": sample,
                    "audio_unit_index": int(unit["audio_unit_index"]),
                    "input_tokens": current_count,
                    "output_tokens": output_count,
                    "kept_absolute_sequence_positions": kept_absolute.detach(),
                    "indices_chronological": True,
                    "query_source": getattr(self.strategy, "query_source", "all_prompt"),
                    "answer_query_leakage": getattr(self.strategy, "answer_query_leakage", "paper_defined_full_prompt"),
                    **details,
                }
            )

        if save_replay_plan:
            active.merge_replay_units[layer_index] = tuple(replay_units)
        compact_hidden, exit_positions = _replace_audio_units(
            output_hidden,
            positions,
            valid_mask,
            replacements,
        )
        active.stage_reports.append(
            {
                **self._position_report(layer_index),
                "compression_boundary": self._compression_boundary,
                "operation": getattr(self.strategy, "token_operation", "merge"),
                "input_sequence_width": int(positions.shape[1]),
                "output_sequence_width": int(exit_positions.shape[1]),
                "input_audio_tokens": sum(
                    item["input_tokens"] for item in unit_reports
                ),
                "output_audio_tokens": sum(
                    item["output_tokens"] for item in unit_reports
                ),
                "position_policy": "preserve_absolute" if getattr(self.strategy, "token_operation", None) == "keep" else getattr(self.strategy, "merge_position_policy", "preserve_absolute_cluster_start"),
                "units": unit_reports if self.debug_enabled else None,
            }
        )
        return compact_hidden, exit_positions

    def _thinker_forward(
        self,
        thinker: Any,
        original_forward: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active = self._active_request
        if active is None:
            return original_forward(*args, **kwargs)
        bound = signature(original_forward).bind_partial(*args, **kwargs)
        cache = bound.arguments.get("past_key_values")
        if active.prefill_complete and cache is not None and bound.arguments.get("cache_position") is None:
            # Layer zero's physical KV length can now be shorter than the
            # absolute prompt width. Restore the omitted public coordinate
            # before Thinker constructs position_ids and M-RoPE embeddings.
            prompt = active.layer_entry_positions.get(0)
            inputs = bound.arguments.get("input_ids")
            if inputs is None:
                inputs = bound.arguments.get("inputs_embeds")
            if prompt is not None and inputs is not None:
                start = active.input_width + int(cache.get_seq_length(0)) - int(prompt.shape[1])
                bound.arguments["cache_position"] = _torch().arange(
                    start, start + int(inputs.shape[1]), device=inputs.device,
                )
        labels = bound.arguments.get("labels")
        if labels is None:
            # ms-swift may consume labels in an outer PEFT/template wrapper before
            # the patched Thinker is called.  The request owns the original labels,
            # so use that stable copy when computing loss for the compact layout.
            labels = active.original_labels
        if labels is None:
            return original_forward(*bound.args, **bound.kwargs)
        bound.arguments["labels"] = None
        output = original_forward(*bound.args, **bound.kwargs)
        return self.attach_compact_training_loss(output, labels=labels)

    def compact_training_labels(self, labels: Any | None = None) -> Any:
        """Return labels in the request's final layout for loss and metrics."""
        active = self._active_request or self._replay_request
        if active is None:
            raise InvariantError(
                "Compact LLM training loss requires the active or most recently "
                "completed compression request."
            )
        labels = active.original_labels if labels is None else labels
        if labels is None:
            raise InvariantError(
                "Compact LLM training loss requires the original training labels."
            )
        return _gather_labels(labels, active.current_positions)

    def attach_compact_training_loss(
        self,
        output: Any,
        *,
        labels: Any | None = None,
    ) -> Any:
        """Attach loss after a provider forward using the request's final layout."""
        compact_labels = self.compact_training_labels(labels)
        logits = output[0] if isinstance(output, tuple) else output.logits
        loss = self.thinker.loss_function(
            logits=logits,
            labels=compact_labels,
            vocab_size=self.thinker.config.get_text_config().vocab_size,
        )
        if isinstance(output, tuple):
            return (loss, *output)
        if isinstance(output, dict):
            # Transformers ModelOutput omits fields initialized as None from its
            # mapping.  Swift consumes the mapping interface, so assigning only
            # output.loss would still leave "loss" absent from output.keys().
            output["loss"] = loss
        else:
            output.loss = loss
        return output


def _hidden_from_call(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[Any, bool]:
    if args:
        return args[0], True
    hidden = kwargs.get("hidden_states")
    if hidden is None:
        raise InvariantError("LLM layer call omitted hidden_states.")
    return hidden, False


def _replace_hidden_call(args, kwargs, hidden, positional):
    if positional:
        return (hidden, *args[1:]), kwargs
    return args, {**kwargs, "hidden_states": hidden}


def _score_value_to_device(value: Any, device: Any) -> Any:
    """Move Q/K context, RoPE tuples and FastAdaSP's tensor score record."""

    if _torch().is_tensor(value):
        return value.to(device)
    if isinstance(value, (tuple, list)):
        return type(value)(_score_value_to_device(item, device) for item in value)
    if is_dataclass(value):
        return replace(value, **{
            item.name: _score_value_to_device(getattr(value, item.name), device)
            for item in fields(value)
        })
    return value


def _move_request_layout(active: _ActiveLLMRequest, device: Any) -> None:
    if active.current_positions.device == device:
        return
    active.current_positions = active.current_positions.to(device=device)
    active.original_attention_mask = active.original_attention_mask.to(device=device)
    active.instruction_by_original = active.instruction_by_original.to(device=device)


def _validate_supported_cache(cache: Any | None) -> None:
    if cache is None:
        return
    if type(cache).__name__ != "DynamicCache":
        raise InvariantError(
            "LLM-token compression currently supports the standard DynamicCache "
            f"only, got {type(cache).__qualname__}."
        )


def _valid_layout_mask(active: _ActiveLLMRequest, positions: Any) -> Any:
    return _layout_mask_from_original(
        active.original_attention_mask.to(device=positions.device).bool(),
        positions,
    ) & positions.ge(0)


def _layout_mask_from_original(original: Any, positions: Any) -> Any:
    torch = _torch()
    result = torch.zeros_like(positions, dtype=torch.bool)
    for batch_index in range(int(positions.shape[0])):
        valid = positions[batch_index].ge(0)
        if bool(valid.any().item()):
            result[batch_index, valid] = original[
                batch_index,
                positions[batch_index, valid].to(device=original.device),
            ].to(device=positions.device, dtype=torch.bool)
    return result


def _replace_audio_units(
    hidden_states: Any,
    positions: Any,
    valid_mask: Any,
    replacements: Mapping[int, list[dict[str, Any]]],
) -> tuple[Any, Any]:
    """Replace disjoint audio spans with differentiable merged sequences."""

    torch = _torch()
    hidden_rows: list[Any] = []
    position_rows: list[Any] = []
    for sample_index in range(int(hidden_states.shape[0])):
        valid_slots = torch.nonzero(
            valid_mask[sample_index], as_tuple=False
        ).flatten()
        hidden_parts: list[Any] = []
        position_parts: list[Any] = []
        cursor = 0
        ordered = sorted(
            (item for item in replacements.get(sample_index, []) if item["slots"].numel()),
            key=lambda item: int(item["slots"][0].item()),
        )
        for replacement in ordered:
            slots = replacement["slots"].to(
                device=valid_slots.device, dtype=torch.long
            )
            start = int(torch.searchsorted(valid_slots, slots[0]).item())
            end = start + int(slots.numel())
            if start < cursor or not torch.equal(slots, valid_slots[start:end]):
                raise InvariantError(
                    "LLM-token merge requires non-overlapping contiguous audio "
                    "units in the valid sequence layout."
                )
            before = valid_slots[cursor:start]
            hidden_parts.append(hidden_states[sample_index].index_select(0, before))
            position_parts.append(positions[sample_index].index_select(0, before))
            hidden_parts.append(replacement["hidden_states"])
            position_parts.append(replacement["positions"])
            cursor = end

        remaining = valid_slots[cursor:]
        hidden_parts.append(hidden_states[sample_index].index_select(0, remaining))
        position_parts.append(positions[sample_index].index_select(0, remaining))
        hidden_rows.append(torch.cat(hidden_parts, dim=0))
        position_rows.append(torch.cat(position_parts, dim=0))

    return (
        torch.nn.utils.rnn.pad_sequence(hidden_rows, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(position_rows, batch_first=True, padding_value=-1),
    )


def _compact_hidden_and_positions(
    hidden_states: Any,
    positions: Any,
    keep_masks: Any,
) -> tuple[Any, Any]:
    rows: list[Any] = []
    position_rows: list[Any] = []
    for batch_index in range(int(hidden_states.shape[0])):
        slots = _torch().nonzero(keep_masks[batch_index], as_tuple=False).flatten()
        rows.append(hidden_states[batch_index].index_select(0, slots))
        position_rows.append(positions[batch_index].index_select(0, slots))
    return (
        _torch().nn.utils.rnn.pad_sequence(rows, batch_first=True),
        _torch().nn.utils.rnn.pad_sequence(position_rows, batch_first=True, padding_value=-1),
    )


def _semantic_token_layouts_by_layer_entry(
    active: _ActiveLLMRequest,
) -> dict[int, list[list[str]]]:
    result: dict[int, list[list[str]]] = {}
    for layer_index, positions in active.layer_entry_positions.items():
        position_rows = positions.detach().cpu().tolist()
        rows: list[list[str]] = []
        for sample_index, position_row in enumerate(position_rows):
            source = active.semantic_rows[sample_index]
            labels: list[str] = []
            for raw_position in position_row:
                position = int(raw_position)
                if position < 0:
                    labels.append("pad")
                elif position >= len(source):
                    raise InvariantError(
                        "LLM-token semantic layout contains an out-of-range "
                        "absolute position."
                    )
                else:
                    labels.append(str(source[position]))
            rows.append(labels)
        result[int(layer_index)] = rows
    return result


def _gather_by_absolute_layout(
    hidden_states: Any,
    entry_positions: Any,
    exit_positions: Any,
) -> Any:
    torch = _torch()
    keep_masks = torch.zeros_like(entry_positions, dtype=torch.bool)
    for batch_index in range(int(entry_positions.shape[0])):
        entry = entry_positions[batch_index]
        exit_row = exit_positions[batch_index]
        kept = exit_row[exit_row.ge(0)]
        keep_masks[batch_index] = entry.ge(0) & (
            entry.unsqueeze(1).eq(kept.unsqueeze(0)).any(dim=1)
        )
    compact_hidden, replayed_positions = _compact_hidden_and_positions(
        hidden_states,
        entry_positions,
        keep_masks,
    )
    if not torch.equal(replayed_positions, exit_positions):
        raise InvariantError(
            "Gradient-checkpoint replay did not reproduce the saved token layout."
        )
    return compact_hidden


def _gather_sequence_value(
    value: Any,
    positions: Any,
    *,
    original_width: int,
    pad_value: int | float,
) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(
            _gather_sequence_value(
                item,
                positions,
                original_width=original_width,
                pad_value=pad_value,
            )
            for item in value
        )
    shape = getattr(value, "shape", None)
    if shape is None:
        return value
    batch_size, width = int(positions.shape[0]), int(positions.shape[1])
    if len(shape) >= 2 and int(shape[0]) == batch_size and int(shape[1]) == original_width:
        output = value.new_full((batch_size, width, *shape[2:]), pad_value)
        for batch_index in range(batch_size):
            valid = positions[batch_index].ge(0)
            if bool(valid.any().item()):
                output[batch_index, valid] = value[
                    batch_index,
                    positions[batch_index, valid].to(device=value.device),
                ]
        return output
    if len(shape) >= 3 and int(shape[1]) == batch_size and int(shape[2]) == original_width:
        output = value.new_full((shape[0], batch_size, width, *shape[3:]), pad_value)
        for batch_index in range(batch_size):
            valid = positions[batch_index].ge(0)
            if bool(valid.any().item()):
                output[:, batch_index, valid] = value[
                    :,
                    batch_index,
                    positions[batch_index, valid].to(device=value.device),
                ]
        return output
    return value


def _compact_prefill_attention_mask(mask: Any, positions: Any) -> Any:
    if mask.ndim == 2:
        return _gather_sequence_value(
            mask,
            positions,
            original_width=int(mask.shape[-1]),
            pad_value=0,
        )
    if mask.ndim != 4:
        raise InvariantError(
            f"Unsupported Qwen prefill attention mask rank {mask.ndim}."
        )
    batch_size, heads, _, _ = mask.shape
    width = int(positions.shape[1])
    minimum = _mask_minimum(mask)
    output = mask.new_full((batch_size, heads, width, width), minimum)
    for batch_index in range(batch_size):
        valid_slots = _torch().nonzero(
            positions[batch_index].ge(0), as_tuple=False
        ).flatten()
        if int(valid_slots.numel()) == 0:
            continue
        original = positions[batch_index, valid_slots].to(device=mask.device)
        selected = mask[batch_index].index_select(1, original).index_select(2, original)
        output[batch_index, :, : int(valid_slots.numel()), : int(valid_slots.numel())] = selected
    return output


def _compact_decode_attention_mask(
    mask: Any,
    prompt_positions: Any,
    *,
    original_width: int,
) -> Any:
    if mask.ndim == 2:
        # FlashAttention accepts a 2D padding mask. Keep every physical compact
        # prompt slot, masking synthetic padding, then append decode columns.
        batch_size = int(mask.shape[0])
        decode_columns = int(mask.shape[1]) - original_width
        width = int(prompt_positions.shape[1]) + max(0, decode_columns)
        output = mask.new_zeros((batch_size, width))
        for batch_index in range(batch_size):
            for slot, position in enumerate(prompt_positions[batch_index].tolist()):
                if int(position) >= 0:
                    output[batch_index, slot] = mask[batch_index, int(position)]
            if decode_columns > 0:
                output[batch_index, int(prompt_positions.shape[1]) :] = mask[
                    batch_index, original_width:
                ]
        return output
    if mask.ndim != 4:
        raise InvariantError(
            f"Unsupported Qwen decode attention mask rank {mask.ndim}."
        )
    batch_size, heads, query_length, global_key_length = mask.shape
    decode_columns = int(global_key_length) - original_width
    physical_key_length = int(prompt_positions.shape[1]) + max(0, decode_columns)
    output = mask.new_full(
        (batch_size, heads, query_length, physical_key_length),
        _mask_minimum(mask),
    )
    for batch_index in range(batch_size):
        for slot, position in enumerate(prompt_positions[batch_index].tolist()):
            if int(position) >= 0:
                output[batch_index, :, :, slot] = mask[
                    batch_index, :, :, int(position)
                ]
        if decode_columns > 0:
            output[batch_index, :, :, int(prompt_positions.shape[1]) :] = mask[
                batch_index, :, :, original_width:
            ]
    return output


def _mask_minimum(mask: Any) -> int | float:
    if bool(getattr(mask.dtype, "is_floating_point", False)):
        return float(_torch().finfo(mask.dtype).min)
    return 0


def _gather_labels(labels: Any, positions: Any) -> Any:
    batch_size, width = int(positions.shape[0]), int(positions.shape[1])
    output = labels.new_full((batch_size, width), -100)
    for batch_index in range(batch_size):
        valid = positions[batch_index].ge(0)
        if bool(valid.any().item()):
            output[batch_index, valid] = labels[
                batch_index,
                positions[batch_index, valid].to(device=labels.device),
            ]
    return output


__all__ = ["Qwen25OmniLLMTokenCompressionController"]
