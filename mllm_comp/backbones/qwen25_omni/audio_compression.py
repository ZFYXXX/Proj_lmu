"""Qwen2.5-Omni runtime support for pre-projector audio-token pruning.

This adapter is deliberately version-specific to the verified Transformers
4.57.3 implementation. It keeps the upstream audio encoder intact, captures
the final encoder attention inputs, prunes ``ln_post`` outputs immediately
before ``audio_tower.proj``, and compacts the matching LLM placeholders.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from time import perf_counter
from types import MethodType
from typing import Any

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.core.errors import InvariantError
from mllm_comp.core.metadata import StageMetadata
from mllm_comp.core.plan import CompressionPlan
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.core.vocabulary import ExecutionPhase, Placement

_VERIFIED_TRANSFORMERS_VERSION = "4.57.3"
_AUDIO_TOKEN_TARGET_SITES = frozenset(
    {
        "audio_encoder_output_pre_projector",
        "audio_projector_output_post_projector",
        "audio_encoder_layer_post_mhsa_pre_ffn",
        "audio_encoder_layer_output_post_block",
    }
)

@dataclass
class _ActiveAudioRequest:
    original_counts: list[int]
    keep_counts: list[int]
    sample_indices: list[int]
    audio_unit_indices: list[int]
    preflight_details: list[dict[str, Any] | None] = field(default_factory=list)
    aftercnn_lengths: list[int] = field(default_factory=list)
    encoder_output_lengths: list[int] = field(default_factory=list)
    projector_call_index: int = 0
    attention_scores: Any | None = None
    sa_map_merge_importance: Any | None = None
    sa_map_prune_importance: Any | None = None
    attention_source: Any | None = None
    attention_source_chunks: list[int] = field(default_factory=list)
    attention_needs_refresh: bool = False
    attention_score_seconds: float = 0.0
    item_reports: list[dict[str, Any]] = field(default_factory=list)

    @property
    def identity(self) -> bool:
        return self.original_counts == self.keep_counts


def _compressed_audio_rope_index(
    original: Any, keep_counts: Sequence[int], *args: Any, **kwargs: Any
) -> Any:
    """Request compact audio positions from the pinned provider's length formula."""

    args_list = list(args)
    audio_arg_index = 5
    audio_seqlens = (
        args_list[audio_arg_index]
        if len(args_list) > audio_arg_index
        else kwargs.get("audio_seqlens")
    )
    if audio_seqlens is not None:
        # A raw feature length of 4*K maps to exactly K LLM audio positions.
        # The encoder's real feature_attention_mask remains untouched.
        compact_raw_lengths = audio_seqlens.new_tensor([4 * count for count in keep_counts])
        # A downstream controller may already have compacted these lengths.
        # Earlier encoder/projector budgets must not expand them again.
        compact_raw_lengths = _torch().minimum(audio_seqlens, compact_raw_lengths)
        if len(args_list) > audio_arg_index:
            args_list[audio_arg_index] = compact_raw_lengths
        else:
            kwargs["audio_seqlens"] = compact_raw_lengths
    return original(*args_list, **kwargs)


def audio_placeholder_keep_positions(
    *,
    sequence_length: int,
    batch_size: int,
    token_actions: Iterable[TokenAction],
) -> list[list[int]]:
    """Return per-sample sequence positions kept by deferred audio actions."""

    if sequence_length < 0 or batch_size < 1:
        raise ValueError("sequence_length must be non-negative and batch_size positive")
    removed_by_sample: list[set[int]] = [set() for _ in range(batch_size)]
    for action in token_actions:
        metadata = action.metadata
        sample_index = int(metadata["sample_index"])
        if not 0 <= sample_index < batch_size:
            raise InvariantError(
                f"audio token action sample_index {sample_index} is outside batch size {batch_size}"
            )
        start = int(metadata["span_start"])
        end = int(metadata["span_end"])
        keep_count = int(metadata["keep_count"])
        if not 0 <= start <= end <= sequence_length:
            raise InvariantError(
                f"audio token action span [{start}, {end}) is outside sequence length {sequence_length}"
            )
        if not 0 <= keep_count <= end - start:
            raise InvariantError(
                f"audio keep_count {keep_count} is invalid for span length {end - start}"
            )
        removed_by_sample[sample_index].update(range(start + keep_count, end))
    return [
        [index for index in range(sequence_length) if index not in removed]
        for removed in removed_by_sample
    ]


def legalize_qwen25_omni_audio_tokens(
    state: ForwardState,
    *,
    token_actions: list[TokenAction],
    policy: LegalizationPolicy,
    stage_name: str,
    expected_target_site: str | None = None,
) -> ForwardState:
    """Compact Qwen audio placeholders while preserving full encoder inputs."""

    if not token_actions:
        return replace(state)
    if policy.position_policy != "compact":
        raise InvariantError(
            "Qwen2.5-Omni audio pruning currently requires position_policy='compact'."
        )
    allowed_target_sites = (
        {expected_target_site}
        if expected_target_site is not None
        else _AUDIO_TOKEN_TARGET_SITES
    )
    if any(
        action.metadata.get("target_site") not in allowed_target_sites
        for action in token_actions
    ):
        raise InvariantError(
            "Qwen2.5-Omni legalizer received an unsupported token action."
        )

    input_ids = state.input_ids
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) != 2:
        raise InvariantError(
            "Qwen2.5-Omni audio compression requires rank-2 input_ids."
        )
    batch_size, sequence_length = int(shape[0]), int(shape[1])
    keep_positions = audio_placeholder_keep_positions(
        sequence_length=sequence_length,
        batch_size=batch_size,
        token_actions=token_actions,
    )
    metadata = dict(state.modality_metadata)
    audio_token_id = metadata.get("token_ids", {}).get("audio")
    if audio_token_id is None:
        raise InvariantError(
            "Qwen2.5-Omni audio compression requires the processor audio token id."
        )
    for action in token_actions:
        sample_index = int(action.metadata["sample_index"])
        start = int(action.metadata["span_start"])
        end = int(action.metadata["span_end"])
        values = input_ids[sample_index, start:end].tolist()
        if any(int(value) != int(audio_token_id) for value in values):
            raise InvariantError(
                "The audio TokenAction span does not contain only audio placeholders."
            )

    padding_side = str(metadata.get("padding_side") or "right")
    if padding_side not in {"left", "right"}:
        raise InvariantError(f"Unsupported tokenizer padding_side {padding_side!r}.")
    pad_token_id = int(metadata.get("pad_token_id") or 0)
    compact_input_ids = _compact_batch_first_tensor(
        input_ids,
        keep_positions,
        pad_value=pad_token_id,
        padding_side=padding_side,
    )
    compact_attention_mask = _compact_batch_first_tensor(
        state.attention_mask,
        keep_positions,
        pad_value=0,
        padding_side=padding_side,
    )
    compact_inputs_embeds = _compact_batch_first_tensor(
        state.inputs_embeds,
        keep_positions,
        pad_value=0,
        padding_side=padding_side,
    )
    compact_labels = _compact_batch_first_tensor(
        state.labels,
        keep_positions,
        pad_value=-100,
        padding_side=padding_side,
    )
    compact_token_type_mask = _compact_batch_first_tensor(
        state.token_type_mask,
        keep_positions,
        pad_value=0,
        padding_side=padding_side,
    )

    removed_by_sample = [
        sorted(set(range(sequence_length)) - set(positions))
        for positions in keep_positions
    ]
    compact_width = max(len(positions) for positions in keep_positions)
    added_prefix_by_sample = [
        compact_width - len(positions) if padding_side == "left" else 0
        for positions in keep_positions
    ]
    semantic_rows = metadata.get("semantic_token_labels_by_sample")
    if semantic_rows is not None:
        if not isinstance(semantic_rows, list) or len(semantic_rows) != batch_size:
            raise InvariantError(
                "Qwen semantic token labels do not match the compression batch."
            )
        compact_semantic_rows: list[list[str]] = []
        for sample_index, positions in enumerate(keep_positions):
            row = semantic_rows[sample_index]
            if not isinstance(row, list) or len(row) != sequence_length:
                raise InvariantError(
                    "Qwen semantic token labels do not match input sequence width."
                )
            selected = [str(row[position]) for position in positions]
            padding = ["pad"] * (compact_width - len(selected))
            compact_semantic_rows.append(
                padding + selected if padding_side == "left" else selected + padding
            )
        metadata["semantic_token_labels_by_sample"] = compact_semantic_rows
    compact_spans = _shift_token_spans(
        state.modality_token_spans,
        removed_by_sample=removed_by_sample,
        added_prefix_by_sample=added_prefix_by_sample,
    )
    expected_counts = [
        dict(item)
        for item in metadata.get("expected_token_counts_by_sample", [])
    ]
    while len(expected_counts) < batch_size:
        expected_counts.append({})
    audio_counts_by_sample = [0] * batch_size
    for action in token_actions:
        audio_counts_by_sample[int(action.metadata["sample_index"])] += int(
            action.metadata["keep_count"]
        )
    for sample_index, count in enumerate(audio_counts_by_sample):
        expected_counts[sample_index]["audio"] = count

    original_lengths = [
        int(action.metadata["original_count"]) for action in token_actions
    ]
    keep_counts = [int(action.metadata["keep_count"]) for action in token_actions]
    selectors = {str(action.metadata["selector"]) for action in token_actions}
    target_sites = {str(action.metadata["target_site"]) for action in token_actions}
    if len(selectors) != 1 or len(target_sites) != 1:
        raise InvariantError(
            "Qwen2.5-Omni audio legalization requires one selector and target site."
        )
    selector = next(iter(selectors))
    target_site = next(iter(target_sites))
    compression_metadata = {
        "enabled": True,
        "stage_name": stage_name,
        "plugin": selector,
        "target_site": target_site,
        "position_policy": "compact",
        "original_audio_tokens_per_unit": original_lengths,
        "kept_audio_tokens_per_unit": keep_counts,
        "original_audio_tokens": sum(original_lengths),
        "kept_audio_tokens": sum(keep_counts),
        "removed_sequence_positions_by_sample": removed_by_sample,
        "original_sequence_length": sequence_length,
        "compacted_sequence_length": int(compact_input_ids.shape[1]),
    }
    metadata["expected_token_counts_by_sample"] = expected_counts
    metadata["audio_compression"] = compression_metadata

    native_inputs = dict(state.native_model_inputs)
    _replace_or_remove(native_inputs, "input_ids", compact_input_ids)
    _replace_or_remove(native_inputs, "attention_mask", compact_attention_mask)
    _replace_or_remove(native_inputs, "inputs_embeds", compact_inputs_embeds)
    _replace_or_remove(native_inputs, "labels", compact_labels)
    _replace_or_remove(native_inputs, "token_type_ids", compact_token_type_mask)
    # Discard metadata for the pre-compression layout. Native attention must
    # derive lengths and RoPE positions from the compact tensors and mask.
    for key in (
        "position_ids", "cu_seq_lens_q", "cu_seq_lens_k",
        "max_length_q", "max_length_k",
    ):
        native_inputs.pop(key, None)
    stage_history = dict(state.stage_history)
    stage_history[stage_name] = StageMetadata(
        stage_name=stage_name,
        changed_indices=removed_by_sample,
        changed_spans=[
            span for span in compact_spans if span.modality == "audio"
        ],
        stats=dict(compression_metadata),
    )
    return replace(
        state,
        input_ids=compact_input_ids,
        inputs_embeds=compact_inputs_embeds,
        attention_mask=compact_attention_mask,
        position_ids=None,
        labels=compact_labels,
        native_model_inputs=native_inputs,
        token_type_mask=compact_token_type_mask,
        modality_token_spans=compact_spans,
        modality_metadata=metadata,
        stage_history=stage_history,
    )


def _run_audio_preflight(audio_tower: Any, state: ForwardState) -> None:
    """Run the real audio path for content-dependent budgets, preserving RNG."""
    torch = _torch()
    audio_spans = [span for span in state.modality_token_spans if span.modality == "audio"]
    if not audio_spans:
        return
    input_features = state.audio_values
    if input_features is None:
        input_features = state.native_model_inputs.get("input_features")
    if input_features is None:
        raise InvariantError(
            "Data-dependent audio planning requires input_features."
        )
    feature_lengths = state.audio_feature_lengths
    if feature_lengths is None:
        feature_mask = state.audio_attention_mask
        if feature_mask is None:
            feature_mask = state.native_model_inputs.get(
                "feature_attention_mask"
            )
        if feature_mask is not None:
            feature_lengths = feature_mask.sum(dim=1)
    if feature_lengths is None:
        raise InvariantError(
            "Data-dependent audio planning requires feature_attention_mask "
            "or audio_feature_lengths."
        )
    if int(feature_lengths.numel()) != len(audio_spans):
        raise InvariantError(
            "Audio feature lengths do not match the number of audio spans: "
            f"expected {len(audio_spans)}, got {int(feature_lengths.numel())}."
        )

    reference_parameter = next(audio_tower.parameters(), None)
    reference_device = torch.device(
        _module_execution_device(audio_tower.conv1)
        if reference_parameter is not None
        else input_features.device
    )
    reference_dtype = (
        reference_parameter.dtype
        if reference_parameter is not None
        else input_features.dtype
    )
    probe_features = input_features.to(device=reference_device)
    if bool(probe_features.is_floating_point()):
        probe_features = probe_features.to(dtype=reference_dtype)
    probe_feature_lengths = feature_lengths.to(
        device=reference_device,
        dtype=torch.long,
    )
    aftercnn_lengths, output_lengths = audio_tower._get_feat_extract_output_lengths(
        probe_feature_lengths
    )
    planned_originals = [span.end - span.start for span in audio_spans]
    probed_originals = [int(value) for value in output_lengths.tolist()]
    if probed_originals != planned_originals:
        raise InvariantError(
            "Audio preflight output lengths do not match processor "
            f"placeholders: spans={planned_originals}, encoder={probed_originals}."
        )
    cuda_devices = (
        [reference_device.index]
        if reference_device.type == "cuda"
        and reference_device.index is not None
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        with torch.no_grad():
            if probe_features.ndim == 3:
                feature_mask = state.audio_attention_mask
                if feature_mask is None:
                    feature_mask = state.native_model_inputs.get("feature_attention_mask")
                if feature_mask is None:
                    feature_mask = torch.arange(probe_features.shape[-1], device=reference_device)[None, :] < probe_feature_lengths[:, None]
                # Native get_audio_features packs this way, but validates a
                # count computed before forward. Natural Encoder counts only
                # become known during this first pass, so validate afterwards.
                probe_features = probe_features.permute(0, 2, 1)[feature_mask.to(reference_device).bool()].permute(1, 0)
            output = audio_tower(
                probe_features,
                feature_lens=probe_feature_lengths,
                aftercnn_lens=aftercnn_lengths,
            )
            _, realized_lengths = audio_tower._get_feat_extract_output_lengths(probe_feature_lengths)
            if int(output.last_hidden_state.shape[0]) != int(realized_lengths.sum()):
                raise InvariantError("Audio preflight features do not match the realized output lengths.")


def _register_strategy_module(controller: Any) -> None:
    """Register a learned rule once, before optimizer or trainer setup."""
    controller.strategy_runtime_module = None
    controller.strategy_runtime_module_name = None
    rule = getattr(controller.strategy, "rule", controller.strategy)
    builder = getattr(rule, "build_runtime_module", None)
    if not callable(builder):
        return
    family = controller.execution.executor_family
    if family == "projector_token":
        reference = controller.audio_projector
        hidden_size = getattr(reference, "out_features" if controller._post_projector else "in_features", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(controller.audio_tower, "config", None), "output_dim" if controller._post_projector else "d_model", None)
    else:
        layer = controller.layers[min(controller.application_positions[0].index, len(controller.layers) - 1)]
        reference = layer.fc1 if family == "encoder_token" else layer.self_attn.q_proj
        hidden_size = reference.in_features
    if hidden_size is None:
        raise InvariantError("Cannot resolve the selected compression boundary's width.")
    module_name = f"_mllm_comp_{controller.execution.name}"
    if family != "projector_token":
        module_name += f"_{family}"
    tower = getattr(controller.thinker, "audio_tower", None)
    runtime_module = builder({
        "model": controller.model, "thinker": controller.thinker,
        "audio_encoder": tower, "audio_projector": getattr(tower, "proj", None),
        "compression_hidden_size": hidden_size, "runtime_module_name": module_name,
    })
    if runtime_module is None:
        raise InvariantError("A compression strategy runtime-module builder returned None.")
    existing = getattr(controller.thinker, module_name, None)
    if existing is not None and existing is not runtime_module:
        raise InvariantError(f"Qwen2.5-Omni already has a different compression runtime module registered as {module_name!r}.")
    setattr(controller.thinker, module_name, runtime_module)
    parameter = next(reference.parameters(), None)
    if parameter is not None:
        runtime_module.to(device=_module_execution_device(reference), dtype=parameter.dtype)
    runtime_module.train(bool(getattr(reference, "training", False)))
    controller.strategy_runtime_module = runtime_module
    controller.strategy_runtime_module_name = module_name


def _refresh_strategy_module(controller: Any) -> None:
    """Follow a trainer's replacement of the existing registered child."""
    name = getattr(controller, "strategy_runtime_module_name", None)
    if name is None:
        return
    module = getattr(controller.thinker, name, None)
    if module is None:
        raise InvariantError("The registered compression strategy runtime module is missing.")
    if module is not controller.strategy_runtime_module:
        rule = getattr(controller.strategy, "rule", controller.strategy)
        setter = getattr(rule, "set_runtime_module", None)
        if not callable(setter):
            raise InvariantError("A trainer replaced the registered strategy module, but the strategy cannot accept the replacement.")
        setter(module)
        controller.strategy_runtime_module = module


class Qwen25OmniAudioCompressionController:
    """Coordinate one registered pre- or post-projector audio compressor."""

    def __init__(
        self,
        model: Any,
        stage: Any,
        *,
        profile_enabled: bool = False,
        execution: Any | None = None,
    ) -> None:
        _require_verified_transformers(model)
        if execution is None:
            from .v2_compression import resolve_qwen25_omni_stage_execution

            execution = resolve_qwen25_omni_stage_execution(stage)
        if execution.executor_family != "projector_token":
            raise InvariantError(
                f"{stage.plugin} does not use the projector-token executor."
            )
        self.model = model
        self.stage = stage
        self.profile_enabled = bool(profile_enabled)
        self.execution = execution
        self.plugin = execution.create_plugin()
        self.plugin_config = stage.plugin_config()
        self.plugin_config["_stage_name"] = stage.name
        self.plugin_config["_execution_phase"] = stage.execution_phase.value
        self.plugin_config, self.spec = execution.prepare_config(self.plugin_config)
        self.strategy = execution.create_strategy(self.spec, stage=stage)
        self._post_projector = (
            execution.target_site == "audio_projector_output_post_projector"
        )
        self.thinker = getattr(model, "thinker", model)
        self.audio_tower = getattr(self.thinker, "audio_tower", None)
        layers = list(getattr(self.audio_tower, "layers", ()))
        attention_layer_index = self.strategy.attention_layer_offset
        self.last_attention = (
            getattr(layers[attention_layer_index], "self_attn", None)
            if attention_layer_index is not None
            and len(layers) >= abs(attention_layer_index)
            else None
        )
        self._requires_attention = attention_layer_index is not None
        self.audio_projector = getattr(self.audio_tower, "proj", None)
        required = [self.thinker, self.audio_tower, self.audio_projector]
        if self._requires_attention:
            required.append(self.last_attention)
        if any(item is None for item in required):
            raise InvariantError(
                "Qwen2.5-Omni audio compression could not resolve thinker.audio_tower, "
                "the required attention layer, and audio_tower.proj."
            )
        self.encoder_references = {}
        if execution.binding.canonical_executor_family == "encoder_token":
            from .audio_attention_reference import AudioAttentionReference

            sources = self.strategy.stage_layer_indices(len(layers))
            self.encoder_references = {
                source: AudioAttentionReference(self.audio_tower, source - len(layers), profile_enabled=self.profile_enabled,
                                                probe_kind=self.strategy.encoder_reference_kind)
                for source in sources
            }
        _register_strategy_module(self)
        self._active_request: _ActiveAudioRequest | None = None
        self._last_report: dict[str, Any] | None = None
        self._hook_handles: list[Any] = []
        self._encoder_controller: Any | None = None
        self._layout_observer: Any | None = None
        self._capture_encoder_source = False
        self._installed = False

    @property
    def last_report(self) -> dict[str, Any] | None:
        """Materialize device telemetry only after generation timing has ended."""

        if self._last_report is not None:
            self._last_report = _materialize_report_value(self._last_report)
        return self._last_report

    @last_report.setter
    def last_report(self, value: dict[str, Any] | None) -> None:
        self._last_report = value

    def install(self) -> None:
        if self._installed:
            return
        self._original_output_lengths = self.audio_tower._get_feat_extract_output_lengths
        # References observe preceding stages, not this projector's own output.
        for reference in self.encoder_references.values():
            reference.install()
        self.audio_tower._mllm_projector_compression = self
        self._original_get_rope_index = self.thinker.get_rope_index
        if self._requires_attention:
            assert self.last_attention is not None
            self._encoder_controller = getattr(self.audio_tower, "_mllm_encoder_compression", None)
            if self._encoder_controller is not None:
                encoder = self._encoder_controller
                encoder._layout_observer = self._on_encoder_layout
                source_layer = next(i for i, layer in enumerate(self.audio_tower.layers)
                                    if layer.self_attn is self.last_attention)
                self._capture_encoder_source = any(
                    encoder._compression_enabled(rule) and (
                        position.index > source_layer or (
                            position.boundary == "post_attention_pre_ffn" and position.index == source_layer
                        )
                    ) for rule, position in encoder._position_by_rule_layer.items()
                )
            self._hook_handles.append(
                self.last_attention.register_forward_pre_hook(
                    self._capture_attention_scores,
                    with_kwargs=True,
                )
            )
        if self._post_projector:
            self._hook_handles.append(
                self.audio_projector.register_forward_hook(
                    self._compress_projector_output
                )
            )
        else:
            self._hook_handles.append(
                self.audio_projector.register_forward_pre_hook(
                    self._compress_projector_input
                )
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
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for reference in reversed(tuple(self.encoder_references.values())):
            reference.uninstall()
        if self._encoder_controller is not None:
            self._encoder_controller._layout_observer = None
            self._encoder_controller = None
        self._capture_encoder_source = False
        del self.audio_tower._mllm_projector_compression
        self.audio_tower._get_feat_extract_output_lengths = self._original_output_lengths
        self.thinker.get_rope_index = self._original_get_rope_index
        self._installed = False

    def prepare_state(self, state: ForwardState) -> tuple[ForwardState, CompressionPlan]:
        planning_extra: dict[str, Any] = {"audio_rules": getattr(self.strategy, "rules", {})}
        adapt_plan = getattr(self.strategy, "adapt_plan", None)
        if callable(adapt_plan):
            planning_extra["defer_token_budget"] = True
        if bool(
            getattr(
                self.plugin,
                "requires_projector_hidden_state_plan",
                False,
            )
        ):
            planning_extra["projector_hidden_states_by_audio_unit"] = (
                self._probe_projector_hidden_states(state)
            )
        plan = self.plugin.plan(
            state=state,
            context=CompressionContext(
                phase=(
                    "train"
                    if self.stage.execution_phase is ExecutionPhase.FORWARD
                    else "eval"
                ),
                model=self.model,
                structural_handles={
                    "audio_encoder": self.audio_tower,
                    "audio_projector": self.audio_projector,
                },
                extra=planning_extra,
            ),
            config=self.plugin_config,
        )
        if callable(adapt_plan):
            plan = adapt_plan(plan)
        if plan.legalization_policy is None:
            raise InvariantError("Audio compression plan omitted its legalization policy.")
        legalized = legalize_qwen25_omni_audio_tokens(
            state,
            token_actions=plan.token_actions,
            policy=plan.legalization_policy,
            stage_name=plan.stage_name,
            expected_target_site=self.execution.target_site,
        )
        return legalized, plan

    def _probe_projector_hidden_states(self, state: ForwardState) -> list[Any]:
        """Capture the selected projector boundary for data-dependent planning.

        Qwen constructs its audio placeholders before entering the audio
        encoder. A no-grad encoder preflight is therefore required for methods
        whose natural output length depends on encoder values. The actual
        forward still recomputes the encoder states with gradients enabled;
        this pass only determines placeholder counts.
        """

        audio_spans = [
            span for span in state.modality_token_spans if span.modality == "audio"
        ]
        if not audio_spans:
            return []
        planned_originals = [span.end - span.start for span in audio_spans]

        captured: list[Any] = []

        def capture_projector_input(
            module: Any,
            args: tuple[Any, ...],
        ) -> None:
            del module
            if not args:
                raise InvariantError(
                    "Audio preflight projector hook received no hidden states."
                )
            captured.append(args[0].detach())

        def capture_projector_output(module: Any, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            captured.append(output.detach())

        handle = (
            self.audio_projector.register_forward_hook(capture_projector_output)
            if self._post_projector
            else self.audio_projector.register_forward_pre_hook(capture_projector_input)
        )
        try:
            _run_audio_preflight(self.audio_tower, state)
        finally:
            handle.remove()
        if len(captured) == 1 and int(captured[0].shape[0]) == sum(planned_originals):
            return list(captured[0].split(planned_originals, dim=0))
        if len(captured) != len(audio_spans):
            raise InvariantError(
                "Audio preflight projector call count does not match audio spans: "
                f"expected {len(audio_spans)}, got {len(captured)}."
            )
        return captured

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "target_site": self.execution.target_site,
            **self.strategy.runtime_metadata(),
        }

    @contextmanager
    def request(self, plan: CompressionPlan) -> Any:
        if self._active_request is not None:
            raise InvariantError("Nested Qwen2.5-Omni audio compression requests are unsupported.")
        active = _ActiveAudioRequest(
            original_counts=[
                int(action.metadata["original_count"])
                for action in plan.token_actions
            ],
            keep_counts=[
                int(action.metadata["keep_count"])
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
            preflight_details=[action.metadata.get("_audio_preflight_details") for action in plan.token_actions],
        )
        self._active_request = active
        self.last_report = None
        for reference in self.encoder_references.values():
            reference.start(enabled=not active.identity)
        try:
            yield active
            if active.original_counts and active.projector_call_index != len(
                active.original_counts
            ):
                raise InvariantError(
                    "Qwen audio projector call count did not match planned audio units: "
                    f"expected {len(active.original_counts)}, got "
                    f"{active.projector_call_index}."
                )
            self.last_report = {
                "stage_name": plan.stage_name,
                "plugin": self.plugin.name,
                "execution_phase": plan.execution_phase.value,
                "placement": Placement.PROJECTOR_LEVEL.value,
                **self.runtime_metadata(),
                "requested_keep_ratio": getattr(self.spec, "keep_ratio", None),
                "original_audio_tokens": sum(active.original_counts),
                "kept_audio_tokens": sum(active.keep_counts),
                "achieved_keep_ratio": (
                    sum(active.keep_counts) / sum(active.original_counts)
                    if active.original_counts and sum(active.original_counts)
                    else 1.0
                ),
                "attention_score_seconds": (
                    active.attention_score_seconds + sum(reference.score_seconds for reference in self.encoder_references.values())
                    if self.profile_enabled else None
                ),
                "telemetry_materialized_after_generation_timing": (
                    self.profile_enabled
                ),
                "items": list(active.item_reports),
            }
        finally:
            for reference in self.encoder_references.values():
                reference.clear()
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
        originals = [int(value) for value in audio_output_lengths.tolist()]
        if originals != active.original_counts:
            raise InvariantError(
                "Processor audio placeholders do not match encoder output lengths: "
                f"spans={active.original_counts}, encoder={originals}."
            )
        active.aftercnn_lengths = [int(value) for value in audio_feat_lengths.tolist()]
        active.encoder_output_lengths = originals
        if active.identity:
            return audio_feat_lengths, audio_output_lengths
        compressed_lengths = audio_output_lengths.new_tensor(active.keep_counts)
        return audio_feat_lengths, compressed_lengths

    def _compressed_rope_index(self, thinker: Any, *args: Any, **kwargs: Any) -> Any:
        del thinker
        active = self._active_request
        if active is None or active.identity or not active.keep_counts:
            return self._original_get_rope_index(*args, **kwargs)
        return _compressed_audio_rope_index(
            self._original_get_rope_index, active.keep_counts, *args, **kwargs
        )

    def _on_encoder_layout(self, strategy: Any, plan: Any, chunk_lengths: list[int], chunk_counts: list[int]) -> None:
        active = self._active_request
        if active is None:
            return
        active.aftercnn_lengths = []
        offset = 0
        for count in chunk_counts:
            active.aftercnn_lengths.append(sum(chunk_lengths[offset:offset + count]))
            offset += count
        if active.attention_source is not None:
            with _torch().no_grad():
                active.attention_source = getattr(strategy, "map_reference", strategy.replay)(active.attention_source, plan).detach()
            active.attention_source_chunks = list(chunk_lengths)
            active.attention_needs_refresh = True

    def _capture_attention_scores(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *, retain_source: bool = True,
    ) -> None:
        active = self._active_request
        if active is None or active.identity:
            return
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is None and len(args) > 1:
            cu_seqlens = args[1]
        if hidden_states is None or cu_seqlens is None:
            raise InvariantError(
                "Cannot capture final audio attention without hidden_states and cu_seqlens."
            )
        if self._capture_encoder_source and retain_source:
            active.attention_source = hidden_states.detach()
            active.attention_source_chunks = [int(n) for n in cu_seqlens.diff().tolist()]
            # A later encoder event changes the scoring layout. Compute once,
            # from the mapped original source, when that final layout is known.
            return
        started = perf_counter() if self.profile_enabled else None
        chunks = [int(n) for n in cu_seqlens.diff().tolist()]
        if self.strategy.uses_distinct_merge_prune_importance:
            scores = sa_map_audio_importance(module, hidden_states, chunks, active.aftercnn_lengths)
            active.sa_map_merge_importance = scores["merge"]
            active.sa_map_prune_importance = scores["prune"]
        else:
            from .encoder_compression import mean_received_attention_scores

            active.attention_scores = mean_received_attention_scores(module, hidden_states, chunks)
        if started is not None:
            active.attention_score_seconds += perf_counter() - started

    def _compress_projector_input(
        self,
        module: Any,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...] | None:
        del module
        if self._active_request is None:
            return None
        if not args:
            raise InvariantError("Audio projector pre-hook received no hidden states.")
        compressed = self._compress_projector_tensor(args[0])
        return (compressed, *args[1:])

    def _compress_projector_output(
        self,
        module: Any,
        args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        del module, args
        if self._active_request is None:
            return output
        return self._compress_projector_tensor(output)

    def _compress_projector_tensor(self, hidden_states: Any) -> Any:
        active = self._active_request
        if active is None:
            return hidden_states
        _refresh_strategy_module(self)
        index = active.projector_call_index
        if index >= len(active.keep_counts):
            raise InvariantError(
                "Audio projector received more calls than planned audio units."
            )
        original_count = int(hidden_states.shape[0])
        remaining_original_counts = active.original_counts[index:]
        remaining_total = sum(remaining_original_counts)
        if original_count == remaining_total:
            item_count = len(remaining_original_counts)
        elif original_count == remaining_original_counts[0]:
            # Retain compatibility with an implementation that invokes the
            # projector once per audio unit instead of concatenating them.
            item_count = 1
        else:
            raise InvariantError(
                "Pre-projector audio length does not match the planned placeholder span: "
                f"expected next unit {remaining_original_counts[0]} or remaining "
                f"concatenated units {remaining_total}, got {original_count}."
            )

        input_offset = 0
        compressed_chunks: list[Any] = []
        for item_offset in range(item_count):
            item_index = index + item_offset
            item_original_count = active.original_counts[item_index]
            item_keep_count = active.keep_counts[item_index]
            item_hidden_states = hidden_states[
                input_offset : input_offset + item_original_count
            ]
            if item_keep_count == item_original_count:
                kept_tensor: Any | None = _torch().arange(
                    item_original_count,
                    dtype=_torch().long,
                    device=hidden_states.device,
                )
                compressed_chunk = item_hidden_states
                details: dict[str, Any] = {}
                scores = None
            else:
                scores = self._projector_scores(
                    active,
                    item_index,
                    item_original_count,
                )
                preflight = active.preflight_details[item_index]
                if preflight is not None:
                    # Variable-length custom planning fixes the actual groups, not just their count.
                    details = dict(preflight)
                    replay_plan = self.strategy.replay_plan(item_keep_count, details, None, scores=scores)
                    compressed_chunk = self.strategy.replay(item_hidden_states, replay_plan)
                    kept_tensor = None
                else:
                    compressed_chunk, kept_tensor, details = self.strategy.compress(
                        item_hidden_states,
                        scores=scores,
                        keep_count=item_keep_count,
                        item_index=item_index,
                    )

            indices_chronological = True
            if kept_tensor is not None and int(kept_tensor.numel()) >= 2:
                indices_chronological = (
                    kept_tensor[1:] >= kept_tensor[:-1]
                ).all().detach()

            compressed_chunks.append(compressed_chunk)
            observer = getattr(self, "_layout_observer", None)
            if observer is not None and item_keep_count != item_original_count:
                observer(self.strategy, item_index, kept_tensor, details, scores, item_keep_count)
            # All downstream references have consumed the rule's opaque plans.
            # Keep them request-local even when there is no layout observer.
            details.pop("_reference_replay_plans", None)
            score_summary: dict[str, Any] = {}
            if isinstance(scores, Mapping):
                for score_name, values in scores.items():
                    score_summary.update(
                        {
                            f"{score_name}_score_min": values.min().detach(),
                            f"{score_name}_score_max": values.max().detach(),
                            f"{score_name}_score_sum": values.sum().detach(),
                        }
                    )
            elif scores is not None:
                score_summary = {
                    "score_min": scores.min().detach(),
                    "score_max": scores.max().detach(),
                    "score_sum": scores.sum().detach(),
                }
            active.item_reports.append(
                {
                    "audio_unit_index": active.audio_unit_indices[item_index],
                    "sample_index": active.sample_indices[item_index],
                    "original_tokens": item_original_count,
                    "kept_tokens": item_keep_count,
                    "kept_indices": (
                        kept_tensor.detach() if kept_tensor is not None else None
                    ),
                    "indices_chronological": indices_chronological,
                    **score_summary,
                    **details,
                }
            )
            input_offset += item_original_count

        active.projector_call_index += item_count
        if all(
            active.keep_counts[item_index] == active.original_counts[item_index]
            for item_index in range(index, index + item_count)
        ):
            # Preserve exact identity semantics, including tensor identity, at
            # keep_ratio=1.0.
            compressed = hidden_states
        else:
            compressed = _torch().cat(compressed_chunks, dim=0)
        return compressed

    def _projector_scores(
        self,
        active: _ActiveAudioRequest,
        item_index: int,
        output_length: int,
    ) -> Any | None:
        if self.encoder_references:
            scores = {}
            for source, reference in self.encoder_references.items():
                reference._pool_scores()
                values = reference.audio_scores[item_index]
                if len(values) != output_length:
                    raise InvariantError("Encoder attention source does not match the projector audio span.")
                scores[source] = values
            return scores
        if active.attention_needs_refresh:
            from itertools import accumulate

            self._capture_attention_scores(self.last_attention, (), {
                "hidden_states": active.attention_source,
                "cu_seqlens": _torch().tensor([0, *accumulate(active.attention_source_chunks)], dtype=_torch().int32),
            }, retain_source=False)
            active.attention_source = None
            active.attention_source_chunks.clear()
            active.attention_needs_refresh = False
        if self.spec.score_source not in {
            "encoder_last_layer_attention",
            "encoder_penultimate_layer_attention",
        }:
            return None
        if not active.aftercnn_lengths:
            raise InvariantError(
                "Final encoder attention scores were not captured before projector pruning."
            )
        if bool(
            getattr(
                self.strategy,
                "uses_distinct_merge_prune_importance",
                False,
            )
        ):
            if (
                active.sa_map_merge_importance is None
                or active.sa_map_prune_importance is None
                or not active.encoder_output_lengths
            ):
                raise InvariantError(
                    "SA-MAP merge/prune attention importance was not captured."
                )
            start = sum(active.encoder_output_lengths[:item_index])
            end = start + output_length
            merge = active.sa_map_merge_importance[start:end]
            prune = active.sa_map_prune_importance[start:end]
            if int(merge.shape[0]) != output_length or int(prune.shape[0]) != output_length:
                raise InvariantError(
                    "SA-MAP projector importance does not match projector output length."
                )
            return {"merge": merge, "prune": prune}

        if active.attention_scores is None:
            raise InvariantError(
                "Final encoder attention scores were not captured before projector pruning."
            )
        raw_start = sum(active.aftercnn_lengths[:item_index])
        raw_end = raw_start + active.aftercnn_lengths[item_index]
        raw_scores = active.attention_scores[raw_start:raw_end]
        needed = output_length * 2
        if int(raw_scores.shape[0]) < needed:
            raise InvariantError(
                "Final encoder attention score length is shorter than the pooled projector input."
            )
        return raw_scores[:needed].reshape(output_length, 2).mean(dim=1)


def sa_map_audio_importance(
    attention: Any, hidden_states: Any, chunk_lengths: Sequence[int],
    audio_lengths: Sequence[int], *, pooled: bool = True,
) -> dict[str, Any]:
    """Original SA-MAP reductions, with pooling only across a real pooler."""
    torch = _torch()
    length = int(hidden_states.shape[0])
    if sum(chunk_lengths) != length or not audio_lengths or sum(audio_lengths) != length:
        raise InvariantError("SA-MAP encoder lengths do not match captured attention.")
    with torch.no_grad():
        merge_received_sum = hidden_states.new_empty(length, dtype=torch.float32)
        prune_diagonal = torch.empty_like(merge_received_sum)
        prune_upper = torch.zeros_like(merge_received_sum) if pooled else None
        prune_lower = torch.zeros_like(merge_received_sum) if pooled else None
        # Odd per-audio tails have no downstream pooled query. They still
        # participate in their chunk's original attention softmax.
        query_mask = torch.ones(length, dtype=torch.bool, device=hidden_states.device)
        if pooled:
            start = 0
            for raw_length in audio_lengths:
                if raw_length % 2:
                    query_mask[start + raw_length - 1] = False
                start += raw_length
        start = 0
        for count in chunk_lengths:
            end = start + count
            chunk = hidden_states[start:end]
            query = attention.q_proj(chunk).reshape(count, attention.num_heads, -1).transpose(0, 1)
            key = attention.k_proj(chunk).reshape(count, attention.num_heads, -1).transpose(0, 1)
            weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * attention.scaling, dim=-1, dtype=torch.float32)
            head_max = weights.max(dim=0).values
            merge_received_sum[start:end] = head_max[query_mask[start:end]].sum(dim=0)
            head_mean = weights.mean(dim=0)
            prune_diagonal[start:end] = head_mean.diagonal()
            if pooled and count > 1:
                prune_upper[start:end - 1] = head_mean.diagonal(offset=1)
                prune_lower[start + 1:end] = head_mean.diagonal(offset=-1)
            start = end
        merge_parts, prune_parts = [], []
        start = 0
        for raw_length in audio_lengths:
            if pooled:
                merge, prune = _pool_sa_map_attention_bands(
                    merge_received_sum, prune_diagonal, prune_upper, prune_lower,
                    start=start, raw_length=raw_length,
                )
            else:
                merge = merge_received_sum[start:start + raw_length] / raw_length
                prune = prune_diagonal[start:start + raw_length]
            merge_parts.append(merge)
            prune_parts.append(prune)
            start += raw_length
        result = {"merge": torch.cat(merge_parts), "prune": torch.cat(prune_parts)}
        if any(not bool(torch.isfinite(values).all()) for values in result.values()):
            raise InvariantError("SA-MAP attention importance contains NaN or Inf.")
        return result


def _pool_sa_map_attention_bands(
    merge_received_sum: Any,
    prune_diagonal: Any,
    prune_upper: Any,
    prune_lower: Any,
    *,
    start: int,
    raw_length: int,
) -> tuple[Any, Any]:
    """Map encoder attention reductions through Qwen's two-token pooler.

    This is algebraically identical to reshaping a per-audio attention matrix
    to ``[N, 2, N, 2]``. Eq. (4) averages the query-pair axis, sums the
    key-pair axis and then averages over pooled queries. ADPruner instead takes
    the diagonal after the same query/key pooling.
    """

    if raw_length < 2:
        raise InvariantError(
            "SA-MAP requires at least two encoder tokens before projector pooling."
        )
    torch = _torch()
    output_length = raw_length // 2
    used_length = output_length * 2
    end = start + used_length
    merge = (
        merge_received_sum[start:end].reshape(output_length, 2).sum(dim=1)
        / used_length
    )
    even = torch.arange(start, end, 2, device=merge_received_sum.device)
    odd = even + 1
    prune = 0.5 * (
        prune_diagonal[even]
        + prune_upper[even]
        + prune_lower[odd]
        + prune_diagonal[odd]
    )
    return merge, prune

def _compact_batch_first_tensor(
    value: Any | None,
    keep_positions: Sequence[Sequence[int]],
    *,
    pad_value: int | float,
    padding_side: str,
) -> Any | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2 or int(shape[0]) != len(keep_positions):
        raise InvariantError(
            "Sequence-aligned Qwen tensor is not batch-first or has the wrong batch size."
        )
    torch = _torch()
    rows = []
    for sample_index, positions in enumerate(keep_positions):
        index = torch.tensor(positions, dtype=torch.long, device=value.device)
        rows.append(value[sample_index].index_select(0, index))
    return torch.nn.utils.rnn.pad_sequence(
        rows, batch_first=True, padding_value=pad_value, padding_side=padding_side,
    )


def _shift_token_spans(
    spans: Iterable[TokenSpan],
    *,
    removed_by_sample: Sequence[Sequence[int]],
    added_prefix_by_sample: Sequence[int],
) -> list[TokenSpan]:
    shifted: list[TokenSpan] = []
    for span in spans:
        removed = removed_by_sample[span.sample_index]
        added_prefix = added_prefix_by_sample[span.sample_index]
        removed_before = sum(position < span.start for position in removed)
        removed_inside = sum(span.start <= position < span.end for position in removed)
        shifted.append(
            replace(
                span,
                start=span.start - removed_before + added_prefix,
                end=span.end - removed_before - removed_inside + added_prefix,
            )
        )
    return sorted(shifted, key=lambda span: (span.sample_index, span.start, span.end))


def _replace_or_remove(mapping: dict[str, Any], key: str, value: Any | None) -> None:
    if value is None:
        mapping.pop(key, None)
    else:
        mapping[key] = value


def _materialize_report_value(value: Any) -> Any:
    """Convert deferred tensor telemetry into JSON-safe host values."""

    if isinstance(value, dict):
        return {
            key: _materialize_report_value(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_materialize_report_value(child) for child in value]
    if isinstance(value, tuple):
        return [_materialize_report_value(child) for child in value]
    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    tolist = getattr(value, "tolist", None)
    if callable(detach) and callable(cpu) and callable(tolist):
        return value.detach().cpu().tolist()
    return value


def _require_verified_transformers(model: Any) -> None:
    """Reject real Transformers models whose private hook contract may differ."""

    if not type(model).__module__.startswith("transformers."):
        return
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - model already proves dependency
        raise InvariantError("Cannot import Transformers for version validation.") from exc
    actual = str(transformers.__version__)
    if actual != _VERIFIED_TRANSFORMERS_VERSION:
        raise InvariantError(
            "Qwen2.5-Omni audio compression is version-specific: expected "
            f"Transformers {_VERIFIED_TRANSFORMERS_VERSION}, got {actual}."
        )


def _module_execution_device(module: Any) -> Any:
    hook = getattr(module, "_hf_hook", None)
    device = getattr(hook, "execution_device", None)
    if device is not None:
        return device
    weight = getattr(module, "weight", None)
    if weight is not None:
        return weight.device
    # Parameterless containers receive no Accelerate hook. Their first
    # executing child owns the real device while stored weights may be meta.
    for child in module.modules():
        device = getattr(getattr(child, "_hf_hook", None), "execution_device", None)
        if device is not None:
            return device
    return next(module.parameters()).device


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - real runtime dependency
        raise InvariantError(
            "Qwen2.5-Omni audio compression requires PyTorch at runtime."
        ) from exc
    return torch
