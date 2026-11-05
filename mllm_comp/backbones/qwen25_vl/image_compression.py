"""Qwen2.5-VL image compression activations and algorithm-to-provider calls.

Physical declarations live in ``compressors.image.contracts.qwen25_vl`` and
are re-exported here. Activations bind shared algorithms to the native
provider, which preserves visual window/merger structure, image placeholders,
M-RoPE and LLM layouts. Request state and checkpoint replay use the shared
image lifecycle. Exact support evidence is maintained by the registry.
"""

from __future__ import annotations

from mllm_comp.compressors.image.positions import (
    saint_encoder_operation,
    saint_encoder_positions,
    image_projector_sites,
    tome_output_site,
    tome_encoder_score_layer,
    tome_encoder_operation,
    tome_position_metadata,
    decoder_token_operation,
)

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any

from mllm_comp.backbones._decoder_attention import (
    DecoderAttentionImplementationOverride,
)
from mllm_comp.compressors.image.saint import SAINTSpec, saint_image_tokens
from mllm_comp.compressors.image.aim import AIMSpec, aim_image_tokens
from mllm_comp.compressors.image.divprune import DivPruneSpec, divprune_image
from mllm_comp.compressors.image.bifocal import (
    BIFOCAL_CONFIG_KEYS,
    BifocalSpec,
    bifocal_extract_query,
    bifocal_image_tokens,
    bifocal_query_text_from_ids,
    bifocal_spec_from_config,
    bifocal_word_embeddings,
)
from mllm_comp.compressors.image.mmtok import (
    MMTokSpec,
    merge_mmtok_executor_config,
    mmtok_image_tokens,
)
from mllm_comp.compressors.image.vispruner import (
    VisPrunerSpec,
    merge_vispruner_executor_config,
    vispruner_image_tokens,
)
from mllm_comp.compressors.image.zoo_prune import (
    ZOOProjectorProxyCache,
    ZOOPruneSpec,
    merge_zoo_prune_executor_config,
    zoo_prune_image_tokens,
)
from mllm_comp.compressors.image.nuwa import (
    NuwaSpec,
    merge_nuwa_executor_config,
    nuwa_stage1_image_tokens,
)
from mllm_comp.compressors.image.restore import (
    RestoreSpec,
    merge_restore_executor_config,
    restore_image_tokens,
)
from mllm_comp.compressors.image.decoder_selection import (
    SGLDecoderMethod,
    NuwaDecoderMethod,
    FEATHERDecoderMethod,
    FinePrunerDecoderMethod,
    SparseVLMDecoderMethod,
    PACTDecoderMethod,
    IVCPruneDecoderMethod,
    DARTDecoderMethod,
    D2PrunerDecoderMethod,
    FastVDecoderMethod,
    PyramidDecoderMethod,
    BTPDecoderMethod,
    HalfVDecoderMethod,
    TransPruneDecoderMethod,
    execute_decoder_selection,
)
from mllm_comp.compressors.image.decoder_registration import (
    create_registered_decoder_selection,
    prepare_decoder_selection,
)
from mllm_comp.compressors.rules import builtin_rule_specs, load_rule_specs
from mllm_comp.compressors.image.sgl import (
    SGLGuideController,
    SGLSpec,
)
from mllm_comp.compressors.image.decoder_config import (
    merge_decoder_executor_config,
)
from mllm_comp.compressors.image.tome import (
    tome_merge_image,
    tome_preserve_source_order,
)
from mllm_comp.compressors.image.visionzip import visionzip_image
from mllm_comp.compressors.image.prumerge import (
    PruMergeSpec,
    prumerge_image_tokens,
)
from mllm_comp.compressors.image.fixed_visual_merge import (
    FixedVisualMergeSpec,
    fixed_visual_merge,
)
from mllm_comp.compressors.image.stage2_projector import (
    Stage2ProjectorSpec,
    create_stage2_projector,
    load_stage2_projector,
)
from mllm_comp.compressors.image.gap import GAPSpec, gap_image_tokens
from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError
from mllm_comp.core.layout import (
    LayoutDelta,
    LayoutTokenUnit,
    OrderingPolicy,
    layout_delta_record,
)
from mllm_comp.core.vocabulary import ExecutionPhase, Modality


_BRIDGE_METADATA_KEY = "qwen25_vl_image_compression_bridge"
_BRIDGE_MODEL_ATTRIBUTES = (
    "mllm_comp_qwen25_vl_image_compression_bridge",
    "_mllm_comp_qwen25_vl_image_compression_bridge",
)
_ALGORITHM_EXECUTOR_CONFIG_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "tome": frozenset(
            {
                "r",
                "output_r",
                "token_sizes",
                "protected_indices",
                "coordinates",
                "eps",
                "clamp_r",
            }
        ),
        "saint": frozenset(
            {
                "layer_indices",
                "keep_ratios",
                "prune_level",
                "group_size",
                "similarity_threshold",
                "neighbour_count",
                "gamma",
                "max_iterations",
            }
        ),
        "aim": frozenset(
            {"retain_ratio", "target_count", "min_visual_tokens"}
        ),
        "pact": frozenset(
            {
                "reduction_layer",
                "prune_keep_ratio",
                "target_keep_ratio",
                "cutoff",
                "density_scale",
                "include_pruned",
                "pruned_cutoff_coefficient",
                "multiply_by_hidden_norm",
                "proportional_attention",
                "cutoff_search_steps",
                "min_visual_tokens",
            }
        ),
        "visionzip": frozenset(
            {
                "budget",
                "attention_mode",
                "cls_index",
                "protected_indices",
                "dominant_weight",
                "contextual_weight",
                "coordinates",
                "media_unit",
                "source_layout",
                "result_layout",
            }
        ),
        "prumerge": frozenset(
            {
                "variant",
                "adaptive",
                "reduction_ratio",
                "target_count",
                "iqr_multiplier",
                "min_centers",
                "cluster_size",
            }
        ),
        "feature_anchor_mean": frozenset(
            {
                "target_count",
                "stage2_variant",
                "bottleneck_dim",
                "projector_checkpoint",
                "trainable_projector",
                "attention_mass_correction",
            }
        ),
        "grid_mean": frozenset({"target_count"}),
        "gap": frozenset({"retain_ratio", "target_count"}),
        "bifocal": BIFOCAL_CONFIG_KEYS,
        "divprune": frozenset(
            {
                "retain_ratio",
                "target_count",
                "prune_layer",
                "distance",
                "selection",
                "min_visual_tokens",
                "distance_chunk_size",
            }
        ),
        "mmtok": frozenset(
            {
                "retain_ratio",
                "budget_rounding",
                "target_count",
                "alpha",
                "text_temperature",
                "vision_temperature",
                "min_visual_tokens",
            }
        ),
        "vispruner": frozenset(
            {"target_count", "important_ratio", "removal_step", "min_visual_tokens"}
        ),
        "zoo_prune": frozenset(
            {
                "retention_ratio",
                "target_count",
                "num_directions",
                "step_size",
                "projector_proxy",
                "low_rank_rank",
                "low_rank_solver",
                "direction_batch_size",
                "seed",
                "min_visual_tokens",
            }
        ),
        "pyramid_drop": frozenset(
            {
                "keep_ratio",
                "minimum_keep_tokens",
                "num_stages",
                "restore_source_order",
                "layer_indices",
                "protected_indices",
                "media_unit",
                "source_layout",
                "result_layout",
            }
        ),
    }
)
_TOME_KERNEL_CONFIG_KEYS = frozenset(
    {"r", "token_sizes", "protected_indices", "coordinates", "eps", "clamp_r"}
)


from mllm_comp.compressors.image.contracts.qwen25_vl import (
    Qwen25VLImageBindingContract,
    QWEN25_VL_IMAGE_BINDING_CONTRACTS,
)


@dataclass(frozen=True)
class Qwen25VLDenseRestorePlan:
    """Expand compact ToMe values to the original dense visual patch layout."""

    compact_delta: LayoutDelta

    def __post_init__(self) -> None:
        if self.compact_delta.media_unit != LayoutTokenUnit.IMAGE_PATCH.value:
            raise InvariantError(
                "Qwen2.5-VL ToMe dense restore requires image_patch layout units."
            )
        if self.compact_delta.result_count < 1:
            raise InvariantError("A dense restore cannot expand an empty patch layout.")

    @property
    def dense_boundary_delta(self) -> LayoutDelta:
        """Layout seen by the spatial merger, relative to its dense input."""

        return LayoutDelta.identity(
            self.compact_delta.source_count,
            media_unit=LayoutTokenUnit.IMAGE_PATCH,
            coordinates=self.compact_delta.coordinates,
            protected_indices=self.compact_delta.protected_indices,
            layout={
                "algorithm": "tome",
                "boundary": "visual_output_pre_spatial_merger",
                "dense_restored": True,
                "media_unit": LayoutTokenUnit.IMAGE_PATCH.value,
            },
        )

    def restore(self, compact_tokens: Any, *, sequence_dim: int = -2) -> Any:
        """Replicate each merged value over its source patch positions."""

        compact_count = _sequence_length(compact_tokens, sequence_dim)
        if compact_count != self.compact_delta.result_count:
            raise InvariantError(
                "ToMe compact tensor count does not match its LayoutDelta: "
                f"{compact_count} != {self.compact_delta.result_count}."
            )
        gather = tuple(
            _required_result_index(self.compact_delta, source_index)
            for source_index in range(self.compact_delta.source_count)
        )
        return _take_sequence(compact_tokens, gather, sequence_dim=sequence_dim)


@dataclass(frozen=True)
class Qwen25VLSparseMRoPERequest:
    """Inputs a provider must use to rebuild sparse image M-RoPE positions.

    ``image_grid_thw`` remains the *source* dense geometry.  A provider must
    consume ``layout_delta`` (and its coordinates/source groups) rather than
    applying Qwen2.5-VL's standard dense placeholder-count formula.
    """

    input_ids: Any
    attention_mask: Any | None
    image_grid_thw: Any
    layout_delta: LayoutDelta
    source_placeholder_indices: tuple[int, ...]
    retained_placeholder_indices: tuple[int, ...]
    source_position_ids: Any | None = None
    source_rope_deltas: Any | None = None
    target_modality: Modality = Modality.IMAGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_modality", Modality(self.target_modality))
        if self.target_modality is not Modality.IMAGE:
            raise InvariantError("Qwen2.5-VL sparse M-RoPE request must be image-only.")


@dataclass(frozen=True)
class Qwen25VLMRoPEResult:
    """Provider output after rebuilding positions for a sparse image span."""

    position_ids: Any
    rope_deltas: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Qwen25VLPromptRepairResult:
    """Prompt fields after post-merger compression and M-RoPE legalization."""

    input_ids: Any
    attention_mask: Any | None
    labels: Any | None
    position_ids: Any
    rope_deltas: Any
    layout_delta: LayoutDelta
    removed_placeholder_indices: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Qwen25VLPlaceholderRepairPlan:
    """Map one compressed image-feature layout into its serialized prompt span."""

    layout_delta: LayoutDelta
    placeholder_indices: tuple[int, ...]
    source_sequence_length: int

    def __post_init__(self) -> None:
        placeholders = _strict_indices(
            self.placeholder_indices,
            upper=self.source_sequence_length,
            name="placeholder_indices",
        )
        object.__setattr__(self, "placeholder_indices", placeholders)
        if len(placeholders) != self.layout_delta.source_count:
            raise InvariantError(
                "Image placeholder count must match the compression source layout: "
                f"{len(placeholders)} != {self.layout_delta.source_count}."
            )
        if self.layout_delta.media_unit != "image_feature":
            raise InvariantError(
                "Qwen2.5-VL post-merger image compression requires image_feature "
                "layout units."
            )
        representatives = _result_representatives(self.layout_delta)
        source_ordered = representatives == tuple(sorted(representatives))
        if (
            self.layout_delta.ordering_policy is not OrderingPolicy.PRESERVE_SOURCE
            and not (
                self.layout_delta.ordering_policy is OrderingPolicy.EXPLICIT
                and source_ordered
            )
        ):
            raise InvariantError(
                "Sparse prompt legalization requires source-ordered representatives."
            )

    @property
    def retained_placeholder_indices(self) -> tuple[int, ...]:
        representatives = _result_representatives(self.layout_delta)
        return tuple(self.placeholder_indices[index] for index in representatives)

    @property
    def removed_placeholder_indices(self) -> tuple[int, ...]:
        retained = set(self.retained_placeholder_indices)
        return tuple(
            index for index in self.placeholder_indices if index not in retained
        )

    @property
    def kept_sequence_indices(self) -> tuple[int, ...]:
        removed = set(self.removed_placeholder_indices)
        return tuple(
            index
            for index in range(self.source_sequence_length)
            if index not in removed
        )

    def repair(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None,
        labels: Any | None,
        image_grid_thw: Any,
        mrope_rebuilder: Callable[[Qwen25VLSparseMRoPERequest], Qwen25VLMRoPEResult],
        source_position_ids: Any | None = None,
        source_rope_deltas: Any | None = None,
        sequence_dim: int = -1,
    ) -> Qwen25VLPromptRepairResult:
        """Compact prompt fields, then require an explicit sparse M-RoPE rebuild."""

        if not callable(mrope_rebuilder):
            raise BackendIntegrationError(
                "Post-merger image compression on Qwen2.5-VL requires a sparse "
                "M-RoPE rebuilder."
            )
        compact_ids = self._compact_field(input_ids, "input_ids", sequence_dim)
        compact_mask = self._compact_optional(
            attention_mask, "attention_mask", sequence_dim
        )
        compact_labels = self._compact_optional(labels, "labels", sequence_dim)
        request = Qwen25VLSparseMRoPERequest(
            input_ids=compact_ids,
            attention_mask=compact_mask,
            image_grid_thw=image_grid_thw,
            layout_delta=self.layout_delta,
            source_placeholder_indices=self.placeholder_indices,
            retained_placeholder_indices=self.retained_placeholder_indices,
            source_position_ids=source_position_ids,
            source_rope_deltas=source_rope_deltas,
        )
        rebuilt = mrope_rebuilder(request)
        if not isinstance(rebuilt, Qwen25VLMRoPEResult):
            raise BackendIntegrationError(
                "Sparse M-RoPE rebuilder must return Qwen25VLMRoPEResult."
            )
        return Qwen25VLPromptRepairResult(
            input_ids=compact_ids,
            attention_mask=compact_mask,
            labels=compact_labels,
            position_ids=rebuilt.position_ids,
            rope_deltas=rebuilt.rope_deltas,
            layout_delta=self.layout_delta,
            removed_placeholder_indices=self.removed_placeholder_indices,
            metadata={
                "target_modality": Modality.IMAGE.value,
                "source_placeholders": self.layout_delta.source_count,
                "result_placeholders": self.layout_delta.result_count,
                "sparse_mrope_rebuilt": True,
                **dict(rebuilt.metadata),
            },
        )

    def _compact_optional(self, value: Any | None, name: str, dim: int) -> Any | None:
        if value is None:
            return None
        return self._compact_field(value, name, dim)

    def _compact_field(self, value: Any, name: str, dim: int) -> Any:
        length = _sequence_length(value, dim)
        if length != self.source_sequence_length:
            raise InvariantError(
                f"{name} sequence length {length} does not match prompt layout "
                f"{self.source_sequence_length}."
            )
        return _take_sequence(value, self.kept_sequence_indices, sequence_dim=dim)


@dataclass(frozen=True)
class Qwen25VLDecoderCacheRepairRequest:
    """Provider-specific cached-generation repair at one decoder boundary."""

    past_key_values: Any
    layout_delta: LayoutDelta
    boundary_layer: int
    kept_sequence_indices: tuple[int, ...]
    position_ids: Any | None
    cache_position: Any | None


@dataclass(frozen=True)
class Qwen25VLDecoderRepairResult:
    """Sequence-aligned fields after one decoder compression boundary."""

    attention_mask: Any | None
    position_ids: Any | None
    cache_position: Any | None
    labels: Any | None
    past_key_values: Any | None
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Qwen25VLDecoderRepairPlan:
    """Legalize masks, positions, labels, and caches after decoder compression."""

    layout_delta: LayoutDelta
    boundary_layer: int
    allow_explicit_reorder: bool = False

    def __post_init__(self) -> None:
        if self.layout_delta.media_unit != "llm_token":
            raise InvariantError("Decoder repair requires llm_token units.")
        if not isinstance(self.allow_explicit_reorder, bool):
            raise TypeError("allow_explicit_reorder must be a boolean.")
        if (
            self.layout_delta.ordering_policy is not OrderingPolicy.PRESERVE_SOURCE
            and not self.allow_explicit_reorder
        ):
            raise InvariantError(
                "Decoder compression must preserve serialized source order."
            )
        if (
            isinstance(self.boundary_layer, bool)
            or not isinstance(self.boundary_layer, int)
            or self.boundary_layer < 0
        ):
            raise InvariantError("boundary_layer must be a non-negative integer.")

    @property
    def kept_sequence_indices(self) -> tuple[int, ...]:
        return _result_representatives(self.layout_delta)

    def repair(
        self,
        *,
        attention_mask: Any | None,
        position_ids: Any | None,
        cache_position: Any | None,
        labels: Any | None,
        past_key_values: Any | None,
        execution_phase: ExecutionPhase | str,
        cache_rebuilder: (
            Callable[[Qwen25VLDecoderCacheRepairRequest], Any] | None
        ) = None,
        sequence_dim: int = -1,
    ) -> Qwen25VLDecoderRepairResult:
        phase = ExecutionPhase(execution_phase)
        repaired_mask = self._compact_attention_mask(attention_mask, sequence_dim)
        repaired_positions = self._compact_optional(
            position_ids, "position_ids", sequence_dim
        )
        repaired_cache_position = self._compact_optional(
            cache_position, "cache_position", sequence_dim
        )
        repaired_labels = self._compact_optional(labels, "labels", sequence_dim)
        repaired_cache = past_key_values
        if past_key_values is not None:
            if phase is ExecutionPhase.GENERATE and not callable(cache_rebuilder):
                raise BackendIntegrationError(
                    "Cached Qwen2.5-VL generation requires an explicit decoder "
                    "KV-cache rebuilder."
                )
            if cache_rebuilder is not None:
                repaired_cache = cache_rebuilder(
                    Qwen25VLDecoderCacheRepairRequest(
                        past_key_values=past_key_values,
                        layout_delta=self.layout_delta,
                        boundary_layer=self.boundary_layer,
                        kept_sequence_indices=self.kept_sequence_indices,
                        position_ids=repaired_positions,
                        cache_position=repaired_cache_position,
                    )
                )
        return Qwen25VLDecoderRepairResult(
            attention_mask=repaired_mask,
            position_ids=repaired_positions,
            cache_position=repaired_cache_position,
            labels=repaired_labels,
            past_key_values=repaired_cache,
            layout_delta=self.layout_delta,
            metadata={
                "target_modality": Modality.IMAGE.value,
                "boundary_layer": self.boundary_layer,
                "source_tokens": self.layout_delta.source_count,
                "result_tokens": self.layout_delta.result_count,
                "cache_rebuilt": past_key_values is not None
                and cache_rebuilder is not None,
            },
        )

    def _compact_optional(self, value: Any | None, name: str, dim: int) -> Any | None:
        if value is None:
            return None
        length = _sequence_length(value, dim)
        if length != self.layout_delta.source_count:
            raise InvariantError(
                f"{name} sequence length {length} does not match decoder layout "
                f"{self.layout_delta.source_count}."
            )
        return _take_sequence(value, self.kept_sequence_indices, sequence_dim=dim)

    def _compact_attention_mask(self, value: Any | None, dim: int) -> Any | None:
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        # A provider may expose either a rank-2 padding mask or an already
        # expanded rank-4 causal mask.  The latter has both query and key axes
        # aligned to the pre-drop sequence and therefore needs two compactions.
        if shape is not None and len(shape) >= 4:
            if (
                int(shape[-2]) != self.layout_delta.source_count
                or int(shape[-1]) != self.layout_delta.source_count
            ):
                raise InvariantError(
                    "Expanded attention_mask query/key axes do not match the "
                    "decoder layout."
                )
            value = _take_sequence(value, self.kept_sequence_indices, sequence_dim=-2)
            return _take_sequence(value, self.kept_sequence_indices, sequence_dim=-1)
        return self._compact_optional(value, "attention_mask", dim)


@dataclass(frozen=True)
class Qwen25VLEncoderCompressionResult:
    kernel_result: Any
    dense_restore: Qwen25VLDenseRestorePlan


@dataclass(frozen=True)
class Qwen25VLProjectorCompressionResult:
    kernel_result: Any
    prompt_repair: Qwen25VLPlaceholderRepairPlan


@dataclass(frozen=True)
class Qwen25VLDecoderCompressionResult:
    kernel_result: Any
    decoder_repair: Qwen25VLDecoderRepairPlan


class Qwen25VLImageCompressionExecutor:
    """Installable callback object consumed by a Qwen2.5-VL provider bridge."""

    def __init__(
        self,
        *,
        model: Any,
        runtime: Any,
        stage: Any,
        contract: Qwen25VLImageBindingContract,
        execution_phase: ExecutionPhase | str,
        config: Mapping[str, Any] | None = None,
        contract_projection: Any = None,
        binding: Any = None,
        binding_config: Any = None,
    ) -> None:
        self.model = model
        self.runtime = runtime
        self.stage = stage
        self.decoder_selection = None
        if contract_projection is not None and binding.strategy_factory is not None:
            (self.decoder_selection, config, self._rules, self._rule_specs) = create_registered_decoder_selection(
                stage=stage, binding=binding, contract_projection=contract_projection,
                backbone="qwen25_vl", execution_phase=execution_phase, binding_config=binding_config)
        else:
            try:
                self._rule_specs = builtin_rule_specs(stage)
                self._rules = load_rule_specs(self._rule_specs)
            except ConfigError as exc:
                raise BackendIntegrationError(f"Invalid image compression rules: {exc}") from exc
        self.contract = contract
        self.execution_phase = ExecutionPhase(execution_phase)
        self.config = dict(config or {})
        if self.contract.algorithm == "tome":
            self.output_target_site = tome_output_site(self.stage, "qwen25_vl")
        if self.contract.algorithm in {"visionzip", "feature_anchor_mean", "grid_mean", "prumerge", "gap", "divprune", "aim", "vispruner", "mmtok", "zoo_prune", "nuwa", "restore"}:
            self.projector_target_sites = image_projector_sites(self.stage, self.contract.algorithm, "qwen25_vl")
            self.target_site = self.projector_target_sites[0]
        self._bifocal_spec_value: BifocalSpec | None = None
        self._bifocal_last_query: str | None = None
        self._projector_generate_summary: dict[str, Any] = {
            "generate_calls": 0,
            "decode_position_verified": 0,
            "decode_position_failed": 0,
            "no_decode_step": 0,
            "prompt_removed_total": 0,
            "last": None,
        }
        # Backward-compatible metadata key for the existing Bifocal tests.
        self._bifocal_generate_summary = self._projector_generate_summary
        if self.contract.algorithm == "bifocal":
            self.projector_target_sites = tuple(self.contract.target_sites)
            self.target_site = self.projector_target_sites[0]
            self._bifocal_spec_value = _bifocal_spec(self.config)
        self._saint_layer_indices_value: tuple[int, ...] = ()
        self._saint_keep_ratios_value: tuple[float, ...] = ()
        self._saint_spec_value: SAINTSpec | None = None
        if self.contract.algorithm == "saint":
            (
                self._saint_layer_indices_value,
                self._saint_keep_ratios_value,
                self._saint_spec_value,
            ) = _saint_config(self.config)
            self.application_positions = saint_encoder_positions(self.stage, self._saint_layer_indices_value)
            self.target_site = saint_encoder_operation(self.stage)["target_site"]
        self._aim_spec_value = (
            _aim_spec(self.config) if self.contract.algorithm == "aim" else None
        )
        self._divprune_spec_value = (
            _divprune_spec(self.config)
            if self.contract.algorithm == "divprune"
            else None
        )
        self._mmtok_spec_value = (
            _mmtok_spec(self.config)
            if self.contract.algorithm == "mmtok"
            else None
        )
        self._vispruner_spec_value = (
            _vispruner_spec(self.config)
            if self.contract.algorithm == "vispruner"
            else None
        )
        self._zoo_prune_spec_value = (
            _zoo_prune_spec(self.config)
            if self.contract.algorithm == "zoo_prune"
            else None
        )
        self._nuwa_spec_value = (
            _nuwa_spec(self.config) if self.contract.algorithm == "nuwa" else None
        )
        if self._nuwa_spec_value is not None:
            try:
                self.decoder_selection = NuwaDecoderMethod(
                    stage=self.stage,
                    spec=self._nuwa_spec_value,
                    rules=self._rules,
                    backbone="qwen25_vl",
                )
            except (TypeError, ValueError) as exc:
                raise BackendIntegrationError(f"Invalid Nüwa config: {exc}") from exc
            self.application_position = self.decoder_selection.positions[0]
            self.decoder_target_site = self.decoder_selection.target_site
        self._restore_spec_value = (
            _restore_spec(self.config)
            if self.contract.algorithm == "restore"
            else None
        )
        self._zoo_proxy_cache = ZOOProjectorProxyCache()
        if self.decoder_selection is None:
            factories = {"fastv": FastVDecoderMethod, "pyramid_drop": PyramidDecoderMethod,
                         "btp": BTPDecoderMethod, "halfv": HalfVDecoderMethod,
                         "transprune": TransPruneDecoderMethod,
                         "dart": DARTDecoderMethod, "d2pruner": D2PrunerDecoderMethod,
                         "feather": FEATHERDecoderMethod, "ivc_prune": IVCPruneDecoderMethod,
                         "finepruner": FinePrunerDecoderMethod, "pact": PACTDecoderMethod, "sparsevlm": SparseVLMDecoderMethod}
            factory = factories.get(self.contract.algorithm)
            if factory is not None:
                self.decoder_selection = factory(stage=self.stage, config=self.config,
                    rules=self._rules, backbone="qwen25_vl")
        self._fastv_spec_value = self.decoder_selection.spec if self.contract.algorithm == "fastv" else None
        if self._fastv_spec_value is not None:
            self._fastv_rules = self._rules
            self.application_position = self.decoder_selection.positions[0]
            self.target_site = self.decoder_selection.target_site
        self._decoder_attention_override = (
            DecoderAttentionImplementationOverride(self.model, require_configs=False)
            if getattr(self.decoder_selection, "requires_eager_attention", False)
            else None
        )
        self._sgl_spec_value = (
            _sgl_spec(self.config) if self.contract.algorithm == "sgl" else None
        )
        self._sgl_guide_value = (
            SGLGuideController(
                rules=self._rules, backbone_name="qwen25_vl",
                target_runtime=self.runtime,
                spec=self._sgl_spec_value,
            )
            if self._sgl_spec_value is not None
            else None
        )
        if self._sgl_spec_value is not None and (
            self.execution_phase is not ExecutionPhase.GENERATE
        ):
            raise BackendIntegrationError(
                "Qwen2.5-VL SGL supports image generation only."
            )
        self._btp_spec_value = (
            self.decoder_selection.spec if self.contract.algorithm == "btp" else None
        )
        if self._btp_spec_value is not None:
            self.application_positions = self.decoder_selection.positions
            self.target_site = self.decoder_selection.target_site
        self._halfv_spec_value = (
            self.decoder_selection.spec
            if self.contract.algorithm == "halfv"
            else None
        )
        if self._halfv_spec_value is not None:
            self.application_positions = self.decoder_selection.positions
            self.target_site = self.decoder_selection.target_site
        self._transprune_spec_value = (
            self.decoder_selection.spec
            if self.contract.algorithm == "transprune"
            else None
        )
        if self._transprune_spec_value is not None:
            self.application_positions = self.decoder_selection.positions
            self.target_site = self.decoder_selection.target_site
        self._prumerge_spec_value = (
            _prumerge_spec(self.config)
            if self.contract.algorithm == "prumerge"
            else None
        )
        self._fixed_visual_merge_spec_value = (
            FixedVisualMergeSpec(
                partition=(
                    "feature_anchor"
                    if self.contract.algorithm == "feature_anchor_mean"
                    else "grid"
                ),
                target_count=int(self.config.get("target_count", 64)),
            )
            if self.contract.algorithm in {"feature_anchor_mean", "grid_mean"}
            else None
        )
        self._stage2_projector = None
        if (
            self.contract.algorithm == "feature_anchor_mean"
            and self.config.get("stage2_variant") is not None
        ):
            text_config = getattr(getattr(self.model, "config", None), "text_config", None)
            hidden_size = getattr(text_config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
            if not isinstance(hidden_size, int) or isinstance(hidden_size, bool):
                raise BackendIntegrationError(
                    "Stage-2 feature-anchor projector could not resolve the LLM hidden size."
                )
            spec = Stage2ProjectorSpec(
                hidden_size=hidden_size,
                variant=str(self.config["stage2_variant"]),
                bottleneck_dim=int(self.config.get("bottleneck_dim", 64)),
            )
            name = "mllm_comp_feature_anchor_projector"
            existing = getattr(self.model, name, None)
            if existing is not None:
                if getattr(existing, "spec", None) != spec:
                    raise BackendIntegrationError(
                        "The model already owns an incompatible Stage-2 projector."
                    )
                module = existing
            else:
                module = create_stage2_projector(spec)
                checkpoint = self.config.get("projector_checkpoint")
                if checkpoint:
                    try:
                        load_stage2_projector(module, checkpoint)
                    except (FileNotFoundError, RuntimeError, ValueError) as exc:
                        raise BackendIntegrationError(
                            f"Could not load Stage-2 projector checkpoint: {exc}"
                        ) from exc
                try:
                    parameter = next(self.model.parameters())
                except (AttributeError, StopIteration) as exc:
                    raise BackendIntegrationError(
                        "Stage-2 projector requires a parameterized Qwen2.5-VL model."
                    ) from exc
                module.to(device=parameter.device, dtype=parameter.dtype)
                self.model.add_module(name, module)
            trainable = bool(self.config.get("trainable_projector", False))
            module.train(trainable)
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
            self._stage2_projector = module
        self._gap_spec_value = (
            _gap_spec(self.config) if self.contract.algorithm == "gap" else None
        )
        self._bridge: Any | None = None
        self._bridge_handle: Any | None = None
        self._owns_provider_bridge = False
        self._installed = False
        self._invocation_count = 0
        self._layout_delta_summaries: list[dict[str, Any]] = []
        self._layout_delta_records: list[dict[str, Any]] = []

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        if self._installed:
            return
        prepare_decoder_selection(self.decoder_selection, self.runtime)
        bridge, owns_provider_bridge = _resolve_or_install_provider_bridge(
            self.model,
            self.runtime,
        )
        install = getattr(bridge, "install_image_compression", None)
        uninstall = getattr(bridge, "uninstall_image_compression", None)
        if not callable(install) or not callable(uninstall):
            raise BackendIntegrationError(
                "Qwen2.5-VL image compression bridge must implement "
                "install_image_compression(executor) and "
                "uninstall_image_compression(handle)."
            )
        try:
            if self._decoder_attention_override is not None:
                self._decoder_attention_override.install()
            handle = install(self)
            if handle is None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL image compression bridge returned an empty handle."
                )
            if self._sgl_guide_value is not None:
                self._sgl_guide_value.attach()
        except BaseException:
            if self._decoder_attention_override is not None:
                self._decoder_attention_override.restore()
            if self._sgl_guide_value is not None:
                self._sgl_guide_value.detach()
            if "handle" in locals() and handle is not None:
                try:
                    uninstall(handle)
                except Exception:
                    pass
            if owns_provider_bridge:
                _uninstall_owned_provider_bridge(self.model)
            raise
        self._bridge = bridge
        self._bridge_handle = handle
        self._owns_provider_bridge = owns_provider_bridge
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            self._zoo_proxy_cache.clear()
            if self._decoder_attention_override is not None:
                self._decoder_attention_override.restore()
            if self._sgl_guide_value is not None:
                self._sgl_guide_value.detach()
            return
        assert self._bridge is not None
        owns_provider_bridge = self._owns_provider_bridge
        try:
            self._bridge.uninstall_image_compression(self._bridge_handle)
        finally:
            self._bridge = None
            self._bridge_handle = None
            self._owns_provider_bridge = False
            self._installed = False
            self._zoo_proxy_cache.clear()
            if self._decoder_attention_override is not None:
                self._decoder_attention_override.restore()
            if self._sgl_guide_value is not None:
                self._sgl_guide_value.detach()
        if owns_provider_bridge:
            _uninstall_owned_provider_bridge(self.model)

    def encoder_dense_boundary(
        self,
        hidden_states: Any,
        keys: Any,
        **kwargs: Any,
    ) -> Qwen25VLEncoderCompressionResult:
        self._require_callback("encoder_dense_boundary")
        call = {
            key: value
            for key, value in {**self.config, **kwargs}.items()
            if key in _TOME_KERNEL_CONFIG_KEYS
        }
        result = tome_merge_image(hidden_states, keys, rules={name.removeprefix("encoder_"): rule for name, rule in self._rules.items() if name.startswith("encoder_")}, **call)
        delta = _single_layout_delta(result)
        self._record_layout_delta(
            delta, operation=tome_encoder_operation(self.stage, "qwen25_vl")["target_site"],
            boundary_layer=kwargs.get("boundary_layer"),
            score_layer=tome_encoder_score_layer(self.stage, "qwen25_vl") if "score_layer" in (getattr(self.stage, "criterion", {}) or {}) else kwargs.get("boundary_layer"),
            **({"application_position": kwargs["application_position"]} if "application_position" in kwargs else {}),
        )
        return Qwen25VLEncoderCompressionResult(
            kernel_result=result,
            dense_restore=Qwen25VLDenseRestorePlan(delta),
        )

    def encoder_progressive_prune(
        self,
        hidden_states: Any,
        keys: Any,
        *,
        boundary_layer: int,
        image_index: int,
    ) -> Any:
        self._require_callback("encoder_progressive_prune")
        if boundary_layer not in self._saint_layer_indices_value:
            raise BackendIntegrationError(
                "SAINT callback ran outside its configured encoder schedule."
            )
        stage_index = self._saint_layer_indices_value.index(boundary_layer)
        point = self.application_positions[stage_index]
        position_layout = ({"attention_visual_layer": point.index} if point.boundary == "post_attention_pre_ffn"
            else {"before_visual_layer": 0} if point.index == 0 else {"after_visual_layer": point.index - 1})
        assert self._saint_spec_value is not None
        spec = SAINTSpec(
            keep_ratio=self._saint_keep_ratios_value[stage_index],
            prune_level=self._saint_spec_value.prune_level,
            group_size=self._saint_spec_value.group_size,
            similarity_threshold=self._saint_spec_value.similarity_threshold,
            neighbour_count=self._saint_spec_value.neighbour_count,
            gamma=self._saint_spec_value.gamma,
            max_iterations=self._saint_spec_value.max_iterations,
        )
        result = saint_image_tokens(
            hidden_states,
            keys,
            rules=self._rules, spec=spec,
            media_unit="image_patch",
            source_layout={
                "backbone": "qwen25_vl",
                "image_index": image_index,
                **position_layout,
                "application_position": point.as_dict(),
                "key_projection_layer": boundary_layer,
                "target_modality": Modality.IMAGE.value,
            },
            result_layout={
                "backbone": "qwen25_vl",
                "image_index": image_index,
                "target_site": self.target_site,
                "application_position": point.as_dict(),
                "key_projection_layer": boundary_layer,
                "position_policy": "selected_patch_rotary",
            },
        )
        self._record_layout_delta(
            result.layout_delta,
            operation="encoder_progressive_prune",
            boundary_layer=boundary_layer,
            stage_index=stage_index,
            application_position=point.as_dict(),
            key_projection_layer=boundary_layer,
            image_index=image_index,
        )
        return result

    def compact_visual_output(
        self,
        hidden_states: Any,
        keys: Any,
        *,
        protected_indices: Sequence[int] = (),
        coordinates: Sequence[Any] = (),
    ) -> Any:
        """Permanently merge complete image groups at the selected merger side.

        Encoder-local merges are restored for window attention and complete
        native 2x2 groups. The output merge consumes either those raw groups
        or the native merger's projected features.
        This second paper-consistent ToMe step is the layout legalizer that
        makes the final image-feature sequence genuinely compact for the LLM.
        """

        self._require_callback("encoder_dense_boundary")
        call = {
            key: value
            for key, value in self.config.items()
            if key in {"eps", "clamp_r"}
        }
        call.update(
            {
                "r": self.config.get("output_r", self.config.get("r", 1)),
                "protected_indices": tuple(protected_indices),
                "coordinates": tuple(coordinates),
                "media_unit": "image_feature",
            }
        )
        result = tome_preserve_source_order(
            tome_merge_image(hidden_states, keys.to(hidden_states.device), rules={name.removeprefix("output_"): rule for name, rule in self._rules.items() if name.startswith("output_")}, **call)
        )
        delta = _single_layout_delta(result)
        before = self.output_target_site == "pre_vision_merger"
        delta = replace(delta,
            source_layout={**dict(delta.source_layout), "key_source": "final_visual_group_keys"},
            result_layout={**dict(delta.result_layout), "target_site": self.output_target_site,
                "projector_position": "before" if before else "after",
                "projector_input_group_count": delta.result_count if before else delta.source_count})
        result = replace(result, layout_deltas=(delta,))
        self._record_layout_delta(delta, operation="permanent_visual_output",
            target_site=self.output_target_site, projector_position="before" if before else "after")
        return result

    def record_projector_generate(self, state: Mapping[str, Any]) -> None:
        summary = self._projector_generate_summary
        summary["generate_calls"] += 1
        verified = state.get("decode_position_verified")
        if verified is True:
            summary["decode_position_verified"] += 1
        elif verified is False:
            summary["decode_position_failed"] += 1
        else:
            summary["no_decode_step"] += 1
        summary["prompt_removed_total"] += int(state.get("prompt_removed") or 0)
        summary["last"] = dict(state)

    def mmtok_prompt_embeddings(self, input_ids: Any, attention_mask: Any = None) -> Any:
        """Released single-turn Qwen text path: tokenize 'Question: <user text>'."""
        from mllm_comp.compressors.comparison.eadp import extract_qwen2_user_instruction_ids
        ids = input_ids[0].clone()
        core = self.model.model
        config = core.config
        for name in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
            value = getattr(config, name, None)
            if value is not None:
                ids[ids == value] = -200
        # Synthetic fixtures deliberately use a non-ChatML vocabulary. Real
        # ChatML never falls back to full serialized system/assistant text.
        if not bool((ids == 151644).any()):
            valid = ids >= 0
            if attention_mask is not None:
                valid &= attention_mask[0].bool()
            self._mmtok_text_scope = "non_chatml_input_embeddings"
            return core.get_input_embeddings()(ids[valid])
        user = extract_qwen2_user_instruction_ids(ids,
            attention_mask=attention_mask[0] if attention_mask is not None else None)
        tokenizer = self._bifocal_tokenizer()
        text = tokenizer.decode(user.tolist(), skip_special_tokens=True).strip()
        encoded = tokenizer(f"Question: {text}".split(), is_split_into_words=True,
            return_tensors="pt", padding=True, truncation=True)["input_ids"].to(ids.device)
        if encoded.shape[1] > 1 and int(encoded[0, 0]) == getattr(tokenizer, "bos_token_id", None):
            encoded = encoded[:, 1:]
        if encoded.numel() == 0:
            raise BackendIntegrationError("MMTok found no user-question tokens")
        self._mmtok_text_scope = "official_question_prefix_last_user_no_answer"
        return core.get_input_embeddings()(encoded[0])

    def _bifocal_tokenizer(self) -> Any:
        cached = getattr(self, "_bifocal_tokenizer_value", None)
        if cached is not None:
            return cached
        processor = getattr(self.runtime, "processor", None)
        tokenizer = getattr(processor, "tokenizer", processor)
        if tokenizer is None or not callable(getattr(tokenizer, "decode", None)):
            config = getattr(self.model, "config", None)
            source = getattr(config, "_name_or_path", None) or getattr(
                self.runtime, "model_path_or_id", None
            )
            if not source:
                raise BackendIntegrationError(
                    "Qwen2.5-VL Bifocal needs a tokenizer to read the query: "
                    "runtime.processor is missing and the model has no _name_or_path."
                )
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    str(source), local_files_only=True
                )
            except Exception as exc:  # noqa: BLE001
                raise BackendIntegrationError(
                    f"Qwen2.5-VL Bifocal could not load the tokenizer from {source!r}: {exc}"
                ) from exc
        self._bifocal_tokenizer_value = tokenizer
        return tokenizer

    def _bifocal_query(
        self,
        prompt_input_ids: Any | None,
        *,
        device: Any,
    ) -> tuple[str | None, Any | None]:
        if prompt_input_ids is None:
            return None, None
        tokenizer = self._bifocal_tokenizer()
        config = getattr(self.model, "config", None)
        vision_end = getattr(config, "vision_end_token_id", None)
        if vision_end is None:
            vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        ids = prompt_input_ids
        if getattr(ids, "ndim", 1) == 2:
            if int(ids.shape[0]) != 1:
                raise BackendIntegrationError("Qwen2.5-VL Bifocal requires batch size one.")
            ids = ids[0]
        text = bifocal_extract_query(
            bifocal_query_text_from_ids(ids, tokenizer, int(vision_end), int(im_end))
        )
        if not text:
            return text, None
        embedding = self.model.get_input_embeddings()
        spec = self._bifocal_spec_value
        assert spec is not None
        word_embeddings = bifocal_word_embeddings(
            tokenizer,
            embedding,
            embedding.weight.device,
            text,
            mode=spec.word_space,
        )
        if word_embeddings is not None:
            word_embeddings = word_embeddings.to(device)
        return text, word_embeddings

    def post_visual_merger(
        self,
        tokens: Any,
        attention: Any,
        keys: Any,
        *,
        placeholder_indices: Sequence[int],
        source_sequence_length: int,
        image_index: int,
        vision_reference: Any | None = None,
        projected_reference: Any | None = None,
        text_embeddings: Any | None = None,
        projector_inputs: Any | None = None,
        projector_layers: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Qwen25VLProjectorCompressionResult:
        self._require_callback("post_visual_merger")
        if (
            isinstance(image_index, bool)
            or not isinstance(image_index, int)
            or image_index < 0
        ):
            raise BackendIntegrationError("image_index must be a non-negative integer.")
        call = {**self.config, **kwargs}
        # Both projector positions operate on complete 2x2 spatial groups,
        # so one layout token always corresponds to one prompt placeholder.
        call.setdefault("media_unit", "image_feature")
        score_tokens = tokens if projected_reference is None else projected_reference
        if projected_reference is not None and (getattr(tokens, "ndim", None) != 2
                or not tokens.is_floating_point() or tokens.shape[0] != projected_reference.shape[0]):
            raise BackendIntegrationError("Projected scoring references must align with floating rank-2 projector values.")
        if self.contract.algorithm == "visionzip":
            call["source_layout"] = {"backbone": "qwen25_vl", "image_index": image_index,
                "source_sequence_length": source_sequence_length, **dict(call.get("source_layout") or {}),
                "target_site": self.target_site}
            call["result_layout"] = {"backbone": "qwen25_vl", "image_index": image_index,
                **dict(call.get("result_layout") or {}), "target_site": self.target_site}
            result = visionzip_image(tokens, attention, keys, rules=self._rules, **call)
        elif self.contract.algorithm in {"feature_anchor_mean", "grid_mean"}:
            media_unit = call.pop("media_unit")
            feature_grid = call.pop("feature_grid", None)
            for name in (
                "target_count",
                "stage2_variant",
                "bottleneck_dim",
                "projector_checkpoint",
                "trainable_projector",
                "attention_mass_correction",
            ):
                call.pop(name, None)
            if call:
                raise BackendIntegrationError(
                    "Fixed visual merge callback-time config overrides are unsupported."
                )
            if (
                not isinstance(feature_grid, Sequence)
                or len(feature_grid) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    for value in feature_grid
                )
            ):
                raise BackendIntegrationError(
                    "Fixed visual merge requires an exact post-merger (height, width)."
                )
            assert self._fixed_visual_merge_spec_value is not None
            try:
                result = fixed_visual_merge(
                    tokens,
                    grid_height=int(feature_grid[0]),
                    grid_width=int(feature_grid[1]),
                    spec=self._fixed_visual_merge_spec_value,
                    media_unit=media_unit,
                    source_layout={
                        "backbone": "qwen25_vl",
                        "image_index": image_index,
                        "source_sequence_length": source_sequence_length,
                        "target_modality": Modality.IMAGE.value,
                    },
                    result_layout={
                        "backbone": "qwen25_vl",
                        "image_index": image_index,
                        "target_site": self.target_site,
                    },
                    projector=self._stage2_projector,
                )
            except ValueError as exc:
                raise BackendIntegrationError(
                    f"Fixed visual merge failed: {exc}"
                ) from exc
        elif self.contract.algorithm == "prumerge":
            media_unit = call.pop("media_unit")
            for name in (
                "variant",
                "adaptive",
                "reduction_ratio",
                "target_count",
                "iqr_multiplier",
                "min_centers",
                "cluster_size",
            ):
                call.pop(name, None)
            if call:
                raise BackendIntegrationError(
                    "PruMerge callback-time config overrides are unsupported."
                )
            assert self._prumerge_spec_value is not None
            result = prumerge_image_tokens(
                tokens,
                attention,
                keys,
                rules=self._rules, spec=self._prumerge_spec_value,
                media_unit=media_unit,
                source_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "source_sequence_length": source_sequence_length,
                    "importance_source": "final_visual_mean_received",
                },
                result_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "target_site": self.target_site,
                    "position_policy": "native_sparse_mrope_representatives",
                },
            )
        elif self.contract.algorithm == "gap":
            media_unit = call.pop("media_unit")
            call.pop("retain_ratio", None)
            call.pop("target_count", None)
            if call:
                raise BackendIntegrationError(
                    "GAP callback-time config overrides are unsupported."
                )
            assert self._gap_spec_value is not None
            result = gap_image_tokens(
                tokens,
                attention,
                rules=self._rules, spec=self._gap_spec_value,
                media_unit=media_unit,
                source_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "source_sequence_length": source_sequence_length,
                    "importance_source": "final_visual_mean_received",
                },
                result_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "target_site": self.target_site,
                    "position_policy": "preserve_original",
                },
            )
        elif self.contract.algorithm == "aim":
            media_unit = call.pop("media_unit")
            if kwargs:
                raise BackendIntegrationError(
                    "AIM callback-time config overrides are unsupported."
                )
            assert self._aim_spec_value is not None
            result = aim_image_tokens(
                tokens,
                rules=self._rules, spec=self._aim_spec_value,
                media_unit=media_unit,
                source_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "source_sequence_length": source_sequence_length,
                    "target_modality": Modality.IMAGE.value,
                },
                result_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "target_site": self.target_site,
                    "position_policy": "sparse_mrope_rebuild",
                },
            )
        elif self.contract.algorithm == "divprune":
            media_unit = call.pop("media_unit")
            if kwargs:
                raise BackendIntegrationError(
                    "DivPrune callback-time config overrides are unsupported."
                )
            assert self._divprune_spec_value is not None
            result = divprune_image(
                tokens,
                rules=self._rules, spec=self._divprune_spec_value,
                media_unit=media_unit,
                source_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "source_sequence_length": source_sequence_length,
                    "target_modality": Modality.IMAGE.value,
                },
                result_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "target_site": self.target_site,
                    "position_policy": "sparse_mrope_rebuild",
                },
            )
        elif self.contract.algorithm == "mmtok":
            media_unit = call.pop("media_unit")
            if kwargs:
                raise BackendIntegrationError(
                    "MMTok callback-time config overrides are unsupported."
                )
            if vision_reference is None or text_embeddings is None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL MMTok requires pre-merger visual features and "
                    "prompt input embeddings."
                )
            assert self._mmtok_spec_value is not None
            result = mmtok_image_tokens(
                score_tokens,
                # Official Qwen MMTok averages the four pre-merger patches;
                # concatenation defines a different visual coverage geometry.
                vision_reference.reshape(vision_reference.shape[0],
                    int(self.model.model.visual.spatial_merge_size) ** 2, -1).mean(dim=1).to(device=score_tokens.device),
                text_embeddings.to(device=score_tokens.device),
                rules=self._rules, spec=self._mmtok_spec_value,
                media_unit=media_unit,
                source_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "source_sequence_length": source_sequence_length,
                    "projected_source": "complete_native_merger_output",
                    "target_modality": Modality.IMAGE.value,
                    "vision_source": "pre_merger_2x2_mean_features",
                    "text_source": getattr(self, "_mmtok_text_scope", "explicit_text_embeddings"),
                },
                result_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "target_site": self.target_site,
                    "dense_projected_reference_required": projected_reference is not None,
                    "position_policy": "sparse_mrope_rebuild",
                },
            )
            if projected_reference is not None:
                result = replace(result, tokens=tokens.index_select(0, result.indices.to(tokens.device)))
        elif self.contract.algorithm == "vispruner":
            media_unit = call.pop("media_unit")
            if kwargs:
                raise BackendIntegrationError(
                    "VisPruner callback-time config overrides are unsupported."
                )
            if attention is None or vision_reference is None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL VisPruner requires final visual attention and "
                    "pre-merger visual features."
                )
            assert self._vispruner_spec_value is not None
            result = vispruner_image_tokens(
                tokens,
                vision_reference.to(device=tokens.device),
                attention.to(device=tokens.device),
                rules=self._rules, spec=self._vispruner_spec_value,
                media_unit=media_unit,
                source_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "source_sequence_length": source_sequence_length,
                    "target_modality": Modality.IMAGE.value,
                    "attention_source": "final_visual_mean_received",
                    "selection_feature_source": "pre_merger_visual_features",
                },
                result_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "target_site": self.target_site,
                    "position_policy": "sparse_mrope_rebuild",
                },
            )
        elif self.contract.algorithm == "zoo_prune":
            media_unit = call.pop("media_unit")
            if kwargs:
                raise BackendIntegrationError(
                    "ZOO-Prune callback-time config overrides are unsupported."
                )
            if projector_inputs is None or not projector_layers:
                raise BackendIntegrationError(
                    "Qwen2.5-VL ZOO-Prune requires native merger MLP inputs "
                    "and its exact projection layers."
                )
            assert self._zoo_prune_spec_value is not None
            if projected_reference is not None:
                score_tokens = score_tokens.to(device=projector_inputs.device)
            result = zoo_prune_image_tokens(
                score_tokens,
                projector_inputs,
                projector_layers,
                rules=self._rules, spec=self._zoo_prune_spec_value,
                proxy_cache=self._zoo_proxy_cache,
                media_unit=media_unit,
                source_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "source_sequence_length": source_sequence_length,
                    "projected_source": "complete_native_merger_output",
                    "target_modality": Modality.IMAGE.value,
                    "projector_input": "native_merger_mlp_input",
                    "selection_scope": "per_image",
                },
                result_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "target_site": self.target_site,
                    "dense_projected_reference_required": projected_reference is not None,
                    "position_policy": "sparse_mrope_rebuild",
                },
            )
            if projected_reference is not None:
                result = replace(result, tokens=tokens.index_select(0, result.indices.to(tokens.device)))
        elif self.contract.algorithm == "nuwa":
            if set(kwargs) - {"feature_grid"}:
                raise BackendIntegrationError(
                    "Nüwa callback-time config overrides are unsupported."
                )
            feature_grid = call.pop("feature_grid", None)
            if (
                not isinstance(feature_grid, Sequence)
                or len(feature_grid) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 1
                    for value in feature_grid
                )
            ):
                raise BackendIntegrationError(
                    "Qwen2.5-VL Nüwa requires an exact post-merger (height, width)."
                )
            if vision_reference is None or attention is None or keys is None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL Nüwa requires semantic visual features, attention, "
                    "and keys."
                )
            assert self._nuwa_spec_value is not None
            result = nuwa_stage1_image_tokens(
                tokens,
                vision_reference,
                attention,
                keys,
                rules=self._rules, grid_height=int(feature_grid[0]),
                grid_width=int(feature_grid[1]),
                spec=self._nuwa_spec_value,
                media_unit=call.pop("media_unit"),
                source_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "source_sequence_length": source_sequence_length,
                    "target_modality": Modality.IMAGE.value,
                    "stage": "stage1",
                    "semantic_source": "native_visual_hidden_range",
                },
                result_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "target_site": self.target_site,
                    "position_policy": "native_sparse_mrope_rpme",
                },
            )
        elif self.contract.algorithm == "bifocal":
            unexpected = set(kwargs) - {
                "feature_grid",
                "vision_maps",
                "prompt_input_ids",
            }
            if unexpected:
                raise BackendIntegrationError(
                    f"Bifocal callback-time overrides are unsupported: {sorted(unexpected)}."
                )
            feature_grid = kwargs.get("feature_grid")
            if (
                not isinstance(feature_grid, Sequence)
                or len(feature_grid) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 1
                    for value in feature_grid
                )
            ):
                raise BackendIntegrationError(
                    "Qwen2.5-VL Bifocal requires an exact post-merger (height, width)."
                )
            spec = self._bifocal_spec_value
            assert spec is not None
            vision_maps = kwargs.get("vision_maps")
            if spec.needs_vit_maps and vision_maps is None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL Bifocal metric=vit_attn requires the ViT column-mass "
                    f"maps of block {spec.vit_block}; the provider bridge did not "
                    "capture them."
                )
            query_text, word_embeddings = self._bifocal_query(
                kwargs.get("prompt_input_ids"), device=tokens.device
            )
            try:
                result = bifocal_image_tokens(
                    tokens,
                    vision_maps,
                    grid_height=int(feature_grid[0]),
                    grid_width=int(feature_grid[1]),
                    word_embeddings=word_embeddings,
                    spec=spec,
                    media_unit=call.pop("media_unit"),
                    source_layout={
                        "backbone": "qwen25_vl",
                        "image_index": image_index,
                        "source_sequence_length": source_sequence_length,
                        "target_modality": Modality.IMAGE.value,
                        "stage": "bifocal_stage1",
                    },
                    result_layout={
                        "backbone": "qwen25_vl",
                        "image_index": image_index,
                        "target_site": self.target_site,
                    },
                )
            except ValueError as exc:
                raise BackendIntegrationError(f"Bifocal Stage-1 failed: {exc}") from exc
            self._bifocal_last_query = query_text
        elif self.contract.algorithm == "restore":
            media_unit = call.pop("media_unit")
            if kwargs:
                raise BackendIntegrationError(
                    "RESTORE callback-time config overrides are unsupported."
                )
            if attention is None or vision_reference is None:
                raise BackendIntegrationError(
                    "Qwen2.5-VL RESTORE requires final visual attention and "
                    "pre-merger visual features."
                )
            assert self._restore_spec_value is not None
            result = restore_image_tokens(
                tokens,
                vision_reference,
                attention,
                rules=self._rules, spec=self._restore_spec_value,
                media_unit=media_unit,
                source_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "source_sequence_length": source_sequence_length,
                    "target_modality": Modality.IMAGE.value,
                    "selection_feature_source": "pre_merger_visual_features",
                    "attention_source": "final_visual_mean_received",
                },
                result_layout={
                    "backbone": "qwen25_vl",
                    "image_index": image_index,
                    "target_site": self.target_site,
                    "position_policy": "native_sparse_mrope_representatives",
                },
            )
        else:  # guarded by the exact binding contract table
            raise BackendIntegrationError(
                f"Unsupported post-visual-merger algorithm {self.contract.algorithm!r}."
            )
        selected_indices = getattr(
            result, "indices", result.layout_delta.kept_indices
        )
        if hasattr(selected_indices, "tolist"):
            selected_indices = selected_indices.tolist()
        self._record_layout_delta(
            result.layout_delta,
            operation="post_visual_merger",
            **(
                {
                    "planned_count": result.metadata.get(
                        "effective_target_count",
                        result.metadata.get(
                            "requested_tokens",
                            result.metadata.get(
                                "target_count",
                                self.config.get("target_count"),
                            ),
                        ),
                    ),
                    "selected_indices": [int(value) for value in selected_indices],
                    "image_index": image_index,
                }
                if self.contract.algorithm
                in {
                    "feature_anchor_mean",
                    "grid_mean",
                    "prumerge",
                    "gap",
                    "aim",
                    "divprune",
                    "mmtok",
                    "vispruner",
                    "zoo_prune",
                    "nuwa",
                    "restore",
                    "bifocal",
                }
                else {}
            ),
            **(
                {
                    "bifocal_query": (self._bifocal_last_query or "")[:200],
                    "bifocal_representatives": [
                        int(value) for value in result.indices.tolist()
                    ],
                    "bifocal_chain": result.metadata.get("chain"),
                    "bifocal_passthrough": result.metadata.get("passthrough"),
                    "bifocal_block": result.metadata.get("block"),
                    "bifocal_coarse_tokens": result.metadata.get("coarse_tokens"),
                    "bifocal_k_seats": result.metadata.get("k_seats"),
                    "bifocal_respool_fallback": result.metadata.get("respool_fallback"),
                    "bifocal_near_merged": result.metadata.get("near_merged"),
                }
                if self.contract.algorithm == "bifocal"
                else {}
            ),
            **(
                {
                    "prumerge_variant": result.metadata.get("variant"),
                    "prumerge_center_indices": list(
                        result.metadata.get("center_indices", ())
                    ),
                    "prumerge_neighbour_indices": [
                        list(row)
                        for row in result.metadata.get("neighbour_indices", ())
                    ],
                    "prumerge_global_count": result.metadata.get("global_count"),
                    "prumerge_cluster_size": result.metadata.get("cluster_size"),
                }
                if self.contract.algorithm == "prumerge"
                else {}
            ),
            **(
                {
                    "partition_hash": result.metadata.get("partition_hash"),
                    "cluster_sizes": list(result.metadata.get("cluster_sizes", ())),
                    "cluster_members": [
                        list(row)
                        for row in result.metadata.get("cluster_members", ())
                    ],
                    "grid_centroids": [
                        list(row)
                        for row in result.metadata.get("grid_centroids", ())
                    ],
                    "position_policy": result.metadata.get("position_policy"),
                }
                if self.contract.algorithm
                in {"feature_anchor_mean", "grid_mean"}
                else {}
            ),
        )
        return Qwen25VLProjectorCompressionResult(
            kernel_result=result,
            prompt_repair=Qwen25VLPlaceholderRepairPlan(
                layout_delta=result.layout_delta,
                placeholder_indices=tuple(placeholder_indices),
                source_sequence_length=source_sequence_length,
            ),
        )

    @property
    def pyramid_rule_layers(self):
        return self.decoder_selection.schedule.layer_indices

    def apply_decoder_selection(self, selection, point, hidden_states, *, image_indices):
        result = execute_decoder_selection(selection, point, hidden_states, image_indices=image_indices,
            media_unit="llm_token",
            source_layout={"backbone": "qwen25_vl", "application_position": point.as_dict()},
            result_layout={"backbone": "qwen25_vl", "application_position": point.as_dict(),
                "target_site": self.decoder_selection.target_site,
                "position_policy": self.decoder_selection.position_policy})
        self._record_layout_delta(result.layout_delta, operation="decoder_layer_loop",
            boundary_layer=point.index, application_position=point.as_dict())
        return Qwen25VLDecoderCompressionResult(kernel_result=result,
            decoder_repair=Qwen25VLDecoderRepairPlan(layout_delta=result.layout_delta,
                boundary_layer=point.index, allow_explicit_reorder=True))

    def configure_sgl_decoder_depth(self, num_layers: int) -> int:
        self.decoder_selection = SGLDecoderMethod(
            stage=self.stage,
            spec=self._sgl_spec_value,
            rules=self._rules,
            consume_scores=self._sgl_guide_value.consume_scores,
            num_layers=num_layers,
        )
        self.application_position = self.decoder_selection.positions[0]
        self.target_site = self.decoder_selection.target_site
        return self.application_position.index

    def runtime_metadata(self) -> Mapping[str, Any]:
        metadata = {
            "backbone": "qwen25_vl",
            **({"rules": self._rule_specs, "rules_changed": True, "canonical_evidence_applicable": False, "development_runtime": True} if self._rules else {}),
            "binding": self.contract.binding,
            "algorithm": self.contract.algorithm,
            "variant": self.contract.variant,
            "callback": self.contract.callback,
            "target_modality": Modality.IMAGE.value,
            "supports_video": False,
            "target_sites": list(self.contract.target_sites),
            "repairs": list(self.contract.repairs),
            "installed": self._installed,
            "invocation_count": self._invocation_count,
            "latest_layout_delta": (
                dict(self._layout_delta_summaries[-1])
                if self._layout_delta_summaries
                else None
            ),
            "layout_delta_summaries": [
                dict(item) for item in self._layout_delta_summaries
            ],
            "layout_delta_records": [dict(item) for item in self._layout_delta_records],
            "checkpoint_evidence": "not_claimed",
        }
        if self.contract.algorithm in {"visionzip", "feature_anchor_mean", "grid_mean", "prumerge", "gap", "divprune", "aim", "vispruner", "mmtok", "zoo_prune"}:
            metadata.update(target_sites=list(self.projector_target_sites), projector_target_sites=list(self.projector_target_sites))
        if self.contract.algorithm in {
            "feature_anchor_mean",
            "grid_mean",
            "prumerge",
            "bifocal",
            "visionzip",
            "divprune",
            "aim",
            "gap",
            "mmtok",
        }:
            metadata["projector_generate"] = {
                key: (dict(value) if isinstance(value, Mapping) else value)
                for key, value in self._projector_generate_summary.items()
            }
        if self.contract.algorithm in {"mmtok", "zoo_prune"}:
            metadata["dense_projected_reference_required"] = self.target_site == "pre_vision_merger"
        if self._bifocal_spec_value is not None:
            metadata.update(
                target_sites=list(self.projector_target_sites),
                projector_target_sites=list(self.projector_target_sites),
                bifocal_spec={
                    item.name: getattr(self._bifocal_spec_value, item.name)
                    for item in fields(self._bifocal_spec_value)
                },
                bifocal_generate={
                    key: (dict(value) if isinstance(value, Mapping) else value)
                    for key, value in self._bifocal_generate_summary.items()
                },
            )
        if self.contract.algorithm == "saint":
            metadata.update(target_sites=[self.target_site],
                application_positions=[point.as_dict() for point in self.application_positions],
                key_projection_layers=list(self._saint_layer_indices_value),
                visual_execution="manual_attention_and_mlp_without_block_checkpoint")
        if self.contract.algorithm == "tome":
            metadata.update(tome_position_metadata(self.stage, "qwen25_vl"))
            metadata["target_sites"] = [tome_encoder_operation(self.stage, "qwen25_vl")["target_site"],
                *self.contract.target_sites[1:-1], self.output_target_site]
            metadata["permanent_output_projector_position"] = "before" if self.output_target_site == "pre_vision_merger" else "after"
            if self.output_target_site == "pre_vision_merger":
                metadata["canonical_evidence_applicable"] = False
        if self.contract.algorithm == "pyramid_drop" and hasattr(self, "application_positions"):
            metadata.update(target_sites=[self.target_site], application_positions=[point.as_dict() for point in self.application_positions],
                score_layers=list(self.pyramid_rule_layers))
        if self._halfv_spec_value is not None:
            metadata.update(target_sites=[self.target_site], application_positions=[point.as_dict() for point in self.application_positions],
                score_layers=list(self._halfv_spec_value.boundary_after_layer_indices))
        if self._fastv_spec_value is not None:
            metadata.update(
                target_sites=[self.target_site], application_position=self.application_position.as_dict(),
                prune_layer=self._fastv_spec_value.prune_layer,
                score_layer=self._fastv_spec_value.prune_layer - 1 if self._fastv_spec_value.prune_layer else None,
                **({"rules": self._rule_specs, "rules_changed": True,
                    "canonical_evidence_applicable": False, "development_runtime": True} if self._fastv_rules else {}),
            )
        if self.contract.algorithm in {"nuwa", "restore"}:
            metadata.update(target_sites=[*self.projector_target_sites, *self.contract.target_sites[1:]],
                projector_target_sites=list(self.projector_target_sites))
        if self._nuwa_spec_value is not None:
            metadata.update(target_sites=[*self.projector_target_sites, self.decoder_target_site],
                application_position=self.application_position.as_dict(),
                decoder_target_site=self.decoder_target_site, score_layer=self._nuwa_spec_value.stage2_score_layer)
        if self._btp_spec_value is not None:
            metadata.update(target_sites=[self.target_site], application_positions=[point.as_dict() for point in self.application_positions],
                score_layers=list(self._btp_spec_value.boundary_after_layer_indices[:-1]))
        if self._sgl_guide_value is not None:
            metadata.update(self._sgl_guide_value.runtime_metadata())
            metadata["prune_layer"] = None
            if self.decoder_selection is not None:
                metadata.update(target_sites=[self.target_site], application_position=self.application_position.as_dict(),
                    prune_layer=self.application_position.index)
        if self._decoder_attention_override is not None:
            metadata["decoder_attention_override"] = (
                self._decoder_attention_override.metadata()
            )
        if self._rules:
            metadata["canonical_evidence_applicable"] = False
        if self.decoder_selection is not None:
            metadata.update(
                target_sites=[
                    *getattr(self, "projector_target_sites", ()),
                    self.decoder_selection.target_site,
                ],
                application_positions=[
                    point.as_dict() for point in self.decoder_selection.positions
                ],
                score_layers=list(self.decoder_selection.score_layers),
                cache_policy=self.decoder_selection.cache_policy.value,
                position_policy=self.decoder_selection.position_policy,
            )
        return metadata

    def clear_diagnostic_records(self) -> None:
        """Start a request without retaining previous diagnostic layouts."""

        self._layout_delta_records.clear()
        self._layout_delta_summaries.clear()

    def _record_layout_delta(
        self,
        delta: LayoutDelta,
        *,
        operation: str,
        **metadata: Any,
    ) -> None:
        if not isinstance(delta, LayoutDelta):
            raise InvariantError("Compression callback did not return LayoutDelta.")
        self._invocation_count += 1
        invocation_id = f"{self.contract.binding}:{self._invocation_count}"
        summary = {
            "source_count": delta.source_count,
            "result_count": delta.result_count,
            "media_unit": delta.media_unit,
            "operation": _layout_operation(delta),
            "site": operation,
            "invocation_id": invocation_id,
            **metadata,
        }
        self._layout_delta_summaries.append(summary)
        record = layout_delta_record(delta)
        record.update({"site": operation, "invocation_id": invocation_id, **metadata})
        self._layout_delta_records.append(record)

    def _require_callback(self, name: str) -> None:
        if self.contract.algorithm in {"nuwa", "restore"} and name in {
            "post_visual_merger",
            "decoder_layer_loop",
        }:
            return
        if name != self.contract.callback:
            raise BackendIntegrationError(
                f"Binding {self.contract.binding!r} exposes callback "
                f"{self.contract.callback!r}, not {name!r}."
            )


class Qwen25VLImageCompressionAdapter:
    """Resolve exact registry bindings into installable Qwen2.5-VL executors."""

    def resolve(
        self,
        *,
        model: Any,
        runtime: Any,
        stage: Any,
        binding: Any,
        binding_config: Any | None = None,
        execution_phase: ExecutionPhase | str,
        facade: Any | None = None,
        contract_projection: Any = None,
    ) -> Qwen25VLImageCompressionExecutor:
        del facade
        if contract_projection is not None and binding.strategy_factory is not None:
            contract = Qwen25VLImageBindingContract(
                binding=binding.name, algorithm=contract_projection.algorithm.name,
                variant=contract_projection.variant.name, callback="decoder_layer_loop",
                target_sites=tuple(target.target_site for target in binding.targets),
                repairs=("image_mask_position_compaction", "native_dynamic_cache_repair"),
                required_handles=tuple(sorted(binding.required_handles)),
            )
            return Qwen25VLImageCompressionExecutor(model=model, runtime=runtime, stage=stage,
                contract=contract, execution_phase=execution_phase, contract_projection=contract_projection,
                binding=binding, binding_config=binding_config)
        name = getattr(binding, "name", binding)
        try:
            contract = QWEN25_VL_IMAGE_BINDING_CONTRACTS[name]
        except (KeyError, TypeError) as exc:
            raise BackendIntegrationError(
                f"Unknown Qwen2.5-VL image compression binding {name!r}."
            ) from exc
        if getattr(binding, "position_resolver", None) is not None:
            try:
                binding = binding.position_resolver(stage=stage, binding=binding, binding_config=binding_config)
            except (TypeError, ValueError) as exc:
                raise BackendIntegrationError(f"Invalid image compression config: {exc}") from exc
        _validate_binding_and_stage(
            contract=contract,
            binding=binding,
            stage=stage,
            execution_phase=execution_phase,
        )
        if contract.algorithm in {
            "pact",
            "fastv",
            "sgl",
            "dart",
            "d2pruner",
            "sparsevlm",
            "finepruner",
            "btp",
            "halfv",
            "transprune",
            "feather",
            "ivc_prune",
        }:
            try:
                config = merge_decoder_executor_config(
                    contract.algorithm,
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        elif contract.algorithm == "mmtok":
            try:
                config = merge_mmtok_executor_config(
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        elif contract.algorithm == "vispruner":
            try:
                config = merge_vispruner_executor_config(
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        elif contract.algorithm == "zoo_prune":
            try:
                config = merge_zoo_prune_executor_config(
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        elif contract.algorithm == "nuwa":
            try:
                config = merge_nuwa_executor_config(
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        elif contract.algorithm == "restore":
            try:
                config = merge_restore_executor_config(
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        else:
            config = {}
            allowed_keys = _ALGORITHM_EXECUTOR_CONFIG_KEYS[contract.algorithm]
            # Control-plane dictionaries contain schema fields unrelated to
            # the kernel. Only recognized algorithm knobs cross this boundary.
            for name in ("operation", "criterion", "schedule", "config"):
                value = getattr(stage, name, None)
                if isinstance(value, Mapping):
                    config.update(
                        {
                            key: item
                            for key, item in value.items()
                            if key in allowed_keys
                        }
                    )
            config.update(
                {
                    key: item
                    for key, item in dict(
                        getattr(binding_config, "config", {}) or {}
                    ).items()
                    if key in allowed_keys
                }
            )
        return Qwen25VLImageCompressionExecutor(
            model=model,
            runtime=runtime,
            stage=stage,
            contract=contract,
            execution_phase=execution_phase,
            config=config,
        )


def create_qwen25_vl_image_compression_executor(
    **kwargs: Any,
) -> Qwen25VLImageCompressionExecutor:
    """Executor-factory signature consumed by ``BackbonePlacementBinding``."""

    return Qwen25VLImageCompressionAdapter().resolve(**kwargs)


def _validate_binding_and_stage(
    *,
    contract: Qwen25VLImageBindingContract,
    binding: Any,
    stage: Any,
    execution_phase: ExecutionPhase | str,
) -> None:
    phase = ExecutionPhase(execution_phase)
    supported = getattr(binding, "supported_phases", {phase})
    if phase not in supported:
        raise BackendIntegrationError(
            f"Binding {contract.binding!r} does not support phase {phase.value!r}."
        )
    backbone = getattr(binding, "backbone", "qwen25_vl")
    if backbone != "qwen25_vl":
        raise BackendIntegrationError(
            f"Binding {contract.binding!r} targets {backbone!r}, not 'qwen25_vl'."
        )
    if getattr(binding, "variant", contract.variant) != contract.variant:
        raise BackendIntegrationError(
            f"Binding {contract.binding!r} has the wrong modality variant."
        )
    stage_algorithm = getattr(stage, "algorithm", contract.algorithm)
    stage_variant = getattr(stage, "variant", contract.variant)
    if stage_algorithm != contract.algorithm or stage_variant != contract.variant:
        raise BackendIntegrationError(
            f"Stage identity does not match binding {contract.binding!r}."
        )
    targets = set(getattr(stage, "target_modalities", {Modality.IMAGE}) or ())
    targets = {Modality(item) for item in targets}
    if targets != {Modality.IMAGE}:
        raise BackendIntegrationError(
            "Qwen2.5-VL ImageLLM bindings target image only; video requires a "
            "separate support unit and adapter contract."
        )
    target_sites = tuple(
        getattr(target, "target_site", "")
        for target in getattr(binding, "ordered_targets", lambda: ())()
    )
    expected_sites = contract.target_sites
    if contract.algorithm in {"visionzip", "feature_anchor_mean", "grid_mean", "prumerge", "gap", "divprune", "aim", "vispruner", "mmtok", "zoo_prune"}:
        expected_sites = image_projector_sites(stage, contract.algorithm, "qwen25_vl")
    if contract.algorithm == "saint":
        expected_sites = (saint_encoder_operation(stage)["target_site"],)
    if contract.algorithm == "tome":
        expected_sites = (tome_encoder_operation(stage, "qwen25_vl")["target_site"], *expected_sites[1:-1], tome_output_site(stage, "qwen25_vl"))
    if contract.algorithm in {"fastv", "dart", "d2pruner", "sgl", "ivc_prune", "feather", "btp", "halfv", "pyramid_drop", "finepruner", "transprune", "pact", "sparsevlm"}:
        expected_sites = (decoder_token_operation(stage, contract.algorithm, backbone="qwen25_vl")["target_site"],)
    if contract.algorithm in {"nuwa", "restore"}:
        expected_sites = (*image_projector_sites(stage, contract.algorithm, "qwen25_vl"), *expected_sites[1:])
    if contract.algorithm == "nuwa":
        expected_sites = (*expected_sites[:-1], decoder_token_operation(stage, "nuwa", backbone="qwen25_vl")["target_site"])
    if target_sites and target_sites != expected_sites:
        raise BackendIntegrationError(
            f"Binding {contract.binding!r} physical sites do not match the "
            "Qwen2.5-VL adapter contract."
        )


def _bifocal_spec(config: Mapping[str, Any]) -> BifocalSpec:
    try:
        return bifocal_spec_from_config(config)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid Bifocal config: {exc}") from exc


def _divprune_spec(config: Mapping[str, Any]) -> DivPruneSpec:
    try:
        return DivPruneSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid DivPrune config: {exc}") from exc


def _saint_config(
    config: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[float, ...], SAINTSpec]:
    values = dict(config)
    layers = tuple(values.pop("layer_indices", (15, 23)))
    ratios = tuple(values.pop("keep_ratios", (0.5, 0.4)))
    if (
        not layers
        or len(layers) != len(ratios)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in layers)
        or layers != tuple(sorted(set(layers)))
    ):
        raise BackendIntegrationError(
            "SAINT layer_indices must be a non-empty sorted unique integer tuple "
            "matching keep_ratios."
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 < float(value) <= 1.0
        for value in ratios
    ):
        raise BackendIntegrationError("SAINT keep_ratios must be in (0, 1].")
    try:
        spec = SAINTSpec(keep_ratio=float(ratios[0]), **values)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid SAINT config: {exc}") from exc
    return layers, tuple(float(value) for value in ratios), spec


def _aim_spec(config: Mapping[str, Any]) -> AIMSpec:
    try:
        return AIMSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid AIM config: {exc}") from exc


def _mmtok_spec(config: Mapping[str, Any]) -> MMTokSpec:
    try:
        # Official Qwen release uses tau_t=.01 (LLaVA uses .02).
        return MMTokSpec(**{"text_temperature": 0.01, **dict(config)})
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid MMTok config: {exc}") from exc


def _vispruner_spec(config: Mapping[str, Any]) -> VisPrunerSpec:
    try:
        return VisPrunerSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid VisPruner config: {exc}") from exc


def _zoo_prune_spec(config: Mapping[str, Any]) -> ZOOPruneSpec:
    try:
        return ZOOPruneSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid ZOO-Prune config: {exc}") from exc


def _nuwa_spec(config: Mapping[str, Any]) -> NuwaSpec:
    try:
        return NuwaSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid Nüwa config: {exc}") from exc


def _restore_spec(config: Mapping[str, Any]) -> RestoreSpec:
    values = dict(config)
    tokens_per_crop = values.pop("tokens_per_crop", None)
    if tokens_per_crop is not None:
        if "target_count" in values:
            raise BackendIntegrationError(
                "RESTORE accepts either target_count or tokens_per_crop, not both."
            )
        values["target_count"] = tokens_per_crop
    try:
        return RestoreSpec(**values)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid RESTORE config: {exc}") from exc


def _sgl_spec(config: Mapping[str, Any]) -> SGLSpec:
    try:
        return SGLSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid SGL config: {exc}") from exc


def _prumerge_spec(config: Mapping[str, Any]) -> PruMergeSpec:
    try:
        return PruMergeSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid PruMerge config: {exc}") from exc


def _gap_spec(config: Mapping[str, Any]) -> GAPSpec:
    try:
        return GAPSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid GAP config: {exc}") from exc


def _resolve_provider_bridge(model: Any, runtime: Any) -> Any:
    metadata = getattr(runtime, "metadata", {})
    if isinstance(metadata, Mapping) and metadata.get(_BRIDGE_METADATA_KEY) is not None:
        return metadata[_BRIDGE_METADATA_KEY]
    for attribute in _BRIDGE_MODEL_ATTRIBUTES:
        value = getattr(model, attribute, None)
        if value is not None:
            return value
    raise BackendIntegrationError(
        "Qwen2.5-VL checkpoint integration has not installed an image compression "
        f"bridge. Provide runtime.metadata[{_BRIDGE_METADATA_KEY!r}] or one of "
        f"model attributes {list(_BRIDGE_MODEL_ATTRIBUTES)!r}."
    )


def _resolve_or_install_provider_bridge(model: Any, runtime: Any) -> tuple[Any, bool]:
    """Resolve an injected bridge or install the pinned production bridge.

    Fake/custom bridges remain injectable through runtime metadata.  A normal
    Transformers checkpoint needs no out-of-band installer: the first exact
    activation owns the reversible provider patch and releases it after the
    final callback is removed.
    """

    try:
        return _resolve_provider_bridge(model, runtime), False
    except BackendIntegrationError:
        try:
            from .provider_bridge import install_qwen25_vl_transformers_4573_bridge

            return install_qwen25_vl_transformers_4573_bridge(model), True
        except BaseException as exc:
            raise BackendIntegrationError(
                "Qwen2.5-VL could not resolve an injected bridge or install the "
                "pinned Transformers 4.57.3 provider bridge."
            ) from exc


def _uninstall_owned_provider_bridge(model: Any) -> None:
    from .provider_bridge import uninstall_qwen25_vl_transformers_4573_bridge

    uninstall_qwen25_vl_transformers_4573_bridge(model)


def _single_layout_delta(result: Any) -> LayoutDelta:
    delta = getattr(result, "layout_delta", None)
    if not isinstance(delta, LayoutDelta):
        raise InvariantError(
            "Qwen2.5-VL encoder_dense_boundary currently requires a single-image "
            "ToMe result with one LayoutDelta."
        )
    return delta


def _layout_operation(delta: LayoutDelta) -> str:
    if delta.is_identity:
        return "identity"
    if delta.dropped_indices:
        return "drop"
    if delta.merge_groups:
        return "merge"
    return "remap"


def _required_result_index(delta: LayoutDelta, source_index: int) -> int:
    result = delta.old_to_new[source_index]
    if result is None:
        raise InvariantError("Dense restore cannot recover a dropped source patch.")
    return result


def _result_representatives(delta: LayoutDelta) -> tuple[int, ...]:
    explicit = delta.result_layout.get("representative_indices")
    if explicit is not None:
        representatives = tuple(int(value) for value in explicit)
        if (
            len(representatives) != delta.result_count
            or len(set(representatives)) != len(representatives)
            or any(
                value < 0 or value >= delta.source_count
                for value in representatives
            )
        ):
            raise InvariantError(
                "LayoutDelta representative_indices are not valid source positions."
            )
        for result_index, source_index in enumerate(representatives):
            if delta.old_to_new[source_index] != result_index:
                raise InvariantError(
                    "LayoutDelta representative does not belong to its result group."
                )
        return representatives
    representatives: list[int] = []
    for result_index in range(delta.result_count):
        preimage = tuple(
            source_index
            for source_index in range(delta.source_count)
            if delta.old_to_new[source_index] == result_index
        )
        if not preimage:
            raise InvariantError("LayoutDelta result has no source representative.")
        representatives.append(preimage[0])
    return tuple(representatives)


def _strict_indices(values: Sequence[int], *, upper: int, name: str) -> tuple[int, ...]:
    if isinstance(upper, bool) or not isinstance(upper, int) or upper < 0:
        raise InvariantError("source_sequence_length must be a non-negative integer.")
    result = tuple(values)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in result):
        raise InvariantError(f"{name} must contain integers.")
    if tuple(sorted(set(result))) != result:
        raise InvariantError(f"{name} must be unique and source ordered.")
    if any(index < 0 or index >= upper for index in result):
        raise InvariantError(f"{name} contains an index outside the prompt.")
    return result


def _sequence_length(value: Any, sequence_dim: int) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        ndim = len(shape)
        resolved = sequence_dim if sequence_dim >= 0 else ndim + sequence_dim
        if not 0 <= resolved < ndim:
            raise InvariantError("sequence_dim is outside the value rank.")
        return int(shape[resolved])
    if sequence_dim not in (-1, 0):
        raise InvariantError("Pure-Python sequences support only sequence_dim=-1 or 0.")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InvariantError("Sequence-aligned value has no readable sequence axis.")
    return len(value)


def _take_sequence(value: Any, indices: Sequence[int], *, sequence_dim: int) -> Any:
    index_select = getattr(value, "index_select", None)
    shape = getattr(value, "shape", None)
    if callable(index_select) and shape is not None:
        import torch

        ndim = len(shape)
        resolved = sequence_dim if sequence_dim >= 0 else ndim + sequence_dim
        index = torch.tensor(tuple(indices), dtype=torch.long, device=value.device)
        return index_select(resolved, index)
    if sequence_dim not in (-1, 0):
        raise InvariantError("Pure-Python sequences support only sequence_dim=-1 or 0.")
    selected = [value[index] for index in indices]
    return tuple(selected) if isinstance(value, tuple) else selected


__all__ = [
    "QWEN25_VL_IMAGE_BINDING_CONTRACTS",
    "Qwen25VLDecoderCacheRepairRequest",
    "Qwen25VLDecoderCompressionResult",
    "Qwen25VLDecoderRepairPlan",
    "Qwen25VLDecoderRepairResult",
    "Qwen25VLDenseRestorePlan",
    "Qwen25VLEncoderCompressionResult",
    "Qwen25VLImageBindingContract",
    "Qwen25VLImageCompressionAdapter",
    "Qwen25VLImageCompressionExecutor",
    "Qwen25VLMRoPEResult",
    "Qwen25VLPlaceholderRepairPlan",
    "Qwen25VLProjectorCompressionResult",
    "Qwen25VLPromptRepairResult",
    "Qwen25VLSparseMRoPERequest",
    "create_qwen25_vl_image_compression_executor",
]
