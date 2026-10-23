"""Backbone-neutral SGL/SGP guide scoring and target token selection.

The compact guide is executed through the same native backbone facade as the
target.  This module owns the released SGP attention aggregation, one-shot
selection, and guide resource lifetime.  Backbone adapters own only their
decoder boundary and layout/cache legalization.
"""

from __future__ import annotations

from ._rules import rule_metadata, score_rule, selection_rule

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.layout import LayoutDelta


SGL_CONFIG_KEYS = frozenset(
    {
        "guide_model_path",
        "guide_revision",
        "guide_device",
        "prune_layer_fraction",
        "keep_ratio",
        "min_visual_tokens",
        "mode",
        "attention_sources",
        "attention_aggregation",
        "score_accumulation",
        "require_matching_visual_tokens",
        "single_image_only",
        "log_first_n",
    }
)


@dataclass(frozen=True)
class SGLSpec:
    """Released SGP configuration with an explicit compact-guide checkpoint."""

    guide_model_path: str
    guide_revision: str | None = None
    guide_device: str | None = None
    prune_layer_fraction: float = 0.05
    keep_ratio: float = 0.05
    min_visual_tokens: int = 1
    mode: str = "sgp"
    attention_sources: str = "prompt_and_generated"
    attention_aggregation: str = "sum_layers_heads_queries"
    score_accumulation: str = "fp32"
    require_matching_visual_tokens: bool = True
    single_image_only: bool = True
    log_first_n: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.guide_model_path, str) or not self.guide_model_path.strip():
            raise ValueError("guide_model_path must be a non-empty string.")
        if self.guide_revision is not None and (
            not isinstance(self.guide_revision, str) or not self.guide_revision.strip()
        ):
            raise ValueError("guide_revision must be None or a non-empty string.")
        if self.guide_device is not None and (
            not isinstance(self.guide_device, str) or not self.guide_device.strip()
        ):
            raise ValueError("guide_device must be None or a non-empty string.")
        for name in ("prune_layer_fraction", "keep_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, not a boolean.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 <= float(self.prune_layer_fraction) < 1.0:
            raise ValueError("prune_layer_fraction must lie in [0, 1).")
        if not 0.0 < float(self.keep_ratio) <= 1.0:
            raise ValueError("keep_ratio must lie in (0, 1].")
        for name in ("min_visual_tokens", "log_first_n"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.min_visual_tokens < 1:
            raise ValueError("min_visual_tokens must be at least one.")
        if self.log_first_n < 0:
            raise ValueError("log_first_n must be non-negative.")
        if self.mode != "sgp":
            raise ValueError("Only the released SGL mode='sgp' path is supported.")
        if self.attention_sources != "prompt_and_generated":
            raise ValueError(
                "attention_sources must be 'prompt_and_generated'."
            )
        if self.attention_aggregation != "sum_layers_heads_queries":
            raise ValueError(
                "attention_aggregation must be 'sum_layers_heads_queries'."
            )
        if self.score_accumulation != "fp32":
            raise ValueError("score_accumulation must be 'fp32'.")
        if self.require_matching_visual_tokens is not True:
            raise ValueError("SGL requires matching guide and target visual tokens.")
        if self.single_image_only is not True:
            raise ValueError("SGL supports exactly one aligned image per request.")

    def prune_layer(self, layer_count: int) -> int:
        if isinstance(layer_count, bool) or not isinstance(layer_count, int):
            raise TypeError("layer_count must be an integer.")
        if layer_count < 1:
            raise ValueError("SGL requires at least one target decoder layer.")
        return min(
            layer_count - 1,
            max(0, int(layer_count * float(self.prune_layer_fraction))),
        )

    def keep_count(self, visual_count: int) -> int:
        if isinstance(visual_count, bool) or not isinstance(visual_count, int):
            raise TypeError("visual_count must be an integer.")
        if visual_count < 1:
            raise ValueError("SGL requires at least one target visual token.")
        return min(
            visual_count,
            max(self.min_visual_tokens, int(visual_count * float(self.keep_ratio))),
        )


@dataclass(frozen=True)
class SGLAttentionSummary:
    visual_scores: Any
    visual_start: int
    visual_end: int
    prompt_query_tokens: int
    generated_query_tokens: int
    guide_layers: int
    guide_layer_calls: int


@dataclass(frozen=True)
class SGLResult:
    hidden_states: Any
    kept_indices: Any
    dropped_indices: Any
    kept_image_indices: Any
    image_scores: Any
    layout_delta: LayoutDelta
    metadata: Mapping[str, Any] = field(default_factory=dict)


def aggregate_sgl_visual_attention(
    attentions: Any,
    *,
    input_ids: Any,
    image_token_id: int,
    rules: Mapping[str, Any] | None = None,
) -> SGLAttentionSummary:
    """Sum guide attention over layers, heads, prompt, and decode queries.

    The supplied values are the model's full causal attention probabilities,
    so every visual column retains the released full-key softmax denominator.
    """

    torch = _torch()
    steps = _attention_steps(attentions)
    if not steps:
        raise ValueError("SGL guide returned no text attention steps.")
    first_layers = steps[0]
    first = _attention_tensor(first_layers[0], step=0, layer=0)
    key_length = int(first.shape[-1])
    visual_start, visual_end = _expanded_visual_span(
        input_ids,
        image_token_id=image_token_id,
        expanded_length=key_length,
    )
    scores = torch.zeros(
        (visual_end - visual_start,),
        dtype=torch.float32,
        device=first.device,
    )
    prompt_queries: int | None = None
    generated_queries = 0
    layer_calls = 0
    guide_layers = len(first_layers)
    for step_index, layers in enumerate(steps):
        if len(layers) != guide_layers:
            raise ValueError("SGL guide layer count changed across generation steps.")
        step_queries: int | None = None
        for layer_index, value in enumerate(layers):
            attention = _attention_tensor(
                value,
                step=step_index,
                layer=layer_index,
            )
            if int(attention.shape[0]) != 1:
                raise ValueError("SGL guide attention requires batch size one.")
            if int(attention.shape[-1]) < visual_end:
                raise ValueError("SGL guide attention lost visual key columns.")
            query_count = int(attention.shape[-2])
            if step_index == 0:
                if query_count < visual_end:
                    raise ValueError(
                        "SGL prefill attention does not expose post-image queries."
                    )
                selected = attention[0, :, visual_end:, visual_start:visual_end]
                current_queries = query_count - visual_end
                if current_queries < 1:
                    raise ValueError("SGL prompt has no post-image text query.")
            else:
                selected = attention[0, :, :, visual_start:visual_end]
                current_queries = query_count
            if step_queries is None:
                step_queries = current_queries
            elif step_queries != current_queries:
                raise ValueError("SGL guide query count differs across layers.")
            contribution = selected.float().sum(dim=(0, 1))
            if not bool(torch.isfinite(contribution).all().item()):
                raise ValueError("SGL guide produced non-finite visual scores.")
            scores = scores + contribution
            layer_calls += 1
        assert step_queries is not None
        if step_index == 0:
            prompt_queries = step_queries
        else:
            generated_queries += step_queries
    assert prompt_queries is not None
    scores = score_rule(rules, "scorer", {"attentions": attentions,
        "input_ids": input_ids, "image_token_id": image_token_id,
        "visual_start": visual_start, "visual_end": visual_end}, scores)
    return SGLAttentionSummary(
        visual_scores=scores.detach().cpu(),
        visual_start=visual_start,
        visual_end=visual_end,
        prompt_query_tokens=prompt_queries,
        generated_query_tokens=generated_queries,
        guide_layers=guide_layers,
        guide_layer_calls=layer_calls,
    )


def sgl_image_tokens(
    hidden_states: Any,
    *,
    image_indices: Any,
    image_scores: Any,
    spec: SGLSpec,
    protected_indices: Sequence[int] = (),
    media_unit: str = "llm_token",
    source_layout: Mapping[str, Any] | None = None,
    result_layout: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> SGLResult:
    """Transfer compact-guide importance to one target decoder boundary."""

    torch = _torch()
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("hidden_states must be a rank-2 torch.Tensor.")
    if not hidden_states.is_floating_point():
        raise TypeError("hidden_states must use a floating dtype.")
    sequence_length = int(hidden_states.shape[0])
    images = torch.as_tensor(
        image_indices,
        dtype=torch.long,
        device=hidden_states.device,
    ).flatten()
    image_values = images.tolist()
    if image_values != sorted(set(image_values)):
        raise ValueError("image_indices must be unique and in source order.")
    if not image_values or any(index < 0 or index >= sequence_length for index in image_values):
        raise ValueError("image_indices must identify target sequence tokens.")
    scores = torch.as_tensor(
        image_scores,
        dtype=torch.float32,
        device=hidden_states.device,
    ).flatten()
    if int(scores.numel()) != len(image_values):
        raise ValueError(
            "SGL guide/target visual-token mismatch: "
            f"guide={scores.numel()}, target={len(image_values)}."
        )
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("SGL image_scores must be finite.")
    protected = tuple(sorted(set(int(value) for value in protected_indices)))
    if any(index < 0 or index >= sequence_length for index in protected):
        raise ValueError("protected_indices contains an out-of-range index.")
    image_positions = {value: index for index, value in enumerate(image_values)}
    protected_positions = tuple(
        image_positions[index] for index in protected if index in image_positions
    )
    keep_count = max(spec.keep_count(len(image_values)), len(protected_positions))
    keep_count = min(len(image_values), keep_count)
    ranking = _stable_descending_indices(scores)
    selected_positions = list(protected_positions)
    protected_set = set(protected_positions)
    selected_positions.extend(
        int(value)
        for value in ranking.tolist()
        if int(value) not in protected_set
    )
    selected_positions = sorted(selected_positions[:keep_count])
    selected = torch.tensor(
        selected_positions,
        dtype=torch.long,
        device=hidden_states.device,
    )
    if keep_count < len(image_values):
        custom = selection_rule(rules, "selector",
            {"hidden_states": hidden_states, "image_indices": images, "scores": scores,
             "keep_count": keep_count, "protected_indices": protected_positions, "spec": spec},
            count=len(image_values), keep_count=keep_count, device=hidden_states.device, protected=protected_positions)
        if custom is not None:
            selected = custom.sort().values
            selected_positions = selected.tolist()
    kept_images = images.index_select(0, selected)
    kept_image_set = set(int(value) for value in kept_images.tolist())
    image_set = set(image_values)
    dropped = tuple(
        index
        for index in range(sequence_length)
        if index in image_set and index not in kept_image_set
    )
    dropped_set = set(dropped)
    kept = torch.tensor(
        [index for index in range(sequence_length) if index not in dropped_set],
        dtype=torch.long,
        device=hidden_states.device,
    )
    dropped_tensor = torch.tensor(
        dropped,
        dtype=torch.long,
        device=hidden_states.device,
    )
    non_image = tuple(index for index in range(sequence_length) if index not in image_set)
    delta = LayoutDelta.drop(
        sequence_length,
        dropped,
        media_unit=media_unit,
        protected_indices=tuple(sorted(set(non_image) | set(protected))),
        source_layout=source_layout,
        result_layout=result_layout,
    )
    output = hidden_states.index_select(0, kept)
    return SGLResult(
        hidden_states=output,
        kept_indices=kept,
        dropped_indices=dropped_tensor,
        kept_image_indices=kept_images,
        image_scores=scores,
        layout_delta=delta,
        metadata={
            "algorithm": "sgl",
            "variant": "sgl.image",
            "mode": "sgp",
            "ranking": "custom" if rules else "guide_attention_sum",
            **rule_metadata(rules),
            "source_tokens": sequence_length,
            "result_tokens": int(output.shape[0]),
            "source_image_tokens": len(image_values),
            "result_image_tokens": int(kept_images.numel()),
            "selected_visual_indices": tuple(selected_positions),
            "selected_indices": tuple(int(value) for value in kept.tolist()),
            "protected_image_tokens": len(protected_positions),
            "compression_applied": bool(dropped),
        },
    )


class SGLGuideController:
    """Request-local score staging plus lazily loaded compact guide runtime."""

    _RUNTIME_KEY = "_sgl_controller"

    def __init__(
        self,
        *,
        backbone_name: str,
        target_runtime: Any,
        spec: SGLSpec,
        rules: Mapping[str, Any] | None = None,
    ) -> None:
        self.backbone_name = backbone_name
        self.target_runtime = target_runtime
        self.spec = spec
        self.rules = rules
        self._facade: Any | None = None
        self._guide_runtime: Any | None = None
        self._pending_scores: Any | None = None
        self._attached = False
        self._status = "resolved"
        self._request_count = 0
        self._failure_count = 0
        self._last_error: str | None = None
        self._last_request: dict[str, Any] = {}
        self._model_controller_owner: Any | None = None

    def attach(self) -> None:
        metadata = getattr(self.target_runtime, "metadata", None)
        if not isinstance(metadata, dict):
            raise BackendIntegrationError("SGL target runtime metadata must be mutable.")
        existing = metadata.get(self._RUNTIME_KEY)
        if existing not in (None, self):
            raise BackendIntegrationError("The target runtime already owns an SGL guide.")
        metadata[self._RUNTIME_KEY] = self
        model = getattr(self.target_runtime, "model", None)
        if model is not None:
            existing_model_controller = getattr(model, self._RUNTIME_KEY, None)
            if existing_model_controller not in (None, self):
                metadata.pop(self._RUNTIME_KEY, None)
                raise BackendIntegrationError("The target model already owns an SGL guide.")
            try:
                setattr(model, self._RUNTIME_KEY, self)
                self._model_controller_owner = model
            except (AttributeError, TypeError):
                self._model_controller_owner = None
        self._attached = True
        self._status = "attached"

    def detach(self) -> None:
        metadata = getattr(self.target_runtime, "metadata", None)
        if isinstance(metadata, dict) and metadata.get(self._RUNTIME_KEY) is self:
            metadata.pop(self._RUNTIME_KEY, None)
        model = self._model_controller_owner
        if model is not None and getattr(model, self._RUNTIME_KEY, None) is self:
            try:
                delattr(model, self._RUNTIME_KEY)
            except AttributeError:
                pass
        self._model_controller_owner = None
        self._pending_scores = None
        self._guide_runtime = None
        self._facade = None
        self._attached = False
        self._status = "released"

    def stage_scores(
        self,
        scores: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        torch = _torch()
        value = torch.as_tensor(scores, dtype=torch.float32).detach().cpu().flatten()
        if int(value.numel()) < 1 or not bool(torch.isfinite(value).all().item()):
            raise BackendIntegrationError("SGL guide scores must be finite and non-empty.")
        if self._pending_scores is not None:
            raise BackendIntegrationError("SGL already has unconsumed guide scores.")
        self._pending_scores = value
        if metadata is not None:
            self._last_request.update(dict(metadata))
        self._last_request["guide_visual_tokens"] = int(value.numel())

    def consume_scores(self, *, target_visual_tokens: int, device: Any) -> Any:
        if self._pending_scores is None:
            raise BackendIntegrationError(
                "SGL target pruning reached its boundary without guide scores."
            )
        if int(self._pending_scores.numel()) != int(target_visual_tokens):
            raise BackendIntegrationError(
                "SGL guide/target visual-token mismatch: "
                f"guide={self._pending_scores.numel()}, target={target_visual_tokens}."
            )
        value = self._pending_scores.to(device=device)
        self._pending_scores = None
        self._last_request["target_visual_tokens"] = int(target_visual_tokens)
        self._last_request["matching_visual_tokens"] = True
        return value

    def prepare(
        self,
        *,
        processed_batch: Any,
        generation_kwargs: Mapping[str, Any],
    ) -> None:
        if not self._attached:
            raise BackendIntegrationError("SGL guide controller is not attached.")
        source_batch = getattr(processed_batch, "source_batch", None)
        if source_batch is None:
            raise BackendIntegrationError(
                "SGL requires ProcessedBatch.source_batch for guide preprocessing."
            )
        if self._pending_scores is not None:
            raise BackendIntegrationError("SGL found stale guide scores before a request.")
        self._ensure_guide_loaded()
        assert self._facade is not None and self._guide_runtime is not None
        image_count = _processed_image_count(processed_batch)
        if image_count != 1:
            raise BackendIntegrationError(
                f"SGL supports exactly one image, received {image_count}."
            )
        guide_processed = self._facade.processor_adapter.prepare_inputs(
            self._guide_runtime.processor,
            source_batch,
            phase="eval",
        )
        guide_image_count = _processed_image_count(guide_processed)
        if guide_image_count != 1:
            raise BackendIntegrationError(
                "SGL guide preprocessing must produce exactly one image."
            )
        guide_state = self._facade.state_builder.build(
            guide_processed,
            raw_batch=source_batch,
            phase="eval",
        )
        guide_kwargs = _guide_generation_kwargs(generation_kwargs)
        torch = _torch()
        telemetry = _GuideMemoryTelemetry(torch, self._guide_runtime.device)
        try:
            with _preserve_rng_state(torch), torch.inference_mode():
                telemetry.start()
                from mllm_comp.backbones.base import TraceRequest

                result = self._facade.runner.generate(
                    self._guide_runtime,
                    guide_state,
                    processed_batch=guide_processed,
                    trace_request=TraceRequest(
                        _full_generation_text_attentions=True
                    ),
                    **guide_kwargs,
                )
                telemetry.finish()
            traces = getattr(result, "traces", None)
            attentions = getattr(traces, "text_attentions", None)
            token_ids = guide_processed.model_inputs.get("input_ids")
            image_token_id = guide_processed.metadata.get("token_ids", {}).get("image")
            if image_token_id is None:
                raise BackendIntegrationError(
                    "SGL guide processor did not publish its image token ID."
                )
            summary = aggregate_sgl_visual_attention(
                attentions,
                rules=self.rules,
                input_ids=token_ids,
                image_token_id=int(image_token_id),
            )
            self._request_count += 1
            request_metadata = {
                "request_index": self._request_count,
                "guide_model_path": self.spec.guide_model_path,
                "guide_revision": self.spec.guide_revision,
                "target_model_path": getattr(
                    self.target_runtime, "model_path_or_id", None
                ),
                "target_revision": getattr(self.target_runtime, "revision", None),
                "guide_layers": summary.guide_layers,
                "guide_layer_calls": summary.guide_layer_calls,
                "prompt_query_tokens": summary.prompt_query_tokens,
                "generated_query_tokens": summary.generated_query_tokens,
                "attention_sources": self.spec.attention_sources,
                "attention_aggregation": self.spec.attention_aggregation,
                "score_accumulation": self.spec.score_accumulation,
                **telemetry.metadata(),
            }
            self._last_request = request_metadata
            self.stage_scores(summary.visual_scores, metadata=request_metadata)
            self._last_error = None
            self._status = "scores_ready"
        except BaseException as exc:
            self._pending_scores = None
            self._failure_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._status = "guide_failed"
            raise

    def prepare_teacher_forced(
        self,
        *,
        model_inputs: Mapping[str, Any],
        image_token_id: int,
    ) -> None:
        """Stage guide scores from a no-cache teacher-forced guide forward."""

        if not self._attached:
            raise BackendIntegrationError("SGL guide controller is not attached.")
        if self._pending_scores is not None:
            raise BackendIntegrationError("SGL found stale guide scores before a request.")
        self._ensure_guide_loaded()
        assert self._guide_runtime is not None
        torch = _torch()

        def move(value: Any) -> Any:
            if torch.is_tensor(value):
                return value.to(self._guide_runtime.device)
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, dict):
                return {name: move(item) for name, item in value.items()}
            return value

        guide_inputs = {
            name: move(value)
            for name, value in model_inputs.items()
            if name not in {"labels", "past_key_values", "cache_position"}
        }
        guide_inputs.update(
            {
                "use_cache": False,
                "output_attentions": True,
                "return_dict": True,
            }
        )
        input_ids = guide_inputs.get("input_ids")
        if input_ids is None:
            raise BackendIntegrationError(
                "SGL teacher forcing requires native input_ids."
            )
        try:
            with _preserve_rng_state(torch), torch.inference_mode():
                output = self._guide_runtime.model(**guide_inputs)
            attentions = getattr(output, "attentions", None)
            if attentions is None and isinstance(output, Mapping):
                attentions = output.get("attentions")
            summary = aggregate_sgl_visual_attention(
                attentions,
                rules=self.rules,
                input_ids=input_ids,
                image_token_id=int(image_token_id),
            )
            self._request_count += 1
            request_metadata = {
                "request_index": self._request_count,
                "guide_model_path": self.spec.guide_model_path,
                "guide_revision": self.spec.guide_revision,
                "target_model_path": getattr(
                    self.target_runtime, "model_path_or_id", None
                ),
                "target_revision": getattr(self.target_runtime, "revision", None),
                "guide_layers": summary.guide_layers,
                "guide_layer_calls": summary.guide_layer_calls,
                "prompt_query_tokens": summary.prompt_query_tokens,
                "generated_query_tokens": 0,
                "attention_sources": "teacher_forced_prompt",
                "attention_aggregation": self.spec.attention_aggregation,
                "score_accumulation": self.spec.score_accumulation,
                "use_cache": False,
            }
            self._last_request = request_metadata
            self.stage_scores(summary.visual_scores, metadata=request_metadata)
            self._last_error = None
            self._status = "scores_ready"
        except BaseException as exc:
            self._pending_scores = None
            self._failure_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._status = "guide_failed"
            raise

    def finish_target(self, *, success: bool) -> None:
        if success and self._pending_scores is not None:
            self._pending_scores = None
            self._failure_count += 1
            self._last_error = "SGL target generation returned without pruning."
            self._status = "target_failed"
            raise BackendIntegrationError(self._last_error)
        self._pending_scores = None
        self._status = "completed" if success else "target_failed"

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "guide_status": self._status,
            "guide_loaded": self._guide_runtime is not None,
            "guide_attached": self._attached,
            "guide_model_path": self.spec.guide_model_path,
            "guide_revision": self.spec.guide_revision,
            "guide_device": self.spec.guide_device,
            "guide_request_count": self._request_count,
            "guide_failure_count": self._failure_count,
            "guide_scores_pending": self._pending_scores is not None,
            "last_guide_error": self._last_error,
            "last_guide_request": dict(self._last_request),
        }

    def _ensure_guide_loaded(self) -> None:
        if self._guide_runtime is not None:
            return
        from mllm_comp.backbones.base import BackboneRuntime
        from mllm_comp.registry.backbones import get_backbone

        torch = _torch()
        facade = get_backbone(self.backbone_name)
        device_name = self.spec.guide_device or str(
            getattr(self.target_runtime, "device", None) or "cpu"
        )
        processor = facade.processor_adapter.load_processor(
            self.spec.guide_model_path,
            revision=self.spec.guide_revision,
        )
        load_kwargs: dict[str, Any] = {"attn_implementation": "eager"}
        target_dtype = getattr(self.target_runtime, "dtype", None)
        if target_dtype is not None:
            load_kwargs["torch_dtype"] = target_dtype
        if device_name:
            load_kwargs["device_map"] = device_name
        model = facade.model_adapter.load_model(
            self.spec.guide_model_path,
            train=False,
            revision=self.spec.guide_revision,
            **load_kwargs,
        )
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()
        requires_grad = getattr(model, "requires_grad_", None)
        if callable(requires_grad):
            requires_grad(False)
        self._facade = facade
        self._guide_runtime = BackboneRuntime(
            model=model,
            processor=processor,
            model_path_or_id=self.spec.guide_model_path,
            revision=self.spec.guide_revision,
            device=torch.device(device_name),
            dtype=getattr(model, "dtype", target_dtype),
            metadata={
                "runtime_mode": "default",
                "compression_mode": "none",
                "sgl_role": "compact_guide",
            },
        )
        self._status = "guide_loaded"


@contextmanager
def sgl_target_generation(
    runtime: Any,
    processed_batch: Any,
    generation_kwargs: Mapping[str, Any],
) -> Iterator[None]:
    """Prepare and consume one guide score vector around target generation."""

    metadata = getattr(runtime, "metadata", None)
    controller = (
        metadata.get(SGLGuideController._RUNTIME_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    if not callable(getattr(controller, "prepare", None)):
        controller = getattr(
            getattr(runtime, "model", None),
            SGLGuideController._RUNTIME_KEY,
            None,
        )
    if not (
        callable(getattr(controller, "prepare", None))
        and callable(getattr(controller, "finish_target", None))
    ):
        yield
        return
    controller.prepare(
        processed_batch=processed_batch,
        generation_kwargs=generation_kwargs,
    )
    try:
        yield
    except BaseException:
        controller.finish_target(success=False)
        raise
    else:
        controller.finish_target(success=True)


def _attention_steps(attentions: Any) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(attentions, Sequence) or isinstance(
        attentions, (str, bytes, bytearray)
    ):
        raise ValueError("SGL text attentions must be a generation sequence.")
    raw = tuple(attentions)
    if not raw:
        return ()
    first = raw[0]
    if isinstance(first, Sequence) and not isinstance(
        first, (str, bytes, bytearray)
    ):
        return tuple(tuple(step) for step in raw)
    return (raw,)


def _attention_tensor(value: Any, *, step: int, layer: int) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(
            f"SGL attention step {step} layer {layer} must have shape [B,H,Q,K]."
        )
    return value


def _expanded_visual_span(
    input_ids: Any,
    *,
    image_token_id: int,
    expanded_length: int,
) -> tuple[int, int]:
    torch = _torch()
    ids = torch.as_tensor(input_ids)
    if ids.ndim != 2 or int(ids.shape[0]) != 1:
        raise ValueError("SGL guide input_ids must have shape [1,L].")
    positions = (ids[0] == int(image_token_id)).nonzero(as_tuple=False).flatten()
    values = [int(value) for value in positions.tolist()]
    if not values or values != list(range(values[0], values[-1] + 1)):
        raise ValueError("SGL requires one contiguous image-token span.")
    visual_count = int(expanded_length) - int(ids.shape[1]) + len(values)
    start = values[0]
    end = start + visual_count
    if visual_count < 1 or not 0 <= start < end <= int(expanded_length):
        raise ValueError("SGL could not align the expanded guide visual span.")
    return start, end


def _processed_image_count(processed_batch: Any) -> int:
    metadata = getattr(processed_batch, "metadata", {})
    if not isinstance(metadata, Mapping):
        return 0
    direct = metadata.get("image_count")
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    counts = metadata.get("modality_counts_by_sample")
    if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)):
        return 0
    total = 0
    for item in counts:
        if isinstance(item, Mapping):
            total += int(item.get("image", 0))
    return total


def _guide_generation_kwargs(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for name in (
        "stream",
        "streamer",
        "return_audio",
        "output_attentions",
        "output_hidden_states",
        "output_scores",
        "return_dict_in_generate",
    ):
        result.pop(name, None)
    num_beams = result.get("num_beams", 1)
    num_return_sequences = result.get("num_return_sequences", 1)
    if num_beams != 1 or num_return_sequences != 1:
        raise BackendIntegrationError(
            "SGL guide execution supports one greedy/sampled sequence without beams."
        )
    result["use_cache"] = True
    return result


@contextmanager
def _preserve_rng_state(torch: Any) -> Iterator[None]:
    devices = (
        list(range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else []
    )
    with torch.random.fork_rng(devices=devices):
        yield


class _GuideMemoryTelemetry:
    def __init__(self, torch: Any, device: Any) -> None:
        self.torch = torch
        self.device = device
        self.before: int | None = None
        self.after: int | None = None
        self.peak: int | None = None

    def start(self) -> None:
        if not self._cuda():
            return
        self.before = int(self.torch.cuda.memory_allocated(self.device))
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def finish(self) -> None:
        if not self._cuda():
            return
        self.after = int(self.torch.cuda.memory_allocated(self.device))
        self.peak = int(self.torch.cuda.max_memory_allocated(self.device))

    def metadata(self) -> dict[str, Any]:
        return {
            "guide_memory_device": str(self.device),
            "guide_memory_allocated_before_bytes": self.before,
            "guide_memory_allocated_after_bytes": self.after,
            "guide_memory_peak_allocated_bytes": self.peak,
            "guide_memory_incremental_peak_bytes": (
                None
                if self.before is None or self.peak is None
                else max(0, self.peak - self.before)
            ),
        }

    def _cuda(self) -> bool:
        return str(self.device).startswith("cuda") and self.torch.cuda.is_available()


def _stable_descending_indices(scores: Any) -> Any:
    torch = _torch()
    return torch.argsort(scores, descending=True, stable=True)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("SGL execution requires PyTorch.") from exc
    return torch


__all__ = [
    "SGLAttentionSummary",
    "SGLGuideController",
    "SGLResult",
    "SGLSpec",
    "SGL_CONFIG_KEYS",
    "aggregate_sgl_visual_attention",
    "sgl_image_tokens",
    "sgl_target_generation",
]
