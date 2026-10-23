"""LLaMA-VID local-BERT provider and independent three-backbone executor."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from mllm_comp.backbones._qwen_video_inputs import restore_qwen_video_raster
from mllm_comp.backbones._video_bridge import (
    install_video_bridge,
    register_video_callback,
    release_video_callback,
)
from mllm_comp.compressors.video._rules import load_video_rules, video_rule_metadata
from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.compressors.video import (
    LLaMAVIDVideoKernel,
    VideoTokenCoordinate,
    VideoTokenPayload,
    replay_llamavid_value_lineage,
)
from mllm_comp.compressors.video.config import resolve_canonical_video_config
from mllm_comp.compressors.video.positions import (
    video_learned_projector_position, video_learned_projector_site,
)
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.layout import layout_delta_record
from mllm_comp.core.vocabulary import ExecutionPhase


_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
_PROVIDER_ATTRIBUTE = "_mllm_comp_llamavid_video_signal_provider"
_QFORMER_ATTRIBUTE = "mllm_comp_llamavid_qformer"


@dataclass
class _LLaMAVIDCapture:
    frames: Any | None = None
    grid_thw: Any | None = None
    raw_features: Any | None = None
    calls: int = 0


class LLaMAVIDVideoSignalProvider:
    """Attach native raw rasters, user text, local BERT tokens, and QFormer."""

    def __init__(
        self,
        *,
        backbone: str,
        model: Any,
        bert_path: str | Path | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown LLaMA-VID video backbone {backbone!r}")
        self.backbone = backbone
        self.model = model
        self.bert_path = None if bert_path is None else Path(bert_path)
        self._tokenizer = tokenizer
        self._qformer_module: Any | None = None
        self._projector_dimensions: Mapping[str, int] | None = None
        self._active: ContextVar[_LLaMAVIDCapture | None] = ContextVar(
            f"llamavid_video_signals_{backbone}_{id(self)}",
            default=None,
        )
        self._patch: MethodPatch | None = None
        self._installed = False
        self._capture_count = 0

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def tokenizer(self) -> Any:
        return self.prepare_tokenizer()

    def prepare_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        if self.bert_path is None or not self.bert_path.is_dir():
            raise BackendIntegrationError(
                "LLaMA-VID requires runtime.metadata['llamavid_bert_path'] naming "
                "an existing local BERT Base Uncased directory; downloads are forbidden"
            )
        try:
            from transformers import BertTokenizer
        except ImportError as exc:
            raise BackendIntegrationError(
                "LLaMA-VID requires BERT support from the existing Transformers runtime"
            ) from exc
        try:
            tokenizer = BertTokenizer.from_pretrained(
                str(self.bert_path),
                truncation_side="left",
                local_files_only=True,
            )
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        except Exception as exc:
            raise BackendIntegrationError(
                "LLaMA-VID could not load the supplied local BERT tokenizer"
            ) from exc
        self._tokenizer = tokenizer
        return tokenizer

    def bind_qformer(
        self,
        module: Any,
        dimensions: Mapping[str, int],
    ) -> None:
        if module is None or not isinstance(dimensions, Mapping):
            raise BackendIntegrationError("LLaMA-VID QFormer binding is invalid")
        self._qformer_module = module
        self._projector_dimensions = dict(dimensions)

    def install(self) -> None:
        if self._installed:
            return
        self.prepare_tokenizer()
        if self._qformer_module is None or self._projector_dimensions is None:
            raise BackendIntegrationError(
                "LLaMA-VID QFormer must be registered before provider installation"
            )
        existing = getattr(self.model, _PROVIDER_ATTRIBUTE, None)
        if existing is not None and existing is not self:
            raise BackendIntegrationError(
                "A different LLaMA-VID video signal provider is already installed"
            )
        owner = self._capture_owner()
        original = getattr(owner, "forward", None)
        if not callable(original):
            raise BackendIntegrationError(
                "LLaMA-VID could not resolve the native visual forward"
            )
        patch = MethodPatch.apply(
            owner,
            "forward",
            self._capture_replacement(original),
            original,
            label="LLaMA-VID",
        )
        try:
            setattr(self.model, _PROVIDER_ATTRIBUTE, self)
        except BaseException:
            patch.restore()
            raise
        self._patch = patch
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        errors = []
        if getattr(self.model, _PROVIDER_ATTRIBUTE, None) is not self:
            errors.append("provider ownership changed while active")
        elif _PROVIDER_ATTRIBUTE in vars(self.model):
            delattr(self.model, _PROVIDER_ATTRIBUTE)
        if self._patch is not None:
            try:
                self._patch.restore()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        self._patch = None
        self._active.set(None)
        self._installed = False
        if errors:
            raise BackendIntegrationError(
                f"LLaMA-VID signal-provider uninstall failed: {errors}"
            )

    @contextmanager
    def capture_context(
        self,
        *,
        frames: Any | None = None,
        grid_thw: Any | None = None,
    ) -> Iterator[_LLaMAVIDCapture]:
        if not self._installed:
            raise BackendIntegrationError("LLaMA-VID signal provider is not installed")
        if self._active.get() is not None:
            raise BackendIntegrationError("Nested LLaMA-VID capture is unsupported")
        if self.backbone == "llava_onevision":
            if getattr(frames, "ndim", None) != 4 or grid_thw is not None:
                raise BackendIntegrationError(
                    "LLaVA LLaMA-VID capture requires one rank-four frame tensor"
                )
            capture = _LLaMAVIDCapture(frames=frames)
        else:
            if frames is not None or getattr(grid_thw, "ndim", None) != 2:
                raise BackendIntegrationError(
                    "Qwen LLaMA-VID capture requires rank-two video_grid_thw"
                )
            if int(grid_thw.shape[0]) != 1:
                raise BackendIntegrationError(
                    "LLaMA-VID supports exactly one Qwen video grid row"
                )
            capture = _LLaMAVIDCapture(grid_thw=grid_thw)
        token = self._active.set(capture)
        try:
            yield capture
        finally:
            self._active.reset(token)

    def payload_signals(
        self,
        capture: _LLaMAVIDCapture,
        *,
        token_count: int,
        coordinates: Sequence[Any],
    ) -> Mapping[str, Any]:
        if not isinstance(capture, _LLaMAVIDCapture) or capture.calls != 1:
            raise BackendIntegrationError(
                "LLaMA-VID capture did not observe exactly one visual pass"
            )
        if capture.raw_features is None:
            raise BackendIntegrationError("LLaMA-VID capture is missing raw features")
        if token_count < 1 or len(tuple(coordinates)) != token_count:
            raise BackendIntegrationError(
                "LLaMA-VID projected coordinates do not match the payload"
            )
        return {}

    def enrich_payload(
        self,
        payload: VideoTokenPayload,
        capture: _LLaMAVIDCapture,
    ) -> VideoTokenPayload:
        if not isinstance(payload, VideoTokenPayload):
            raise TypeError("payload must be a VideoTokenPayload")
        if payload.video_count != 1:
            raise BackendIntegrationError(
                "LLaMA-VID provider requires exactly one video"
            )
        if capture.calls != 1 or capture.raw_features is None:
            raise BackendIntegrationError("LLaMA-VID capture is incomplete")
        if self._qformer_module is None or self._projector_dimensions is None:
            raise BackendIntegrationError("LLaMA-VID provider lost its QFormer")
        if self.backbone == "llava_onevision":
            raw, raw_coordinates = self._llava_raw(capture)
            raw_source = "llava_outer_vision_pre_projector"
        else:
            raw, raw_coordinates = self._qwen_raw(capture)
            raw_source = "qwen_final_visual_block_pre_merger_raster"
        text = payload.signals.get("text_context")
        if not isinstance(text, str) or not text.strip():
            raise BackendIntegrationError(
                "LLaMA-VID requires one non-empty user-text context"
            )
        tokenizer = self.prepare_tokenizer()
        tokens = tokenizer(
            [text.strip()],
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = getattr(tokens, "input_ids", None)
        attention_mask = getattr(tokens, "attention_mask", None)
        if input_ids is None and isinstance(tokens, Mapping):
            input_ids = tokens.get("input_ids")
            attention_mask = tokens.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise BackendIntegrationError(
                "LLaMA-VID BERT tokenizer omitted input ids or attention mask"
            )
        raw = raw.to(device=payload.features.device, dtype=payload.features.dtype)
        self._capture_count += 1
        return VideoTokenPayload(
            features=payload.features,
            media_ids=payload.media_ids,
            video_offsets=payload.video_offsets,
            coordinates=payload.coordinates,
            raw_features=raw,
            raw_video_offsets=(0, int(raw.shape[0])),
            raw_coordinates=raw_coordinates,
            token_mask=payload.token_mask,
            signals={
                **dict(payload.signals),
                "bert_qformer": self._qformer_module,
                "projector_dimensions": dict(self._projector_dimensions),
                "llamavid.bert_tokens_by_video": (
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    },
                ),
            },
            metadata={
                **dict(payload.metadata),
                "llamavid_raw_stream": raw_source,
                "llamavid_text_policy": "user_text_without_teacher_forced_answer",
                "llamavid_bert_local_only": True,
            },
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": f"{self.backbone}_llamavid_video_signals",
            "installed": self._installed,
            "capture_count": self._capture_count,
            "video_only": True,
            "raw_stream_attached": True,
            "bert_path": None if self.bert_path is None else str(self.bert_path),
            "bert_tokenizer_loaded": self._tokenizer is not None,
            "qformer_bound": self._qformer_module is not None,
            "text_max_length": 256,
            "text_truncation_side": "left",
            "real_checkpoint_verified": False,
        }

    def _capture_owner(self) -> Any:
        if self.backbone == "llava_onevision":
            get_vision = getattr(self.model, "get_vision_tower", None)
            outer = get_vision() if callable(get_vision) else None
            if outer is None:
                raise BackendIntegrationError(
                    "LLaVA LLaMA-VID requires the native outer vision tower"
                )
            return outer
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        blocks = tuple(getattr(visual, "blocks", ()))
        if not blocks:
            raise BackendIntegrationError(
                "LLaMA-VID requires native Qwen visual blocks"
            )
        return blocks[-1]

    def _capture_replacement(
        self,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        def replacement(_owner: Any, *args: Any, **kwargs: Any) -> Any:
            capture = self._active.get()
            output = original(*args, **kwargs)
            if capture is None:
                return output
            if capture.calls:
                raise BackendIntegrationError(
                    "LLaMA-VID observed multiple visual encoder calls"
                )
            if getattr(output, "ndim", None) not in {2, 3}:
                raise BackendIntegrationError(
                    "LLaMA-VID native visual encoder returned an unknown raw shape"
                )
            capture.raw_features = output
            capture.calls = 1
            return output

        return replacement

    def _llava_raw(
        self,
        capture: _LLaMAVIDCapture,
    ) -> tuple[Any, tuple[VideoTokenCoordinate, ...]]:
        raw = capture.raw_features
        frames = capture.frames
        if getattr(raw, "ndim", None) != 3 or int(raw.shape[0]) != int(frames.shape[0]):
            raise BackendIntegrationError(
                "LLaVA LLaMA-VID raw features do not match the frame count"
            )
        token_count = int(raw.shape[1])
        side = int(token_count**0.5)
        if side * side != token_count:
            raise BackendIntegrationError(
                "LLaVA LLaMA-VID raw features are not a square patch raster"
            )
        coordinates = tuple(
            VideoTokenCoordinate(t=frame, h=row, w=column, frame_index=frame)
            for frame in range(int(raw.shape[0]))
            for row in range(side)
            for column in range(side)
        )
        return raw.reshape(-1, int(raw.shape[-1])), coordinates

    def _qwen_raw(
        self,
        capture: _LLaMAVIDCapture,
    ) -> tuple[Any, tuple[VideoTokenCoordinate, ...]]:
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        merge_size = _first_positive_integer(
            getattr(visual, "spatial_merge_size", None),
            getattr(getattr(visual, "config", None), "spatial_merge_size", None),
        )
        return restore_qwen_video_raster(
            capture.raw_features,
            capture.grid_thw,
            visual,
            backbone=self.backbone,
            merge_size=merge_size,
            merge_unit=_first_positive_integer(
                getattr(visual, "spatial_merge_unit", None), merge_size * merge_size
            ),
            method_name="LLaMA-VID",
        )


class LLaMAVIDVideoExecutor:
    """Install local BERT/QFormer state before the native optimizer is built."""

    algorithm = "llamavid"
    variant = "llamavid.video"

    def __init__(
        self,
        *,
        backbone: str,
        model: Any,
        runtime: Any | None = None,
        facade: Any | None = None,
        stage: Any | None = None,
        binding: Any | None = None,
        binding_config: Any | None = None,
        contract_projection: Any | None = None,
        execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
        signal_provider: Any | None = None,
        qformer_module: Any | None = None,
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown LLaMA-VID video backbone {backbone!r}")
        self.backbone = backbone
        self.model = model
        self.runtime = runtime
        self.facade = facade
        self.stage = stage
        self.binding = binding
        self.binding_config = binding_config
        self.execution_phase = ExecutionPhase(execution_phase)
        self.config = (
            contract_projection.algorithm_config
            if contract_projection is not None
            else _validated_config(stage, binding_config)
        )
        self.pool_position = video_learned_projector_position(stage, backbone, target_name="qformer_context_content_pool")
        self.target_site = video_learned_projector_site(self.algorithm, self.pool_position)
        _validate_binding(binding, stage, backbone, self.execution_phase)
        runtime_metadata = (
            getattr(runtime, "metadata", {}) if runtime is not None else {}
        )
        self.bert_path = (
            runtime_metadata.get("llamavid_bert_path")
            if isinstance(runtime_metadata, Mapping)
            else None
        )
        self.kernel = LLaMAVIDVideoKernel()
        self._rules, self._rule_specs = load_video_rules(stage, contract_projection=contract_projection)
        self.signal_provider = signal_provider or LLaMAVIDVideoSignalProvider(
            backbone=backbone,
            model=model,
            bert_path=self.bert_path,
        )
        self.qformer_module = qformer_module
        self._qformer_prepared = False
        self._qformer_attached = False
        self._owns_qformer_attribute = False
        self._bridge: Any | None = None
        self._bridge_handle: Any | None = None
        self._owns_bridge = False
        self._installed = False
        self._invocations = 0
        self._layout_records: list[dict[str, Any]] = []
        self._algorithm_records: list[dict[str, Any]] = []

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        if self._installed:
            return
        bridge = None
        handle = None
        owns_bridge = False
        try:
            self._prepare_qformer_module()
            self._attach_qformer_module()
            self.signal_provider.install()
            bridge, owns_bridge, projector_site = install_video_bridge(self.model, self.backbone)
            handle = register_video_callback(
                bridge,
                projector_site,
                self._projector_callback,
                label="LLaMA-VID",
                signal_provider=self.signal_provider,
            )
        except BaseException:
            release_video_callback(bridge, handle, owns_bridge)
            try:
                self.signal_provider.uninstall()
            except Exception:
                pass
            self._detach_qformer_module()
            raise
        self._bridge = bridge
        self._bridge_handle = handle
        self._owns_bridge = owns_bridge
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        errors = release_video_callback(
            self._bridge, self._bridge_handle, self._owns_bridge
        )
        try:
            self.signal_provider.uninstall()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        try:
            self._detach_qformer_module()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        self._bridge_handle = None
        self._bridge = None
        self._owns_bridge = False
        self._installed = False
        if errors:
            raise BackendIntegrationError(f"LLaMA-VID video uninstall failed: {errors}")

    def compression_state_dict(self) -> Mapping[str, Any]:
        self._prepare_qformer_module()
        state_dict = getattr(self.qformer_module, "state_dict", None)
        if not callable(state_dict):
            raise BackendIntegrationError(
                "LLaMA-VID QFormer does not expose state_dict()"
            )
        return state_dict()

    def load_compression_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("LLaMA-VID compression state must be a mapping")
        self._prepare_qformer_module()
        loader = getattr(self.qformer_module, "load_state_dict", None)
        if not callable(loader):
            raise BackendIntegrationError(
                "LLaMA-VID QFormer does not expose load_state_dict()"
            )
        loader(state, strict=True)

    def runtime_metadata(self) -> dict[str, Any]:
        parameter_count, trainable_count = _parameter_counts(self.qformer_module)
        provider_metadata = getattr(self.signal_provider, "runtime_metadata", None)
        binding_metadata = getattr(self.binding, "metadata", {})
        if not isinstance(binding_metadata, Mapping):
            binding_metadata = {}
        verified = self.pool_position == "before" and self.execution_phase.value in binding_metadata.get(
            "verified_phases", ()
        )
        return {
            "executor": f"{self.backbone}_llamavid_video",
            "algorithm": self.algorithm,
            "variant": self.variant,
            "backbone": self.backbone,
            "binding": getattr(self.binding, "name", None),
            "execution_phase": self.execution_phase.value,
            "installed": self._installed,
            "canonical_config": dict(self.config),
            "pool_position": self.pool_position,
            "canonical_evidence_applicable": self.pool_position == "before",
            "target_site": self.target_site,
            "projector_owner": "llamavid_qformer",
            "dense_backbone_projector_reference": True,
            "output_projectors_before_pool": self.pool_position == "after",
            "projector_invocations": self._invocations,
            "records_scope": "latest_video_request",
            "projector_layout_deltas": list(self._layout_records),
            "algorithm_records": list(self._algorithm_records),
            "signal_provider": (
                provider_metadata() if callable(provider_metadata) else None
            ),
            "bridge": (
                self._bridge.runtime_metadata()
                if self._bridge is not None
                and callable(getattr(self._bridge, "runtime_metadata", None))
                else None
            ),
            "stages": (
                "bert_qformer_cross_attention",
                "context_attention_pool",
                "mean_content_pool",
            ),
            "qformer_attribute": _QFORMER_ATTRIBUTE,
            "qformer_registered_pre_optimizer": self._qformer_attached,
            "qformer_parameter_count": parameter_count,
            "qformer_trainable_parameter_count": trainable_count,
            "bert_path": self.bert_path,
            "method_checkpoint_required": False,
            "compression_checkpoint_api": (
                "compression_state_dict",
                "load_compression_state_dict",
            ),
            "real_checkpoint_verified": verified,
            "support_activation_claimed": verified,
            "evidence_artifact": (
                binding_metadata.get(f"{self.execution_phase.value}_evidence_artifact")
                if verified
                else None
            ),
            **video_rule_metadata(self._rule_specs),
        }

    def clear_diagnostic_records(self) -> None:
        """Discard only diagnostic history; keep counters and replay state."""

        self._layout_records.clear()
        self._algorithm_records.clear()

    def _projector_callback(self, request: Any) -> Any:
        self.clear_diagnostic_records()
        result = self.kernel(
            getattr(request, "payload", None),
            qformer_module=self.qformer_module,
            config=self.config,
            pool_position=self.pool_position,
        **({"rules": self._rules} if self._rules else {}))
        replay_record: dict[str, Any] = {}
        if self.backbone == "qwen3_vl":
            from mllm_comp.backbones.qwen3_vl.video_compression import (
                Qwen3VLVideoCompressionResponse,
            )

            deepstack = tuple(
                replay_llamavid_value_lineage(branch, result, **({"rules": self._rules} if self._rules else {}))
                for branch in request.deepstack_features
            )
            returned = Qwen3VLVideoCompressionResponse.from_result(
                request,
                result,
                deepstack_features=deepstack,
            )
            replay_record = {
                "deepstack_branch_count": len(deepstack),
                "deepstack_source_shapes": tuple(
                    tuple(int(item) for item in branch.shape)
                    for branch in request.deepstack_features
                ),
                "deepstack_result_shapes": tuple(
                    tuple(int(item) for item in branch.shape) for branch in deepstack
                ),
            }
        else:
            returned = result
        self._invocations += 1
        record = layout_delta_record(result.layout_deltas[0])
        record.update(
            {
                "stage": "qformer_context_content_pool",
                "site": self.target_site,
                "pool_position": self.pool_position,
                "invocation_id": getattr(request, "invocation_id", None),
                "media_id": getattr(request, "media_id", None),
                "tokens_per_temporal_unit": 2,
            }
        )
        self._layout_records.append(record)
        algorithm_record = _llamavid_invocation_record(request, result)
        algorithm_record.update(replay_record)
        algorithm_record.update(target_site=self.target_site, pool_position=self.pool_position)
        self._algorithm_records.append(algorithm_record)
        return returned

    def _prepare_qformer_module(self) -> None:
        if self._qformer_prepared:
            return
        raw_hidden, output_hidden = _model_dimensions(self.model, self.backbone)
        dimensions = {
            "raw_hidden_size": raw_hidden,
            "output_hidden_size": output_hidden,
            "num_query": int(self.config["num_query"]),
        }
        if self.qformer_module is None:
            prepare_tokenizer = getattr(self.signal_provider, "prepare_tokenizer", None)
            if not callable(prepare_tokenizer):
                raise BackendIntegrationError(
                    "LLaMA-VID signal provider cannot prepare its tokenizer"
                )
            tokenizer = prepare_tokenizer()
            source = self.bert_path or getattr(self.signal_provider, "bert_path", None)
            if source is None:
                raise BackendIntegrationError(
                    "LLaMA-VID local BERT path was not supplied"
                )
            from mllm_comp.backbones._llamavid_torch import LLaMAVIDQFormerModule

            self.qformer_module = LLaMAVIDQFormerModule.from_local_bert(
                source,
                vocab_size=len(tokenizer),
                raw_hidden_size=raw_hidden,
                output_hidden_size=output_hidden,
                num_query=int(self.config["num_query"]),
            )
            device, dtype = _model_device_dtype(self.model)
            self.qformer_module.to(device=device, dtype=dtype)
        bind = getattr(self.signal_provider, "bind_qformer", None)
        if callable(bind):
            bind(self.qformer_module, dimensions)
        self._qformer_prepared = True

    def _attach_qformer_module(self) -> None:
        if self._qformer_attached:
            return
        existing = getattr(self.model, _QFORMER_ATTRIBUTE, None)
        if existing is not None and existing is not self.qformer_module:
            raise BackendIntegrationError(
                "A different LLaMA-VID QFormer is already registered"
            )
        train = getattr(self.qformer_module, "train", None)
        if callable(train):
            train(self.model.training)
        if existing is None:
            setattr(self.model, _QFORMER_ATTRIBUTE, self.qformer_module)
            self._owns_qformer_attribute = True
        self._qformer_attached = True

    def _detach_qformer_module(self) -> None:
        if not self._qformer_attached:
            return
        if getattr(self.model, _QFORMER_ATTRIBUTE, None) is not self.qformer_module:
            raise BackendIntegrationError(
                "LLaMA-VID QFormer ownership changed while active"
            )
        if self._owns_qformer_attribute:
            delattr(self.model, _QFORMER_ATTRIBUTE)
        self._qformer_attached = False
        self._owns_qformer_attribute = False


def _llamavid_invocation_record(request: Any, result: Any) -> dict[str, Any]:
    """Freeze the defining raw/text/BERT/QFormer signals and value lineage."""

    torch = _torch()
    payload = getattr(request, "payload", None)
    features = getattr(payload, "features", None)
    raw_features = getattr(payload, "raw_features", None)
    signals = getattr(payload, "signals", {})
    source_count = int(getattr(result, "source_count", 0))
    result_count = int(getattr(result, "result_count", 0))
    if (
        getattr(features, "ndim", None) != 2
        or getattr(raw_features, "ndim", None) != 2
        or int(features.shape[0]) != source_count
    ):
        raise BackendIntegrationError(
            "LLaMA-VID invocation evidence lost aligned projected/raw features"
        )
    finite = {
        "projected_features_finite": bool(torch.isfinite(features).all().item()),
        "raw_features_finite": bool(torch.isfinite(raw_features).all().item()),
        "result_features_finite": bool(
            torch.isfinite(getattr(result, "features", None)).all().item()
        ),
    }
    if not all(finite.values()):
        raise BackendIntegrationError(
            "LLaMA-VID invocation evidence contains non-finite data"
        )

    metadata = getattr(result, "metadata", {})
    media_ids = tuple(getattr(result, "media_ids", ()))
    source_offsets = tuple(int(value) for value in result.source_video_offsets)
    result_offsets = tuple(int(value) for value in result.video_offsets)
    groups = tuple(
        tuple(tuple(int(index) for index in group) for group in video_groups)
        for video_groups in result.source_groups
    )
    representatives = tuple(
        tuple(int(index) for index in values)
        for values in result.representative_indices
    )
    value_records = tuple(metadata.get("value_records_by_video", ()))
    if (
        len(media_ids) != 1
        or len(source_offsets) != 2
        or len(result_offsets) != 2
        or len(groups) != 1
        or len(representatives) != 1
        or len(value_records) != 1
        or len(groups[0]) != result_count
        or len(representatives[0]) != result_count
        or len(value_records[0]) != result_count
    ):
        raise BackendIntegrationError(
            "LLaMA-VID invocation evidence requires one complete video record"
        )
    text_context = signals.get("text_context") if isinstance(signals, Mapping) else None
    tokens_by_video = (
        signals.get("llamavid.bert_tokens_by_video")
        if isinstance(signals, Mapping)
        else None
    )
    token_shapes: tuple[dict[str, tuple[int, ...]], ...] = ()
    if tokens_by_video is not None:
        token_shapes = tuple(
            {
                name: tuple(int(item) for item in value.shape)
                for name, value in token_record.items()
            }
            for token_record in tokens_by_video
        )
    payload_metadata = getattr(payload, "metadata", {})
    return {
        **dict(metadata),
        **finite,
        "algorithm": "llamavid",
        "variant": "llamavid.video",
        "source_count": source_count,
        "result_count": result_count,
        "video_count": 1,
        "per_video_source_counts": (source_offsets[1] - source_offsets[0],),
        "per_video_result_counts": (result_offsets[1] - result_offsets[0],),
        "exact_budgets": tuple(
            int(value) for value in metadata.get("exact_budgets", ())
        ),
        "projected_features_shape": tuple(int(item) for item in features.shape),
        "raw_features_shape": tuple(int(item) for item in raw_features.shape),
        "result_features_shape": tuple(int(item) for item in result.features.shape),
        "source_groups": groups,
        "representative_indices": representatives,
        "text_context_present": isinstance(text_context, str)
        and bool(text_context.strip()),
        "bert_token_shapes": token_shapes,
        "raw_stream": payload_metadata.get("llamavid_raw_stream"),
        "text_policy": payload_metadata.get("llamavid_text_policy"),
        "bert_local_only": payload_metadata.get("llamavid_bert_local_only"),
    }


def create_llamavid_video_executor(
    *,
    backbone: str,
    **kwargs: Any,
) -> LLaMAVIDVideoExecutor:
    return LLaMAVIDVideoExecutor(backbone=backbone, **kwargs)


def _validated_config(stage: Any, binding_config: Any) -> Mapping[str, Any]:
    stage_config = getattr(stage, "config", {}) if stage is not None else {}
    if stage_config is None:
        stage_config = {}
    if not isinstance(stage_config, Mapping):
        raise BackendIntegrationError("LLaMA-VID stage config must be a mapping")
    binding_values = (
        getattr(binding_config, "config", {}) if binding_config is not None else {}
    )
    if binding_values not in ({}, None):
        raise BackendIntegrationError(
            "LLaMA-VID binding does not accept binding-local config"
        )
    return resolve_canonical_video_config("llamavid", stage_config)


def _validate_binding(
    binding: Any,
    stage: Any,
    backbone: str,
    phase: ExecutionPhase,
) -> None:
    expected = f"llamavid.video.{backbone}"
    if binding is not None:
        if getattr(binding, "name", None) != expected:
            raise BackendIntegrationError(
                f"LLaMA-VID executor requires binding {expected!r}"
            )
        if phase not in getattr(binding, "supported_phases", ()):
            raise BackendIntegrationError(
                f"LLaMA-VID binding {expected!r} does not support {phase.value}"
            )
        targets = tuple(
            getattr(item, "name", None) for item in binding.ordered_targets()
        )
        if targets != ("qformer_context_content_pool",):
            raise BackendIntegrationError("LLaMA-VID binding target changed")
    if stage is not None:
        if getattr(stage, "algorithm", None) != "llamavid":
            raise BackendIntegrationError(
                "LLaMA-VID executor received another algorithm"
            )
        if getattr(stage, "variant", None) != "llamavid.video":
            raise BackendIntegrationError("LLaMA-VID executor received another variant")


def _model_dimensions(model: Any, backbone: str) -> tuple[int, int]:
    if backbone == "llava_onevision":
        get_vision = getattr(model, "get_vision_tower", None)
        outer = get_vision() if callable(get_vision) else None
        raw = _first_positive_integer(
            getattr(outer, "hidden_size", None),
            getattr(getattr(outer, "config", None), "hidden_size", None),
            getattr(
                getattr(getattr(outer, "vision_tower", None), "config", None),
                "hidden_size",
                None,
            ),
        )
        config = getattr(model, "config", None)
        output = _first_positive_integer(
            getattr(config, "hidden_size", None),
            getattr(getattr(config, "text_config", None), "hidden_size", None),
        )
        return raw, output
    visual = getattr(getattr(model, "model", None), "visual", None)
    blocks = tuple(getattr(visual, "blocks", ()))
    if not blocks:
        raise BackendIntegrationError(
            "LLaMA-VID could not infer Qwen visual dimensions"
        )
    norm = getattr(blocks[-1], "norm1", None)
    shape = getattr(norm, "normalized_shape", None)
    norm_hidden = shape[-1] if isinstance(shape, Sequence) and shape else None
    raw = _first_positive_integer(
        norm_hidden,
        getattr(getattr(visual, "config", None), "hidden_size", None),
        getattr(getattr(visual, "config", None), "embed_dim", None),
    )
    config = getattr(model, "config", None)
    output = _first_positive_integer(
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(config, "hidden_size", None),
    )
    return raw, output


def _model_device_dtype(model: Any) -> tuple[Any, Any]:
    torch = _torch()
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise BackendIntegrationError(
            "LLaMA-VID requires a loaded PyTorch model before registration"
        )
    parameter = next(iter(parameters()), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


def _parameter_counts(module: Any) -> tuple[int, int]:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0, 0
    values = tuple(parameters())
    return (
        sum(int(value.numel()) for value in values),
        sum(int(value.numel()) for value in values if bool(value.requires_grad)),
    )


def _first_positive_integer(*values: Any) -> int:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            sequence = tuple(value)
            if sequence and all(
                isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in sequence
            ):
                return int(sequence[-1])
    raise BackendIntegrationError(
        "LLaMA-VID could not infer a required model dimension"
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise BackendIntegrationError(
            "LLaMA-VID provider requires the existing PyTorch runtime"
        ) from exc
    return torch


__all__ = [
    "LLaMAVIDVideoExecutor",
    "LLaMAVIDVideoSignalProvider",
    "create_llamavid_video_executor",
]
