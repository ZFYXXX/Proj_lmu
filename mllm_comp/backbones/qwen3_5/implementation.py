"""Qwen3.5 image/video adapter for the MLLM-Comp runtime.

This is intentionally separate from the Qwen3-VL adapter: Qwen3.5 has a
different Transformers implementation, no DeepStack visual branches, and uses
the Transformers 5.x loading path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from mllm_comp.backbones.base import (
    BackboneFacade,
    BackboneRuntime,
    GenerationResult,
    TextGenerationOutput,
    TraceRequest,
)
from mllm_comp.backbones.profiles import QWEN3_5_PROFILE
from mllm_comp.backbones.qwen3_vl.implementation import (
    Qwen3VLSample,
    Qwen3VLProcessorAdapter,
    Qwen3VLStateBuilder,
    Qwen3VLLegalizer,
    _conversation_modality_counts,
    _normalize_samples,
    _move_value,
    _raise_violations,
    _shape,
    _slice_generated_ids,
)
from .validation import Qwen3_5NativeValidator
from mllm_comp.core.errors import BackendIntegrationError, InvariantError


# The sample protocol is deliberately shared; model/provider implementations
# remain isolated even though their conversation envelope is identical.
Qwen3_5Sample = Qwen3VLSample


@dataclass(frozen=True)
class Qwen3_5ProcessorAdapter(Qwen3VLProcessorAdapter):
    """Reuse tested Qwen3-VL media normalization for Qwen3.5 image/video inputs."""

    profile: Any = QWEN3_5_PROFILE

    def prepare_inputs(self, processor: Any, raw_sample: Any, **kwargs: Any):
        samples = []
        for sample in _normalize_samples(raw_sample):
            counts = _conversation_modality_counts(sample.conversation)
            if counts.get("audio", 0):
                raise BackendIntegrationError(
                    "The Qwen3.5 MLLM-Comp adapter supports image/video input, not audio."
                )
            # Match the direct Qwen3.5 smoke and keep short-answer benchmark
            # prompts from spending their token budget on an implicit <think>.
            if sample.enable_thinking is None:
                sample = replace(sample, enable_thinking=False)
            samples.append(sample)
        return super().prepare_inputs(processor, samples, **kwargs)


@dataclass(frozen=True)
class Qwen3_5ModelAdapter:
    """Load the Transformers 5.x Qwen3.5 model and expose shared handles."""

    profile: Any = QWEN3_5_PROFILE

    def capabilities(self):
        return self.profile

    def load_model(
        self,
        model_path_or_id: str,
        *,
        train: bool,
        revision: str | None = None,
        **load_kwargs: Any,
    ) -> Any:
        try:
            from transformers import AutoConfig, AutoModelForMultimodalLM
        except ImportError as exc:
            raise BackendIntegrationError(
                "Qwen3.5 requires the isolated Transformers 5.x runtime exposing "
                "AutoModelForMultimodalLM."
            ) from exc
        if revision is not None:
            load_kwargs["revision"] = revision
        config_kwargs = {
            key: load_kwargs[key]
            for key in ("revision", "trust_remote_code", "local_files_only")
            if key in load_kwargs
        }
        config = AutoConfig.from_pretrained(model_path_or_id, **config_kwargs)
        if getattr(config, "model_type", None) != "qwen3_5":
            raise BackendIntegrationError(
                f"Expected Qwen3.5 model_type='qwen3_5', got "
                f"{getattr(config, 'model_type', None)!r}."
            )
        if "torch_dtype" in load_kwargs and "dtype" not in load_kwargs:
            load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
        model = AutoModelForMultimodalLM.from_pretrained(model_path_or_id, **load_kwargs)
        model.train(mode=train)
        return model

    def structural_handles(self, model: Any) -> Mapping[str, Any]:
        core = getattr(model, "model", None)
        visual = getattr(core, "visual", None)
        language_model = getattr(core, "language_model", None)
        get_embeddings = getattr(model, "get_input_embeddings", None)
        handles = {
            "model": core,
            "llm": language_model,
            "language_model": language_model,
            "llm.input_embeddings": get_embeddings() if callable(get_embeddings) else None,
            "llm.final_norm": getattr(language_model, "norm", None),
            "lm_head": getattr(model, "lm_head", None),
            "vision_encoder": visual,
            "vision_projector": getattr(visual, "merger", None),
            "image_features_pre_llm_scatter": getattr(core, "get_image_features", None),
            "video_features_pre_llm_scatter": getattr(core, "get_video_features", None),
            "mrope_owner": core,
        }
        return {name: value for name, value in handles.items() if value is not None}

    def external_code_info(self) -> dict[str, Any]:
        return {
            "code_origin": self.profile.code_origin,
            "source_ref": self.profile.source_ref,
            "loader_strategy": self.profile.loader_strategy,
            "model_type": "qwen3_5",
            "minimum_transformers_version": "5.0",
            "verified_transformers_version": "5.19.0",
            "supported_input_slice": "text/image/video -> text",
            "video_support": True,
            "audio_support": False,
        }


@dataclass(frozen=True)
class Qwen3_5Runner:
    """Run image/text inference through the native Qwen3.5 generate API."""

    processor_adapter: Qwen3_5ProcessorAdapter
    native_validator: Qwen3_5NativeValidator
    profile: Any = QWEN3_5_PROFILE

    def forward_train(self, runtime: BackboneRuntime, state: Any) -> Any:
        session = runtime.metadata.get("_compression_session")
        if session is not None:
            session.clear_diagnostic_records()
        model_inputs = {
            name: _move_value(value, device=runtime.device, dtype=runtime.dtype)
            for name, value in state.native_model_inputs.items()
        }
        return runtime.model(**model_inputs)

    def generate(
        self,
        runtime: BackboneRuntime,
        state: Any,
        *,
        processed_batch: Any,
        trace_request: TraceRequest | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        session = runtime.metadata.get("_compression_session")
        if session is not None:
            session.clear_diagnostic_records()
        _raise_violations(
            "Qwen3.5 native input validation failed",
            self.native_validator.validate_state(runtime, state, processed_batch),
        )
        if trace_request is not None and trace_request.enabled():
            raise InvariantError("Qwen3.5 tracing is not part of the initial image/video-compression slice.")
        if gen_kwargs.pop("return_audio", False):
            raise InvariantError("Qwen3.5 image/video adapter has no audio output branch.")
        if gen_kwargs.pop("stream", False) or gen_kwargs.get("streamer") is not None:
            raise InvariantError("Qwen3.5 adapter currently returns complete generations only.")
        model_inputs = {
            name: _move_value(value, device=runtime.device, dtype=runtime.dtype)
            for name, value in state.native_model_inputs.items()
        }
        shape = _shape(model_inputs.get("input_ids"))
        if shape is None or len(shape) != 2:
            raise InvariantError("Qwen3.5 generation requires rank-2 input_ids.")
        input_width = shape[1]
        if gen_kwargs.get("do_sample") is False:
            gen_kwargs.setdefault("temperature", None)
            gen_kwargs.setdefault("top_p", None)
            gen_kwargs.setdefault("top_k", None)
        raw_output = runtime.model.generate(**model_inputs, **gen_kwargs)
        sequences = getattr(raw_output, "sequences", raw_output)
        generated_ids = _slice_generated_ids(sequences, input_width)
        decoded = self.processor_adapter.detokenize(runtime.processor, generated_ids)
        result = GenerationResult(
            text=TextGenerationOutput(token_ids=generated_ids, text=decoded),
            speech=None,
            traces=None,
            sample_ids=list(processed_batch.sample_ids),
            metadata={
                "provider": "transformers.AutoModelForMultimodalLM.generate",
                "backbone": "qwen3_5",
                "input_sequence_length": input_width,
                "streaming_supported": False,
            },
        )
        _raise_violations(
            "Qwen3.5 native output validation failed",
            self.native_validator.validate_generation(runtime, state, processed_batch, result),
        )
        return result


def create_qwen3_5_backbone() -> BackboneFacade:
    """Compose the isolated Qwen3.5 Transformers-5.x backbone."""

    processor_adapter = Qwen3_5ProcessorAdapter(profile=QWEN3_5_PROFILE)
    validator = Qwen3_5NativeValidator()
    return BackboneFacade(
        processor_adapter=processor_adapter,
        model_adapter=Qwen3_5ModelAdapter(),
        state_builder=Qwen3VLStateBuilder(profile=QWEN3_5_PROFILE),
        legalizer=Qwen3VLLegalizer(profile=QWEN3_5_PROFILE),
        runner=Qwen3_5Runner(processor_adapter, validator),
        native_validator=validator,
    )
