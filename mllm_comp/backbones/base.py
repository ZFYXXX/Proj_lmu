"""Backbone adapter protocols and the composed facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol

from mllm_comp.core.actions import LegalizationPolicy, OperationAction, OperationApplyResult, TokenAction
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.invariants import InvariantViolation
from mllm_comp.core.state import ForwardState
from mllm_comp.backbones.profiles import BackboneCapabilities


@dataclass
class BackboneRuntime:
    """Heavyweight objects owned by one command/inference lifecycle."""

    model: Any
    processor: Any
    model_path_or_id: str | None = None
    revision: str | None = None
    device: Any | None = None
    dtype: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        """Uninstall owned runtime hooks and release this experiment's objects.

        Callers must collect result metadata before closing. Closing again is
        harmless; a closed runtime is not reusable for another experiment.
        """

        errors: list[str] = []
        try:
            # The loader installs compression first, then the debug collector.
            for key in ("_token_debug_collector", "_modality_split_session", "_compression_session"):
                resource = self.metadata.pop(key, None)
                if resource is None:
                    continue
                try:
                    resource.uninstall()
                except Exception as exc:
                    errors.append(f"{key}: {type(exc).__name__}: {exc}")
        finally:
            # Metadata may also hold structural handles or a provider bridge
            # that retains model modules. It belongs to this runtime as well.
            self.metadata.clear()
            self.model = None
            self.processor = None
        if errors:
            raise BackendIntegrationError(
                "Backbone runtime cleanup failed: " + "; ".join(errors)
            )


@dataclass
class ProcessedBatch:
    """Processor-native model inputs plus platform-side batch metadata."""

    model_inputs: dict[str, Any]
    sample_ids: list[str | int]
    prompt_lengths: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_batch: Any | None = None


@dataclass
class TextGenerationOutput:
    """Text branch of a generative backbone result."""

    token_ids: Any
    text: list[str]


@dataclass
class SpeechGenerationOutput:
    """Speech branch after Talker and waveform decoding."""

    waveform: Any
    sample_rate: int
    speaker: str | None = None
    codec_token_ids: Any | None = None


@dataclass(frozen=True)
class TraceRequest:
    """Opt-in model internals; all fields are disabled by default."""

    # Generic text-generation branch used by non-Omni backbones.
    text_logits: bool = False
    text_hidden_states: bool = False
    text_attentions: bool = False
    text_kv_cache: bool = False
    # Internal runner mode used by SGL's compact guide. Public trace requests
    # keep their established last-forward-step semantics.
    _full_generation_text_attentions: bool = False

    # Qwen2.5-Omni-specific Thinker/Talker branches.
    thinker_logits: bool = False
    thinker_hidden_states: bool = False
    thinker_attentions: bool = False
    thinker_kv_cache: bool = False
    thinker_reply_part: bool = False
    talker_logits: bool = False
    talker_hidden_states: bool = False
    talker_attentions: bool = False
    talker_kv_cache: bool = False
    talker_codec_token_ids: bool = False
    token2wav_intermediates: bool = False

    def enabled(self) -> bool:
        return any(vars(self).values())


@dataclass
class TraceArtifacts:
    """Optional probe/hook artifacts, never populated by default."""

    text_logits: Any | None = None
    text_hidden_states: Any | None = None
    text_attentions: Any | None = None
    text_kv_cache: Any | None = None

    thinker_logits: Any | None = None
    thinker_hidden_states: Any | None = None
    thinker_attentions: Any | None = None
    thinker_kv_cache: Any | None = None
    thinker_reply_part: Any | None = None
    talker_logits: Any | None = None
    talker_hidden_states: Any | None = None
    talker_attentions: Any | None = None
    talker_kv_cache: Any | None = None
    talker_codec_token_ids: Any | None = None
    token2wav_intermediates: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Normalized text and optional speech output from a backbone runner."""

    text: TextGenerationOutput | None = None
    speech: SpeechGenerationOutput | None = None
    traces: TraceArtifacts | None = None
    sample_ids: list[str | int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ProcessorAdapter(Protocol):
    def capabilities(self) -> BackboneCapabilities: ...

    def load_processor(
        self,
        model_path_or_id: str,
        *,
        revision: str | None = None,
        **load_kwargs: Any,
    ) -> Any: ...

    def prepare_inputs(
        self,
        processor: Any,
        raw_sample: Any,
        *,
        phase: str,
        runtime_mode: str = "default",
        **processor_kwargs: Any,
    ) -> ProcessedBatch: ...

    def detokenize(
        self,
        processor: Any,
        token_ids: Any,
        **decode_kwargs: Any,
    ) -> list[str]: ...


class TeacherForcedProcessorAdapter(Protocol):
    """Optional analysis capability implemented by activated target backbones."""

    def prepare_teacher_forced_inputs(
        self,
        processor: Any,
        raw_sample: Any,
        *,
        target_text: str,
        prompt_batch: ProcessedBatch | None = None,
        phase: str,
        runtime_mode: str = "default",
    ) -> ProcessedBatch: ...


class ModelAdapter(Protocol):
    def capabilities(self) -> BackboneCapabilities: ...

    def load_model(
        self,
        model_path_or_id: str,
        *,
        train: bool,
        revision: str | None = None,
        **load_kwargs: Any,
    ) -> Any: ...

    def structural_handles(self, model: Any) -> Mapping[str, Any]: ...

    def external_code_info(self) -> dict[str, Any]: ...


class StateBuilder(Protocol):
    def build(
        self,
        processed_batch: ProcessedBatch,
        *,
        raw_batch: Any | None = None,
        phase: Literal["dry_run", "train", "eval"] = "train",
    ) -> ForwardState: ...


class Legalizer(Protocol):
    def apply(
        self,
        state: ForwardState,
        *,
        token_actions: list[TokenAction],
        policy: LegalizationPolicy,
        stage_name: str,
    ) -> ForwardState: ...


class NativeValidator(Protocol):
    """Validate one backbone's native state and generated outputs."""

    def validate_state(
        self,
        runtime: BackboneRuntime,
        state: ForwardState,
        processed_batch: ProcessedBatch,
    ) -> list[InvariantViolation]: ...

    def validate_generation(
        self,
        runtime: BackboneRuntime,
        state: ForwardState,
        processed_batch: ProcessedBatch,
        result: GenerationResult,
    ) -> list[InvariantViolation]: ...


class Runner(Protocol):
    def forward_train(self, runtime: BackboneRuntime, state: ForwardState) -> Any: ...

    def generate(
        self,
        runtime: BackboneRuntime,
        state: ForwardState,
        *,
        processed_batch: ProcessedBatch,
        trace_request: TraceRequest | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult: ...


class OperationApplier(Protocol):
    def validate(
        self,
        *,
        model: Any,
        structural_handles: Mapping[str, Any],
        actions: list[OperationAction],
    ) -> list[InvariantViolation]: ...

    def apply(
        self,
        *,
        model: Any,
        structural_handles: Mapping[str, Any],
        actions: list[OperationAction],
        stage_name: str,
        dry_run: bool = False,
    ) -> OperationApplyResult: ...


@dataclass(frozen=True)
class BackboneFacade:
    """Composed entry point for one backbone's adapter responsibilities.

    Runtime processor and model instances are deliberately not stored here.
    The facade owns stateless adapter components; command lifecycles own the
    heavyweight objects returned by those adapters.
    """

    processor_adapter: ProcessorAdapter
    model_adapter: ModelAdapter
    state_builder: StateBuilder
    legalizer: Legalizer
    runner: Runner
    operation_applier: OperationApplier | None = None
    native_validator: NativeValidator | None = None

    def __post_init__(self) -> None:
        self.capabilities()

    def capabilities(self) -> BackboneCapabilities:
        """Return the shared profile and reject incorrectly wired components."""

        processor_capabilities = self.processor_adapter.capabilities()
        model_capabilities = self.model_adapter.capabilities()
        if processor_capabilities != model_capabilities:
            raise ValueError(
                "Backbone facade components expose different capability profiles: "
                f"processor={processor_capabilities.name!r}, "
                f"model={model_capabilities.name!r}."
            )
        return model_capabilities
