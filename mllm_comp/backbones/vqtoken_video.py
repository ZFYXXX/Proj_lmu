"""Installable canonical VQToken video executors for three backbones.

VQToken is a projector-stage-only method.  The existing independent video
bridges own native prompt/grid/MRoPE/DeepStack legalization. This activation
supplies fixed-K clustering and normalized group replay to VideoProjectorExecutor.
Support cells remain declared until real checkpoint and train/reload evidence
is attached separately.
"""

from __future__ import annotations

from typing import Any, Mapping

from mllm_comp.backbones.video_projector import VideoProjectorExecutor, builtin_video_projector_projection
from mllm_comp.compressors.video.config import resolve_canonical_video_config
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.vocabulary import ExecutionPhase


_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")


class VQTokenVideoExecutor(VideoProjectorExecutor):
    """Canonical fixed-K method using the shared video Projector lifecycle."""

    algorithm = "vqtoken"
    variant = "vqtoken.video"

    def __init__(self, *, backbone: str, model: Any, runtime: Any = None,
                 facade: Any = None, stage: Any = None, binding: Any = None,
                 binding_config: Any = None, contract_projection: Any = None,
                 execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown VQToken video backbone {backbone!r}")
        phase = ExecutionPhase(execution_phase)
        if contract_projection is None or contract_projection.algorithm.name == "vqtoken":
            _validate_binding(binding, stage, backbone, phase)
        self._evidence_binding = binding
        if contract_projection is None:
            contract_projection = builtin_video_projector_projection(
                algorithm_name="vqtoken", backbone=backbone, stage=stage, binding=binding,
                config=_validated_config(stage, binding_config),
            )
            binding = contract_projection.bindings[0]
        super().__init__(backbone=backbone, model=model, runtime=runtime, facade=facade,
            stage=stage, binding=binding, binding_config=binding_config,
            contract_projection=contract_projection, execution_phase=phase)

    def runtime_metadata(self) -> dict[str, Any]:
        if self.algorithm != "vqtoken":
            return super().runtime_metadata()
        binding_metadata = getattr(self._evidence_binding, "metadata", {})
        if not isinstance(binding_metadata, Mapping):
            binding_metadata = {}
        verified_phases = tuple(binding_metadata.get("verified_phases", ()))
        applicable = self.projector_position == "after" and not self._rules
        verified = applicable and self.execution_phase.value in verified_phases
        return {
            **super().runtime_metadata(),
            "canonical_config": dict(self.config),
            "canonical_evidence_applicable": applicable,
            "global_fixed_k": 32, "adaptive": False, "cross_attention": False,
            "training_gradients_preserved": True,
            "real_checkpoint_verified": verified, "support_activation_claimed": verified,
            "evidence_artifact": binding_metadata.get("generate_evidence_artifact") if verified else None,
        }

    def _projector_callback(self, request: Any) -> Any:
        returned = super()._projector_callback(request)
        if self.algorithm == "vqtoken":
            for record in self._layout_records:
                record["num_clusters"] = 32
        return returned


def create_vqtoken_video_executor(
    *,
    backbone: str,
    **kwargs: Any,
) -> VQTokenVideoExecutor:
    return VQTokenVideoExecutor(backbone=backbone, **kwargs)


def _validated_config(stage: Any, binding_config: Any) -> Mapping[str, Any]:
    stage_config = getattr(stage, "config", {}) if stage is not None else {}
    if stage_config is None:
        stage_config = {}
    if not isinstance(stage_config, Mapping):
        raise BackendIntegrationError("VQToken stage config must be a mapping")
    binding_values = (
        getattr(binding_config, "config", {}) if binding_config is not None else {}
    )
    if binding_values not in ({}, None):
        raise BackendIntegrationError(
            "VQToken video binding does not accept binding-local config"
        )
    return resolve_canonical_video_config("vqtoken", stage_config)


def _validate_binding(
    binding: Any,
    stage: Any,
    backbone: str,
    phase: ExecutionPhase,
) -> None:
    expected = f"vqtoken.video.{backbone}"
    if binding is not None:
        if getattr(binding, "name", None) != expected:
            raise BackendIntegrationError(
                f"VQToken executor requires binding {expected!r}"
            )
        if phase not in getattr(binding, "supported_phases", ()):
            raise BackendIntegrationError(
                f"VQToken binding {expected!r} does not support {phase.value}"
            )
        targets = tuple(
            getattr(item, "name", None) for item in binding.ordered_targets()
        )
        if targets != ("fixed_k_cluster",):
            raise BackendIntegrationError("VQToken binding target changed")
    if stage is not None:
        if getattr(stage, "algorithm", None) != "vqtoken":
            raise BackendIntegrationError("VQToken executor received another algorithm")
        if getattr(stage, "variant", None) != "vqtoken.video":
            raise BackendIntegrationError("VQToken executor received another variant")


__all__ = [
    "VQTokenVideoExecutor",
    "create_vqtoken_video_executor",
]
