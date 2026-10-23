"""Official lmms-eval model entry point backed by MLLM-Comp runtimes."""

from __future__ import annotations

import json
import atexit
from pathlib import Path
from typing import Any, Sequence

from lmms_eval.api.model import lmms

from mllm_comp.config.loader import load_config
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.vocabulary import ExecutionPhase, RuntimeMode
from mllm_comp.runtime import load_backbone_runtime

from .model_bridge import LMMSEvalRequestBridge


_SIMPLE_CLASS_PATH = "mllm_comp.integrations.lmms_eval.plugin.MLLMCompLMM"
_CHAT_CLASS_PATH = "mllm_comp.integrations.lmms_eval.plugin.MLLMCompChatLMM"
_MODEL_IDS = (
    "qwen2_5_omni",
    "mllm_comp_qwen25_vl",
    "mllm_comp_qwen3_vl",
    "mllm_comp_qwen3_5",
    "mllm_comp_llava_onevision",
)


def get_model_manifests() -> list[Any]:
    """Expose platform-backed chat and simple models through the public registry."""

    from lmms_eval.models.registry_v2 import ModelManifest

    return [
        ModelManifest(
            model_id=model_id,
            simple_class_path=_SIMPLE_CLASS_PATH,
            chat_class_path=_CHAT_CLASS_PATH,
        )
        for model_id in _MODEL_IDS
    ]


def _load_plugin_spec(path: str) -> dict[str, Any]:
    spec_path = Path(path)
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendIntegrationError(
            f"Cannot read MLLM-Comp lmms-eval plugin spec {spec_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") not in (1, 2):
        raise BackendIntegrationError(
            "The MLLM-Comp lmms-eval plugin spec must be a schema_version=1 or 2 JSON object."
        )
    return payload


def _ranked_trace_path(path: str, *, rank: int, world_size: int) -> str:
    """Give each worker its own diagnostic trace file in distributed runs."""

    if world_size == 1:
        return path
    trace_path = Path(path)
    suffix = trace_path.suffix
    stem = trace_path.name[: -len(suffix)] if suffix else trace_path.name
    return str(trace_path.with_name(f"{stem}.rank{rank}{suffix}"))


class _MLLMCompLMMBase(lmms):
    """Shared loader for platform-backed lmms-eval protocol implementations."""

    is_simple = True
    _request_mode = "simple"

    def __init__(
        self,
        *,
        pretrained: str,
        mllm_comp_spec: str,
        batch_size: int | str = 1,
        max_batch_size: int | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del max_batch_size, kwargs
        spec = _load_plugin_spec(mllm_comp_spec)
        config_path = spec.get("config_path")
        if spec["schema_version"] == 2:
            effective_config = spec.get("effective_config")
            if not isinstance(effective_config, dict):
                raise BackendIntegrationError("Plugin spec effective_config must be a config mapping.")
            config = experiment_config_from_dict(effective_config, path=config_path)
        else:
            # Saved schema-1 launch specs predate effective config snapshots.
            if not isinstance(config_path, str) or not config_path:
                raise BackendIntegrationError("Plugin spec config_path must be a non-empty string.")
            config = load_config(config_path)
        expected_backbone = spec.get("backbone_name")
        if expected_backbone != config.backbone.name:
            raise BackendIntegrationError(
                "Plugin spec backbone identity does not match the experiment config: "
                f"{expected_backbone!r} != {config.backbone.name!r}."
            )
        checkpoint = str(spec.get("checkpoint_path") or pretrained)
        if str(pretrained) != checkpoint:
            raise BackendIntegrationError(
                "lmms-eval pretrained and the platform plugin checkpoint must match."
            )

        compression_mode = str(spec.get("compression_mode", "none"))
        enabled_stages = [stage for stage in config.pipeline.stages if stage.enabled]
        eval_section = getattr(config, "eval", None)
        configured_mode_value = getattr(
            eval_section,
            "compression_mode",
            compression_mode,
        )
        configured_mode = str(
            getattr(configured_mode_value, "value", configured_mode_value)
        )
        if compression_mode != configured_mode:
            raise BackendIntegrationError(
                "Plugin spec compression_mode does not match eval.compression_mode: "
                f"{compression_mode!r} != {configured_mode!r}."
            )
        has_split = any(config.backbone.config.get(key) is not None for key in
                        ("modality_split", "modality_split_checkpoint"))
        if compression_mode == "compressed" and not enabled_stages and not has_split:
            raise BackendIntegrationError(
                "compression_mode=compressed requires an enabled pipeline stage."
            )
        try:
            runtime_mode = RuntimeMode(spec.get("runtime_mode", RuntimeMode.DEFAULT))
        except ValueError as exc:
            raise BackendIntegrationError(
                f"Plugin spec runtime_mode is invalid: {spec.get('runtime_mode')!r}."
            ) from exc
        request_mode = spec.get("request_mode", "generate")
        if request_mode not in {"generate", "forward_probe"}:
            raise BackendIntegrationError(
                f"Plugin spec request_mode is invalid: {request_mode!r}."
            )
        if request_mode == "forward_probe" and expected_backbone == "qwen25_omni":
            raise BackendIntegrationError(
                "forward_probe is not implemented for qwen25_omni."
            )
        try:
            from accelerate import Accelerator
        except ImportError as exc:
            raise BackendIntegrationError(
                "The lmms-eval model plugin requires lmms-eval's accelerate dependency."
            ) from exc
        self.accelerator = Accelerator()
        self._rank = int(self.accelerator.process_index)
        self._world_size = int(self.accelerator.num_processes)
        expected_world_size = spec.get("expected_world_size")
        if expected_world_size is not None:
            if (
                isinstance(expected_world_size, bool)
                or not isinstance(expected_world_size, int)
                or expected_world_size < 1
            ):
                raise BackendIntegrationError(
                    "Plugin spec expected_world_size must be a positive integer."
                )
            if self._world_size != expected_world_size:
                raise BackendIntegrationError(
                    "lmms-eval worker world size does not match the platform launch "
                    f"plan: expected {expected_world_size}, got {self._world_size}."
                )
        resolved_device = (
            device or spec.get("device") or config.backbone.config.get("device")
            or str(self.accelerator.device)
        )
        facade, runtime = load_backbone_runtime(
            config,
            checkpoint_path=checkpoint,
            device=str(resolved_device),
            train=False,
            enable_talker=False,
            compression_mode=compression_mode,
            runtime_mode=runtime_mode,
            execution_phase=(
                ExecutionPhase.FORWARD
                if request_mode == "forward_probe"
                else ExecutionPhase.GENERATE
            ),
        )
        if facade.capabilities().name != expected_backbone:
            raise BackendIntegrationError(
                "The loaded platform facade does not match the requested backbone identity."
            )
        self._platform_bridge = LMMSEvalRequestBridge(
            facade=facade,
            runtime=runtime,
            system_prompt=(
                None
                if spec.get("system_prompt") is None
                else str(spec["system_prompt"])
            ),
            use_audio_in_video=bool(spec.get("use_audio_in_video", False)),
            generation_overrides=(
                spec.get("generation")
                if isinstance(spec.get("generation"), dict)
                else {}
            ),
            diagnostic_trace_path=(
                None
                if spec.get("diagnostic_trace_path") is None
                else _ranked_trace_path(
                    str(spec["diagnostic_trace_path"]),
                    rank=self._rank,
                    world_size=self._world_size,
                )
            ),
            compression_manifest_path=(
                None
                if spec.get("compression_manifest_path") is None
                else _ranked_trace_path(
                    str(spec["compression_manifest_path"]),
                    rank=self._rank,
                    world_size=self._world_size,
                )
            ),
            request_mode=str(request_mode),
        )
        # lmms-eval owns the process-level CLI lifecycle and does not provide
        # a reliable per-model close callback. Ensure platform-installed
        # compression hooks are always uninstalled at worker shutdown.
        self._runtime_cleanup_callback = self.close
        atexit.register(self._runtime_cleanup_callback)
        self.batch_size_per_gpu = batch_size
        self._install_grounding_scorer()

    def close(self) -> None:
        bridge = getattr(self, "_platform_bridge", None)
        if bridge is not None:
            bridge.runtime.close()
        callback = getattr(self, "_runtime_cleanup_callback", None)
        if callback is not None:
            try:
                atexit.unregister(callback)
            except Exception:
                pass
            self._runtime_cleanup_callback = None

    def _install_grounding_scorer(self) -> None:
        from .grounding import install_grounding_scorer
        bridge = self._platform_bridge
        install_grounding_scorer(bridge.backbone_name, bridge.min_pixels, bridge.max_pixels)

    @property
    def model(self) -> Any:
        return self._platform_bridge.runtime.model

    @property
    def processor(self) -> Any:
        return self._platform_bridge.runtime.processor

    @property
    def tokenizer(self) -> Any:
        return self._platform_bridge.tokenizer

    @property
    def config(self) -> Any:
        return self._platform_bridge.config

    @property
    def device(self) -> Any:
        return self._platform_bridge.device

    @property
    def batch_size(self) -> int | str:
        return self.batch_size_per_gpu

    def generate_until(self, requests: Sequence[Any]) -> list[str]:
        self._platform_bridge.task_dict = self.task_dict
        generate = (
            self._platform_bridge.generate_until
            if self._request_mode == "simple"
            else self._platform_bridge.generate_until_chat
        )
        return generate(requests, cache_hook=self.cache_hook, show_progress=self.rank == 0)

    def loglikelihood(self, requests: Sequence[Any]) -> list[tuple[float, bool]]:
        return self._platform_bridge.loglikelihood(requests)

    def generate_until_multi_round(self, requests: Sequence[Any]) -> list[str]:
        return self._platform_bridge.generate_until_multi_round(requests)


class MLLMCompLMM(_MLLMCompLMMBase):
    """Platform implementation of lmms-eval's simple request protocol."""

    is_simple = True
    _request_mode = "simple"

    @classmethod
    def from_runtime(cls, facade, runtime, *, compression_manifest_path=None):
        """Evaluate a caller-owned single-process runtime without loading it twice.

        The caller retains lifecycle ownership and must close/uninstall afterwards.
        Used by operator smoke matrices immediately after their optimizer step.
        """
        instance = cls.__new__(cls)
        lmms.__init__(instance)
        instance._rank, instance._world_size = 0, 1
        instance.batch_size_per_gpu = 1
        instance._platform_bridge = LMMSEvalRequestBridge(
            facade=facade, runtime=runtime,
            compression_manifest_path=compression_manifest_path,
        )
        instance._install_grounding_scorer()
        return instance


class MLLMCompChatLMM(_MLLMCompLMMBase):
    """Platform implementation of lmms-eval's ``doc_to_messages`` protocol."""

    is_simple = False
    _request_mode = "chat"
