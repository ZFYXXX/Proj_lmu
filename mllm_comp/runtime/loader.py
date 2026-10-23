"""Runtime construction shared by executable platform commands."""

from __future__ import annotations

from typing import Any

from mllm_comp.backbones.base import BackboneFacade, BackboneRuntime
from mllm_comp.config.schema import ExperimentConfig
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.vocabulary import CompressionMode, ExecutionPhase, RuntimeMode
from mllm_comp.registry.backbones import get_backbone


_MODEL_LOAD_KEYS = {
    "attn_implementation",
    "device_map",
    "low_cpu_mem_usage",
    "torch_dtype",
}


def _resolve_torch_dtype(torch: Any, value: Any) -> Any:
    if value is None or value == "auto":
        return value
    if not isinstance(value, str):
        return value
    normalized = value.removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if dtype is None:
        raise BackendIntegrationError(f"Unknown torch dtype {value!r} in backbone.config.")
    return dtype


def load_backbone_runtime(
    config: ExperimentConfig,
    *,
    checkpoint_path: str | None = None,
    device: str | None = None,
    train: bool = False,
    enable_talker: bool | None = None,
    compression_mode: CompressionMode | str | None = None,
    runtime_mode: RuntimeMode | str | None = None,
    execution_phase: ExecutionPhase | str | None = None,
) -> tuple[BackboneFacade, BackboneRuntime]:
    """Load one configured backbone without giving ownership to a framework bridge."""

    try:
        import torch
    except ImportError as exc:
        raise BackendIntegrationError("A real backbone runtime requires PyTorch.") from exc

    facade = get_backbone(config.backbone.name)
    resolved_runtime_mode = RuntimeMode(
        runtime_mode
        if runtime_mode is not None
        else (config.train.runtime_mode if train else config.eval.runtime_mode)
    )
    if (
        resolved_runtime_mode is not RuntimeMode.DEFAULT
        and facade.capabilities().name != "qwen25_omni"
    ):
        raise BackendIntegrationError(
            "profile/debug runtime modes are currently implemented only for qwen25_omni."
        )
    model_path = checkpoint_path or config.backbone.model_path or config.backbone.model_id
    if not model_path:
        raise BackendIntegrationError(
            "Backbone loading needs a checkpoint override, backbone.model_path or backbone.model_id."
        )
    from pathlib import Path
    from mllm_comp.integrations.ms_swift.split_lifecycle import inspect_split_checkpoint, validate_checkpoint_plan
    split_contract = inspect_split_checkpoint(model_path)
    if compression_mode is None:
        compression_mode = config.train.compression_mode if train else config.eval.compression_mode
    from mllm_comp.fixed_mask.checkpoint import validate_checkpoint
    mask_contract = validate_checkpoint(model_path, config.backbone.name, config.pipeline.stages,
                                        compression_mode=CompressionMode(compression_mode).value)
    split_plan = None
    adapter_path = Path(model_path) / "adapter_config.json"
    split_adapter = split_contract is not None and adapter_path.is_file()
    mask_adapter = mask_contract is not None and adapter_path.is_file()
    load_path = model_path
    if split_contract is not None:
        if CompressionMode(compression_mode) is not CompressionMode.COMPRESSED:
            raise ValueError("A split checkpoint requires compression_mode=compressed.")
        from mllm_comp.modality_split.pipeline import compile_pipeline_split
        split_plan = compile_pipeline_split(config)
        validate_checkpoint_plan(model_path, split_plan)
        if split_adapter:
            import json
            load_path = json.loads(adapter_path.read_text()).get("base_model_name_or_path")
            if not load_path or str(load_path) == str(model_path):
                raise ValueError("Split adapter checkpoint does not identify its native base model.")

    if mask_adapter:
        import json
        load_path = json.loads(adapter_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
        if not load_path or str(load_path) == str(model_path):
            raise ValueError("Fixed-mask adapter checkpoint does not identify its native base model")

    backbone_args = dict(config.backbone.config)
    nested_model_args = backbone_args.pop("model_load_kwargs", {})
    nested_processor_args = backbone_args.pop("processor_load_kwargs", {})
    if not isinstance(nested_model_args, dict) or not isinstance(nested_processor_args, dict):
        raise BackendIntegrationError(
            "backbone.config.model_load_kwargs and processor_load_kwargs must be mappings."
        )

    model_load_kwargs = {
        key: backbone_args[key]
        for key in _MODEL_LOAD_KEYS
        if key in backbone_args
    }
    model_load_kwargs.update(nested_model_args)
    if "torch_dtype" in model_load_kwargs:
        model_load_kwargs["torch_dtype"] = _resolve_torch_dtype(
            torch, model_load_kwargs["torch_dtype"]
        )

    resolved_device = device or str(backbone_args.get("device") or "")
    if not resolved_device:
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if resolved_device.startswith("cuda") and "device_map" not in model_load_kwargs:
        model_load_kwargs["device_map"] = resolved_device

    if config.backbone.trust_remote_code is not None:
        model_load_kwargs.setdefault("trust_remote_code", config.backbone.trust_remote_code)
        nested_processor_args.setdefault(
            "trust_remote_code", config.backbone.trust_remote_code
        )
    supports_speech = "speech" in facade.capabilities().output_modalities
    if supports_speech:
        configured_talker = bool(backbone_args.get("enable_talker", True))
        model_load_kwargs["enable_talker"] = (
            configured_talker if enable_talker is None else bool(enable_talker)
        )
    elif enable_talker:
        raise BackendIntegrationError(
            f"Backbone {config.backbone.name!r} does not expose a speech/Talker branch."
        )

    processor = facade.processor_adapter.load_processor(
        str(load_path),
        revision=config.backbone.model_revision,
        **nested_processor_args,
    )
    model = facade.model_adapter.load_model(
        str(load_path),
        train=train,
        revision=config.backbone.model_revision,
        **model_load_kwargs,
    )
    if split_adapter:
        from peft import PeftModel
        from mllm_comp.modality_split.pipeline import install_pipeline_split
        from mllm_comp.integrations.ms_swift.split_lifecycle import capture_projection_aliases, retie_projection_aliases
        install_pipeline_split(model, split_plan, checkpoint=model_path)
        aliases = capture_projection_aliases(model)
        peft_model = PeftModel.from_pretrained(model, model_path)
        retie_projection_aliases(model, aliases)
        model = peft_model.merge_and_unload()
        retie_projection_aliases(model, aliases)
    if mask_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
    runtime_dtype = model_load_kwargs.get("torch_dtype")
    if runtime_dtype in (None, "auto"):
        runtime_dtype = getattr(model, "dtype", None)
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        revision=config.backbone.model_revision,
        device=torch.device(resolved_device),
        dtype=runtime_dtype,
        metadata={
            "phase": "train" if train else "eval",
            "runtime_mode": resolved_runtime_mode.value,
            "speech_sample_rate": (
                int(backbone_args.get("speech_sample_rate", 24000))
                if supports_speech
                else None
            ),
            "enable_talker": bool(model_load_kwargs.get("enable_talker", False)),
            "device_map": model_load_kwargs.get("device_map"),
        },
    )
    if compression_mode is None:
        compression_mode = (
            config.train.compression_mode
            if train
            else config.eval.compression_mode
        )
    from mllm_comp.pipeline.runtime_compression import activate_runtime_compression

    activate_runtime_compression(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode=compression_mode,
        execution_phase=(
            ExecutionPhase(execution_phase)
            if execution_phase is not None
            else (ExecutionPhase.FORWARD if train else ExecutionPhase.GENERATE)
        ),
    )
    if resolved_runtime_mode is RuntimeMode.DEBUG:
        from mllm_comp.backbones.qwen25_omni.token_debug import (
            Qwen25OmniTokenDebugCollector,
        )

        collector = Qwen25OmniTokenDebugCollector(
            runtime.model,
            processor=runtime.processor,
        )
        collector.install()
        runtime.metadata["_token_debug_collector"] = collector
    return facade, runtime
