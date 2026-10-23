"""Compile native pipeline OP plans into physical image/text partitions.

No algorithm YAML is parsed here: the standard definition validator and OP
planners own stage targets, configuration validation and parameter defaults.
The persisted stage contract is distinct from the derived installation layout.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.llm_level import Qwen3VLAttentionReplacement, Qwen3VLFFNReplacement
from mllm_comp.compressors.rules import rule_specs
from mllm_comp.core.errors import BackendIntegrationError, ConfigError
from mllm_comp.core.serialization import atomic_write_json, to_jsonable
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import CompressionMode, ExecutionPhase, Modality
from mllm_comp.pipeline.definition import resolve_compression_definition


_METHODS = {method.name: method for method in (Qwen3VLAttentionReplacement, Qwen3VLFFNReplacement)}
_IMAGE_OPS = {"attn_identity": "skip", "attn_linear_attention": "linear", "attn_cnn": "cnn",
              "attn_lolcats_linear_window": "lolcats", "attn_nha": "nha",
              "attn_sparse_window": "window", "ffn_identity": "skip",
              "ffn_low_rank": "low_rank", "ffn_two_layer_mlp": "two_layer_mlp"}
_TARGET = re.compile(r"^llm\.layer\[(\d+)\]\.(attn|ffn)$")


@dataclass(frozen=True)
class PipelineSplitPlan:
    contract: dict
    spec: dict
    stages: tuple[dict, ...]


@dataclass
class _StageOrder:
    name: str
    dependencies: tuple[str, ...]
    config_order: int


def _ordered_stages(stages, definition):
    from mllm_comp.pipeline.runtime_session import _dependency_order, _resolve_dependency_aliases

    records, by_binding = [], {}
    for index, stage in enumerate(stages):
        dependencies = tuple(stage.depends_on)
        projection = definition.projections.get(stage.name)
        if projection is not None:
            dependencies += tuple(dep for selection in projection.selections for dep in selection.depends_on)
            dependencies += tuple(dep for binding in projection.bindings for dep in binding.dependencies)
        if stage.name in dependencies:
            raise ConfigError(f"stage {stage.name}: dependency on itself.")
        record = _StageOrder(stage.name, dependencies, index)
        records.append(record)
        for binding in projection.bindings if projection is not None else ():
            by_binding.setdefault(binding.name, []).append(record)
    try:
        _resolve_dependency_aliases(records, records_by_stage={r.name: [r] for r in records},
                                    binding_to_records=by_binding)
        ordered = _dependency_order(tuple(records))
    except BackendIntegrationError as exc:
        raise ConfigError(str(exc)) from exc
    by_name = {stage.name: stage for stage in stages}
    return tuple(by_name[record.name] for record in ordered)


def compile_pipeline_split(config, *, model=None, backbone_name=None, definition=None):
    """Return a validated derived plan, or None for an unchanged legacy pipeline.

    Model is optional; when supplied, native backbone/layer bounds are checked
    immediately. Installation always repeats those model-dependent checks.
    """
    candidates = config.pipeline.stages if definition is None else definition.stages
    if not any(getattr(stage, "enabled", True) and stage.plugin in _METHODS
               and stage.modalities == {Modality.IMAGE} for stage in candidates):
        return None
    backbone = backbone_name or config.backbone.name
    if definition is None:
        definition = resolve_compression_definition(
            config, compression_mode="compressed", backbone_name=backbone)
    if definition.compression_mode is CompressionMode.NONE:
        return None
    stages = definition.stages
    image = [stage for stage in stages if stage.method.value == "operation_compression"
             and stage.modalities == {Modality.IMAGE}]
    if not image:
        return None
    if not any(stage.plugin in _METHODS for stage in image):
        return None
    if len(image) != len(stages):
        raise ConfigError("Image-scoped split stages cannot combine with ambiguous legacy/non-image stages.")
    if backbone not in {"qwen25_vl", "qwen3_vl"}:
        raise ConfigError(f"Image-scoped split does not support backbone {backbone!r}.")
    if any(config.backbone.config.get(key) is not None
           for key in ("modality_split", "modality_split_checkpoint")):
        raise ConfigError("pipeline.stages cannot combine with standalone modality_split configuration.")
    stages = _ordered_stages(stages, definition)
    operations, records, assigned = [], [], set()
    for stage in stages:
        if stage.plugin not in _METHODS or stage.execution_phase is not ExecutionPhase.PREPARE:
            raise ConfigError(f"stage {stage.name}: image split requires an existing prepare-time OP planner.")
        rule_specs(stage, supported_rules=())
        plugin_config = stage.plugin_config()
        projection = definition.projections.get(stage.name)
        if projection is not None:
            plugin_config["config"] = dict(projection.algorithm_config)
        method = _METHODS[stage.plugin]()
        try:
            planned = method.plan(state=ForwardState(), context=CompressionContext(
                phase="dry_run", model=model, extra={"stage_name": stage.name}), config=plugin_config)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"stage {stage.name}: {exc}") from exc
        handles = []
        for action in planned.operation_actions:
            site = action.module_path
            if site in assigned:
                raise ConfigError(f"Duplicate image-operation target assignment: {site}.")
            assigned.add(site)
            match = _TARGET.fullmatch(site)
            if match is None or action.strategy_name not in _IMAGE_OPS:
                raise ConfigError(f"Unsupported planned image operation {site!r}.")
            op = _IMAGE_OPS[action.strategy_name]
            parameters = {} if op == "skip" else deepcopy(action.metadata)
            operations.append({"branch": "attention" if match[2] == "attn" else "ffn",
                "layer": int(match[1]), "image_operator": op,
                "strategy": action.strategy_name, "config": parameters, "stage_name": stage.name})
            handles.append(site)
        records.append({"stage_name": stage.name, "plugin": stage.plugin,
                        "installed_handles": handles, "operation_count": len(handles)})
    plan = PipelineSplitPlan(
        contract={"schema_version": 1, "backbone": backbone,
                  "stages": to_jsonable(config.pipeline.stages),
                  "extensions": list(config.pipeline.extensions)},
        spec={"schema_version": 2, "backbone": backbone, "operations": operations},
        stages=tuple(records))
    if model is not None:
        from .core import _validate_spec
        _validate_spec(model, plan.spec)
    return plan


class PipelineSplitSession:
    """Joint installation with native per-stage runtime evidence."""

    def __init__(self, core_session, plan):
        self.core_session = core_session
        self.plan = deepcopy(plan)
        self.model = core_session.model

    def metadata(self):
        result = self.core_session.metadata()
        installed = set(result["installed_handles"])
        stages = []
        for record in self.plan.stages:
            handles = [site for site in record["installed_handles"] if site in installed]
            stages.append({**deepcopy(record), "installed_handles": handles,
                           "replacement_count": len(handles), "prepared": bool(handles),
                           "execution_phase": "prepare", "reversible": True})
        return {**result, "contract": deepcopy(self.plan.contract), "stages": stages,
                "image_window": "planner-configured-image-subsequence",
                "mode": "compressed", "enabled": result["active"],
                "status": "installed" if result["active"] else "uninstalled",
                "activations": stages, "configured_order": [s["stage_name"] for s in stages]}

    runtime_metadata = metadata

    def clear_diagnostic_records(self):
        """Begin a native runner request without altering caches or trained state."""
        for wrapper in self.core_session.replacements.values():
            image_op = wrapper.image_op
            if image_op is not None and hasattr(image_op, "last_debug"):
                image_op.last_debug = {}

    def uninstall(self):
        self.core_session.uninstall()
        if getattr(self.model, "_pipeline_split_session", None) is self:
            del self.model._pipeline_split_session


def install_pipeline_split(model, plan, *, checkpoint=None):
    """Install atomically; return an existing exact-contract session unchanged."""
    if plan is None:
        raise ValueError("A compiled image-scoped pipeline plan is required.")
    if checkpoint is not None:
        # An existing installation must not hide an explicitly supplied
        # checkpoint mismatch. Restore validates its contract before returning
        # the unchanged existing session (without reloading trained weights).
        return restore_pipeline_split(model, checkpoint, plan)
    existing = getattr(model, "_pipeline_split_session", None)
    if existing is not None:
        if (existing.plan.contract != plan.contract or existing.plan.spec != plan.spec
                or not existing.metadata()["active"]):
            raise ValueError("Installed split pipeline contract differs from the requested contract.")
        return existing
    from .core import install_modality_split
    core = install_modality_split(model, plan.spec)
    session = PipelineSplitSession(core, plan)
    model._pipeline_split_session = session
    return session


def save_pipeline_split(model, directory):
    """Save stage contract and selected native/image state; native model saves separately."""
    from .core import save_modality_split
    session = model._pipeline_split_session
    if not session.metadata()["active"]:
        raise ValueError("Cannot save an inactive pipeline split session.")
    save_modality_split(model, directory)
    atomic_write_json(Path(directory) / "pipeline_split.json", session.plan.contract)


def restore_pipeline_split(model, directory, plan):
    """Require exact stage contract/layout and validate all weights before mutation."""
    directory = Path(directory)
    contract = json.loads((directory / "pipeline_split.json").read_text())
    spec = json.loads((directory / "modality_split.json").read_text())
    if contract != plan.contract or spec != plan.spec:
        raise ValueError("Saved split checkpoint contract/layout differs from the requested pipeline contract.")
    existing = getattr(model, "_pipeline_split_session", None)
    if existing is not None:
        return install_pipeline_split(model, plan)
    from .core import load_modality_split
    core = load_modality_split(model, directory)
    session = PipelineSplitSession(core, plan)
    model._pipeline_split_session = session
    return session


def activate_pipeline_split(runtime, plan):
    """Install through the standard runtime and auto-detect colocated saved state."""
    source = getattr(runtime, "model_path_or_id", None)
    checkpoint = None
    if source is not None and any((Path(source) / name).is_file()
                                  for name in ("pipeline_split.json", "modality_split.json")):
        checkpoint = source
    session = install_pipeline_split(runtime.model, plan, checkpoint=checkpoint)
    runtime.metadata["_pipeline_split_session"] = session
    runtime.metadata["_modality_split_session"] = session.core_session
    runtime.metadata["_compression_session"] = session
    runtime.metadata["compression"] = session.metadata()
    runtime.metadata["modality_split"] = session.metadata()
    runtime.metadata["compression_mode"] = "compressed"
    return session
