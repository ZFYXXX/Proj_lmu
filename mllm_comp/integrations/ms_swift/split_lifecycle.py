"""Native trainer checkpoint plumbing for image-scoped pipeline operations.

The trainer still owns weights/optimizer/scheduler/RNG saving. This module only
maps native parameter names and persists the extra image branch and its contract.
"""
from __future__ import annotations

import json
from pathlib import Path

from mllm_comp.core.serialization import atomic_write_json

MARKER = "mllm_comp_pipeline_split_required"
SIDECARS = ("pipeline_split.json", "modality_split.json", "modality_split.pt")


def split_model(model):
    """Find the actual installed model, not PEFT's attribute-forwarding facade."""
    for module in model.modules():
        if "_pipeline_split_session" in vars(module):
            return module
    return None


def inspect_split_checkpoint(directory):
    directory = Path(directory)
    marked = False
    for filename in ("config.json", "adapter_config.json"):
        path = directory / filename
        if path.is_file():
            marked |= bool(json.loads(path.read_text()).get(MARKER))
    present = [bool((directory / name).is_file()) for name in SIDECARS]
    if not marked and not any(present):
        return None
    if not all(present):
        raise ValueError("Split checkpoint is missing required sidecar files.")
    contract = json.loads((directory / SIDECARS[0]).read_text())
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported split checkpoint sidecar contract.")
    return contract


def validate_checkpoint_plan(directory, plan):
    contract = inspect_split_checkpoint(directory)
    if contract is None or plan is None or contract != plan.contract:
        raise ValueError("Checkpoint split sidecar contract differs from pipeline.stages.")
    spec = json.loads((Path(directory) / "modality_split.json").read_text())
    if spec != plan.spec:
        raise ValueError("Checkpoint split sidecar layout differs from pipeline.stages.")


def capture_projection_aliases(model):
    paths = {}
    for name, module in model.named_modules(remove_duplicate=False):
        paths.setdefault(id(module), []).append(name)
    aliases = []
    for names in paths.values():
        native = next((n for n in names if ".native." in n), None)
        if native:
            aliases.extend((native, n) for n in names if ".image_op." in n)
    # Parents first; children then resolve through the newly installed parents.
    return sorted(aliases, key=lambda pair: pair[0].count("."))


def retie_projection_aliases(model, aliases):
    for source, target in aliases:
        parent, name = target.rsplit(".", 1)
        setattr(model.get_submodule(parent), name, model.get_submodule(source))


def _clean_key(name):
    if any(piece.startswith("lora_") for piece in name.split(".")):
        return None
    return name.replace(".base_layer.", ".")


def _checked_state(model, state_dict=None):
    import torch
    expected = model.state_dict()
    supplied = expected if state_dict is None else state_dict
    if set(expected) != set(supplied):
        raise ValueError("Incomplete/partitioned state_dict cannot save a split checkpoint.")
    for name, value in supplied.items():
        target = expected[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                or value.is_meta or value.is_quantized or value.layout != target.layout):
            raise ValueError(f"State {name} must be a complete materialized tensor.")
    return supplied


def native_save_state(model, state_dict=None):
    setattr(model.config, MARKER, True)
    result = {}
    for name, value in _checked_state(model, state_dict).items():
        if ".self_attn.image_op." in name or ".mlp.image_op." in name:
            continue
        name = _clean_key(name)
        if name is not None:
            name = name.replace(".self_attn.native.", ".self_attn.").replace(".mlp.native.", ".mlp.")
            result[name] = value
    return result


def _selected_states(model, supplied):
    session = model._pipeline_split_session.core_session
    paths = {id(module): name for name, module in model.named_modules()}
    states = {}
    for (index, attr), wrapper in session.replacements.items():
        prefix = paths[id(wrapper)] + "."
        state = {}
        for name, value in supplied.items():
            if name.startswith(prefix):
                key = _clean_key(name[len(prefix):])
                if key is not None:
                    state[key] = value.detach().cpu().clone()
        states[f"{index}:{attr}"] = state
    return states


def save_split_checkpoint(model, directory, *, state_dict=None):
    import torch
    supplied = _checked_state(model, state_dict)
    session = model._pipeline_split_session
    states = _selected_states(model, supplied)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    setattr(model.config, MARKER, True)
    # A marker survives deletion of all sidecars, so reload fails closed.
    for filename in ("config.json", "adapter_config.json"):
        path = directory / filename
        if path.is_file():
            payload = json.loads(path.read_text())
            payload[MARKER] = True
            atomic_write_json(path, payload)
    torch.save(states, directory / "modality_split.pt")
    atomic_write_json(directory / "modality_split.json", session.plan.spec)
    atomic_write_json(directory / "pipeline_split.json", session.plan.contract)


def restore_training_checkpoint(model, directory):
    import torch
    session = model._pipeline_split_session
    validate_checkpoint_plan(directory, session.plan)
    states = torch.load(Path(directory) / "modality_split.pt", map_location="cpu", weights_only=True)
    expected = _selected_states(model, model.state_dict())
    if not isinstance(states, dict) or set(states) != set(expected):
        raise ValueError("Split checkpoint state keys differ.")
    for site, target in expected.items():
        saved = states[site]
        if not isinstance(saved, dict) or set(saved) != set(target):
            raise ValueError(f"Split checkpoint state keys differ at {site}.")
        for name, tensor in saved.items():
            dest = target[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != dest.shape
                    or tensor.is_meta or tensor.is_quantized or tensor.layout != dest.layout):
                raise ValueError(f"Split checkpoint has invalid materialized state at {site}/{name}.")
            torch.empty_like(dest).copy_(tensor)
    with torch.no_grad():
        for (index, attr), wrapper in session.core_session.replacements.items():
            for name, dest in wrapper.state_dict().items():
                canonical = _clean_key(name)
                if canonical is not None:
                    dest.copy_(states[f"{index}:{attr}"][canonical])


def install_native_load_mapping(model):
    """Adapt native Trainer's partial shard loads without taking over resume."""
    if getattr(model, "_split_native_load_hook", None) is not None:
        return
    mapping = {}
    for name in model.state_dict():
        if ".image_op." in name or _clean_key(name) is None:
            continue
        native = name.replace(".self_attn.native.", ".self_attn.").replace(".mlp.native.", ".mlp.")
        if native != name:
            mapping[native] = name

    def remap(module, state, prefix, *unused):
        for native, wrapped in mapping.items():
            key = prefix + native
            if key in state:
                state[prefix + wrapped] = state.pop(key)

    model._split_native_load_hook = model.register_load_state_dict_pre_hook(remap)


def unpacked_swift_inputs(kwargs):
    """Swift emits FA boundary hints even for a single ordinary SDPA sequence.

Accept only boundaries exactly matching batch rows; internal segment boundaries
remain an error. The original attention mask and MRoPE positions are preserved.
"""
    clean = dict(kwargs)
    if not any(clean.get(k) is not None for k in ("cu_seq_lens_q", "cu_seq_lens_k")):
        return clean
    ids = clean.get("input_ids")
    if ids is None or ids.ndim != 2:
        raise ValueError("Cannot establish unpacked boundaries without batched input_ids.")
    batch, length = ids.shape
    expected = list(range(0, (batch + 1) * length, length))
    for key in ("cu_seq_lens_q", "cu_seq_lens_k"):
        value = clean.pop(key, None)
        if value is None or value.ndim != 1 or value.tolist() != expected:
            raise ValueError("Swift packed/internal sequence boundaries are unsupported.")
    for key in ("max_length_q", "max_length_k"):
        value = clean.pop(key, None)
        if value is not None and int(value) != length:
            raise ValueError("Swift sequence boundary maximum length is inconsistent.")
    return clean


def install_training_split(model, plan, *, recorder, checkpoint=None):
    from functools import wraps
    from types import MethodType
    from mllm_comp.modality_split.pipeline import install_pipeline_split
    from mllm_comp.pipeline.stage import CompressionStageConfig
    if checkpoint is not None:
        validate_checkpoint_plan(checkpoint, plan)
    session = install_pipeline_split(model, plan, checkpoint=checkpoint)
    setattr(model.config, MARKER, True)
    model._mllm_comp_training_session = session
    model._split_recorder = recorder
    model._split_projection_aliases = capture_projection_aliases(model)
    model._split_forward_index = 0
    stages = {s["name"]: CompressionStageConfig.from_dict(s) for s in plan.contract["stages"]}
    for record in session.metadata()["stages"]:
        recorder.record("install", stage=stages[record["stage_name"]],
                        executor_family="image_operation", lifecycle="post_load_pre_tuner_pre_optimizer",
                        replacement_count=record["replacement_count"], installed_handles=record["installed_handles"])
    original = model.forward

    @wraps(original)
    def forward(self, *args, **kwargs):
        kwargs = unpacked_swift_inputs(kwargs)
        kwargs["use_cache"] = False
        output = original(*args, **kwargs)
        loss = getattr(output, "loss", None)
        if loss is None and isinstance(output, dict):
            loss = output.get("loss")
        if model.training and loss is not None and loss.requires_grad:
            index = model._split_forward_index
            model._split_forward_index += 1
            for record in session.metadata()["stages"]:
                stage = stages[record["stage_name"]]
                recorder.record("forward", stage=stage, executor_family="image_operation",
                                forward_index=index, loss_requires_grad=True, runtime_mode="default",
                                replacement_count=record["replacement_count"], installed_handles=record["installed_handles"])

                def backward(gradient, stage=stage, index=index):
                    recorder.record("backward", stage=stage, executor_family="image_operation",
                                    forward_index=index, gradient_present=gradient is not None)
                    return gradient
                loss.register_hook(backward)
        return output

    model.forward = MethodType(forward, model)
    install_native_load_mapping(model)
    return model


def patch_native_lifecycle(tuner_class=None, trainer_class=None):
    """Narrow hooks; optimizer, Trainer state, datasets and training stay native."""
    from functools import wraps
    if tuner_class is None:
        from swift.pipelines.train.tuner import TunerMixin as tuner_class
    if trainer_class is None:
        from swift.trainers.mixin import SwiftMixin as trainer_class
    if not vars(tuner_class).get("_split_tuner_hook", False):
        original_prepare = tuner_class.prepare_model.__func__

        @classmethod
        @wraps(original_prepare)
        def prepare(cls, args, model, **kwargs):
            base = split_model(model)
            result = original_prepare(cls, args, model, **kwargs)
            if base is not None:
                retie_projection_aliases(base, base._split_projection_aliases)
                # Independent image parameters (not native shared projections)
                # are small enough to keep exact CPU snapshots for save evidence.
                base._split_image_before = {n: p.detach().cpu().clone() for n, p in base.named_parameters()
                                            if ".image_op." in n and p.requires_grad}
                base._split_image_gradients = {}
                for name, parameter in base.named_parameters():
                    if name not in base._split_image_before:
                        continue
                    def gradient_hook(gradient, name=name):
                        import torch
                        base._split_image_gradients[name] = {
                            "parameter": name, "norm": float(gradient.float().norm()),
                            "finite": bool(torch.isfinite(gradient).all())}
                        return gradient
                    parameter.register_hook(gradient_hook)
                base._split_recorder.record("tuner_ready", executor_family="image_operation",
                                            scope="all_image_stages",
                                            trainable_image_parameters=list(base._split_image_before))
            return result

        tuner_class.prepare_model = prepare
        tuner_class._split_tuner_hook = True
    if not vars(trainer_class).get("_split_save_hook", False):
        original_save = trainer_class._save_model

        @wraps(original_save)
        def save(trainer, output_dir=None, state_dict=None):
            base = split_model(trainer.model)
            if base is None:
                return original_save(trainer, output_dir, state_dict)
            # Distributed partition modes are rejected before launching Swift.
            # A gathered dict may still be supplied for ordinary full/DDP saves.
            supplied = state_dict
            if supplied is not None and trainer.model is not base:
                path = next(n for n, m in trainer.model.named_modules() if m is base)
                prefix = path + "." if path else ""
                supplied = {k[len(prefix):]: v for k, v in supplied.items() if k.startswith(prefix)}
            _checked_state(base, supplied)
            peft = bool(getattr(trainer.model, "peft_config", None))
            save_state = state_dict if peft else native_save_state(base, supplied)
            result = original_save(trainer, output_dir, save_state)
            save_split_checkpoint(base, output_dir, state_dict=supplied)
            import torch
            params = dict(base.named_parameters())
            updates = []
            for name, before in getattr(base, "_split_image_before", {}).items():
                after = params[name].detach().cpu()
                updates.append({"parameter": name, "changed": not torch.equal(before, after),
                                "max_abs_delta": float((after.float() - before.float()).abs().max())})
            base._split_recorder.record("save", executor_family="image_operation", output_dir=str(output_dir),
                                        scope="all_image_stages", pipeline_contract=base._pipeline_split_session.plan.contract,
                                        image_parameter_updates=updates,
                                        image_parameter_gradients=list(getattr(base, "_split_image_gradients", {}).values()))
            return result

        trainer_class._save_model = save
        trainer_class._split_save_hook = True
