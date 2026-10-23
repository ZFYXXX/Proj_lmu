"""Dense, differentiable reference execution on native Qwen decoders.

No parameters or projection names change. Layout travels with layer kwargs so
checkpoint recomputation uses the layout of its own forward, not the latest call.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from copy import deepcopy
import inspect
import json
from pathlib import Path

from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.core.serialization import atomic_write_json
from .spec import KINDS, normalize_spec, validate_geometry

KEY = "fixed_mask_layout"
ATTRIBUTE = "_mllm_fixed_mask"
CONFIG_KEY = "mllm_comp_fixed_mask"
SIDECAR = "fixed_mask.json"


@dataclass(frozen=True)
class MaskLayout:
    vision: object
    valid: object
    suffix: object

    def select(self, indices):
        return MaskLayout(*(x.index_select(-1, indices.to(x.device))
                            for x in (self.vision, self.valid, self.suffix)))


def make_layout(vision, padding=None):
    import torch
    vision = vision.bool().clone()
    if vision.ndim != 2 or vision.shape[0] != 1:
        raise ValueError("fixed_mask currently requires unpacked batch size one")
    valid = torch.ones_like(vision) if padding is None else padding.to(vision.device).bool().clone()
    if valid.shape != vision.shape:
        raise ValueError("fixed_mask requires a sequence-aligned 2D padding mask")
    vision &= valid
    positions = torch.arange(vision.shape[-1], device=vision.device)[None, :]
    last = torch.where(vision, positions, -1).max(-1).values[:, None]
    suffix = valid & ~vision & (positions > last) & (last >= 0)
    return MaskLayout(vision, valid, suffix)


def _call(original, args, kwargs):
    bound = inspect.signature(original).bind_partial(*args, **kwargs)
    call = dict(bound.arguments)
    for name, param in inspect.signature(original).parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            call.update(call.pop(name, {}))
    return call


def _cached(past):
    return past is not None and int(past.get_seq_length()) > 0


def session_for(model):
    return vars(model).get(ATTRIBUTE)


def attach_layout(model, call, *, vision=None, padding=None):
    """Provider hook before native decoder execution or token compaction."""
    session = session_for(model)
    if session is None:
        return
    if _cached(call.get("past_key_values")):
        call[KEY] = None
    elif vision is not None:
        call[KEY] = make_layout(vision, padding)
    elif KEY not in call:
        call[KEY] = session.incoming.get()


def compact_layout(call, indices):
    layout = call.get(KEY)
    if layout is not None:
        call[KEY] = layout.select(indices)


def record_llava_layout(model, image_indices, length, padding, device):
    """Exact packed/AnyRes layout published by the existing LLaVA provider."""
    session = session_for(model)
    if session is not None:
        import torch
        vision = torch.zeros((1, length), dtype=torch.bool, device=device)
        vision[:, image_indices] = True
        session.incoming.set(make_layout(vision, padding))


def masked_attention(mask, layout, selections, layer, hidden, heads):
    """Add hard path cuts without exposing future keys; zero fully blocked rows."""
    import torch
    batch, length = hidden.shape[:2]
    valid = layout.valid.to(hidden.device)
    if tuple(valid.shape) != (batch, length):
        raise ValueError("fixed_mask layout differs from the current decoder sequence")
    if mask is None:
        allowed = torch.ones((length, length), dtype=torch.bool, device=hidden.device).tril()
        allowed = allowed[None, None] & valid[:, None, None, :]
        result = torch.zeros((batch, 1, length, length), device=hidden.device, dtype=hidden.dtype)
        result = result.masked_fill(~allowed, float("-inf"))
    else:
        if mask.ndim != 4 or mask.shape[-2:] != (length, length):
            raise ValueError("fixed_mask paths require a prefill 4D causal mask")
        result = mask.to(device=hidden.device)
        if result.dtype == torch.bool:
            result = torch.zeros_like(result, dtype=hidden.dtype).masked_fill(~result, float("-inf"))
    result = result.expand(batch, heads, length, length).clone()
    vision, suffix = layout.vision.to(hidden.device), layout.suffix.to(hidden.device)
    for kind, query, key in (("v2v", vision, vision), ("t2v", suffix, vision), ("t2t", suffix, suffix)):
        indices = selections[kind].get(str(layer), [])
        if indices:
            selected = torch.zeros(heads, dtype=torch.bool, device=hidden.device)
            selected[indices] = True
            cut = selected[None, :, None, None] & query[:, None, :, None] & key[:, None, None, :]
            result = result.masked_fill(cut, float("-inf"))
    allowed = torch.isfinite(result) & (result > torch.finfo(result.dtype).min / 2)
    empty = ~allowed.any(-1)
    # Eager softmax cannot consume a fully -inf row. A temporary self edge is
    # harmless because its entire head output is explicitly zeroed before O.
    if bool(empty.any()):
        diagonal = torch.eye(length, dtype=torch.bool, device=hidden.device)[None, None]
        result = result.masked_fill(empty[..., None] & diagonal, 0)
    return result, empty


class FixedMaskSession:
    def __init__(self, model, spec, *, backbone):
        if backbone not in {"qwen25_vl", "llava_onevision"}:
            raise ValueError(f"fixed_mask does not support {backbone!r}")
        self.model, self.backbone, self.spec = model, backbone, normalize_spec(spec)
        self.decoder = model.model.language_model if backbone == "qwen25_vl" else model.get_model()
        self.layers = self.decoder.layers
        validate_geometry(self.spec, self.layers)
        self.incoming = ContextVar(f"fixed_mask_input_{id(self)}", default=None)
        self.current = ContextVar(f"fixed_mask_layer_{id(self)}", default=None)
        self.patches = []
        self.active = False
        self.old_config = getattr(model.config, CONFIG_KEY, None)
        self.forward_count = 0
        self.masked_layer_calls = 0

    def _patch(self, owner, name, replacement):
        self.patches.append(MethodPatch.apply(owner, name, replacement, getattr(owner, name), label="fixed_mask"))

    def install(self):
        if self.active:
            return self
        if session_for(self.model) is not None:
            raise ValueError("Only one fixed_mask stage may be installed")
        if getattr(self.model, "_modality_split_session", None) is not None:
            raise ValueError("fixed_mask cannot share modules with modality_split")
        if any(getattr(layer.self_attn.config, "_attn_implementation", None) not in {"eager", "sdpa"}
               for layer in self.layers):
            raise ValueError("fixed_mask requires eager or sdpa attention")
        setattr(self.model, ATTRIBUTE, self)
        try:
            if self.backbone == "qwen25_vl":
                self._install_qwen_input()
            else:
                self._install_llava_input()
            self._install_decoder()
            for index, layer in enumerate(self.layers):
                self._install_layer(layer, index)
            original_save = self.model.save_pretrained

            def save(_model, directory, *args, **kwargs):
                result = original_save(directory, *args, **kwargs)
                save_fixed_mask(self.model, directory)
                return result

            self._patch(self.model, "save_pretrained", save)
            setattr(self.model.config, CONFIG_KEY, self.payload())
            self.active = True
        except BaseException:
            self.uninstall()
            raise
        return self

    def _install_qwen_input(self):
        original = self.model.forward

        def forward(_model, *args, **kwargs):
            call = _call(original, args, kwargs)
            ids = call.get("input_ids")
            if call.get("pixel_values_videos") is not None or call.get("video_grid_thw") is not None:
                raise ValueError("fixed_mask supports images, not videos")
            layout = None
            if not _cached(call.get("past_key_values")):
                if ids is None:
                    raise ValueError("fixed_mask requires input_ids to identify image positions")
                layout = make_layout(ids == self.model.config.image_token_id, call.get("attention_mask"))
            token = self.incoming.set(layout)
            try:
                return original(*args, **kwargs)
            finally:
                self.incoming.reset(token)
        self._patch(self.model, "forward", forward)

    def _install_llava_input(self):
        original = self.model.prepare_inputs_labels_for_multimodal

        def prepare(_model, *args, **kwargs):
            import torch
            call = _call(original, args, kwargs)
            ids = call["input_ids"]
            if ids.shape[0] != 1 or any(m != "image" for m in call.get("modalities", ["image"])):
                raise ValueError("fixed_mask LLaVA requires batch-one image inputs")
            if _cached(call.get("past_key_values")):
                self.incoming.set(None)
                return original(*args, **kwargs)
            labels = call.get("labels")
            # Transport source positions through the native expansion/truncation
            # using its existing label packing. Restore the user's labels exactly.
            call["labels"] = torch.arange(ids.shape[-1], device=ids.device)[None, :]
            output = list(original(**call))
            mapping = output[5]
            vision = mapping == -100
            self.incoming.set(make_layout(vision, output[2]))
            output[5] = None if labels is None else labels.gather(1, mapping.clamp_min(0)).masked_fill(vision, -100)
            return tuple(output)
        self._patch(self.model, "prepare_inputs_labels_for_multimodal", prepare)

    def _install_decoder(self):
        original = self.decoder.forward

        def forward(_decoder, *args, **kwargs):
            call = _call(original, args, kwargs)
            attach_layout(self.model, call)
            layout = call.get(KEY)
            if not _cached(call.get("past_key_values")) and layout is None:
                ids = call.get("input_ids")
                if ids is None:
                    raise ValueError("fixed_mask decoder is missing the prepared image layout")
                layout = make_layout(ids == -200, call.get("attention_mask"))
                call[KEY] = layout
            self.forward_count += 1
            try:
                return original(**call)
            finally:
                if self.backbone == "llava_onevision":
                    self.incoming.set(None)
        self._patch(self.decoder, "forward", forward)

    def _install_layer(self, layer, index):
        original_layer = layer.forward

        def forward(_layer, *args, **kwargs):
            layout = kwargs.pop(KEY, None)
            if layout is not None:
                self.masked_layer_calls += 1
            token = self.current.set(layout)
            try:
                return original_layer(*args, **kwargs)
            finally:
                self.current.reset(token)
        self._patch(layer, "forward", forward)
        original_attn = layer.self_attn.forward

        def attention(module, *args, **kwargs):
            import torch
            layout = self.current.get()
            if layout is None:
                return original_attn(*args, **kwargs)
            call = _call(original_attn, args, kwargs)
            hidden = call["hidden_states"]
            heads = module.config.num_attention_heads
            path_selected = any(self.spec[k].get(str(index)) for k in ("v2v", "t2v", "t2t"))
            disabled = self.spec["heads"].get(str(index), [])
            if not path_selected and not disabled:
                return original_attn(*args, **kwargs)
            empty = None
            if path_selected:
                call["attention_mask"], empty = masked_attention(call.get("attention_mask"), layout, self.spec, index, hidden, heads)

            def mask_output(_projection, inputs):
                value = inputs[0]
                keep = torch.ones((*value.shape[:2], heads), dtype=torch.bool, device=value.device)
                if disabled:
                    keep[..., disabled] = ~layout.valid.to(value.device)[..., None]
                if empty is not None:
                    keep &= ~empty.to(value.device).transpose(1, 2)
                value = (value.reshape(*value.shape[:2], heads, -1) * keep[..., None]).reshape_as(value)
                return (value, *inputs[1:])

            hook = module.o_proj.register_forward_pre_hook(mask_output)
            try:
                return original_attn(**call)
            finally:
                hook.remove()
        self._patch(layer.self_attn, "forward", attention)
        original_ffn = layer.mlp.forward

        def ffn(module, *args, **kwargs):
            import torch
            layout = self.current.get()
            if layout is None or not any(self.spec[k].get(str(index)) for k in ("ffn_all", "ffn_vision", "ffn_text")):
                return original_ffn(*args, **kwargs)

            def mask_input(_projection, inputs):
                value = inputs[0]
                keep = torch.ones_like(value)
                groups = {"ffn_all": layout.valid, "ffn_vision": layout.vision,
                          "ffn_text": layout.valid & ~layout.vision}
                for kind, group in groups.items():
                    indices = self.spec[kind].get(str(index), [])
                    if indices:
                        keep[..., indices] *= (~group.to(value.device))[..., None]
                return (value * keep, *inputs[1:])
            hook = module.down_proj.register_forward_pre_hook(mask_input)
            try:
                return original_ffn(*args, **kwargs)
            finally:
                hook.remove()
        self._patch(layer.mlp, "forward", ffn)

    def payload(self):
        return {"backbone": self.backbone, "spec": deepcopy(self.spec)}

    def runtime_metadata(self):
        return {"prepared": self.active, "reversible": True, "spec": deepcopy(self.spec),
                "forward_calls": self.forward_count, "execution": "dense_mask_reference",
                "masked_layer_calls": self.masked_layer_calls,
                "physical_acceleration": False,
                "selected_counts": {k: sum(map(len, self.spec[k].values())) for k in KINDS}}

    def uninstall(self):
        for patch in reversed(self.patches):
            patch.restore()
        self.patches.clear()
        self.active = False
        if session_for(self.model) is self:
            delattr(self.model, ATTRIBUTE)
        if self.old_config is None:
            if hasattr(self.model.config, CONFIG_KEY):
                delattr(self.model.config, CONFIG_KEY)
        else:
            setattr(self.model.config, CONFIG_KEY, self.old_config)


def install_fixed_mask(model, spec, *, backbone):
    return FixedMaskSession(model, spec, backbone=backbone).install()


def save_fixed_mask(model, directory):
    for module in model.modules():
        session = session_for(module)
        if session is not None:
            atomic_write_json(Path(directory) / SIDECAR, session.payload())
            return
    raise ValueError("No fixed_mask is installed")


def load_fixed_mask(model, directory):
    value = json.loads((Path(directory) / SIDECAR).read_text(encoding="utf-8"))
    return install_fixed_mask(model, value["spec"], backbone=value["backbone"])
