"""Differentiable physical modality partitions for dense Qwen VL decoders."""
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
import json
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from mllm_comp.backbones.qwen3_vl import attention_replacement as kernels
from mllm_comp.backbones.qwen3_vl.ffn_replacement import Qwen3VLLowRankMLP, Qwen3VLTwoLayerMLP
from .cache import ModalitySplitCache


@dataclass(frozen=True)
class Layout:
    image: torch.Tensor
    valid: torch.Tensor


def _rope_for_sample(position_embeddings, sample, indices, backbone, native):
    if position_embeddings is None:
        raise ValueError("Native position embeddings are required; positions must not be renumbered.")
    cos, sin = position_embeddings
    if backbone == "qwen25_vl":
        sections = native.rope_scaling["mrope_section"] * 2
        cos = torch.cat([part[i % 3] for i, part in enumerate(cos.split(sections, dim=-1))], dim=-1)
        sin = torch.cat([part[i % 3] for i, part in enumerate(sin.split(sections, dim=-1))], dim=-1)
    return tuple(p[sample:sample + 1].index_select(1, indices) for p in (cos, sin))


def _native_attention(native, x, rope, cache):
    """Same projections, norms, RoPE, scale and dropout as native dense attention."""
    shape = (*x.shape[:-1], -1, native.head_dim)
    q = getattr(native, "q_norm", nn.Identity())(native.q_proj(x).view(shape)).transpose(1, 2)
    k = getattr(native, "k_norm", nn.Identity())(native.k_proj(x).view(shape)).transpose(1, 2)
    v = native.v_proj(x).view(shape).transpose(1, 2)
    q, k = kernels._apply_qwen3_vl_rope(q, k, rope)
    if cache is not None:
        k, v = cache.update(k, v, native.layer_idx)
    k = kernels._repeat_kv(k, native.num_key_value_groups)
    v = kernels._repeat_kv(v, native.num_key_value_groups)
    n, total = q.shape[-2], k.shape[-2]
    rows = torch.arange(total - n, total, device=x.device)[:, None]
    cols = torch.arange(total, device=x.device)[None, :]
    causal = cols <= rows
    sliding = getattr(native, "sliding_window", None)
    if sliding is not None:
        causal = causal & (cols > rows - sliding)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal,
                                      dropout_p=native.attention_dropout if native.training else 0.0,
                                      scale=native.scaling)
    return native.o_proj(y.transpose(1, 2).reshape(*x.shape[:-1], -1))


class SplitAttention(nn.Module):
    def __init__(self, native, branch, backbone, context):
        super().__init__()
        self.native = native
        self.operator = branch["image_operator"]
        self.backbone = backbone
        self.context = context
        factories = {"window": kernels.Qwen3VLSparseWindowAttention,
                     "cnn": kernels.Qwen3VLCausalConvAttention,
                     "linear": kernels.Qwen3VLLinearAttention,
                     "lolcats": kernels.Qwen3VLLoLCATsLinearWindowAttention,
                     "nha": kernels.Qwen3VLNativeHybridAttention}
        config = dict(branch.get("config", {}))
        if self.operator == "window" and "strategy" not in branch:
            if set(config) - {"local_left", "debug"}:
                raise ValueError("Image window accepts local_left and debug only; it is strictly causal.")
        if "strategy" in branch and self.operator != "skip":
            from mllm_comp.compressors.llm_level import Qwen3VLAttentionReplacement
            self.image_op = Qwen3VLAttentionReplacement.build_replacement(
                native, strategy=branch["strategy"], config=config)
        else:
            self.image_op = factories[self.operator](native, **config) if self.operator in factories else None
        self.train(native.training)

    def forward(self, hidden_states, position_embeddings=None, past_key_values=None, **kwargs):
        layout = self.context.get()
        if layout is None:
            raise RuntimeError("Modality layout must be supplied through the installed model.")
        out = torch.zeros_like(hidden_states)
        for sample in range(hidden_states.shape[0]):
            for modality, mask in (("text", ~layout.image), ("image", layout.image)):
                if modality == "image" and self.operator == "skip":
                    continue  # Strict skip: before gathering, projections and all cache allocation.
                indices = (mask[sample] & layout.valid[sample]).nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue
                x = hidden_states[sample:sample + 1].index_select(1, indices)
                rope = _rope_for_sample(position_embeddings, sample, indices, self.backbone, self.native)
                cache = None if past_key_values is None else past_key_values.branch(
                    self.native.layer_idx, sample, modality)
                if modality == "text" or self.operator == "full":
                    y = _native_attention(self.native, x, rope, cache)
                else:
                    y, _ = self.image_op(x, rope, None, past_key_values=cache)
                out[sample] = out[sample].index_copy(0, indices, y[0])
        return out, None


class SplitFFN(nn.Module):
    def __init__(self, native, branch, context):
        super().__init__()
        self.native = native
        self.operator = branch["image_operator"]
        self.context = context
        factories = {"low_rank": Qwen3VLLowRankMLP, "two_layer_mlp": Qwen3VLTwoLayerMLP}
        if "strategy" in branch and self.operator != "skip":
            from mllm_comp.compressors.llm_level import Qwen3VLFFNReplacement
            self.image_op = Qwen3VLFFNReplacement.build_replacement(
                native, strategy=branch["strategy"], config=branch.get("config", {}))
        else:
            self.image_op = factories[self.operator](native, **branch.get("config", {})) if self.operator in factories else None
        self.train(native.training)

    def forward(self, hidden_states):
        layout = self.context.get()
        if layout is None:
            raise RuntimeError("FFN modality layout unavailable outside decoder context.")
        out = torch.zeros_like(hidden_states)
        for sample in range(hidden_states.shape[0]):
            for is_image in (False, True):
                if is_image and self.operator == "skip":
                    continue
                mask = layout.image[sample] if is_image else ~layout.image[sample]
                indices = (mask & layout.valid[sample]).nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue
                x = hidden_states[sample:sample + 1].index_select(1, indices)
                op = self.image_op if is_image and self.image_op is not None else self.native
                out[sample] = out[sample].index_copy(0, indices, op(x)[0])
        return out


def _language_model(model):
    for item in (model, getattr(model, "model", None), getattr(model, "language_model", None),
                 getattr(getattr(model, "model", None), "language_model", None)):
        if item is not None and hasattr(item, "layers") and hasattr(item, "rotary_emb"):
            return item
    raise ValueError("Expected a dense Qwen2.5-VL or Qwen3-VL decoder model.")


def _branches(spec):
    """Yield per-selection branches from legacy homogeneous or derived layouts."""
    if spec.get("schema_version") == 2:
        for operation in spec.get("operations", []):
            yield operation["branch"], {**operation, "layers": [operation["layer"]]}
    else:
        for name in ("attention", "ffn"):
            if name in spec:
                yield name, spec[name]


def _validate_spec(model, spec):
    spec = deepcopy(spec)
    if spec.get("schema_version") not in (1, 2) or spec.get("backbone") not in ("qwen25_vl", "qwen3_vl"):
        raise ValueError("Expected schema_version=1|2 and backbone=qwen25_vl|qwen3_vl.")
    language = _language_model(model)
    if (getattr(language.config, "use_sliding_window", False) or
            "sliding_attention" in (getattr(language.config, "layer_types", None) or []) or
            any(getattr(layer.self_attn, "sliding_window", None) is not None for layer in language.layers)):
        raise ValueError("Native sliding-attention configurations are unsupported by modality split.")
    actual = language.config.model_type
    # Legacy flat Qwen2.5 checkpoints propagate their top-level model_type into
    # the TextConfig through Transformers' backward-compatible constructor.
    expected = {"qwen25_vl": {"qwen2_5_vl_text", "qwen2_5_vl"},
                "qwen3_vl": {"qwen3_vl_text"}}[spec["backbone"]]
    if actual not in expected:
        raise ValueError(f"Spec backbone does not match model: {expected} != {actual}.")
    branches = list(_branches(spec))
    if not any(branch.get("layers") for _, branch in branches):
        raise ValueError("At least one selected attention or FFN layer is required.")
    seen = set()
    allowed_ops = {"attention": {"full", "skip", "window", "cnn", "linear", "lolcats", "nha"},
                   "ffn": {"native", "skip", "low_rank", "two_layer_mlp"}}
    for name, branch in branches:
        if name not in allowed_ops:
            raise ValueError(f"Invalid operation branch {name!r}.")
        indices = branch.get("layers", [])
        if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < len(language.layers) for i in indices):
            raise ValueError(f"Invalid {name} layer indices.")
        if (len(set(indices)) != len(indices) or branch.get("image_operator") not in allowed_ops[name]
                or any((name, index) in seen for index in indices)):
            raise ValueError(f"Invalid {name} operator or duplicate layers.")
        seen.update((name, index) for index in indices)
        branch.setdefault("config", {})
        if branch["image_operator"] in {"native", "skip", "full"} and branch["config"]:
            raise ValueError("Native/full/skip operators take no configuration.")
    return language, spec


class ModalitySplitSession:
    def __init__(self, model, spec, *, states=None):
        self.language, self.spec = _validate_spec(model, spec)
        self.model = model
        self.original_forward = model.forward
        self.original_layers = {}
        self.replacements = {}
        contexts = {}
        # Build all replacements before mutation, so invalid configs are atomic.
        for name, branch in _branches(self.spec):
            attribute, factory = ("self_attn", SplitAttention) if name == "attention" else ("mlp", SplitFFN)
            for index in branch["layers"]:
                context = contexts.setdefault(index, ContextVar(f"modality_layer_{index}", default=None))
                layer = self.language.layers[index]
                native = getattr(layer, attribute)
                args = (native, branch, self.spec["backbone"], context) if name == "attention" else (native, branch, context)
                self.replacements[index, attribute] = factory(*args)
        if states is not None:
            expected = {f"{index}:{attribute}": wrapper.state_dict()
                        for (index, attribute), wrapper in self.replacements.items()}
            if not isinstance(states, dict) or set(states) != set(expected):
                raise ValueError("Split checkpoint replacement keys do not match the layout.")
            # Constructors alias native projections. Validate the ENTIRE payload
            # before loading even one tensor, not only before installing wrappers.
            for key, target in expected.items():
                saved = states[key]
                if not isinstance(saved, dict) or set(saved) != set(target):
                    raise ValueError(f"Split checkpoint parameter keys mismatch for {key}.")
                if any(not isinstance(saved[name], torch.Tensor) or saved[name].shape != value.shape
                       or saved[name].layout != value.layout for name, value in target.items()):
                    raise ValueError(f"Split checkpoint parameter shapes/layout mismatch for {key}.")
                for name, value in target.items():
                    tensor = saved[name]
                    if tensor.is_quantized or tensor.is_meta or value.is_meta:
                        raise ValueError(f"Split checkpoint parameter {key}/{name} requires materialized, unquantized tensors.")
                    # Shape and layout alone do not establish dtype/device copy
                    # compatibility. Probe disposable storage before any native
                    # alias can be overwritten; memory is bounded by one tensor.
                    try:
                        with torch.no_grad():
                            torch.empty_like(value).copy_(tensor)
                    except (RuntimeError, TypeError, NotImplementedError) as exc:
                        raise ValueError(f"Split checkpoint parameter {key}/{name} cannot be copied safely.") from exc
            for (index, attribute), wrapper in self.replacements.items():
                wrapper.load_state_dict(states[f"{index}:{attribute}"], strict=True)
        for index, context in contexts.items():
            layer = self.language.layers[index]
            original = layer.forward
            self.original_layers[index] = original
            def make_forward(original, context):
                @wraps(original)
                def forward(_layer, *args, **kwargs):
                    layout = kwargs.pop("modality_layout", None)
                    if layout is None:
                        raise RuntimeError("Decoder did not receive immutable modality layout.")
                    token = context.set(layout)
                    try:
                        return original(*args, **kwargs)
                    finally:
                        context.reset(token)
                return forward
            layer.forward = MethodType(make_forward(original, context), layer)
        for (index, attribute), wrapper in self.replacements.items():
            setattr(self.language.layers[index], attribute, wrapper)
        @wraps(self.original_forward)
        def forward(_model, *args, **kwargs):
            return self._forward(*args, **kwargs)
        model.forward = MethodType(forward, model)

    def _forward(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError("Pass optional model inputs by keyword in modality-split mode.")
        ids = kwargs.get("input_ids", args[0] if args else None)
        embeds = kwargs.get("inputs_embeds")
        source = ids if ids is not None else embeds
        if source is None:
            raise ValueError("input_ids or inputs_embeds are required.")
        batch, length = source.shape[:2]
        image = kwargs.pop("modality_image_mask", None)
        config = self.model.config
        video_id = getattr(config, "video_token_id", None)
        if kwargs.get("pixel_values_videos") is not None or kwargs.get("video_grid_thw") is not None or (
                ids is not None and video_id is not None and bool((ids == video_id).any())):
            raise ValueError("Video inputs are not supported by modality split.")
        for key in ("cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens", "seq_lens", "packed_sequence",
                    "sequence_ids", "segment_ids"):
            if kwargs.get(key) is not None:
                raise ValueError("Packed inputs are not supported by modality split.")
        if image is None:
            image_id = getattr(config, "image_token_id", None)
            if ids is None and (image_id is not None or kwargs.get("pixel_values") is not None):
                raise ValueError("Multimodal model inputs_embeds requires explicit modality_image_mask, including all-text inputs.")
            image = ids == image_id if ids is not None and image_id is not None else torch.zeros(
                (batch, length), dtype=torch.bool, device=source.device)
        if tuple(image.shape) != (batch, length):
            raise ValueError("modality_image_mask must match current input batch and sequence.")
        mask = kwargs.get("attention_mask")
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.ndim != 2):
            raise ValueError("Only unpacked 2D padding attention masks are supported.")
        if mask is not None and (mask.shape[0] != batch or mask.shape[1] < length or
                                 not bool(((mask == 0) | (mask == 1)).all())):
            raise ValueError("Padding masks must be binary with matching batch and at least current sequence length.")
        valid = torch.ones((batch, length), dtype=torch.bool, device=source.device) if mask is None else mask[:, -length:].bool()
        positions = kwargs.get("position_ids")
        # Grid axes legitimately reset inside images; only the extra text axis can
        # identify packing in 4-axis inputs. Standard 2-axis text IDs must increase.
        text_pos = positions[0] if positions is not None and positions.ndim == 3 and positions.shape[0] == 4 else (
            positions if positions is not None and positions.ndim == 2 else None)
        if text_pos is not None:
            for sample in range(batch):
                # Padding may separate the two sides of a packed boundary. Compare
                # gathered valid positions, not merely physically adjacent tokens.
                keep = valid[sample].to(text_pos.device)
                if positions.ndim == 2:
                    keep = keep & ~image[sample].bool().to(text_pos.device)
                gathered = text_pos[sample][keep]
                if bool((gathered[1:] <= gathered[:-1]).any()):
                    raise ValueError("Packed/reset text positions are unsupported.")
        if positions is not None and positions.ndim == 3 and positions.shape[0] == 3:
            for sample in range(batch):
                # Grid coordinates can reset within an image, but positions of
                # valid text tokens must strictly increase across the sample.
                text_indices = (valid[sample].to(image.device) & ~image[sample].bool()).nonzero(as_tuple=True)[0]
                text_positions = positions[0, sample].index_select(0, text_indices.to(positions.device))
                if bool((text_positions[1:] <= text_positions[:-1]).any()):
                    raise ValueError("Packed/reset text positions are unsupported.")
        kwargs["modality_layout"] = Layout(image.bool().to(source.device).clone(), valid.to(source.device).clone())
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = getattr(config, "use_cache", getattr(self.language.config, "use_cache", True))
        if self.model.training and getattr(self.language, "gradient_checkpointing", False):
            use_cache = False
            kwargs["use_cache"] = False
        cache = kwargs.get("past_key_values")
        if use_cache:
            if cache is None or not isinstance(cache, ModalitySplitCache):
                if cache is not None and (type(cache).__name__ != "DynamicCache" or cache.get_seq_length() != 0):
                    raise ValueError("Start modality-split generation with an empty dynamic cache.")
                cache = ModalitySplitCache(self.language.config)
                kwargs["past_key_values"] = cache
            if self.model.training and cache.global_length:
                raise ValueError("Cached recurrent training is unsupported; train with use_cache=False.")
        elif cache is not None:
            raise ValueError("past_key_values requires use_cache=True in modality-split mode.")
        output = self.original_forward(*args, **kwargs)
        if use_cache:
            cache.global_length += length
        return output

    def metadata(self):
        installed = []
        counts = {"attention": 0, "ffn": 0}
        for (index, attribute), wrapper in self.replacements.items():
            if getattr(self.language.layers[index], attribute) is not wrapper:
                continue
            branch, suffix = ("attention", "attn") if attribute == "self_attn" else ("ffn", "ffn")
            counts[branch] += 1
            image_op = wrapper.image_op
            installed.append({"handle": f"llm.layer[{index}].{suffix}", "branch": branch, "layer": index,
                              "wrapper_class": type(wrapper).__name__,
                              "image_operator_class": None if image_op is None else type(image_op).__name__,
                              "image_operator": wrapper.operator,
                              "native_class": type(wrapper.native).__name__})
        active = (getattr(self.model, "_modality_split_session", None) is self and
                  len(installed) == len(self.replacements))
        return {"spec": deepcopy(self.spec), "implementation_protocol": "physical-gather-scatter-v1",
                "active": active, "installed_handles": [item["handle"] for item in installed],
                "replacement_count": len(installed), "replacement_counts": counts,
                "installed_modules": installed,
                "text_operator": "native-causal", "cross_modal_edges": False,
                "image_window": "causal-image-subsequence-current-plus-local_left",
                "position_protocol": "original-native-rope-before-gather",
                "cache_protocol": "per-layer-per-sample-per-modality-with-global-model-clock"}

    def uninstall(self):
        for (index, attribute), wrapper in self.replacements.items():
            setattr(self.language.layers[index], attribute, wrapper.native)
        for index, original in self.original_layers.items():
            self.language.layers[index].forward = original
        self.model.forward = self.original_forward
        if getattr(self.model, "_modality_split_session", None) is self:
            del self.model._modality_split_session


def install_modality_split(model, spec, *, states=None):
    if hasattr(model, "_modality_split_session"):
        raise ValueError("Modality split is already installed; uninstall before replacing its spec.")
    session = ModalitySplitSession(model, spec, states=states)
    model._modality_split_session = session
    return session


def native_state_dict(model):
    """Native-compatible *trained* state. Use with model.save_pretrained(state_dict=...)."""
    result = {}
    for name, value in model.state_dict().items():
        if ".self_attn.image_op." in name or ".mlp.image_op." in name:
            continue
        name = name.replace(".self_attn.native.", ".self_attn.").replace(".mlp.native.", ".mlp.")
        result[name] = value
    return result


def save_modality_split(model, directory):
    """Save exact spec and all selected native/replacement weights (not unselected weights).

    Call native_state_dict + HF save_pretrained separately for a standalone trained
    checkpoint. Load that native checkpoint first, then load_modality_split.
    """
    session = model._modality_split_session
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "modality_split.json").write_text(json.dumps(session.spec, indent=2) + "\n")
    states = {f"{index}:{attribute}": wrapper.state_dict() for (index, attribute), wrapper in session.replacements.items()}
    torch.save(states, directory / "modality_split.pt")


def load_modality_split(model, directory):
    directory = Path(directory)
    spec = json.loads((directory / "modality_split.json").read_text())
    states = torch.load(directory / "modality_split.pt", map_location="cpu", weights_only=True)
    return install_modality_split(model, spec, states=states)
