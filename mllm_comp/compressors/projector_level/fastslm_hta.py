"""FastSLM Hierarchical Temporal Abstractor for Qwen audio tokens.

This is a framework adaptation of the official FastSLM HTA implementation.
The official module consumes 50 Hz Whisper features in 1,500-frame windows;
Qwen2.5-Omni's registered pre-projector target is after its two-frame pooler,
so the equivalent default window is 750 tokens and the paper's 1.67 token/s
budget is ``50 / 750 == 1 / 15``.

The plugin owns configuration and budgets, the kernel owns tensor math, and
the Qwen strategy/controller own runtime module installation.  No hook or
provider-specific model path is registered here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from mllm_comp.compressors.audio_rule_utils import AudioTokenStrategyBase, _torch
from mllm_comp.compressors.base import CompressionContext
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.core.errors import InvariantError
from mllm_comp.core.metadata import StageMetadata
from mllm_comp.core.plan import CompressionPlan
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    Modality,
    ObjectFamily,
    Placement,
    TokenRole,
)


@dataclass(frozen=True)
class FastSLMHTASpec:
    """Normalized FastSLM HTA settings.

    ``stage_query_tokens=80``, ``output_query_tokens=50`` and two detail
    recovery blocks are the official model defaults.  ``window_tokens=750``
    is the only Qwen-specific rate adaptation: this target receives 25 Hz
    pooled features instead of the paper's 50 Hz encoder features.
    """

    keep_ratio: float = 1.0 / 15.0
    minimum_keep_tokens: int = 1
    stage_query_tokens: int = 80
    output_query_tokens: int = 50
    window_tokens: int = 750
    detail_recovery_depth: int = 2
    num_attention_heads: int = 0
    score_source: str = "learned_hierarchical_cross_attention"
    target_site: str = "audio_encoder_output_pre_projector"
    checkpoint_path: str | None = None
    checkpoint_strict: bool = True
    trainable: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FastSLMHTASpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        schedule = _mapping(config.get("schedule"), "schedule")
        architecture = _mapping(config.get("config"), "config")
        checkpoint_value = architecture.get("checkpoint_path", cls.checkpoint_path)
        if checkpoint_value is None or checkpoint_value == "":
            checkpoint_path = None
        elif not isinstance(checkpoint_value, str):
            raise ValueError("checkpoint_path must be a string or null")
        else:
            checkpoint_path = checkpoint_value
        spec = cls(
            keep_ratio=float(schedule.get("keep_ratio", cls.keep_ratio)),
            minimum_keep_tokens=_strict_int(
                schedule.get("minimum_keep_tokens", cls.minimum_keep_tokens),
                "minimum_keep_tokens",
            ),
            stage_query_tokens=_strict_int(
                architecture.get("stage_query_tokens", cls.stage_query_tokens),
                "stage_query_tokens",
            ),
            output_query_tokens=_strict_int(
                architecture.get("output_query_tokens", cls.output_query_tokens),
                "output_query_tokens",
            ),
            window_tokens=_strict_int(
                architecture.get("window_tokens", cls.window_tokens),
                "window_tokens",
            ),
            detail_recovery_depth=_strict_int(
                architecture.get(
                    "detail_recovery_depth", cls.detail_recovery_depth
                ),
                "detail_recovery_depth",
            ),
            num_attention_heads=_strict_int(
                architecture.get("num_attention_heads", cls.num_attention_heads),
                "num_attention_heads",
            ),
            score_source=str(
                criterion.get("score_source", cls.score_source)
            ),
            target_site=str(
                next(value for value in (
                    operation.get("target_site"), architecture.get("target_site"), cls.target_site,
                ) if value is not None)
            ),
            checkpoint_path=checkpoint_path,
            checkpoint_strict=_strict_bool(
                architecture.get("checkpoint_strict", cls.checkpoint_strict),
                "checkpoint_strict",
            ),
            trainable=_strict_bool(
                architecture.get("trainable", cls.trainable),
                "trainable",
            ),
        )
        spec.validate()
        if operation.get("type") not in {None, "merge", "distill"}:
            raise ValueError(
                "fastslm_hta_audio requires operation.type=merge or distill"
            )
        return spec

    def validate(self) -> None:
        if not 0 < self.keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        if self.minimum_keep_tokens < 1:
            raise ValueError("minimum_keep_tokens must be at least 1")
        if self.stage_query_tokens < 1:
            raise ValueError("stage_query_tokens must be at least 1")
        if self.output_query_tokens < 1:
            raise ValueError("output_query_tokens must be at least 1")
        if self.window_tokens < 1:
            raise ValueError("window_tokens must be at least 1")
        if self.detail_recovery_depth < 0:
            raise ValueError("detail_recovery_depth must be non-negative")
        if self.num_attention_heads < 0:
            raise ValueError("num_attention_heads must be non-negative")
        if self.minimum_keep_tokens > self.output_query_tokens:
            raise ValueError(
                "minimum_keep_tokens cannot exceed output_query_tokens"
            )
        maximum_ratio = self.output_query_tokens / self.window_tokens
        if self.keep_ratio > maximum_ratio + 1e-12:
            raise ValueError(
                "keep_ratio exceeds the configured per-window query capacity: "
                f"{self.keep_ratio} > {maximum_ratio}"
            )
        if self.score_source != "learned_hierarchical_cross_attention":
            raise ValueError(
                "FastSLM HTA requires "
                "score_source='learned_hierarchical_cross_attention'"
            )


def fastslm_hta_keep_count(
    length: int,
    keep_ratio: float,
    *,
    minimum_keep_tokens: int = 1,
    window_tokens: int = 750,
    output_query_tokens: int = 50,
) -> int:
    """Return the exact per-audio HTA budget supported by its query banks."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if minimum_keep_tokens < 1:
        raise ValueError("minimum_keep_tokens must be at least 1")
    if window_tokens < 1 or output_query_tokens < 1:
        raise ValueError("window_tokens and output_query_tokens must be positive")
    if minimum_keep_tokens > output_query_tokens:
        raise ValueError("minimum_keep_tokens cannot exceed output_query_tokens")
    if length == 0:
        return 0
    window_count = math.ceil(length / window_tokens)
    capacity = window_count * output_query_tokens
    requested = max(minimum_keep_tokens, math.floor(length * keep_ratio))
    return min(length, capacity, requested)


def fastslm_window_budgets(
    window_lengths: list[int],
    *,
    keep_count: int,
    output_query_tokens: int,
) -> list[int]:
    """Distribute an exact global budget proportionally across time windows."""

    if not window_lengths or any(length < 1 for length in window_lengths):
        raise ValueError("window_lengths must contain positive lengths")
    if not 1 <= keep_count <= sum(window_lengths):
        raise ValueError("keep_count is outside the input length")
    capacity = len(window_lengths) * output_query_tokens
    if keep_count > capacity:
        raise ValueError("keep_count exceeds FastSLM query capacity")

    total_length = sum(window_lengths)
    quotas = [keep_count * length / total_length for length in window_lengths]
    budgets = [min(output_query_tokens, math.floor(value)) for value in quotas]
    remaining = keep_count - sum(budgets)
    ranking = sorted(
        range(len(window_lengths)),
        key=lambda index: (-(quotas[index] - math.floor(quotas[index])), index),
    )
    while remaining:
        changed = False
        for index in ranking:
            if budgets[index] >= output_query_tokens:
                continue
            budgets[index] += 1
            remaining -= 1
            changed = True
            if remaining == 0:
                break
        if not changed:
            raise AssertionError("FastSLM budget allocation exhausted capacity")
    return budgets


def build_fastslm_hta(
    *,
    hidden_size: int,
    num_attention_heads: int,
    stage_query_tokens: int,
    output_query_tokens: int,
    window_tokens: int,
    detail_recovery_depth: int,
) -> Any:
    """Construct the official HTA modules without importing torch at catalog time."""

    import torch
    from torch import nn
    from torch.nn import functional as functional

    if hidden_size < 4 or hidden_size % 2:
        raise InvariantError("FastSLM HTA hidden_size must be even and at least 4.")
    if num_attention_heads < 1 or hidden_size % num_attention_heads:
        raise InvariantError(
            "FastSLM HTA attention heads must divide the hidden size."
        )
    if min(stage_query_tokens, output_query_tokens, window_tokens) < 1:
        raise InvariantError("FastSLM HTA token counts must be positive.")
    if detail_recovery_depth < 0:
        raise InvariantError("FastSLM HTA detail recovery depth must be non-negative.")

    def sinusoids(length: int) -> Any:
        half = hidden_size // 2
        log_increment = math.log(10000.0) / (half - 1)
        inverse = torch.exp(-log_increment * torch.arange(half, dtype=torch.float32))
        scaled = torch.arange(length, dtype=torch.float32)[:, None] * inverse[None, :]
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)

    class Compressor(nn.Module):
        def __init__(self, num_query: int, context_tokens: int) -> None:
            super().__init__()
            self.num_heads = num_attention_heads
            self.head_dims = hidden_size // num_attention_heads
            self.query = nn.Parameter(torch.empty(1, num_query, hidden_size))
            nn.init.normal_(self.query, mean=0.0, std=0.02)
            self.q_ln = nn.LayerNorm(hidden_size, eps=1e-5)
            self.kv_ln = nn.LayerNorm(hidden_size, eps=1e-5)
            self.out_proj = nn.Linear(hidden_size, hidden_size)
            self.register_buffer("q_pos_embeds", sinusoids(num_query))
            self.register_buffer("kv_pos_embeds", sinusoids(context_tokens))

        def forward(self, values: Any, *, query_count: int | None = None) -> Any:
            active_queries = self.query
            if query_count is not None:
                if not 1 <= query_count <= int(self.query.shape[1]):
                    raise InvariantError("FastSLM query_count exceeds its query bank.")
                active_queries = active_queries[:, :query_count]
            if int(values.shape[1]) > int(self.kv_pos_embeds.shape[0]):
                raise InvariantError("FastSLM compressor context is too short.")
            query = self.q_ln(active_queries.expand(values.shape[0], -1, -1))
            values = self.kv_ln(values)
            query_pos = self.q_pos_embeds[: query.shape[1]].to(
                device=query.device,
                dtype=query.dtype,
            )
            key_pos = self.kv_pos_embeds[: values.shape[1]].to(
                device=values.device,
                dtype=values.dtype,
            )
            query = (query + query_pos).reshape(
                query.shape[0], query.shape[1], self.num_heads, self.head_dims
            ).transpose(1, 2)
            key = (values + key_pos).reshape(
                values.shape[0], values.shape[1], self.num_heads, self.head_dims
            ).transpose(1, 2)
            value = values.reshape(
                values.shape[0], values.shape[1], self.num_heads, self.head_dims
            ).transpose(1, 2)
            attended = functional.scaled_dot_product_attention(query, key, value)
            attended = attended.transpose(1, 2).reshape(
                attended.shape[0], attended.shape[2], hidden_size
            )
            return self.out_proj(attended)

    class MultiHeadAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(hidden_size, hidden_size, bias=True)
            self.k = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, hidden_size, bias=True)
            self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        def forward(self, values: Any, context: Any | None = None) -> Any:
            source = values if context is None else context
            query = self.q(values).reshape(
                values.shape[0], values.shape[1], num_attention_heads, -1
            ).transpose(1, 2)
            key = self.k(source).reshape(
                source.shape[0], source.shape[1], num_attention_heads, -1
            ).transpose(1, 2)
            value = self.v(source).reshape(
                source.shape[0], source.shape[1], num_attention_heads, -1
            ).transpose(1, 2)
            attended = functional.scaled_dot_product_attention(query, key, value)
            attended = attended.transpose(1, 2).reshape(
                attended.shape[0], attended.shape[2], hidden_size
            )
            return self.out_proj(attended)

    class AttentionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = MultiHeadAttention()
            self.cross_attn = MultiHeadAttention()
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
            self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)

        def forward(self, values: Any, context: Any) -> Any:
            values = values + self.attn(self.norm1(values))
            return values + self.cross_attn(self.norm2(values), context)

    class Downsampler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(
                hidden_size, hidden_size, kernel_size=3, padding=1
            )
            self.conv2 = nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.ln_post = nn.LayerNorm(hidden_size, eps=1e-5)

        def forward(self, values: Any) -> Any:
            values = functional.gelu(self.conv1(values))
            values = functional.gelu(self.conv2(values))
            return self.ln_post(values.transpose(1, 2))

    class HierarchicalTemporalAbstractor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.window_tokens = window_tokens
            self.output_query_tokens = output_query_tokens
            self.compressor1 = Compressor(stage_query_tokens, window_tokens)
            self.stage1 = Downsampler()
            stage2_context = math.ceil(window_tokens / 2)
            self.compressor2 = Compressor(stage_query_tokens, stage2_context)
            self.stage2 = Downsampler()
            stage3_context = math.ceil(stage2_context / 2)
            self.compressor3 = Compressor(stage_query_tokens, stage3_context)
            self.compressor = Compressor(
                output_query_tokens,
                stage_query_tokens * 3,
            )
            self.out_attn = nn.ModuleList(
                AttentionBlock() for _ in range(detail_recovery_depth)
            )

        def _forward_window(self, values: Any, budget: int) -> Any:
            features0 = values.unsqueeze(0)
            stage1_tokens = self.compressor1(features0)
            features1 = self.stage1(features0.transpose(1, 2))
            stage2_tokens = self.compressor2(features1)
            features2 = self.stage2(features1.transpose(1, 2))
            stage3_tokens = self.compressor3(features2)
            hierarchy = torch.cat(
                [stage1_tokens, stage2_tokens, stage3_tokens], dim=1
            )
            compressed = self.compressor(hierarchy, query_count=budget)
            recovery_features = torch.cat(
                [features0, features1, features2], dim=1
            )
            for block in self.out_attn:
                compressed = block(compressed, recovery_features)
            return compressed.squeeze(0)

        def forward(self, values: Any, *, keep_count: int) -> tuple[Any, dict[str, Any]]:
            windows = list(values.split(self.window_tokens, dim=0))
            lengths = [int(window.shape[0]) for window in windows]
            budgets = fastslm_window_budgets(
                lengths,
                keep_count=keep_count,
                output_query_tokens=self.output_query_tokens,
            )
            outputs = [
                self._forward_window(window, budget)
                for window, budget in zip(windows, budgets)
                if budget > 0
            ]
            output = torch.cat(outputs, dim=0)
            return output, {
                "window_lengths": lengths,
                "window_budgets": budgets,
                "stage_query_tokens": stage_query_tokens,
                "output_query_tokens": output_query_tokens,
                "detail_recovery_depth": detail_recovery_depth,
                "token_order": "window_then_query_position",
                "implementation": "official_fastslm_hta_qwen_preprojector_adaptation",
            }

    return HierarchicalTemporalAbstractor()


def fastslm_hierarchical_temporal_abstract(
    hidden_states: Any,
    *,
    keep_count: int,
    abstractor: Any,
    transform: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Run HTA on one rank-2 audio unit and emit exactly ``keep_count`` tokens."""

    if getattr(hidden_states, "ndim", None) != 2:
        raise InvariantError("FastSLM HTA expects rank-2 token hidden states.")
    length = int(hidden_states.shape[0])
    if not 1 <= keep_count <= length:
        raise InvariantError(
            f"FastSLM HTA keep_count must be in [1, {length}], got {keep_count}."
        )
    if keep_count == length:
        return hidden_states, {
            "window_lengths": [length],
            "window_budgets": [length],
            "token_order": "chronological_identity",
            "implementation": "identity",
        }
    if transform is None:
        output, details = abstractor(hidden_states, keep_count=keep_count)
    else:
        from mllm_comp.compressors.audio_rule_utils import validate_values
        output = transform({"values": hidden_states, "keep_count": keep_count, "abstractor": abstractor})
        validate_values(output, hidden_states[:keep_count])
        details = {"implementation": "custom_transform", "token_order": "chronological_window_query_slots"}
    if getattr(output, "ndim", None) != 2:
        raise InvariantError("FastSLM HTA emitted a non-rank-2 tensor.")
    if int(output.shape[0]) != keep_count:
        raise InvariantError(
            "FastSLM HTA output length does not match its planned budget: "
            f"expected {keep_count}, got {int(output.shape[0])}."
        )
    if int(output.shape[1]) != int(hidden_states.shape[1]):
        raise InvariantError("FastSLM HTA changed the hidden width.")
    return output, details


def load_fastslm_hta_checkpoint(
    abstractor: Any,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    module_name: str = "_mllm_comp_fastslm_hta_audio",
) -> dict[str, Any]:
    """Load HTA tensors from an official file or a full platform checkpoint."""

    import torch

    path = Path(checkpoint_path).expanduser()
    if not path.is_file() and not path.is_dir():
        raise InvariantError(f"FastSLM HTA checkpoint does not exist: {path}")

    component_names = (
        "compressor1.",
        "stage1.",
        "compressor2.",
        "stage2.",
        "compressor3.",
        "compressor.",
        "out_attn.",
    )
    def platform_prefix(keys: Any) -> str | None:
        modules = set()
        for raw_key in keys:
            key = str(raw_key)
            position = key.find("_mllm_comp_fastslm_hta_audio")
            if position >= 0:
                modules.add(key[position:].partition(".")[0])
        if module_name in modules:
            return module_name + "."
        # Only an old, single standard module is an unambiguous source for a
        # moved recipe. Never borrow another slot from a multi-HTA checkpoint.
        if modules == {"_mllm_comp_fastslm_hta_audio"}:
            return "_mllm_comp_fastslm_hta_audio."
        if modules:
            raise InvariantError(f"Checkpoint has no HTA weights for {module_name!r}.")
        return None

    def load_safetensors(
        source: Path,
        *,
        selected_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise InvariantError(
                "Loading a safetensors FastSLM checkpoint requires safetensors."
            ) from exc
        with safe_open(str(source), framework="pt", device="cpu") as stream:
            available = set(stream.keys())
            # Unindexed folders can split different HTA modules across files.
            # Select the requested module after assembling those relevant keys.
            keys = (selected_keys if selected_keys is not None else [
                key for key in available if any(component in key for component in component_names)
            ])
            missing = sorted(set(keys) - available)
            if missing:
                raise InvariantError(
                    f"FastSLM checkpoint index references missing tensors in {source}: "
                    f"{missing}"
                )
            return {key: stream.get_tensor(key) for key in keys}

    def unwrap(payload: Any) -> Mapping[str, Any]:
        if isinstance(payload, Mapping):
            for wrapper in ("state_dict", "model_state_dict", "model"):
                nested = payload.get(wrapper)
                if isinstance(nested, Mapping):
                    return nested
        if not isinstance(payload, Mapping):
            raise InvariantError(
                "FastSLM HTA checkpoint is not a state-dict mapping."
            )
        return payload

    def load_pytorch(source: Path) -> Mapping[str, Any]:
        try:
            loaded = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch
            loaded = torch.load(source, map_location="cpu")
        return unwrap(loaded)

    source_kind = "hta_state_dict"
    if path.is_dir():
        payload: dict[str, Any] = {}
        index_paths = [
            candidate
            for candidate in (
                path / "model.safetensors.index.json",
                path / "adapter_model.safetensors.index.json",
            )
            if candidate.is_file()
        ]
        if index_paths:
            index_path = index_paths[0]
            try:
                index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise InvariantError(
                    f"Cannot read FastSLM checkpoint index: {index_path}"
                ) from exc
            weight_map = index_payload.get("weight_map")
            if not isinstance(weight_map, Mapping):
                raise InvariantError(
                    f"FastSLM checkpoint index has no weight_map: {index_path}"
                )
            keys_by_shard: dict[str, list[str]] = {}
            platform_module = platform_prefix(weight_map)
            if platform_module is None:
                raise InvariantError(f"Checkpoint folder does not contain FastSLM HTA tensors: {path}")
            for raw_key, raw_shard in weight_map.items():
                key = str(raw_key)
                if platform_module not in key:
                    continue
                if not isinstance(raw_shard, str) or not raw_shard:
                    raise InvariantError(
                        f"FastSLM checkpoint index has an invalid shard for {key!r}."
                    )
                keys_by_shard.setdefault(raw_shard, []).append(key)
            if not keys_by_shard:
                raise InvariantError(
                    f"Checkpoint folder does not contain FastSLM HTA tensors: {path}"
                )
            for shard_name, selected_keys in sorted(keys_by_shard.items()):
                shard_path = path / shard_name
                if not shard_path.is_file():
                    raise InvariantError(
                        f"FastSLM checkpoint shard does not exist: {shard_path}"
                    )
                payload.update(
                    load_safetensors(
                        shard_path,
                        selected_keys=sorted(selected_keys),
                    )
                )
        else:
            safetensor_paths = sorted(path.glob("*.safetensors"))
            for source in safetensor_paths:
                payload.update(load_safetensors(source))
            if not payload:
                binary_paths = [
                    source
                    for source in (
                        path / "adapter_model.bin",
                        path / "pytorch_model.bin",
                        *sorted(path.glob("pytorch_model-*.bin")),
                    )
                    if source.is_file()
                ]
                for source in binary_paths:
                    payload.update(
                        {
                            str(key): value
                            for key, value in load_pytorch(source).items()
                            if any(component in str(key) for component in component_names)
                        }
                    )
            if not payload:
                raise InvariantError(
                    f"Checkpoint folder does not contain FastSLM HTA tensors: {path}"
                )
        source_kind = "platform_full_checkpoint"
    elif path.suffix == ".safetensors":
        payload = load_safetensors(path)
    else:
        payload = load_pytorch(path)

    payload = unwrap(payload)
    prefix = platform_prefix(payload)
    if prefix is not None:
        payload = {key: value for key, value in payload.items() if prefix in str(key)}

    expected_state = abstractor.state_dict()
    expected_parameters = dict(abstractor.named_parameters())
    remapped: dict[str, Any] = {}
    shape_mismatches: list[str] = []
    for raw_key, value in payload.items():
        key = str(raw_key)
        candidates = [key]
        for prefix in ("module.encoder.", "model.encoder.", "encoder.", "module."):
            if key.startswith(prefix):
                candidates.append(key[len(prefix) :])
        for component in component_names:
            position = key.find(component)
            if position >= 0:
                candidates.append(key[position:])
        target = next((item for item in candidates if item in expected_state), None)
        if target is None:
            continue
        expected_shape = getattr(expected_state[target], "shape", None)
        actual_shape = getattr(value, "shape", None)
        if expected_shape != actual_shape:
            if target.endswith(("q_pos_embeds", "kv_pos_embeds")):
                continue
            shape_mismatches.append(
                f"{target}: expected {tuple(expected_shape)}, got {tuple(actual_shape)}"
            )
            continue
        remapped[target] = value
    if shape_mismatches:
        raise InvariantError(
            "FastSLM HTA checkpoint tensor shapes do not match this module: "
            + "; ".join(shape_mismatches)
        )
    missing_parameters = sorted(set(expected_parameters) - set(remapped))
    if strict and missing_parameters:
        raise InvariantError(
            "FastSLM HTA checkpoint is missing trainable tensors: "
            f"{missing_parameters}"
        )
    abstractor.load_state_dict(remapped, strict=False)
    return {
        "checkpoint_path": str(path.resolve()),
        "loaded_tensors": len(remapped),
        "missing_trainable_tensors": missing_parameters,
        "source": (
            source_kind
            if source_kind == "platform_full_checkpoint"
            else (
                "official_fastslm_full_state_dict"
                if any(
                    str(key).startswith(("encoder.", "module.encoder."))
                    for key in payload
                )
                else source_kind
            )
        ),
    }


class FastSLMHTAProjectorStrategy(AudioTokenStrategyBase):
    """Learned HTA strategy backed by the official FastSLM module topology."""

    layer_families = frozenset({"encoder_token", "llm_token"})
    token_operation = "merge"
    merge_position_policy = "preserve_absolute_window_query_anchor"
    metadata_fields = (
        "keep_ratio",
        "minimum_keep_tokens",
        "stage_query_tokens",
        "output_query_tokens",
        "window_tokens",
        "detail_recovery_depth",
        "trainable",
    )

    def __init__(self, spec: Any) -> None:
        super().__init__(spec)
        self._runtime_module: Any | None = None
        self._checkpoint_report: dict[str, Any] | None = None
        self._parameter_count = 0
        self._num_attention_heads: int | None = None

    def build_runtime_module(self, *, hidden_size: int, num_attention_heads: int, module_name: str) -> Any:
        """Build the learned module from geometry supplied by the backbone."""
        if self._runtime_module is not None:
            return self._runtime_module
        module = build_fastslm_hta(
            hidden_size=int(hidden_size),
            num_attention_heads=num_attention_heads,
            stage_query_tokens=self.spec.stage_query_tokens,
            output_query_tokens=self.spec.output_query_tokens,
            window_tokens=self.spec.window_tokens,
            detail_recovery_depth=self.spec.detail_recovery_depth,
        )
        if self.spec.checkpoint_path is not None:
            self._checkpoint_report = load_fastslm_hta_checkpoint(
                module,
                self.spec.checkpoint_path,
                strict=self.spec.checkpoint_strict,
                module_name=module_name,
            )
        for parameter in module.parameters():
            parameter.requires_grad = bool(self.spec.trainable)
        self._parameter_count = sum(
            int(parameter.numel()) for parameter in module.parameters()
        )
        self._runtime_module = module
        self._num_attention_heads = num_attention_heads
        return module

    def set_runtime_module(self, module: Any) -> None:
        """Follow a trainer wrapper that replaces the registered model child."""

        self._runtime_module = module

    def runtime_metadata(self) -> dict[str, Any]:
        result = super().runtime_metadata()
        result.update(
            runtime_module_built=self._runtime_module is not None,
            checkpoint_loaded=self._checkpoint_report is not None,
            hta_parameters=self._parameter_count or None,
            num_attention_heads=self._num_attention_heads,
        )
        return result

    def output_length(self, length: int) -> int:
        return fastslm_hta_keep_count(
            length, self.spec.keep_ratio, minimum_keep_tokens=self.spec.minimum_keep_tokens,
            window_tokens=self.spec.window_tokens, output_query_tokens=self.spec.output_query_tokens,
        )

    def representative_indices(self, length: int, keep_count: int, device: Any) -> Any:
        torch = _torch()
        if keep_count == length:
            return torch.arange(length, device=device)
        lengths = [min(self.spec.window_tokens, length - start) for start in range(0, length, self.spec.window_tokens)]
        budgets = fastslm_window_budgets(lengths, keep_count=keep_count, output_query_tokens=self.spec.output_query_tokens)
        anchors = []
        start = 0
        for count, budget in zip(lengths, budgets):
            if budget:
                anchors.append(start + torch.arange(budget, device=device) * count // budget)
            start += count
        return torch.cat(anchors)

    def map_reference(self, reference: Any, *, indices: Any, details: dict[str, Any], scores: Any, keep_count: int) -> Any:
        return self.map_layer_reference(reference, keep_count)

    def map_layer_reference(self, reference: Any, keep_count: int) -> Any:
        if keep_count == int(reference.shape[0]):
            return reference
        raise InvariantError(
            "HTA creates learned query tokens through nonlinear, multi-head and multi-scale transforms; "
            "the existing fixed Encoder-token attention reference has no defined transfer to those queries."
        )

    @property
    def runtime_device(self) -> Any:
        return next(self._runtime_module.parameters()).device

    def compress(
        self,
        hidden_states: Any,
        *,
        scores: Any | None,
        keep_count: int,
        item_index: int,
    ) -> tuple[Any, None, dict[str, Any]]:
        del scores, item_index
        if self._runtime_module is None:
            raise InvariantError(
                "FastSLM HTA runtime module was not built before compression."
            )
        parameter = next(self._runtime_module.parameters())
        values = hidden_states.to(device=self.runtime_device, dtype=parameter.dtype)
        output, details = fastslm_hierarchical_temporal_abstract(
            values, keep_count=keep_count, abstractor=self._runtime_module,
            transform=self.rules.get("transform"),
        )
        return output.to(device=hidden_states.device, dtype=hidden_states.dtype), None, details


class FastSLMHTAAudio:
    """Projector-level plan provider for the learned FastSLM HTA."""

    name = "fastslm_hta_audio"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        FastSLMHTASpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        del context
        spec = FastSLMHTASpec.from_config(config)
        actions: list[TokenAction] = []
        original_total = 0
        planned_total = 0
        audio_spans = [
            span for span in state.modality_token_spans if span.modality == "audio"
        ]
        for audio_unit_index, span in enumerate(audio_spans):
            original_count = span.end - span.start
            keep_count = fastslm_hta_keep_count(
                original_count,
                spec.keep_ratio,
                minimum_keep_tokens=spec.minimum_keep_tokens,
                window_tokens=spec.window_tokens,
                output_query_tokens=spec.output_query_tokens,
            )
            original_total += original_count
            planned_total += keep_count
            actions.append(
                TokenAction(
                    op="merge",
                    target_indices=None,
                    merge_groups=None,
                    metadata={
                        "selection_deferred": True,
                        "selector": self.name,
                        "target_site": spec.target_site,
                        "sample_index": span.sample_index,
                        "audio_unit_index": audio_unit_index,
                        "span_start": span.start,
                        "span_end": span.end,
                        "original_count": original_count,
                        "keep_count": keep_count,
                        "keep_ratio": spec.keep_ratio,
                        "window_tokens": spec.window_tokens,
                        "stage_query_tokens": spec.stage_query_tokens,
                        "output_query_tokens": spec.output_query_tokens,
                        "token_order": "window_then_query_position",
                    },
                )
            )
        stage_name = str(config.get("_stage_name") or self.name)
        execution_phase = ExecutionPhase(
            config.get("_execution_phase", ExecutionPhase.GENERATE)
        )
        return CompressionPlan(
            stage_name=stage_name,
            placement=Placement.PROJECTOR_LEVEL,
            execution_phase=execution_phase,
            modalities={Modality.AUDIO},
            token_roles={TokenRole.OBSERVATION},
            method=CompressionMethod.TOKEN_COMPRESSION,
            object_family=ObjectFamily.TENSOR_SLOT,
            object_subtype=spec.target_site,
            token_actions=actions,
            legalization_policy=LegalizationPolicy(
                position_policy="compact",
                mask_policy="rebuild_from_tokens",
                label_policy="rebuild_ignore_index",
                coordinate_policy="drop_removed",
            ),
            this_stage_metadata=StageMetadata(
                stage_name=stage_name,
                changed_spans=audio_spans,
                stats={
                    "target_site": spec.target_site,
                    "score_source": spec.score_source,
                    "num_audio_units": len(actions),
                    "original_audio_tokens": original_total,
                    "planned_audio_tokens": planned_total,
                    "requested_keep_ratio": spec.keep_ratio,
                    "stage_query_tokens": spec.stage_query_tokens,
                    "output_query_tokens": spec.output_query_tokens,
                    "window_tokens": spec.window_tokens,
                    "detail_recovery_depth": spec.detail_recovery_depth,
                    "trainable": spec.trainable,
                    "checkpoint_configured": spec.checkpoint_path is not None,
                    "selection_deferred_to_runtime": True,
                },
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError(
                "FastSLMHTAAudio plan provider has no state; weights belong to its strategy module"
            )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


__all__ = [
    "FastSLMHTAProjectorStrategy",
    "FastSLMHTAAudio",
    "FastSLMHTASpec",
    "build_fastslm_hta",
    "fastslm_hierarchical_temporal_abstract",
    "fastslm_hta_keep_count",
    "fastslm_window_budgets",
    "load_fastslm_hta_checkpoint",
]
