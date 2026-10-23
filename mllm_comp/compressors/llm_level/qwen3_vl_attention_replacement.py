"""Plan and construct Qwen3-VL attention replacement methods."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.core.actions import OperationAction
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.metadata import StageMetadata
from mllm_comp.core.plan import CompressionPlan
from mllm_comp.core.state import ForwardState


_ATTENTION_PATH = re.compile(r"^llm\.layer\[(\d+)\]\.attn$")


class Qwen3VLAttentionReplacement:
    """Static shape-preserving Qwen3-VL attention replacement method."""

    name = "qwen3_vl_attention_replacement"
    version = "0.1.0"
    strategies = frozenset({
        "attn_linear_attention",
        "attn_cnn",
        "attn_identity",
        "attn_lolcats_linear_window",
        "attn_nha",
        "attn_sparse_window",
    })

    def validate_config(self, config: dict[str, Any]) -> None:
        operation = config.get("operation", {})
        if not isinstance(operation, dict) or operation.get("type") != "replace":
            raise ValueError(
                "qwen3_vl_attention_replacement requires operation.type=replace"
            )
        private = config.get("config", {})
        if not isinstance(private, dict):
            raise ValueError("replacement config must be a mapping")
        strategy = private.get("strategy")
        if strategy not in self.strategies:
            raise ValueError(
                f"replacement config.strategy must be one of {sorted(self.strategies)}, got {strategy!r}"
            )
        if strategy == "attn_linear_attention":
            eps = private.get("eps", 1.0e-6)
            if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps <= 0:
                raise ValueError("linear-attention config.eps must be positive")
        if strategy == "attn_cnn":
            kernel_size = private.get("kernel_size", 3)
            if (
                isinstance(kernel_size, bool)
                or not isinstance(kernel_size, int)
                or kernel_size < 1
                or kernel_size % 2 == 0
            ):
                raise ValueError(
                    "CNN config.kernel_size must be a positive odd integer"
                )
        if strategy == "attn_lolcats_linear_window":
            for name, default in (
                ("window_size", 64),
                ("decode_window_size", private.get("window_size", 64)),
                ("feature_dim", 64),
            ):
                value = private.get(name, default)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"LoLCATs config.{name} must be a positive integer"
                    )
            feature_map = private.get("feature_map", "hedgehog_softmax_dim")
            if feature_map != "hedgehog_softmax_dim":
                raise ValueError(
                    "LoLCATs config.feature_map must be 'hedgehog_softmax_dim'"
                )
            for name, default in (
                ("tie_qk_feature_maps", False),
                ("train_window_weight", True),
                ("affine_attention_factors", False),
            ):
                if not isinstance(private.get(name, default), bool):
                    raise ValueError(f"LoLCATs config.{name} must be a boolean")
            window_weight_init = private.get("window_weight_init", 0.1)
            if (
                isinstance(window_weight_init, bool)
                or not isinstance(window_weight_init, (int, float))
                or not 0 < window_weight_init < 1
            ):
                raise ValueError(
                    "LoLCATs config.window_weight_init must be between 0 and 1"
                )
            eps = private.get("eps", 1.0e-12)
            if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps <= 0:
                raise ValueError("LoLCATs config.eps must be positive")
        if strategy == "attn_nha":
            for name, default in (("num_slots", 16), ("window_size", 16)):
                value = private.get(name, default)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"NHA config.{name} must be a positive integer")
            for name, default in (
                ("gate_logit_normalizer", 8.0),
                ("eps", 1.0e-6),
            ):
                value = private.get(name, default)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value <= 0
                ):
                    raise ValueError(f"NHA config.{name} must be positive")
        if strategy == "attn_sparse_window":
            integer_fields = (
                "local_left",
                "local_right",
                "prefix_size",
                "prompt_suffix_size",
            )
            defaults = {
                "local_left": 127,
                "local_right": 0,
                "prefix_size": 0,
                "prompt_suffix_size": 0,
            }
            values: dict[str, int] = {}
            for name in integer_fields:
                value = private.get(name, defaults[name])
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"sparse-window config.{name} must be non-negative"
                    )
                values[name] = value
            causal = private.get("causal", True)
            include_self = private.get("include_self", True)
            if not isinstance(causal, bool) or not isinstance(include_self, bool):
                raise ValueError("sparse-window causal/include_self must be booleans")
            if causal and values["local_right"]:
                raise ValueError("causal sparse-window requires config.local_right=0")
            fixed_positions = private.get("fixed_positions", [])
            if not isinstance(fixed_positions, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in fixed_positions
            ):
                raise ValueError(
                    "sparse-window config.fixed_positions must be a list of non-negative integers"
                )
            stride = private.get("stride")
            if stride is not None and (
                isinstance(stride, bool) or not isinstance(stride, int) or stride < 1
            ):
                raise ValueError("sparse-window config.stride must be null or positive")
            if not (
                include_self
                or any(values.values())
                or fixed_positions
                or stride is not None
            ):
                raise ValueError("sparse-window pattern must select at least one key")

        schedule = config.get("schedule", {})
        targets = schedule.get("targets") if isinstance(schedule, dict) else None
        if not isinstance(targets, list) or not targets:
            raise ValueError(
                "replacement schedule.targets must contain at least one attention site"
            )
        seen: set[str] = set()
        for target in targets:
            if not isinstance(target, dict) or not isinstance(target.get("site"), str):
                raise ValueError("each replacement target must contain a string site")
            site = target["site"]
            if _ATTENTION_PATH.fullmatch(site) is None:
                raise ValueError(
                    f"replacement target {site!r} must match llm.layer[<index>].attn"
                )
            if site in seen:
                raise ValueError(f"duplicate replacement target {site!r}")
            seen.add(site)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        del state
        self.validate_config(config)
        private = dict(config["config"])
        strategy = str(private["strategy"])
        actions: list[OperationAction] = []
        for target in config["schedule"]["targets"]:
            site = str(target["site"])
            layer_index = int(_ATTENTION_PATH.fullmatch(site).group(1))  # type: ignore[union-attr]
            metadata = {
                "strategy": strategy,
                "debug": bool(private.get("debug", False)),
                "reload_contract": "reapply_config_to_original_checkpoint",
            }
            if strategy == "attn_linear_attention":
                metadata["eps"] = float(private.get("eps", 1.0e-6))
            elif strategy == "attn_cnn":
                metadata["kernel_size"] = int(private.get("kernel_size", 3))
            elif strategy == "attn_lolcats_linear_window":
                metadata.update(
                    {
                        "window_size": int(private.get("window_size", 64)),
                        "decode_window_size": int(
                            private.get(
                                "decode_window_size",
                                private.get("window_size", 64),
                            )
                        ),
                        "feature_dim": int(private.get("feature_dim", 64)),
                        "feature_map": str(
                            private.get("feature_map", "hedgehog_softmax_dim")
                        ),
                        "tie_qk_feature_maps": bool(
                            private.get("tie_qk_feature_maps", False)
                        ),
                        "window_weight_init": float(
                            private.get("window_weight_init", 0.1)
                        ),
                        "train_window_weight": bool(
                            private.get("train_window_weight", True)
                        ),
                        "affine_attention_factors": bool(
                            private.get("affine_attention_factors", False)
                        ),
                        "eps": float(private.get("eps", 1.0e-12)),
                    }
                )
            elif strategy == "attn_nha":
                metadata.update(
                    {
                        "num_slots": int(private.get("num_slots", 16)),
                        "window_size": int(private.get("window_size", 16)),
                        "gate_logit_normalizer": float(
                            private.get("gate_logit_normalizer", 8.0)
                        ),
                        "eps": float(private.get("eps", 1.0e-6)),
                    }
                )
            elif strategy == "attn_sparse_window":
                metadata.update(
                    {
                        "causal": bool(private.get("causal", True)),
                        "include_self": bool(private.get("include_self", True)),
                        "local_left": int(private.get("local_left", 127)),
                        "local_right": int(private.get("local_right", 0)),
                        "prefix_size": int(private.get("prefix_size", 0)),
                        "prompt_suffix_size": int(private.get("prompt_suffix_size", 0)),
                        "fixed_positions": list(private.get("fixed_positions", [])),
                        "stride": private.get("stride"),
                    }
                )
            actions.append(
                OperationAction(
                    action_id=f"replace_qwen3_vl_layer_{layer_index}_attention_with_{strategy}",
                    op="replace",
                    target_kind="attention",
                    target_scope="llm",
                    module_path=site,
                    apply_mode="structural_rewrite",
                    shape_effect="preserve_io_shape",
                    original_shape=None,
                    pruned_shape=None,
                    strategy_name=strategy,
                    requires_checkpoint_export=False,
                    metadata=metadata,
                )
            )
        return CompressionPlan(
            stage_name=str(context.extra.get("stage_name") or self.name),
            placement="llm_level",
            execution_phase="prepare",
            modalities=set(),
            token_roles=set(),
            method="operation_compression",
            object_family="module_unit",
            object_subtype="attention",
            operation_actions=actions,
            this_stage_metadata=StageMetadata(
                stage_name=str(context.extra.get("stage_name") or self.name),
                stats={
                    "strategy": strategy,
                    "replacement_count": len(actions),
                    "targets": [action.module_path for action in actions],
                },
            ),
        )

    @staticmethod
    def build_replacement(
        original_module: Any,
        *,
        strategy: str,
        config: Mapping[str, Any],
    ) -> Any:
        """Construct this method's module without installing it on the model.

        ``config`` is the normalized OperationAction metadata emitted by plan.
        The applier owns parent replacement and rollback. Import tensors only
        here so planning stays usable without torch.
        """
        from mllm_comp.backbones.qwen3_vl.attention_replacement import (
            Qwen3VLCausalConvAttention,
            Qwen3VLIdentityAttention,
            Qwen3VLLinearAttention,
            Qwen3VLLoLCATsLinearWindowAttention,
            Qwen3VLNativeHybridAttention,
            Qwen3VLSparseWindowAttention,
        )

        debug = bool(config.get("debug", False))

        if strategy == "attn_linear_attention":
            return Qwen3VLLinearAttention(
                original_module,
                eps=float(config.get("eps", 1.0e-6)),
                debug=debug,
            )
        if strategy == "attn_cnn":
            return Qwen3VLCausalConvAttention(
                original_module,
                kernel_size=int(config.get("kernel_size", 3)),
                debug=debug,
            )
        if strategy == "attn_identity":
            return Qwen3VLIdentityAttention(
                original_module,
                debug=debug,
            )
        if strategy == "attn_lolcats_linear_window":
            return Qwen3VLLoLCATsLinearWindowAttention(
                original_module,
                window_size=int(config.get("window_size", 64)),
                decode_window_size=int(config.get("decode_window_size", 64)),
                feature_dim=int(config.get("feature_dim", 64)),
                feature_map=str(config.get("feature_map", "hedgehog_softmax_dim")),
                tie_qk_feature_maps=bool(config.get("tie_qk_feature_maps", False)),
                window_weight_init=float(config.get("window_weight_init", 0.1)),
                train_window_weight=bool(config.get("train_window_weight", True)),
                affine_attention_factors=bool(
                    config.get("affine_attention_factors", False)
                ),
                eps=float(config.get("eps", 1.0e-12)),
                debug=debug,
            )
        if strategy == "attn_nha":
            return Qwen3VLNativeHybridAttention(
                original_module,
                num_slots=int(config.get("num_slots", 16)),
                window_size=int(config.get("window_size", 16)),
                gate_logit_normalizer=float(config.get("gate_logit_normalizer", 8.0)),
                eps=float(config.get("eps", 1.0e-6)),
                debug=debug,
            )
        if strategy == "attn_sparse_window":
            return Qwen3VLSparseWindowAttention(
                original_module,
                causal=bool(config.get("causal", True)),
                include_self=bool(config.get("include_self", True)),
                local_left=int(config.get("local_left", 127)),
                local_right=int(config.get("local_right", 0)),
                prefix_size=int(config.get("prefix_size", 0)),
                prompt_suffix_size=int(config.get("prompt_suffix_size", 0)),
                fixed_positions=list(config.get("fixed_positions", [])),
                stride=(
                    None if config.get("stride") is None else int(config["stride"])
                ),
                debug=debug,
            )
        raise BackendIntegrationError(
            f"Unknown Qwen3-VL attention replacement strategy {strategy!r}."
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("Qwen3VLAttentionReplacement has no learned planner state")


__all__ = ["Qwen3VLAttentionReplacement"]
