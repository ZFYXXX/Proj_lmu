"""Plan and construct Qwen3-VL FFN replacement methods."""

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


_FFN_PATH = re.compile(r"^llm\.layer\[(\d+)\]\.ffn$")


class Qwen3VLFFNReplacement:
    """Static shape-preserving Qwen3-VL FFN replacement method."""

    name = "qwen3_vl_ffn_replacement"
    version = "0.2.0"
    strategies = frozenset({"ffn_low_rank", "ffn_identity", "ffn_two_layer_mlp"})

    def validate_config(self, config: dict[str, Any]) -> None:
        operation = config.get("operation", {})
        if not isinstance(operation, dict) or operation.get("type") != "replace":
            raise ValueError("qwen3_vl_ffn_replacement requires operation.type=replace")
        private = config.get("config", {})
        if not isinstance(private, dict):
            raise ValueError("FFN replacement config must be a mapping")
        strategy = private.get("strategy")
        if strategy not in self.strategies:
            raise ValueError(
                f"FFN replacement config.strategy must be one of {sorted(self.strategies)}, "
                f"got {strategy!r}"
            )
        if strategy == "ffn_low_rank":
            rank = private.get("rank", 256)
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError("low-rank FFN config.rank must be a positive integer")
            initialization = private.get("initialization", "column_sampling")
            if initialization != "column_sampling":
                raise ValueError(
                    "low-rank FFN config.initialization must be 'column_sampling'"
                )
        if strategy == "ffn_two_layer_mlp":
            bottleneck_dim = private.get("bottleneck_dim", 1024)
            if (
                isinstance(bottleneck_dim, bool)
                or not isinstance(bottleneck_dim, int)
                or bottleneck_dim < 1
            ):
                raise ValueError(
                    "two-layer FFN config.bottleneck_dim must be a positive integer"
                )
            bias = private.get("bias", False)
            if not isinstance(bias, bool):
                raise ValueError("two-layer FFN config.bias must be a boolean")

        schedule = config.get("schedule", {})
        targets = schedule.get("targets") if isinstance(schedule, dict) else None
        if not isinstance(targets, list) or not targets:
            raise ValueError(
                "FFN replacement schedule.targets must contain at least one FFN site"
            )
        seen: set[str] = set()
        for target in targets:
            if not isinstance(target, dict) or not isinstance(target.get("site"), str):
                raise ValueError(
                    "each FFN replacement target must contain a string site"
                )
            site = target["site"]
            if _FFN_PATH.fullmatch(site) is None:
                raise ValueError(
                    f"FFN replacement target {site!r} must match llm.layer[<index>].ffn"
                )
            if site in seen:
                raise ValueError(f"duplicate FFN replacement target {site!r}")
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
            layer_index = int(_FFN_PATH.fullmatch(site).group(1))  # type: ignore[union-attr]
            metadata: dict[str, Any] = {
                "strategy": strategy,
                "debug": bool(private.get("debug", False)),
                "reload_contract": "reapply_config_to_original_checkpoint",
            }
            if strategy == "ffn_low_rank":
                metadata["rank"] = int(private.get("rank", 256))
                metadata["initialization"] = str(
                    private.get("initialization", "column_sampling")
                )
            elif strategy == "ffn_two_layer_mlp":
                metadata["bottleneck_dim"] = int(private.get("bottleneck_dim", 1024))
                metadata["bias"] = bool(private.get("bias", False))
            actions.append(
                OperationAction(
                    action_id=f"replace_qwen3_vl_layer_{layer_index}_ffn_with_{strategy}",
                    op="replace",
                    target_kind="ffn",
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
        stage_name = str(context.extra.get("stage_name") or self.name)
        return CompressionPlan(
            stage_name=stage_name,
            placement="llm_level",
            execution_phase="prepare",
            modalities=set(),
            token_roles=set(),
            method="operation_compression",
            object_family="module_unit",
            object_subtype="ffn",
            operation_actions=actions,
            this_stage_metadata=StageMetadata(
                stage_name=stage_name,
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
        from mllm_comp.backbones.qwen3_vl.ffn_replacement import (
            Qwen3VLIdentityMLP,
            Qwen3VLLowRankMLP,
            Qwen3VLTwoLayerMLP,
        )

        debug = bool(config.get("debug", False))

        if strategy == "ffn_low_rank":
            return Qwen3VLLowRankMLP(
                original_module,
                rank=int(config.get("rank", 256)),
                initialization=str(config.get("initialization", "column_sampling")),
                debug=debug,
            )
        if strategy == "ffn_identity":
            return Qwen3VLIdentityMLP(original_module, debug=debug)
        if strategy == "ffn_two_layer_mlp":
            return Qwen3VLTwoLayerMLP(
                original_module,
                bottleneck_dim=int(config.get("bottleneck_dim", 1024)),
                bias=bool(config.get("bias", False)),
                debug=debug,
            )
        raise BackendIntegrationError(
            f"Unknown Qwen3-VL ffn replacement strategy {strategy!r}."
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("Qwen3VLFFNReplacement has no learned planner state")


__all__ = ["Qwen3VLFFNReplacement"]
