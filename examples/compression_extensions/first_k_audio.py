"""An importable research method reusing the standard audio projector mechanism."""

from typing import Any

from mllm_comp.compressors.audio_rule_utils import AudioTokenStrategyBase
from mllm_comp.compressors.contracts import CompressionContractRegistry
from mllm_comp.compressors.extensions import register_method


ALGORITHM = "researcher_first_k_audio"


class FirstKStrategy(AudioTokenStrategyBase):
    """Keep the first K items using the existing exact audio-token budget."""

    attention_layer_offset = None  # This selection does not need attention capture.

    def runtime_metadata(self) -> dict[str, Any]:
        return {"selection": "chronological_first_k"}

    def compress(
        self,
        hidden_states: Any,
        *,
        scores: Any | None,
        keep_count: int,
        item_index: int,
    ) -> tuple[Any, Any, dict[str, Any]]:
        del scores, item_index
        import torch

        indices = torch.arange(
            keep_count, dtype=torch.long, device=hidden_states.device
        )
        return hidden_states.index_select(0, indices), indices, {}


def register(registry: CompressionContractRegistry) -> None:
    register_method(
        registry,
        algorithm=ALGORITHM,
        reuse_bindings=["segmentwise_topk_audio.audio.qwen25_omni"],
        strategy_factory=FirstKStrategy,
        config_defaults={"num_segments": 1, "score_source": "uniform"},
    )
