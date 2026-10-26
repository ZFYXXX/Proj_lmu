"""Prompt-conditioned residual scorer; independent of the old checkpoint format.

MMTok/CDPruner motivate prompt conditioning, not this trainable architecture.
Uses frozen input embeddings, NOT contextual LLM states or answer tokens.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .learned_visual_topk import LearnedVisualTopK, LearnedVisualTopKSpec, continuous_topk, hard_topk_indices

FORMAT = "qwen25_context_visual_topk_v1"


@dataclass(frozen=True)
class ContextVisualTopKSpec(LearnedVisualTopKSpec):
    context_mode: str = "question"

    def __post_init__(self):
        LearnedVisualTopKSpec(self.hidden_size, self.width, self.target_count, self.seed)
        if self.context_mode not in {"question", "image"}:
            raise ValueError("context_mode must be question or image")


class ContextVisualTopK(nn.Module):
    def __init__(self, spec, *, visual=None):
        super().__init__()
        self.spec = spec
        original = LearnedVisualTopKSpec(spec.hidden_size, spec.width, spec.target_count, spec.seed)
        if visual is not None and visual.spec != original:
            raise ValueError("Warm-start specification mismatch")
        self.scorer = (visual or LearnedVisualTopK(original)).scorer
        self.context_norm = nn.LayerNorm(spec.hidden_size, elementwise_affine=False)
        # Zero branch: exactly preserves the old scorer before any new training.
        with torch.random.fork_rng(devices=[]):
            self.context_projection = nn.Linear(spec.hidden_size, spec.width, bias=False)
            nn.init.zeros_(self.context_projection.weight)

    def forward(self, features, *, question=None, soft=False):
        if features.ndim != 2 or features.shape[1] != self.spec.hidden_size or not len(features):
            raise ValueError("Expected nonempty visual [N,D] features")
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite visual features")
        context = question if self.spec.context_mode == "question" else features
        if context is None or context.ndim != 2 or context.shape[1] != self.spec.hidden_size or not len(context):
            raise ValueError("Expected nonempty context [T,D] embeddings")
        if not torch.isfinite(context).all():
            raise ValueError("Nonfinite context embeddings")
        with torch.autocast(device_type=features.device.type, enabled=False):
            h = self.scorer[1](self.scorer[0](features.detach().float()))
            pooled = self.context_norm(context.detach().float().mean(0))
            condition = torch.tanh(self.context_projection(pooled))
            scores = self.scorer[3](self.scorer[2](h + condition[None])).squeeze(-1)
            k = min(len(features), self.spec.target_count)
            indices = hard_topk_indices(scores, k)
            weights = continuous_topk(scores[None], k)[0] if soft else None
        tokens = (features * weights[:, None]).to(features.dtype) if soft else features.index_select(0, indices)
        return {"tokens": tokens, "indices": indices, "scores": scores, "soft_weights": weights,
            "hard_mask": torch.zeros_like(scores).scatter(0, indices, 1).detach(),
            "source_count": len(features), "result_count": len(tokens), "target_count": k,
            "applied_weights": weights,
            "mode": "soft_full_sequence" if soft else "hard_compact"}


def prompt_question(model, prompt):
    """Last complete user span only. Never reads teacher answer embeddings.

    Labels, if supplied by a forward caller, also bound the extraction. Training
    passes the separate answer-free prompt, not its teacher-forced sequence.
    """
    from mllm_comp.compressors.comparison.eadp import extract_qwen2_user_instruction_ids
    ids = prompt.get("input_ids")
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Question conditioning requires batch-one serialized ChatML input_ids")
    ids = ids[0].detach().clone()
    for key in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
        ids[ids == getattr(model.config, key)] = -200
    labels = prompt.get("labels")
    starts = (labels[0] != -100).nonzero(as_tuple=True)[0] if labels is not None else []
    mask = prompt.get("attention_mask")
    selected = extract_qwen2_user_instruction_ids(ids,
        attention_mask=mask[0] if mask is not None else None,
        answer_start=int(starts[0]) if len(starts) else None)
    if not len(selected):
        raise ValueError("Question has no text tokens after removing vision sentinels")
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(selected).detach()
    digest = hashlib.sha256(selected.cpu().contiguous().numpy().tobytes()).hexdigest()
    return embeddings, {"question_tokens": len(selected), "question_ids_sha256": digest,
        "question_source": "last_user_input_embeddings_no_answer_no_extra_llm_forward"}


def implementation_hash():
    from .learned_visual_topk import implementation_hash as operator_hash
    return hashlib.sha256(Path(__file__).read_bytes() + operator_hash().encode()).hexdigest()


def save_selector(module, folder, *, base_identity, provenance):
    from safetensors.torch import save_file
    from mllm_comp.core.serialization import atomic_write_json
    folder = Path(folder)
    if folder.exists():
        raise ValueError(f"Refusing checkpoint overwrite: {folder}")
    folder.mkdir(parents=True)
    weights = folder / "selector.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}, str(weights))
    atomic_write_json(folder / "selector_config.json", {"format": FORMAT, "spec": asdict(module.spec),
        "implementation_hash": implementation_hash(), "base_identity_sha256": base_identity,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "provenance": provenance})


def load_selector(folder, *, base_identity, device):
    from safetensors.torch import load_file
    folder = Path(folder)
    config = json.loads((folder / "selector_config.json").read_text())
    if config["format"] != FORMAT or config["implementation_hash"] != implementation_hash():
        raise ValueError("Context selector format/implementation mismatch")
    if config["base_identity_sha256"] != base_identity:
        raise ValueError("Context selector base identity mismatch")
    if config["weights_sha256"] != hashlib.sha256((folder / "selector.safetensors").read_bytes()).hexdigest():
        raise ValueError("Context selector weight integrity failure")
    module = ContextVisualTopK(ContextVisualTopKSpec(**config["spec"]))
    module.load_state_dict(load_file(str(folder / "selector.safetensors")), strict=True)
    return module.to(device)
