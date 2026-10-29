"""Input-conditioned strength of word/visual matching at fixed K.

Reuses the immutable word selector's matching initialization and visual prior.
The two-layer gate starts at one, making initial scores exactly the old head's
scores. It conditions ranking strength, not token count or a task-type label.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .word_visual_topk import WordVisualTopK, WordVisualTopKSpec, implementation_hash as word_hash
from .learned_visual_topk import continuous_topk, hard_topk_indices

FORMAT = "qwen25_input_adaptive_topk_v1"


@dataclass(frozen=True)
class InputAdaptiveTopKSpec(WordVisualTopKSpec):
    gate_width: int = 64

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.gate_width, bool) or not isinstance(self.gate_width, int) or self.gate_width < 1:
            raise ValueError("gate_width must be a positive integer")


class InputAdaptiveTopK(WordVisualTopK):
    def __init__(self, spec, *, visual=None):
        super().__init__(spec, visual=visual)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed + 271)
            self.input_gate = nn.Sequential(
                nn.Linear(2 * spec.match_width, spec.gate_width), nn.GELU(),
                nn.Linear(spec.gate_width, 1),
            )
            nn.init.zeros_(self.input_gate[-1].weight)
            nn.init.zeros_(self.input_gate[-1].bias)

    def forward(self, features, *, question=None, soft=False, visual_only=False):
        if features.ndim != 2 or not len(features) or features.shape[1] != self.spec.hidden_size:
            raise ValueError("Expected nonempty visual [N,D]")
        if question is None or question.ndim != 2 or not len(question) or question.shape[1] != self.spec.hidden_size:
            raise ValueError("Expected nonempty answer-free question [T,D]")
        if features.device != question.device:
            raise ValueError("Visual/question devices differ")
        if not torch.isfinite(features).all() or not torch.isfinite(question).all():
            raise ValueError("Nonfinite selector input")
        with torch.autocast(device_type=features.device.type, enabled=False):
            with torch.no_grad():
                h = self.visual.scorer[2](self.visual.scorer[1](self.visual.scorer[0](features.detach().float())))
                prior = self.visual.scorer[3](h).squeeze(-1)
            z = F.normalize(self.visual_projection(h), dim=-1)
            u = F.normalize(self.question_projection(self.question_norm(question.detach().float())), dim=-1)
            similarities = z @ u.T
            relevance = self.spec.temperature * (
                torch.logsumexp(similarities / self.spec.temperature, -1) - math.log(len(question)))
            gate = 2 * torch.sigmoid(self.input_gate(torch.cat((z.mean(0), u.mean(0))))).squeeze()
            scale = prior.std(unbiased=False).detach().clamp_min(1e-4)
            mix = F.softplus(self.mix_logit)
            residual = mix * gate * scale * (relevance - relevance.mean())
            scores = prior if visual_only else prior + residual
            k = min(len(features), self.spec.target_count)
            indices = hard_topk_indices(scores, k)
            weights = continuous_topk(scores[None], k)[0] if soft else None
        tokens = features * weights[:, None] if soft else features.index_select(0, indices)
        return {
            "tokens": tokens.to(features.dtype), "scores": scores, "indices": indices,
            "soft_weights": weights, "applied_weights": weights,
            "hard_mask": torch.zeros_like(scores).scatter(0, indices, 1).detach(),
            "source_count": len(features), "result_count": len(tokens), "target_count": k,
            "mode": "soft_full_sequence" if soft else "hard_compact",
            "word_diagnostics": {
                "mix": float(mix.detach()), "input_gate": float(gate.detach()),
                "effective_query_mix": float((mix * gate).detach()), "input_adaptive_gate": True,
                "word_tokens": len(question), "prior_score_std": float(scale),
                "query_residual_rms": float(residual.detach().square().mean().sqrt()),
                "relative_residual_rms": float(residual.detach().square().mean().sqrt() / scale),
                "visual_prior_frozen": not any(p.requires_grad for p in self.visual.parameters()),
                "explicit_task_type_prediction": False, "dynamic_token_budget": False,
            },
        }


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes() + word_hash().encode()).hexdigest()


def save_selector(module, folder, *, base_identity, provenance):
    from safetensors.torch import save_file
    from mllm_comp.core.serialization import atomic_write_json
    folder = Path(folder)
    if folder.exists():
        raise ValueError("Refusing input-adaptive checkpoint overwrite")
    folder.mkdir(parents=True)
    weights = folder / "selector.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}, str(weights))
    atomic_write_json(folder / "selector_config.json", {
        "format": FORMAT, "spec": asdict(module.spec), "implementation_hash": implementation_hash(),
        "base_identity_sha256": base_identity, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "provenance": provenance,
    })


def load_selector(folder, *, base_identity, device):
    from safetensors.torch import load_file
    folder = Path(folder)
    cfg = json.loads((folder / "selector_config.json").read_text())
    if cfg["format"] != FORMAT or cfg["implementation_hash"] != implementation_hash():
        raise ValueError("Input-adaptive selector format/source changed")
    if cfg["base_identity_sha256"] != base_identity:
        raise ValueError("Input-adaptive selector base changed")
    if cfg["weights_sha256"] != hashlib.sha256((folder / "selector.safetensors").read_bytes()).hexdigest():
        raise ValueError("Input-adaptive selector weight integrity failure")
    module = InputAdaptiveTopK(InputAdaptiveTopKSpec(**cfg["spec"]))
    module.load_state_dict(load_file(str(folder / "selector.safetensors")), strict=True)
    return module.to(device)
