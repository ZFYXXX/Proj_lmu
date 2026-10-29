"""Contextual question-to-visual scoring, with independent checkpoint identity."""
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

FORMAT = 'qwen25_query_interaction_topk_v1'


@dataclass(frozen=True)
class QueryInteractionSpec(WordVisualTopKSpec):
    interaction: str = 'contextual'
    heads: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.interaction not in ('word', 'contextual'):
            raise ValueError('Unknown interaction')
        if not isinstance(self.heads, int) or self.heads < 1 or self.match_width % self.heads:
            raise ValueError('Matching width must be divisible by heads')


def positions(length, width, device):
    steps = torch.arange(length, device=device).float()[:, None]
    freq = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000.) / width))
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = (steps * freq).sin()
    result[:, 1::2] = (steps * freq[:width // 2]).cos()
    return result


def attention(q, k, v, heads):
    n, width = q.shape
    m = k.shape[0]
    q = q.reshape(n, heads, width // heads).transpose(0, 1)
    k = k.reshape(m, heads, width // heads).transpose(0, 1)
    v = v.reshape(m, heads, width // heads).transpose(0, 1)
    return ((q @ k.transpose(-1, -2) / math.sqrt(width // heads)).softmax(-1) @ v).transpose(0, 1).reshape(n, width)


class QueryInteractionTopK(WordVisualTopK):
    def __init__(self, spec, *, visual=None):
        super().__init__(spec, visual=visual)
        if spec.interaction == 'contextual':
            w = spec.match_width
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(spec.seed + 397)
                self.text_norm = nn.LayerNorm(w)
                self.text_qkv = nn.Linear(w, 3*w, bias=False)
                self.text_out = nn.Linear(w, w, bias=False)
                self.text_ff = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 2*w), nn.GELU(), nn.Linear(2*w, w))
                self.cross_q = nn.Linear(w, w, bias=False)
                self.cross_k = nn.Linear(w, w, bias=False)
                self.cross_v = nn.Linear(w, w, bias=False)
                self.token_head = nn.Sequential(nn.LayerNorm(2*w), nn.Linear(2*w, w), nn.GELU(), nn.Linear(w, 1, bias=False))

    def forward(self, features, *, question=None, soft=False, visual_only=False):
        if self.spec.interaction == 'word':
            out = super().forward(features, question=question, soft=soft, visual_only=visual_only)
            out['word_diagnostics'].update(contextual_question_encoder=False, global_scalar_gate=False)
            return out
        if features.ndim != 2 or not len(features) or features.shape[1] != self.spec.hidden_size:
            raise ValueError('Expected nonempty visual [N,D]')
        if question is None or question.ndim != 2 or not len(question) or question.shape[1] != self.spec.hidden_size:
            raise ValueError('Expected nonempty answer-free question [T,D]')
        if features.device != question.device or not torch.isfinite(features).all() or not torch.isfinite(question).all():
            raise ValueError('Nonfinite or mismatched selector input')
        with torch.autocast(device_type=features.device.type, enabled=False):
            with torch.no_grad():
                h = self.visual.scorer[2](self.visual.scorer[1](self.visual.scorer[0](features.detach().float())))
                prior = self.visual.scorer[3](h).squeeze(-1)
            z = self.visual_projection(h)
            u = self.question_projection(self.question_norm(question.detach().float()))
            u = u + positions(len(u), self.spec.match_width, u.device)
            q, k, v = self.text_qkv(self.text_norm(u)).chunk(3, -1)
            u = u + self.text_out(attention(q, k, v, self.spec.heads))
            u = u + self.text_ff(u)
            context = attention(self.cross_q(z), self.cross_k(u), self.cross_v(u), self.spec.heads)
            relevance = self.token_head(torch.cat((z, context), -1)).squeeze(-1)
            scale = prior.std(unbiased=False).detach().clamp_min(1e-4)
            residual = F.softplus(self.mix_logit) * scale * (relevance - relevance.mean())
            scores = prior if visual_only else prior + residual
            k = min(len(features), self.spec.target_count)
            indices = hard_topk_indices(scores, k)
            weights = continuous_topk(scores[None], k)[0] if soft else None
        tokens = features * weights[:, None] if soft else features.index_select(0, indices)
        return {'tokens': tokens.to(features.dtype), 'scores': scores, 'indices': indices,
                'soft_weights': weights, 'applied_weights': weights,
                'hard_mask': torch.zeros_like(scores).scatter(0, indices, 1).detach(),
                'source_count': len(features), 'result_count': len(tokens), 'target_count': k,
                'mode': 'soft_full_sequence' if soft else 'hard_compact',
                'word_diagnostics': {'contextual_question_encoder': True, 'per_token_cross_attention': True,
                    'global_scalar_gate': False, 'word_matching': False, 'word_tokens': len(question),
                    'visual_prior_frozen': True, 'query_residual_rms': float(residual.detach().square().mean().sqrt()),
                    'relative_residual_rms': float(residual.detach().square().mean().sqrt()/scale)}}


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes() + word_hash().encode()).hexdigest()


def save_selector(module, folder, *, base_identity, provenance):
    from safetensors.torch import save_file
    from mllm_comp.core.serialization import atomic_write_json
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    weights = folder / 'selector.safetensors'
    save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}, str(weights))
    atomic_write_json(folder / 'selector_config.json', {'format': FORMAT, 'spec': asdict(module.spec),
        'implementation_hash': implementation_hash(), 'base_identity_sha256': base_identity,
        'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(), 'provenance': provenance})


def load_selector(folder, *, base_identity, device):
    from safetensors.torch import load_file
    folder = Path(folder)
    cfg = json.loads((folder / 'selector_config.json').read_text())
    if cfg['format'] != FORMAT or cfg['implementation_hash'] != implementation_hash():
        raise ValueError('Query-interaction source/format changed')
    if cfg['base_identity_sha256'] != base_identity or cfg['weights_sha256'] != hashlib.sha256((folder / 'selector.safetensors').read_bytes()).hexdigest():
        raise ValueError('Query-interaction base/weight integrity changed')
    module = QueryInteractionTopK(QueryInteractionSpec(**cfg['spec']))
    module.load_state_dict(load_file(str(folder / 'selector.safetensors')), strict=True)
    return module.to(device)
