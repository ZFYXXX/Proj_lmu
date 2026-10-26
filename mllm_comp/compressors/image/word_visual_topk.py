"""Learned token-specific question residual over a FROZEN visual prior.

Not OT, a contextual question encoder, or a reproduction of MMTok. Keep raw
question words separate; never infer semantic evidence labels from this head.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .learned_visual_topk import (
    LearnedVisualTopK, LearnedVisualTopKSpec, continuous_topk, hard_topk_indices,
    implementation_hash as operator_hash,
)

FORMAT = "qwen25_word_visual_topk_v1"
FORMAT_SUFFIXES = (
    "Answer the question using a single word or phrase.",
    "Answer the question with a single word.",
    "Answer with the option's letter from the given choices directly.",
)


def template_patterns(tokenizer):
    """Only exact known suffixes; no noun/stopword filter or answer/options cut."""
    return tuple(sorted({tuple(tokenizer.encode(prefix+text, add_special_tokens=False))
        for text in FORMAT_SUFFIXES for prefix in ("", " ", "\n")}, key=lambda x:(-len(x),x)))


def clean_ids(ids, patterns):
    result=list(ids)
    while result and result[-1]==198:result.pop()
    for pattern in sorted(patterns,key=lambda x:-len(x)):
        if len(result)>=len(pattern) and tuple(result[-len(pattern):])==tuple(pattern):
            result=result[:-len(pattern)]
            break
    while result and result[-1]==198:result.pop()
    if not result:raise ValueError("Question became empty after exact suffix cleaning")
    return result


def prompt_word_question(model, prompt, patterns):
    from mllm_comp.compressors.comparison.eadp import extract_qwen2_user_instruction_ids
    from mllm_comp.analysis.common.selector_conditioning import ids_sha
    ids=prompt.get("input_ids")
    if ids is None or ids.ndim!=2 or ids.shape[0]!=1:
        raise ValueError("Word selector needs one answer-free serialized user prompt")
    values=ids[0].detach().clone()
    for name in ("image_token_id","vision_start_token_id","vision_end_token_id"):
        values[values==getattr(model.config,name)]=-200
    labels=prompt.get("labels")
    starts=(labels[0]!=-100).nonzero(as_tuple=True)[0] if labels is not None else []
    mask=prompt.get("attention_mask")
    raw=extract_qwen2_user_instruction_ids(values,attention_mask=mask[0] if mask is not None else None,
        answer_start=int(starts[0]) if len(starts) else None).tolist()
    selected=clean_ids(raw,patterns)
    with torch.no_grad():
        embeddings=model.get_input_embeddings()(torch.tensor(selected,device=ids.device,dtype=torch.long)).detach()
    return embeddings, {"question_tokens":len(selected),"question_raw_tokens":len(raw),
        "question_ids_sha256":ids_sha(selected),"raw_question_ids_sha256":ids_sha(raw),
        "removed_format_tokens":len(raw)-len(selected),
        "question_source":"last_user_raw_word_embeddings_exact_format_suffix_removed_no_answer"}


@dataclass(frozen=True)
class WordVisualTopKSpec:
    hidden_size:int
    width:int=256
    target_count:int=128
    seed:int=42
    match_width:int=128
    temperature:float=.1
    initial_mix:float=1.
    template_token_ids:tuple=()

    def __post_init__(self):
        LearnedVisualTopKSpec(self.hidden_size,self.width,self.target_count,self.seed)
        if isinstance(self.match_width,bool) or not isinstance(self.match_width,int) or self.match_width<1:
            raise ValueError("Invalid word matching width")
        if any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in (self.temperature,self.initial_mix)):
            raise ValueError("Positive finite temperature/mix required")
        patterns=tuple(tuple(p) for p in self.template_token_ids)
        if any(not p or any(isinstance(i,bool) or not isinstance(i,int) or i<0 for i in p) for p in patterns):
            raise ValueError("Invalid exact formatting suffix IDs")
        object.__setattr__(self,"template_token_ids",patterns)


class WordVisualTopK(nn.Module):
    def __init__(self,spec,*,visual=None):
        super().__init__()
        self.spec=spec
        expected=LearnedVisualTopKSpec(spec.hidden_size,spec.width,spec.target_count,spec.seed)
        if visual is not None and visual.spec!=expected:raise ValueError("Wrong visual prior specification")
        self.visual=visual or LearnedVisualTopK(expected)
        self.question_norm=nn.LayerNorm(spec.hidden_size,elementwise_affine=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed+173)
            self.visual_projection=nn.Linear(spec.width,spec.match_width,bias=False)
            self.question_projection=nn.Linear(spec.hidden_size,spec.match_width,bias=False)
        self.mix_logit=nn.Parameter(torch.tensor(math.log(math.expm1(spec.initial_mix))))
        self.freeze_prior()

    def freeze_prior(self):
        self.visual.requires_grad_(False).eval()

    def train(self,mode=True):
        super().train(mode)
        self.freeze_prior()
        return self

    def forward(self,features,*,question=None,soft=False,visual_only=False):
        if features.ndim!=2 or not len(features) or features.shape[1]!=self.spec.hidden_size:
            raise ValueError("Expected visual [N,D]")
        if question is None or question.ndim!=2 or not len(question) or question.shape[1]!=self.spec.hidden_size:
            raise ValueError("Expected answer-free word embeddings [T,D]")
        if not torch.isfinite(features).all() or not torch.isfinite(question).all():raise ValueError("Nonfinite selector input")
        with torch.autocast(device_type=features.device.type,enabled=False):
            with torch.no_grad():
                h=self.visual.scorer[2](self.visual.scorer[1](self.visual.scorer[0](features.detach().float())))
                prior=self.visual.scorer[3](h).squeeze(-1)
            z=F.normalize(self.visual_projection(h),dim=-1)
            u=F.normalize(self.question_projection(self.question_norm(question.detach().float())),dim=-1)
            similarities=z@u.T
            # Log-MEAN-exp: repeating the same words does not change relevance.
            relevance=self.spec.temperature*(torch.logsumexp(similarities/self.spec.temperature,-1)-math.log(len(question)))
            centered=relevance-relevance.mean()
            # Match the prior's units, NOT per-query z-scoring/min-max amplification.
            scale=prior.std(unbiased=False).detach().clamp_min(1e-4)
            mix=F.softplus(self.mix_logit)
            residual=mix*scale*centered
            scores=prior if visual_only else prior+residual
            k=min(len(features),self.spec.target_count)
            indices=hard_topk_indices(scores,k)
            weights=continuous_topk(scores[None],k)[0] if soft else None
        tokens=(features*weights[:,None]).to(features.dtype) if soft else features.index_select(0,indices)
        return {"tokens":tokens,"scores":scores,"indices":indices,"soft_weights":weights,
            "hard_mask":torch.zeros_like(scores).scatter(0,indices,1).detach(),
            "source_count":len(features),"result_count":len(tokens),"target_count":k,
            "applied_weights":weights,"mode":"soft_full_sequence" if soft else "hard_compact",
            "word_diagnostics":{"mix":float(mix.detach()),"word_tokens":len(question),
                "prior_score_std":float(scale),"query_residual_rms":float(residual.detach().square().mean().sqrt()),
                "relative_residual_rms":float(residual.detach().square().mean().sqrt()/scale),
                "visual_prior_frozen":not any(p.requires_grad for p in self.visual.parameters())}}


def evidence_loss(scores,positive,negative):
    """Optional verified-region ranking supervision; not automatic pseudo-GT."""
    positive=torch.as_tensor(positive,device=scores.device,dtype=torch.bool)
    negative=torch.as_tensor(negative,device=scores.device,dtype=torch.bool)
    if positive.shape!=scores.shape or negative.shape!=scores.shape or (positive&negative).any() or not positive.any() or not negative.any():
        raise ValueError("Evidence supervision needs nonempty disjoint verified regions")
    return F.softplus(-(scores[positive].mean()-scores[negative].mean()))


def evidence_masks(annotation,*,height,width,image_fingerprint):
    """Explicit verified positive/negative boxes, not attention-derived labels."""
    if annotation is None:return None
    from mllm_comp.analysis.common.selector_mechanisms import evidence_coverage
    masks=[]
    for field in ("positive_boxes","negative_boxes"):
        evidence_coverage({**annotation,"groups":[{"boxes":annotation.get(field,[])}]},[],
            height=height,width=width,image_fingerprint=image_fingerprint)
        members=[]
        for r in range(height):
            for c in range(width):
                members.append(any(min((c+1)/width,b[2])>max(c/width,b[0]) and
                    min((r+1)/height,b[3])>max(r/height,b[1]) for b in annotation[field]))
        masks.append(members)
    if any(p and n for p,n in zip(*masks)):raise ValueError("Positive/negative evidence shares native tokens")
    return masks


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+operator_hash().encode()).hexdigest()


def save_selector(module,folder,*,base_identity,provenance):
    from safetensors.torch import save_file
    from mllm_comp.core.serialization import atomic_write_json
    folder=Path(folder)
    if folder.exists():raise ValueError("Refusing word-selector checkpoint overwrite")
    folder.mkdir(parents=True)
    path=folder/"selector.safetensors"
    save_file({k:v.detach().cpu().contiguous() for k,v in module.state_dict().items()},str(path))
    atomic_write_json(folder/"selector_config.json",{"format":FORMAT,"spec":asdict(module.spec),
        "implementation_hash":implementation_hash(),"base_identity_sha256":base_identity,
        "weights_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"provenance":provenance})


def load_selector(folder,*,base_identity,device):
    from safetensors.torch import load_file
    folder=Path(folder);cfg=json.loads((folder/"selector_config.json").read_text())
    if cfg["format"]!=FORMAT or cfg["implementation_hash"]!=implementation_hash():raise ValueError("Word selector format/source changed")
    if cfg["base_identity_sha256"]!=base_identity or cfg["weights_sha256"]!=hashlib.sha256((folder/"selector.safetensors").read_bytes()).hexdigest():
        raise ValueError("Word selector base/weights changed")
    module=WordVisualTopK(WordVisualTopKSpec(**cfg["spec"]))
    module.load_state_dict(load_file(str(folder/"selector.safetensors")),strict=True)
    return module.to(device)
