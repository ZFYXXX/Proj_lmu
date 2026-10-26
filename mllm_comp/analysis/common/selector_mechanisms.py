"""Mechanism diagnostics, not semantic-grounding labels or a new compressor."""
import hashlib
import math
import random

import torch

VARIANTS = ('observe', 'restore_tt', 'restore_tc', 'restore_ct', 'restore_cc',
    'remove_tt', 'remove_tc', 'remove_ct', 'remove_cc')


def ranking_stats(true, other, k, *, boundary_band=16):
    from mllm_comp.compressors.image.learned_visual_topk import hard_topk_indices
    if true.ndim != 1 or true.shape != other.shape or not torch.isfinite(true).all() or not torch.isfinite(other).all():
        raise ValueError('Finite aligned score vectors required')
    a, b = true.detach().double(), other.detach().double()
    n = len(a)
    ta, tb = hard_topk_indices(a, k), hard_topk_indices(b, k)
    delta = b-a
    common = delta.mean()
    centered = delta-common
    total = float(delta.square().sum())
    common_energy, centered_energy = float(n*common.square()), float(centered.square().sum())
    order_a, order_b = a.argsort(descending=True, stable=True), b.argsort(descending=True, stable=True)
    rank_a, rank_b = torch.empty_like(order_a), torch.empty_like(order_b)
    rank_a[order_a], rank_b[order_b] = torch.arange(n,device=a.device), torch.arange(n,device=a.device)
    lost = ta[~torch.isin(ta,tb)]
    gained = tb[~torch.isin(tb,ta)]
    crossings = torch.cat((lost,gained))
    near = (rank_a[crossings] >= max(0,k-boundary_band)) & (rank_a[crossings] < min(n,k+boundary_band))
    # scipy's tie-aware ranks: stable tie-break is for selection serialization,
    # not evidence that equal scores have a meaningful ordinal difference.
    from scipy.stats import spearmanr
    correlation = None if float(a.std(unbiased=False)*b.std(unbiased=False)) == 0 else float(spearmanr(a.cpu().numpy(),b.cpu().numpy()).statistic)
    return {'score_change_mean':float(common), 'score_change_rms':math.sqrt(total/n),
        'common_shift_energy_fraction':common_energy/total if total else 0.,
        'relative_score_change_rms':math.sqrt(centered_energy/n),
        'score_energy_closure_relative':abs(total-common_energy-centered_energy)/max(total,1e-30),
        'rank_spearman':correlation, 'mean_absolute_rank_shift':float((rank_a-rank_b).abs().double().mean()),
        'true_k_boundary_gap':float(a[order_a[k-1]]-a[order_a[k]]) if k<n else None,
        'other_k_boundary_gap':float(b[order_b[k-1]]-b[order_b[k]]) if k<n else None,
        'lost_indices':lost.tolist(), 'gained_indices':gained.tolist(), 'swapped_token_count':len(lost),
        'boundary_band':boundary_band,
        'crossings_in_true_boundary_band_fraction':float(near.double().mean()) if len(crossings) else None}


def condition_path_stats(selector, features, question, other_question, true_scores, other_scores):
    """Measure the actual shared-condition -> GELU -> score path.

    First-order prediction is a diagnostic, NOT an exact error decomposition.
    Large residual is allowed; only the independently recomputed path must agree.
    """
    with torch.no_grad(), torch.autocast(device_type=features.device.type,enabled=False):
        h=selector.scorer[1](selector.scorer[0](features.float()))
        c= torch.tanh(selector.context_projection(selector.context_norm(question.float().mean(0))))
        d= torch.tanh(selector.context_projection(selector.context_norm(other_question.float().mean(0))))
        activation=selector.scorer[2]
        if not isinstance(activation,torch.nn.GELU) or activation.approximate!='none':
            raise ValueError('Diagnostic expects the unchanged exact-GELU context scorer')
        z=h+c
        actual_a=selector.scorer[3](activation(z)).flatten()
        actual_b=selector.scorer[3](activation(h+d)).flatten()
        torch.testing.assert_close(actual_a,true_scores.float(),rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(actual_b,other_scores.float(),rtol=1e-5,atol=1e-7)
        derivative=.5*(1+torch.erf(z/math.sqrt(2)))+z*torch.exp(-.5*z.square())/math.sqrt(2*math.pi)
        linearized=(derivative*(d-c)*selector.scorer[3].weight[0]).sum(-1)
        delta=actual_b-actual_a
        rms=float(delta.square().mean().sqrt())
        residual=float((delta-linearized).square().mean().sqrt())
        return {'context_vector_delta_l2':float((d-c).norm()),
            'true_context_vector_l2':float(c.norm()), 'other_context_vector_l2':float(d.norm()),
            'gelu_linearization_residual_rms':residual,
            'gelu_linearization_relative_residual':residual/rms if rms>1e-12 else None,
            'gelu_linearization_is_exact_identity':False}


def spatial_counts(indices,height,width,bins=8):
    result=[0]*(bins*bins)
    for i in indices:
        if not 0<=int(i)<height*width:
            raise ValueError('Spatial index outside native post-merger grid')
        r,c=divmod(int(i),width)
        result[min(bins-1,r*bins//height)*bins+min(bins-1,c*bins//width)]+=1
    return result


def plan_swaps(true_indices,other_indices,true_scores,other_scores,features,*,height,width,key,max_swap=8):
    """Fixed-K 2x2 add/remove candidates and matched controls, both directions.

    Target groups are scorer-affected candidates, NOT verified answer evidence.
    Controls are disjoint members of the same lost/gained populations, matched
    by quadrant and feature norm where possible, with deterministic random tie.
    """
    a,b=set(map(int,true_indices)),set(map(int,other_indices))
    if len(a)!=len(true_indices) or len(b)!=len(other_indices) or len(a)!=len(b) or len(features)!=height*width:
        raise ValueError('Swap plan requires equal unique budgets and aligned rectangular grid')
    lost,gained=sorted(a-b),sorted(b-a)
    result={'eligible':len(lost)>=2,'candidate_is_verified_evidence':False,'lost_count':len(lost),
        'target_lost':[],'control_lost':[],'target_gained':[],'control_gained':[],
        'swap_count':0,'matching_relaxations':0,'states':{'observe':sorted(b)}}
    if not result['eligible']:
        result['inapplicable_reason']='fewer_than_two_exchanged_tokens_for_disjoint_controls'
        for v in VARIANTS[1:]:result['states'][v]=sorted(b if v.startswith('restore') else a)
        return result
    m=min(max_swap,max(1,len(lost)//4))
    delta=(true_scores-other_scores).detach().float().cpu().tolist()
    target_lost=sorted(lost,key=lambda i:(-delta[i],i))[:m]
    target_gained=sorted(gained,key=lambda i:(delta[i],i))[:m]
    norms=features.detach().float().norm(dim=-1).cpu().tolist()
    rng=random.Random(hashlib.sha256(f'{key}:swap-controls-v1'.encode()).hexdigest())
    relaxations=0
    def quadrant(i):
        r,c=divmod(i,width)
        return (int(2*r>=height),int(2*c>=width))
    def controls(target,pool):
        nonlocal relaxations
        available=set(pool)-set(target)
        selected=[]
        for token in target:
            local=[i for i in available if quadrant(i)==quadrant(token)]
            if not local:local=sorted(available);relaxations+=1
            ranked=sorted(local,key=lambda i:(abs(norms[i]-norms[token]),i))[:8]
            chosen=rng.choice(ranked)
            selected.append(chosen);available.remove(chosen)
        return selected
    control_lost=controls(target_lost,lost)
    control_gained=controls(target_gained,gained)
    for left,name_l in (('t',target_lost),('c',control_lost)):
        for right,name_g in (('t',target_gained),('c',control_gained)):
            result['states'][f'restore_{left}{right}']=sorted((b-set(name_g))|set(name_l))
            result['states'][f'remove_{left}{right}']=sorted((a-set(name_l))|set(name_g))
    if any(len(v)!=len(a) or len(set(v))!=len(a) for v in result['states'].values()):
        raise ValueError('A causal intervention changed the total token budget')
    result.update(swap_count=m,target_lost=target_lost,control_lost=control_lost,
        target_gained=target_gained,control_gained=control_gained,matching_relaxations=relaxations)
    return result


def evidence_coverage(annotation,selected,*,height,width,image_fingerprint):
    """Unknown is unknown. Never convert saliency or success into ground truth."""
    if annotation is None:
        return {'status':'pending_review','verified':False,'groups':None}
    if annotation.get('status')!='verified' or annotation.get('source') not in ('human','dataset_ground_truth') or not annotation.get('reviewer'):
        raise ValueError('Evidence annotations require verified human/dataset provenance')
    if annotation.get('image_fingerprint')!=image_fingerprint or annotation.get('coordinate_space')!='normalized_original_image':
        raise ValueError('Evidence image/coordinate provenance mismatch')
    groups=[]
    for group in annotation.get('groups',[]):
        members=set()
        for box in group.get('boxes',[]):
            if len(box)!=4 or not all(math.isfinite(float(v)) for v in box):raise ValueError('Invalid evidence box')
            x0,y0,x1,y1=map(float,box)
            if not (0<=x0<x1<=1 and 0<=y0<y1<=1):raise ValueError('Evidence boxes must be normalized and nonempty')
            for r in range(height):
                for c in range(width):
                    if min((c+1)/width,x1)>max(c/width,x0) and min((r+1)/height,y1)>max(r/height,y0):
                        members.add(r*width+c)
        if not members:raise ValueError('Verified evidence group has no valid boxes')
        kept=members&set(selected)
        groups.append({'name':group.get('name','unnamed'),'source_token_count':len(members),
            'retained_token_count':len(kept),'retained_fraction':len(kept)/len(members),'any_retained':bool(kept)})
    if not groups:raise ValueError('Verified evidence annotation has no groups')
    return {'status':'verified','verified':True,'groups':groups,
        'all_groups_have_a_retained_token':all(g['any_retained'] for g in groups),
        'warning':'Geometric token coverage is not a guarantee of readable or sufficient evidence'}
