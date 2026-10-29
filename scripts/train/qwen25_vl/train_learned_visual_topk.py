"""Frozen 7B, full-sequence soft-mask training. Single GPU or torchrun (2 ranks).

Global batch/order/320-step schedule are invariant to world size. World size is
part of the resume contract: changing it requires a new experiment.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import load_runtime, file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    available_gpus, batch_indices, cooperative_pause, dependency_identity, digest_json, lr_multiplier, owned_output, shared_policy,
    restore_training_state, rng_state, save_training_state, source_identity, training_loss,
)
from mllm_comp.compressors.image.learned_visual_topk import LearnedVisualTopK, LearnedVisualTopKSpec
from mllm_comp.analysis.common.topk_training_modes import (
    TRAINING_MODES, parameter_digest, selection_diagnostics, training_selection,
)
from mllm_comp.core.serialization import atomic_write_json
from scripts.train.qwen25_vl.train_learned_hybrid import load_examples, prepare_example


def now():
    return datetime.now(timezone.utc).isoformat()


def main(args):
    import numpy as np
    import torch
    import torch.distributed as dist
    from mllm_comp.analysis.common.learned_hybrid_runtime import answer_logits
    from mllm_comp.compressors.comparison.learned_topk import QwenLearnedTopKComparison
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    contextual = protocol.get('selector_type') == 'context_visual_topk'
    input_adaptive = protocol.get('selector_type') == 'input_adaptive_topk'
    query_interaction = protocol.get('selector_type') == 'query_interaction_topk'
    wordwise = protocol.get('selector_type') == 'word_visual_topk' or input_adaptive or query_interaction
    save_fn = load_fn = None
    binding = QwenLearnedTopKComparison
    if contextual:
        from mllm_comp.compressors.image.context_visual_topk import (
            ContextVisualTopK, ContextVisualTopKSpec, prompt_question,
            save_selector as save_fn, load_selector as load_fn,
        )
        from mllm_comp.compressors.comparison.context_topk import QwenContextTopKComparison
        binding = QwenContextTopKComparison
    if wordwise:
        from mllm_comp.compressors.image.word_visual_topk import (
            WordVisualTopK,WordVisualTopKSpec,prompt_word_question,evidence_masks,evidence_loss,
            save_selector as save_fn,load_selector as load_fn,
        )
        from mllm_comp.compressors.comparison.word_topk import QwenWordTopKComparison
        binding = QwenWordTopKComparison
    if input_adaptive:
        from mllm_comp.compressors.image.input_adaptive_topk import (
            InputAdaptiveTopK as WordVisualTopK, InputAdaptiveTopKSpec as WordVisualTopKSpec,
            save_selector as save_fn, load_selector as load_fn,
        )
        from mllm_comp.compressors.comparison.input_adaptive_topk import QwenInputAdaptiveTopKComparison
        binding = QwenInputAdaptiveTopKComparison
    if query_interaction:
        from mllm_comp.compressors.image.query_interaction_topk import (
            QueryInteractionTopK as WordVisualTopK, QueryInteractionSpec as WordVisualTopKSpec,
            save_selector as save_fn, load_selector as load_fn,
        )
        from mllm_comp.compressors.comparison.query_interaction_topk import QwenQueryInteractionTopK
        from mllm_comp.analysis.common.task_loss_supervision import load_target, teacher_ranking_loss
        binding = QwenQueryInteractionTopK
    mask_mode = protocol['training'].get('training_mask_mode', 'soft')
    if mask_mode not in TRAINING_MODES:
        raise ValueError('Unknown frozen training mask mode')
    source = source_identity(ROOT)
    if source != protocol['source']:
        raise ValueError('Run from the exact prepared execution worktree')
    if dependency_identity() != protocol['dependencies']:
        raise ValueError('Software/evaluator source changed after experiment preparation')
    for filename, expected in protocol['input_hashes'].items():
        if file_sha(filename) != expected:
            raise ValueError(f'Frozen experiment input changed: {filename}')
    identity = json.loads(Path(protocol['identity_path']).read_text())
    validate_identity(identity)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if world not in (1, 2) or not 0 < args.stop_after <= 320:
        raise ValueError('Expected 1/2 ranks and stop-after in 1..320')
    visible = [int(i) for i in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
    if len(visible) < world or available_gpus(visible[:world], allow_own=True) != visible[:world]:
        raise RuntimeError('GPU resource gate: another user is computing on a requested GPU')
    device = torch.device(f'cuda:{local}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(device)
    policy = shared_policy()
    if policy != protocol.get('resource_policy'):
        raise ValueError('Resource policy differs from the frozen run')
    if policy:
        torch.cuda.set_per_process_memory_fraction(policy['allocator_fraction'], device)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    output = owned_output(args.output)
    contract = {**protocol['training'], 'resource_policy': policy, 'protocol_sha256': file_sha(protocol_path),
                'source': source, 'world_size': world, 'base_identity_sha256': identity['identity_sha256']}
    manifest = {'status': 'running', 'started_utc': now(), 'contract': contract,
                'contract_hash': digest_json(contract), 'steps_completed': 0, 'gpu_ids': visible[:world]}
    if rank == 0:
        if output.exists() and not args.resume:
            raise ValueError('Training directory already exists; use a new run or explicit resume')
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / 'manifest.json', manifest)
    if world > 1:
        dist.barrier()
    step = 0
    try:
        random.seed(42 + rank)
        np.random.seed(42 + rank)
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)
        examples = load_examples(Path(protocol['training_manifest']))
        bundle = load_runtime(identity, device)
        if any(p.requires_grad for p in bundle['model'].parameters()):
            raise ValueError('Backbone must be entirely frozen before training')
        manifest.update(frozen_parameters=sum(p.numel() for p in bundle['model'].parameters()),
                        base_identity_sha256=identity['identity_sha256'], revision=identity['revision'])
        llm = bundle['llm']
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if contextual or wordwise:
            from mllm_comp.compressors.image.learned_visual_topk import load_selector as load_visual
            visual = load_visual(protocol['warmstart_checkpoint'], base_identity=identity['identity_sha256'], device='cpu')
            if wordwise:
                selector=WordVisualTopK(WordVisualTopKSpec(**protocol['word_spec']),visual=visual).to(device)
                frozen_prior_digest=parameter_digest(selector.visual)
                evidence_path=protocol.get('evidence_training_manifest')
                evidence_rows=json.loads(Path(evidence_path).read_text())['rows'] if evidence_path else {}
                if set(evidence_rows)-{e['sample_id'] for e in examples}:raise ValueError('Unknown supervised training sample')
                manifest.update(visual_prior_frozen=True,visual_prior_sha256=frozen_prior_digest,
                    verified_evidence_training_images=len(evidence_rows),evidence_loss_active=bool(evidence_rows))
                task_index = None
                if query_interaction and protocol.get('task_teacher_index'):
                    task_index = json.loads(Path(protocol['task_teacher_index']).read_text())
                    if set(task_index['rows']) != {e['sample_id'] for e in examples}:
                        raise ValueError('Task teacher must cover the exact frozen training manifest')
                    manifest['task_teacher_images'] = len(task_index['rows'])
            else:
                selector = ContextVisualTopK(ContextVisualTopKSpec(llm.config.hidden_size,
                    context_mode=protocol['training']['context_mode']), visual=visual).to(device)
                manifest.update(context_mode=selector.spec.context_mode,
                    context_additional_parameters=selector.context_projection.weight.numel())
            manifest['warmstart_checkpoint']=protocol['warmstart_checkpoint']
        else:
            selector = LearnedVisualTopK(LearnedVisualTopKSpec(llm.config.hidden_size)).to(device)
        manifest.update(training_mask_mode=mask_mode,
            initial_parameter_sha256=parameter_digest(selector),
            training_physical_compression=False,
            backward_estimator='VisionSelector continuous sigmoid-threshold Jacobian')
        if protocol.get('initial_parameter_sha256', manifest['initial_parameter_sha256']) != manifest['initial_parameter_sha256']:
            raise ValueError('Scorer initialization differs from the matched-arm contract')
        trainable=[p for p in selector.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        if args.resume:
            pointer = json.loads((output / 'latest.json').read_text())
            step = restore_training_state(output / pointer['checkpoint'], selector=selector,
                optimizer=optimizer, scheduler=scheduler, contract=contract, rank=rank, device=device, load_fn=load_fn)
        if step >= args.stop_after:
            raise ValueError('Requested stop point is not after the resumed checkpoint')
        if not args.resume:
            # Real-checkpoint gates: identity logits and hard cache lifecycle.
            first = prepare_example(bundle, examples[batch_indices(0, world_size=world, rank=rank)[0]])
            bundle['model'].eval()
            with torch.inference_mode():
                expected = bundle['model'](**first['prompt_inputs'], use_cache=False).logits
                identity_cfg = {'target_count': len(first['visual'])}
                if contextual:
                    identity_cfg['context_mode'] = selector.spec.context_mode
                if wordwise:
                    identity_cfg.update(match_width=selector.spec.match_width,template_token_ids=selector.spec.template_token_ids)
                if query_interaction:
                    identity_cfg.update(interaction=selector.spec.interaction, heads=selector.spec.heads)
                identity_session = binding(bundle['model'], identity_cfg).install()
                try:
                    actual = bundle['model'](**first['prompt_inputs'], use_cache=False).logits
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                finally:
                    identity_session.uninstall()
                del expected, actual
                smoke_cfg = {'target_count': 128}
                if contextual:
                    smoke_cfg['context_mode'] = selector.spec.context_mode
                if wordwise:
                    smoke_cfg.update(match_width=selector.spec.match_width,template_token_ids=selector.spec.template_token_ids)
                if query_interaction:
                    smoke_cfg.update(interaction=selector.spec.interaction, heads=selector.spec.heads)
                session = binding(bundle['model'], smoke_cfg).install()
                try:
                    bundle['model'].generate(**first['prompt_inputs'], max_new_tokens=2, min_new_tokens=2,
                        do_sample=False, repetition_penalty=1.05, use_cache=True,
                        pad_token_id=bundle['processor'].tokenizer.pad_token_id)
                    record = session.runtime_metadata()['selection']
                    if record['result_count'] != min(128, len(first['visual'])) or not record['decode_position_verified']:
                        raise ValueError('7B hard selection/cache smoke failed')
                finally:
                    session.uninstall()
            del first
            manifest['real_checkpoint_identity_and_decode_smoke'] = True
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        attempt_start_step = step
        attempt_id = time.time_ns()
        while step < args.stop_after:
            torch.cuda.synchronize(device)
            active_start = time.perf_counter()
            if policy:
                free_bytes, _ = torch.cuda.mem_get_info(device)
                if free_bytes < policy['minimum_free_mib_at_step'] * 1024**2:
                    raise RuntimeError('Shared GPU headroom fell below 8 GiB; stopping safely at the last checkpoint')
            selector.train()
            totals = {}
            indices = batch_indices(step, world_size=world, rank=rank)
            for index in indices:
                state = prepare_example(bundle, examples[index])
                llm.eval()
                with torch.no_grad():
                    teacher, targets = answer_logits(bundle, state['hidden'], state['positions'], state['teacher_inputs']['labels'])
                llm.train()
                if wordwise:
                    question,audit=prompt_word_question(bundle['model'],state['prompt_inputs'],selector.spec.template_token_ids)
                    selection=training_selection(selector,state['features'],mask_mode,question=question)
                elif contextual:
                    question, audit = prompt_question(bundle['model'], state['prompt_inputs'])
                    selection = training_selection(selector, state['features'], mask_mode, question=question)
                else:
                    selection = training_selection(selector, state['features'], mask_mode)
                visual = torch.tensor(state['visual'], device=device)
                hidden = state['hidden'].index_copy(1, visual, selection['tokens'][None])
                if hidden.shape != state['hidden'].shape:
                    raise ValueError('Soft training must retain the entire physical sequence')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    student, student_targets = answer_logits(bundle, hidden, state['positions'], state['teacher_inputs']['labels'])
                    if not torch.equal(targets, student_targets):
                        raise ValueError('Teacher/student answer labels changed')
                    loss, metrics = training_loss(student, teacher, targets, selection, step)
                    if wordwise:
                        annotation=evidence_rows.get(examples[index]['sample_id'])
                        masks=evidence_masks(annotation,height=state['height'],width=state['width'],
                            image_fingerprint=examples[index]['image_fingerprint'])
                        supervised=evidence_loss(selection['scores'],*masks) if masks else selection['scores'].sum()*0
                        loss=loss+protocol['training']['evidence_weight']*supervised
                        metrics.update(evidence_loss=float(supervised.detach()),evidence_supervised=float(masks is not None),
                            removed_format_tokens=audit['removed_format_tokens'],**selection['word_diagnostics'])
                        if query_interaction and task_index is not None:
                            target = load_target(task_index, examples[index], state, audit, device=device)
                            task_loss = teacher_ranking_loss(selection['scores'], target)
                            loss = loss + protocol['training']['task_teacher_weight'] * task_loss
                            metrics['task_teacher_loss'] = float(task_loss.detach())
                metrics.update(selection_diagnostics(selection))
                (loss / len(indices)).backward()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.) + value / len(indices)
                del loss, student, teacher, hidden, selection, state
            if any(p.grad is not None for p in bundle['model'].parameters()):
                raise ValueError('Frozen backbone accumulated gradients')
            for param in selector.parameters():
                if not param.requires_grad:
                    if param.grad is not None:raise ValueError('Frozen selector prior accumulated gradients')
                    continue
                if param.grad is None:
                    raise ValueError('Selector parameter has no gradient')
                if world > 1:
                    dist.all_reduce(param.grad)
                    param.grad.div_(world)
            branch_gradients={}
            if wordwise:
                for name,param in selector.named_parameters():
                    if param.requires_grad:
                        value=float(param.grad.norm())
                        # The gate's zero-initialized final layer intentionally
                        # blocks its first layer gradient on the very first step.
                        initial_gate = input_adaptive and step == 0 and name.startswith('input_gate.0.')
                        if not np.isfinite(value) or (value<=0 and not initial_gate and not query_interaction):raise ValueError(f'Word matching branch has invalid gradient: {name}')
                        branch_gradients[name]=value
                if query_interaction and branch_gradients.get('question_projection.weight', 0.) <= 0:
                    raise ValueError('Contextual question projection has zero gradient')
                if parameter_digest(selector.visual)!=frozen_prior_digest:raise ValueError('Frozen visual prior changed')
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError('Selector gradient is nonfinite or zero')
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            values = torch.tensor(list(totals.values()), device=device)
            if world > 1:
                dist.all_reduce(values)
                values.div_(world)
            metrics = dict(zip(totals, values.tolist()))
            if rank == 0:
                record = {'step': step, 'global_examples_seen': step * 8, **metrics,
                    'training_mask_mode': mask_mode,
                    'global_sample_ids': [e['sample_id'] for e in examples[(step - 1) * 8:step * 8]],
                    'gradient_norm': float(norm), 'learning_rate': scheduler.get_last_lr()[0],
                    'elapsed_s_this_attempt': time.perf_counter() - start}
                if wordwise:record['branch_gradient_norms']=branch_gradients
                with (output / f'training_from_step_{attempt_start_step}_{attempt_id}.jsonl').open('a') as log:
                    log.write(json.dumps(record, allow_nan=False) + '\n')
                print(json.dumps(record), flush=True)
            if step in (1, 2) or step % 16 == 0 or step == args.stop_after:
                local_rng = rng_state(device)
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, local_rng)
                else:
                    states[0] = local_rng
                if rank == 0:
                    folder = output / 'checkpoints' / f'step_{step:06d}_{time.time_ns()}'
                    save_training_state(folder, selector=selector, optimizer=optimizer, scheduler=scheduler,
                        step=step, contract=contract, random_states=states, save_fn=save_fn)
                    atomic_write_json(output / 'latest.json', {'step': step, 'checkpoint': str(folder.relative_to(output))})
                    manifest.update(steps_completed=step, global_examples_seen=step * 8,
                        trainable_parameters=sum(p.numel() for p in selector.parameters() if p.requires_grad),
                        peak_allocated_bytes_rank0=torch.cuda.max_memory_allocated(device))
                    atomic_write_json(output / 'manifest.json', manifest)
                if world > 1:
                    dist.barrier()
            torch.cuda.synchronize(device)
            cooperative_pause(time.perf_counter() - active_start)
        if rank == 0:
            manifest.update(status='complete' if step == 320 else 'stopped_at_checkpoint', finished_utc=now())
            atomic_write_json(output / 'manifest.json', manifest)
    except BaseException as exc:
        atomic_write_json(output / f'failure_rank{rank}_{time.time_ns()}.json', {
            'step': step, 'error': str(exc), 'traceback': traceback.format_exc(), 'time': now()})
        if rank == 0:
            manifest.update(status='failed', steps_completed=step, error=str(exc))
            atomic_write_json(output / 'manifest.json', manifest)
        raise
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-after', type=int, default=320)
    parser.add_argument('--resume', action='store_true')
    main(parser.parse_args())
