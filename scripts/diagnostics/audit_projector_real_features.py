"""Read-only parity audit; never edits frozen experiments or official sources."""
import argparse
import ast
import base64
import glob
import hashlib
import importlib.util
import json
import math
from io import BytesIO
from pathlib import Path
import sys
import time
import traceback


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--source-root', type=Path, default=Path('/home/ZhouFengyu/fastv-repro/worktrees/context-visual-topk-20261024-v1'))
    parser.add_argument('--skip-official-injection', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from mllm_comp.analysis.common.anchor7b_runtime import load_runtime
    from mllm_comp.analysis.common.learned_hybrid_runtime import move_inputs
    from mllm_comp.backbones.qwen25_vl import Qwen25VLSample
    from mllm_comp.integrations.lmms_eval.model_bridge import LMMSEvalRequestBridge
    from mllm_comp.compressors.comparison.eadp import extract_qwen2_user_instruction_ids, select_eadp_indices, normalize_eadp_config
    from mllm_comp.compressors.comparison.prunesid import select_prunesid_indices
    from mllm_comp.compressors.image.mmtok import MMTokSpec, mmtok_image_tokens
    from lmms_eval.tasks.chartqa.utils import chartqa_doc_to_text
    from lmms_eval.tasks.ocrbench.utils import ocrbench_doc_to_text
    from lmms_eval.tasks.ai2d.utils import ai2d_doc_to_text
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
            default=lambda x: x.detach().cpu().tolist() if torch.is_tensor(x) else str(x)))
    official_root = Path('/home/ZhouFengyu/fastv-repro')
    epath = official_root / 'research_sources/EADP/Qwen_vl/model/pruner.py'
    ppath = official_root / 'research_sources/PruneSID/prunesid/prunesid_qwen/modeling_qwen2_vl.py'
    mpath = official_root / 'MMTok/mmtok/core/semantic_selector.py'
    eofficial = load('audit_eadp', epath)
    mofficial = load('audit_mmtok', mpath)
    tree = ast.parse(ppath.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'pca_group', 'nms'}]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen2VisionTransformerPretrainedModel_prunesid')
    fwd = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    begin = next(i for i, n in enumerate(fwd.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) for t in n.targets))
    fn = ast.FunctionDef(name='reference', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='hidden_states'), ast.arg(arg='need_token_num')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=fwd.body[begin:], decorator_list=[])
    ns = {'torch': torch, 'F': torch.nn.functional, 'np': np, 'math': math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[*funcs, fn], type_ignores=[])), str(ppath), 'exec'), ns)
    ident = json.loads((repo / 'outputs/train/qwen25_vl/anchor_mlp_7b_k128_ddp_20261021_v1/model_identity.json').read_text())
    save('manifest.json', {'status': 'running', 'source': str(source), 'device': args.device,
        'purpose': 'Implementation parity, not a benchmark or independent quality estimate',
        'base_identity': ident['identity_sha256'], 'torch': torch.__version__,
        'official_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (epath, ppath, mpath)}})
    bundle = load_runtime(ident, args.device)
    model, processor = bundle['model'], bundle['processor']
    from transformers import AutoTokenizer
    from mllm_comp.backbones.qwen25_vl.image_compression import Qwen25VLImageCompressionExecutor
    fast_tokenizer = AutoTokenizer.from_pretrained(ident['checkpoint'], local_files_only=True, use_fast=True)
    # Invoke the exact method used by the frozen MMTok binding, without installing patches.
    executor = Qwen25VLImageCompressionExecutor.__new__(Qwen25VLImageCompressionExecutor)
    executor.model, executor.runtime = model, bundle['runtime']
    bridge = LMMSEvalRequestBridge(facade=bundle['facade'], runtime=bundle['runtime'])
    root = '/home/YinPeiran/datasets/lmms_eval_datasets'
    tasks = [('chartqa', 'ChartQA', chartqa_doc_to_text, [0, 1, 2, 10, 100, 500, 1000, 2000]),
             ('ocrbench', 'OCRBench', ocrbench_doc_to_text, [0, 1, 2, 10, 100, 300, 600, 900]),
             ('ai2d', 'ai2d', ai2d_doc_to_text, [0, 6, 21, 31, 60, 63, 100, 500])]
    def overlap(a, b):
        a, b = set(map(int, a)), set(map(int, b))
        return {'same_set': a == b, 'intersection': len(a & b), 'jaccard': len(a & b) / len(a | b)}
    try:
        with torch.inference_mode():
            for task, folder, text_fn, ids in tasks:
                table = pq.read_table(sorted(glob.glob(f'{root}/{folder}/data/test-*.parquet')))
                for idx in ids:
                    start = time.monotonic()
                    doc = table.slice(idx, 1).to_pylist()[0]
                    item = doc['image']
                    image = Image.open(BytesIO(item['bytes']) if item.get('bytes') else item['path']).convert('RGB')
                    extra = {'pre_prompt': '', 'post_prompt': '\nAnswer the question with a single word.' if task == 'chartqa' else ''}
                    if task == 'ai2d':
                        extra.update(prompt_format='mcq', post_prompt="\nAnswer with the option's letter from the given choices directly.")
                    context = text_fn(doc, extra)
                    sample = Qwen25VLSample(sample_id=f'{task}/{idx}', conversation=bridge._conversation(context, [image], audio_sample_rate=16000))
                    batch = bundle['facade'].processor_adapter.prepare_inputs(processor, sample, phase='eval')
                    inputs = move_inputs(batch.model_inputs, bundle['device'])
                    # Independently reconstruct the released lmms Qwen wrapper's JPEG input path.
                    stream = BytesIO()
                    image.save(stream, format='JPEG')
                    uri = 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()
                    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': [{'type': 'image', 'image': uri, 'min_pixels': 200704, 'max_pixels': 1605632},
                         {'type': 'text', 'text': context.replace('<image>', '')}]}]
                    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    images, videos = process_vision_info(messages)
                    manual = processor(text=[rendered], images=images, videos=videos, padding=True, return_tensors='pt')
                    parity = {k: torch.equal(v.cpu(), manual[k]) for k, v in batch.model_inputs.items() if torch.is_tensor(v)}
                    stream85 = BytesIO()
                    image.save(stream85, format='JPEG', quality=85)
                    messages[1]['content'][0]['image'] = 'data:image/jpeg;base64,' + base64.b64encode(stream85.getvalue()).decode()
                    images85, videos85 = process_vision_info(messages)
                    manual85 = processor(text=[rendered], images=images85, videos=videos85, padding=True, return_tensors='pt')
                    parity85 = {k: torch.equal(v.cpu(), manual85[k]) for k, v in batch.model_inputs.items() if torch.is_tensor(v)}
                    captured = []
                    handle = bundle['vision'].merger.register_forward_pre_hook(lambda mod, values: captured.append(values[0].detach()))
                    try:
                        features, = model.model.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'])
                    finally:
                        handle.remove()
                    window, _ = bundle['vision'].get_window_index(inputs['image_grid_thw'])
                    vision = captured[0].reshape(features.shape[0], 4, -1).mean(1)[window.argsort()]
                    user_ids = inputs['input_ids'][0].clone()
                    for token in (model.config.image_token_id, model.config.vision_start_token_id, model.config.vision_end_token_id):
                        user_ids[user_ids == token] = -200
                    user_ids = extract_qwen2_user_instruction_ids(user_ids)
                    text = model.get_input_embeddings()(user_ids)
                    decoded = processor.tokenizer.decode(user_ids.tolist(), skip_special_tokens=True).strip()
                    encoded = processor.tokenizer(f'Question: {decoded}'.split(), is_split_into_words=True, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(bundle['device'])
                    if encoded.shape[1] > 1 and int(encoded[0, 0]) == processor.tokenizer.bos_token_id:
                        encoded = encoded[:, 1:]
                    mtext = model.get_input_embeddings()(encoded[0])
                    binding_text = executor.mmtok_prompt_embeddings(inputs['input_ids'], inputs['attention_mask'])
                    fast_ids = fast_tokenizer(f'Question: {decoded}'.split(), is_split_into_words=True,
                        return_tensors='pt', padding=True, truncation=True)['input_ids'].to(bundle['device'])
                    if fast_ids.shape[1] > 1 and int(fast_ids[0, 0]) == fast_tokenizer.bos_token_id:
                        fast_ids = fast_ids[:, 1:]
                    fast_text = model.get_input_embeddings()(fast_ids[0])
                    h, w = (inputs['image_grid_thw'][0, 1:] // 2).tolist()
                    row = {'task': task, 'doc_id': idx, 'N': len(features), 'grid': [h, w], 'input_parity_jpeg75': parity,
                        'input_parity_jpeg85': parity85, 'selectors': {}}
                    row['mmtok_text'] = {'processor_tokenizer': type(processor.tokenizer).__name__,
                        'processor_is_fast': processor.tokenizer.is_fast, 'slow_encoded_shape': list(encoded.shape),
                        'binding_matches_slow_first_row': torch.equal(binding_text, mtext),
                        'binding_decoded_input': processor.tokenizer.decode(encoded[0].tolist()),
                        'fast_slow_ids_equal': torch.equal(encoded, fast_ids),
                        'official_fast_shape': list(fast_ids.shape),
                        'official_fast_decoded_input': fast_tokenizer.decode(fast_ids[0].tolist())}
                    for dtype in (torch.bfloat16, torch.float32):
                        x, t, mt, v = features.to(dtype), text.to(dtype), mtext.to(dtype), vision.to(dtype)
                        label = str(dtype)
                        actual, _ = select_eadp_indices(x[None], t, patches_per_side=w, grid_height=h, cfg=normalize_eadp_config({'tokens_per_crop': 128, 'beta': 2.}))
                        pruner = eofficial.VisualTokenPruner(visual_token_num=128, beta=2., device=args.device)
                        scores = pruner.compute_importance(x[None], t.mean(0, keepdim=True), t[None], h, w)
                        expect, _ = eofficial._greed_select_impl(scores, eofficial._sim_visual_impl(x[None]), 128)
                        e = overlap(actual[0].tolist(), expect[0].tolist())
                        actual = mmtok_image_tokens(x, v, mt, spec=MMTokSpec(target_count=128, text_temperature=.01)).indices
                        expected, _ = mofficial.SemanticTokenSelector(128).mm_coverage_selection(mt, x, v, tv_temp=.01, vv_temp=.2)
                        m = overlap(actual.tolist(), expected)
                        actual, meta = select_prunesid_indices(x[None], cfg={'tokens_per_crop': 128, 'pca_seed': 0})
                        torch.manual_seed(0)
                        _, expected = ns['reference'](x, 128)
                        p = overlap(actual[0].tolist(), expected.tolist())
                        p['same_order'] = actual[0].tolist() == expected.tolist()
                        p['official_source_order'] = expected.tolist() == expected.sort().values.tolist()
                        p['framework_threshold'] = meta['nms_threshold']
                        row['selectors'][label] = {'eadp': e, 'mmtok': m, 'prunesid': p}
                    official_full, _ = mofficial.SemanticTokenSelector(128).mm_coverage_selection(fast_text, features, vision, tv_temp=.01, vv_temp=.2)
                    current = mmtok_image_tokens(features, vision, binding_text, spec=MMTokSpec(target_count=128, text_temperature=.01)).indices
                    row['mmtok_official_full_question'] = overlap(current.tolist(), official_full)
                    # Check the actual facade generation and EOS behavior on known formatting cases.
                    if task == 'ai2d' and idx in (6, 21, 31):
                        kwargs, _ = bridge._generation_kwargs({'max_new_tokens': 16, 'temperature': 0., 'repetition_penalty': 1.05})
                        original = model.generate(**inputs, **kwargs)[0, inputs['input_ids'].shape[1]:].tolist()
                        state = bundle['facade'].state_builder.build(batch, raw_batch=sample, phase='eval')
                        out = bundle['facade'].runner.generate(bundle['runtime'], state, processed_batch=batch, **kwargs)
                        alternate = dict(kwargs, eos_token_id=model.generation_config.eos_token_id)
                        alt = model.generate(**inputs, **alternate)[0, inputs['input_ids'].shape[1]:].tolist()
                        row['generation'] = {'kwargs': kwargs, 'checkpoint_eos': model.generation_config.eos_token_id,
                            'direct_ids': original, 'facade_ids': out.text.token_ids,
                            'direct_text': processor.tokenizer.decode(original, skip_special_tokens=True),
                            'checkpoint_eos_ids': alt, 'checkpoint_eos_text': processor.tokenizer.decode(alt, skip_special_tokens=True)}
                        quality75 = move_inputs(manual, bundle['device'])
                        alt75 = model.generate(**quality75, **kwargs)[0, quality75['input_ids'].shape[1]:].tolist()
                        row['generation']['jpeg75_ids'] = alt75
                        row['generation']['jpeg75_text'] = processor.tokenizer.decode(alt75, skip_special_tokens=True)
                    row['seconds'] = time.monotonic() - start
                    rows.append(row)
                    save('results.json', rows)
                    print(json.dumps({'task': task, 'doc_id': idx, 'N': len(features), 'selectors': row['selectors']}), flush=True)
        # Attempt the unmodified official injection only after all platform checks.
        # Import package submodules without executing the unrelated LLaVA initializer.
        if args.skip_official_injection:
            save('status.json', {'status': 'complete', 'images': len(rows)})
            manifest = json.loads((args.output / 'manifest.json').read_text())
            manifest.update(status='complete', images=len(rows))
            save('manifest.json', manifest)
            return
        import types
        import importlib
        package = types.ModuleType('mmtok')
        package.__path__ = [str(official_root / 'MMTok/mmtok')]
        sys.modules['mmtok'] = package
        try:
            injection = importlib.import_module('mmtok.qwen.qwen2_5_vl_mmtok')
            injection.mmtok_qwen2_5_vl(model, language_tokenizer=fast_tokenizer,
                processor=processor, retain_ratio=(128.01 / len(features)))
            model.set_question(context)
            kwargs, _ = bridge._generation_kwargs({'max_new_tokens': 16, 'temperature': 0.})
            with torch.inference_mode():
                output = model.generate(**inputs, **kwargs)
            save('official_injection.json', {'status': 'success', 'note': 'Same-K diagnostic ratio, not published 20% protocol',
                'generated_ids': output[0, inputs['input_ids'].shape[1]:]})
        except Exception:
            save('official_injection.json', {'status': 'failed', 'traceback': traceback.format_exc(),
                'note': 'Official source unmodified; failure is a dependency compatibility result, not a benchmark score.'})
        save('status.json', {'status': 'complete', 'images': len(rows)})
        manifest = json.loads((args.output / 'manifest.json').read_text())
        manifest.update(status='complete', images=len(rows))
        save('manifest.json', manifest)
    except BaseException:
        save('status.json', {'status': 'failed', 'images': len(rows), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
