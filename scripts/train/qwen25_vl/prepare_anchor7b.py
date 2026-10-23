"""Pin a full local 7B checkpoint and reuse the exact train/held-out images."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import identify_checkpoint, implementation_hash, CHECKPOINT, file_sha
from mllm_comp.core.serialization import atomic_write_json


def prepare(root, source):
    import yaml
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():
        raise ValueError('Existing frozen run; resume rather than overwrite')
    identity=identify_checkpoint(CHECKPOINT)
    atomic_write_json(root/'model_identity.json',identity)
    train=json.loads((source/'training_samples.json').read_text())
    cohort=json.loads((source/'cohort.json').read_text())
    training_hashes={e['image_fingerprint'] for e in train['entries']}
    evaluation_hashes={e['image_fingerprint'] for e in cohort['entries']}
    if len(training_hashes)!=2560 or training_hashes&evaluation_hashes or train['selection_reads_answers']:
        raise ValueError('Invalid/deduplication-contaminated training manifest')
    if Counter(e['source'] for e in train['entries'])!={'textvqa_train':2048,'scienceqa_train':512}:
        raise ValueError('Unexpected training composition')
    atomic_write_json(root/'training_samples.json',train)
    atomic_write_json(root/'cohort.json',cohort)
    protocol={'default_base_for_future_experiments':'Qwen2.5-VL-7B-Instruct','checkpoint':identity['checkpoint'],
        'revision':identity['revision'],'model_identity_sha256':identity['identity_sha256'],'seed':42,
        'adapter':{'hidden_size':3584,'width':256,'form':'mean + MLP(LayerNorm(mean))','initialization':'final Linear weight and bias exactly zero',
                   'trainable_parameters':1846016},
        'frozen':'vision encoder, native merger/projector, LLM and lm_head','budget':128,
        'partition':'deterministic farthest-first cosine feature anchors; hard assignment; fp32 means',
        'positions':'rounded three-axis native cluster centroid; text positions unchanged',
        'world_size':2,'distributed':'DDP of adapter only; one frozen 7B replica per GPU',
        'per_rank_micro_batch':1,'accumulation_per_rank':4,'effective_global_batch':8,'steps':320,'epochs':1,
        'loss':{'answer_ce':1.,'dense_7b_teacher_KL':.5},'teacher_temperature':1.,'optimizer':'AdamW',
        'learning_rate':.0002,'weight_decay':.01,'warmup_steps':16,'schedule':'cosine','clip_grad_norm':1.,
        'dtype':{'backbone':'bfloat16','adapter_optimizer':'float32'},'attention':'eager',
        'processor':{'use_fast':False,'min_pixels':200704,'max_pixels':1605632},
        'generation':{'do_sample':False,'repetition_penalty':1.,'use_cache':True,'chartqa_max_new_tokens':16,'textvqa_max_new_tokens':32},
        'training_images':2560,'training_sources':{'textvqa_train':2048,'scienceqa_img_train':512},
        'all_eval_source_images_excluded':train['excluded_all_evaluation_source_images'],
        'training_overlap_with_selected_eval_images':0,'source_3b_run':str(source),
        'comparison':'Reuse the same 16 smoke, 200 development, 800 held-out images; do not mix 3B and 7B baseline scores',
        'evaluation_methods':['dense','feature_anchor_untrained','feature_anchor_adapter'],
        'checkpoint_choice':'fixed step320, never selected using held-out accuracy',
        'gates':['CPU DDP vs serial gradient test','7B native/dense and zero-init layout/cache smoke',
                 'two-rank save at step2 and resume through step4','post-train saved-checkpoint 16-image evaluation'],
        'implementation_hash':implementation_hash(),'training_manifest_sha256':file_sha(root/'training_samples.json')}
    atomic_write_json(root/'protocol.json',protocol)
    (root/'configs').mkdir(exist_ok=True)
    for stage in ['smoke','development','confirmation']:
        for shard in [0,1]:
            config=yaml.safe_load((source/'configs'/f'{stage}_{shard}.yaml').read_text())
            config['experiment']['name']=f'qwen25_vl_7b_anchor_k128_{stage}_{shard}'
            config['backbone']['model_path']=identity['checkpoint']
            config['backbone']['model_revision']=identity['revision']
            config['analysis']['sampling']['manifest_path']=str(root/'cohort.json')
            config['analysis']['probes']=[{'name':'anchor7b_evaluation','enabled':True,'artifact_root':str(root),
                'cohort_path':str(root/'cohort.json'),'model_identity_path':str(root/'model_identity.json'),'stage':stage,
                'checkpoint_path':str(root/'model/checkpoints/step_000320'),'save_raw':False}]
            (root/'configs'/f'{stage}_{shard}.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    print(json.dumps({'root':str(root),'checkpoint':identity['checkpoint'],'revision':identity['revision'],
        'training_images':2560,'effective_global_batch':8,'ddp_workers':2}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    args=p.parse_args()
    prepare(args.root.resolve(),args.source.resolve())
