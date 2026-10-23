"""Durable 7B two-GPU smoke/resume/train/evaluate queue; no old artifacts changed."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import implementation_hash,validate_identity,file_sha
from mllm_comp.analysis.common.evidence_interaction import bootstrap,holm
from mllm_comp.core.serialization import atomic_write_json


def frozen_hashes(root):
    paths=[Path(__file__),Path(__file__).with_name('train_anchor7b_ddp.py'),Path(__file__).with_name('prepare_anchor7b.py'),
        ROOT/'mllm_comp/analysis/probes/anchor7b_evaluation.py',ROOT/'mllm_comp/registry/catalog.py',
        ROOT/'mllm_comp/analysis/probes/__init__.py',root/'model_identity.json',root/'training_samples.json',
        root/'cohort.json',root/'protocol.json']+sorted((root/'configs').glob('*.yaml'))
    return {str(p):file_sha(p) for p in paths}


def report(root,stage):
    entries=[e for e in json.loads((root/'cohort.json').read_text())['entries']
             if (e.get('smoke',False) if stage=='smoke' else e['split']==stage)]
    expected={e['sample_id']:e for e in entries}
    summaries=[json.loads(p.read_text()) for p in (root/'samples'/stage).glob('*/summary.json')]
    if len(summaries)!=len(entries) or {s['sample_id'] for s in summaries}!=set(expected):
        raise ValueError('Incomplete 7B held-out evaluation')
    groups={}
    for summary in summaries:
        if summary['implementation_hash']!=implementation_hash():
            raise ValueError('Mixed implementation hashes')
        rows=[json.loads(line) for line in Path(summary['condition_path']).read_text().splitlines()]
        methods={r['variant']:r for r in rows}
        if len(rows)!=3 or set(methods)!={'dense','feature_anchor_untrained','feature_anchor_adapter'}:
            raise ValueError('Incomplete or duplicate 7B conditions')
        for name,row in methods.items():
            if row['image_fingerprint']!=expected[summary['sample_id']]['image_fingerprint'] or row['gold_answer_in_decode_context']:
                raise ValueError('Image mismatch or gold leakage')
            if row['repetition_penalty']!=1. or (name!='dense' and row['retained_visual_tokens']!=128):
                raise ValueError('Unmatched generation/token budget')
        if stage=='smoke' and not methods['dense'].get('native_generation_identity_passed'):
            raise ValueError('7B native/dense identity not verified')
        groups.setdefault(summary['task'],[]).append(methods)
    tables={}
    for task,images in sorted(groups.items()):
        table={}
        for name in ['dense','feature_anchor_untrained','feature_anchor_adapter']:
            table[name]={'n_images':len(images),'mean_score':sum(i[name]['primary_task_score'] for i in images)/len(images),
                'mean_nll':sum(i[name]['answer_nll'] for i in images)/len(images),
                'mean_visual_tokens':sum(i[name]['retained_visual_tokens'] for i in images)/len(images)}
        table['adapter_minus_untrained']=bootstrap([i['feature_anchor_adapter']['primary_task_score']-
            i['feature_anchor_untrained']['primary_task_score'] for i in images])
        tables[task]=table
    if stage=='confirmation':
        if len(summaries)!=800 or len(tables)!=2:
            raise ValueError('Expected both tasks and all 800 held-out images')
        adjusted=holm([t['adapter_minus_untrained']['one_sided_p'] for t in tables.values()])
        for table,p in zip(tables.values(),adjusted):
            table['adapter_minus_untrained']['holm_adjusted_p']=p
    result={'status':'complete','stage':stage,'n_images':len(entries),'tables':tables,
        'base':'Qwen2.5-VL-7B-Instruct','interpretation':'Single seed; same train-excluded images previously evaluated on 3B. Not a new independent discovery confirmation. Do not mix 3B baseline scores.'}
    atomic_write_json(root/f'{stage}_report.json',result)
    lines=[f'# Qwen2.5-VL-7B anchor adapter: {stage}','','All compressed conditions use exactly 128 visual tokens.','',
        '| Task | Method | Native accuracy | Mean visual tokens |','|---|---|---:|---:|']
    for task,table in tables.items():
        for name in ['dense','feature_anchor_untrained','feature_anchor_adapter']:
            m=table[name]
            lines.append(f"| {task} | {name} | {100*m['mean_score']:.2f}% | {m['mean_visual_tokens']:.1f} |")
        effect=table['adapter_minus_untrained']
        lines += ['',f"{task}: adapter minus untrained = {100*effect['mean']:.3f} pp, image bootstrap 95% CI "
            f"[{100*effect['ci95'][0]:.3f}, {100*effect['ci95'][1]:.3f}] pp.",'']
    lines += [result['interpretation'],'','Diagnostic generation/scoring time is not a deployment latency benchmark.']
    (root/f'{stage}_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def run(root):
    root=root.resolve()
    logs=root/'logs';logs.mkdir(exist_ok=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=json.loads((root/'model_identity.json').read_text())
        validate_identity(identity)
        manifest={'status':'running','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
            'base_identity_sha256':identity['identity_sha256'],'implementation_hash':implementation_hash(),
            'frozen_hashes':frozen_hashes(root),'stages':{},
            'software':{n:importlib.metadata.version(n) for n in ['torch','transformers','safetensors','numpy']},
            'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version','--format=csv'],text=True)}
        children=[]
        def execute(name,commands):
            nonlocal children
            if frozen_hashes(root)!=manifest['frozen_hashes'] or implementation_hash()!=manifest['implementation_hash']:
                raise RuntimeError('Frozen 7B training code/configuration changed')
            children=[]
            for number,command in enumerate(commands):
                path=logs/f'{name}_{number}_{int(time.time())}.log'
                stream=path.open('w')
                process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false',
                         'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
                children.append((process,stream,path))
            manifest['stages'][name]={'status':'running','children':[{'pid':p.pid,'log':str(path)} for p,_,path in children]}
            manifest['active_stage']=name
            atomic_write_json(root/'run_manifest.json',manifest)
            while any(p.poll() is None for p,_,_ in children):
                if any(p.poll() not in {None,0} for p,_,_ in children):
                    raise RuntimeError(f'{name} worker failed; inspect owned logs')
                time.sleep(3)
            if any(p.returncode for p,_,_ in children):
                raise RuntimeError(f'{name} exited unsuccessfully')
            for _,stream,_ in children:
                stream.close()
            validate_identity(identity)
            if frozen_hashes(root)!=manifest['frozen_hashes'] or implementation_hash()!=manifest['implementation_hash']:
                raise RuntimeError('Code/config changed during stage')
            manifest['stages'][name]['status']='complete'
            atomic_write_json(root/'run_manifest.json',manifest)
        def training(smoke,stop_after=None):
            folder=root/('smoke_model' if smoke else 'model')
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
                str(Path(__file__).with_name('train_anchor7b_ddp.py')),'--root',str(root),'--output',str(folder),
                '--steps','4' if smoke else '320','--accumulation','4','--resume']
            if smoke:
                command+=['--smoke']
            if stop_after is not None:
                command+=['--stop-after',str(stop_after)]
            return command
        try:
            execute('ddp_smoke_checkpoint',[training(True,2)])
            for rank in [0,1]:
                state=json.loads((root/'smoke_model'/f'rank_{rank}_manifest.json').read_text())
                if state['status'] not in {'checkpointed','complete'} or state['steps_completed'] not in {2,4}:
                    raise RuntimeError('Two-rank checkpoint smoke failed')
            execute('ddp_smoke_resume',[training(True)])
            for rank in [0,1]:
                state=json.loads((root/'smoke_model'/f'rank_{rank}_manifest.json').read_text())
                if state['status']!='complete' or state['steps_completed']!=4 or not all(state.get(k) for k in [
                    'ddp_parameter_identity_passed','checkpoint_roundtrip_passed','native_generation_identity_passed',
                    'zero_init_identity_passed','compressed_cache_smoke_passed']):
                    raise RuntimeError('Resume/native/zero-init/DDP smoke gate failed')
            atomic_write_json(root/'frozen.json',{'frozen_hashes':manifest['frozen_hashes'],
                'implementation_hash':implementation_hash(),'model_identity_sha256':identity['identity_sha256'],
                'smoke_steps':4,'world_size':2,'effective_global_batch':8})
            execute('train_ddp_320',[training(False)])
            pointer=json.loads((root/'model/latest.json').read_text())
            if pointer['step']!=320:
                raise RuntimeError('Main training did not reach fixed checkpoint320')
            visited=[]
            for rank in [0,1]:
                state=json.loads((root/'model'/f'rank_{rank}_manifest.json').read_text())
                if state['status']!='complete' or state['steps_completed']!=320:
                    raise RuntimeError('One DDP worker did not complete')
                for line in (root/'model'/f'rank_{rank}_data_trace.jsonl').read_text().splitlines():
                    visited.extend(json.loads(line)['indices'])
            if sorted(visited)!=list(range(2560)):
                raise RuntimeError('Training image coverage/duplication audit failed')
            for stage in ['smoke','development','confirmation']:
                commands=[[str(Path(sys.executable).parent/'mllm-comp'),'analyze','--config',str(root/'configs'/f'{stage}_{rank}.yaml'),
                    '--device',f'cuda:{rank}','--output_dir',str(root/'lifecycle'/f'{stage}_{rank}_{int(time.time())}')] for rank in [0,1]]
                execute(f'eval_{stage}',commands)
                report(root,stage)
            (root/'final_report.md').write_text((root/'confirmation_report.md').read_text(),encoding='utf-8')
            manifest['status']='complete'
        except BaseException as exc:
            manifest.update({'status':'failed','error':str(exc),'traceback':traceback.format_exc()})
            for process,_,_ in children:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
            for process,_,_ in children:
                process.wait()
            raise
        finally:
            for _,stream,_ in children:
                stream.close()
            manifest['updated_utc']=datetime.now(timezone.utc).isoformat()
            atomic_write_json(root/'run_manifest.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    root=args.root.resolve()
    if args.launch:
        with (root/'supervisor.log').open('a') as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--root',str(root)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'supervisor_pid':child.pid,'root':str(root)}))
    else:
        run(root)
