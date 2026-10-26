"""Queued fixed-budget mechanism study; does not mutate the preceding run."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha
from mllm_comp.analysis.common.learned_topk_experiment import available_gpus,owned_output,source_identity,dependency_identity,shared_policy
from mllm_comp.analysis.common.selector_mechanisms import VARIANTS, evidence_coverage
from mllm_comp.analysis.common.selector_mechanism_cohort import choose_images
from mllm_comp.analysis.common.selector_conditioning import request_key
from mllm_comp.core.serialization import atomic_write_json as write_json,atomic_write_text as write_text
from scripts.diagnostics import run_selector_condition_audit as parent

TASKS=parent.TASKS


def read(path):return json.loads(Path(path).read_text())


def config(root,p,variant,phase,task,run):
    cfg=parent.config(root,p,'shuffled',phase,task,run)
    stage,=cfg['pipeline']['stages']
    name='selector_mechanism_audit'
    stage.update(name=variant,algorithm=name,plugin=name,variant=f'{name}.image',binding=f'{name}.image.qwen25_vl')
    stage['config'].update(mechanism_variant=variant,max_swap=8,evidence_manifest=str(root/'evidence.json'),
        evidence_manifest_sha256=p['evidence_sha256'])
    cfg['experiment']['name']=f'selector_mechanism_{phase}_{variant}_{task}'
    cfg['eval']['args']['include_path']=str(ROOT/'configs/eval/lmms_eval/tasks/selector_mechanisms')
    return cfg


def prepare(root,prerequisite,per_task):
    if root.exists():raise ValueError('Preserve old experiments; choose a new root')
    old=read(prerequisite/'protocol.json')
    if old['dependencies']!=dependency_identity():raise ValueError('Evaluator/software changed')
    conditions=read(prerequisite/'conditions.json')
    if file_sha(prerequisite/'conditions.json')!=old['conditions_sha256']:raise ValueError('Source conditions changed')
    selected=choose_images(conditions['rows'],per_task=per_task)
    cohort={'selection':'image-content hash, one question/image, no outcomes, global image deduplication',
        'seed':20261025,'per_task':per_task,'tasks':{}}
    rows={}
    review=[]
    for task,entries in selected.items():
        cohort['tasks'][task]={'source_doc_ids':[r['doc_id'] for r in entries],
            'image_fingerprints':[r['image_fingerprint'] for r in entries]}
        for i,row in enumerate(entries):
            rows[request_key(f'context_topk_{task}:test:{i}:0')]={**row,'source_doc_id':row['doc_id'],'doc_id':i}
            review.append({'task':task,'source_doc_id':row['doc_id'],'image_fingerprint':row['image_fingerprint'],
                'question_token_ids':row['true_ids'],'coordinate_space':'normalized_original_image',
                'status':'pending_review','reviewer':None,'groups':[],
                'note':'Independently identify all answer-support regions; predictions and selected tokens intentionally absent.'})
    root.mkdir(parents=True)
    write_json(root/'cohort.json',cohort)
    write_json(root/'conditions.json',{**conditions,'rows':rows})
    write_json(root/'evidence.json',{'format':'selector_evidence_annotations_v1','rows':{},
        'status':'no_verified_annotations','note':'No pseudo/3B attention labels are silently used as ground truth.'})
    write_json(root/'review/blind_tasks.json',{'status':'pending_review','independent_images':len(review),'tasks':review})
    prior_report=prerequisite.parents[2]/'analysis/visual_evidence_interaction/20261020_v1/final_report.md'
    p={**old,'created_utc':parent.context.native.now(),'source':source_identity(ROOT),
        'resource_policy':shared_policy(),'prerequisite':str(prerequisite),'per_task':per_task,
        'variants':VARIANTS,'conditions_sha256':file_sha(root/'conditions.json'),
        'cohort_sha256':file_sha(root/'cohort.json'),'evidence_sha256':file_sha(root/'evidence.json'),
        'prerequisite_hashes':{str(f):file_sha(f) for f in (prerequisite/'protocol.json',prerequisite/'conditions.json')},
        'prior_evidence_audit':{'model':'Qwen2.5-VL-3B','report':str(prior_report),'sha256':file_sha(prior_report),
            'conclusion':'insufficient_confirmatory_evidence','use':'background only, not 7B evidence ground truth'},
        'limits':[*old['limits'],
            '750 independently hash-selected images by default; no outcome-based failure/success filtering.',
            'Targets are question-induced lost/gained candidate tokens, NOT verified semantic evidence.',
            'Four crossed additions/removals in each direction separate added-candidate and removed-distractor contrasts.',
            'Candidate/control groups are disjoint and equal size; total K128 and original M-RoPE/content remain unchanged.',
            'Norm/quadrant control matching can relax quadrant when insufficient candidates; every relaxation is recorded.',
            'Fewer than two exchanged tokens is inapplicable for disjoint-control mediation, not a zero causal-effect finding.',
            'Semantic evidence coverage remains pending until independent human/dataset annotations are verified.',
            'The old 3B region-interaction result is not relabeled as a 7B semantic mechanism result.']}
    write_json(root/'protocol.json',p)
    write_json(root/'manifest.json',{'status':'prepared','source':p['source'],'completed_cells':[],
        'protocol_sha256':file_sha(root/'protocol.json'),'semantic_evidence_status':'pending_review',
        'independent_images':3*per_task,'created_utc':parent.context.native.now()})
    print(json.dumps({'images':3*per_task,'variants':len(VARIANTS),'new_full_requests':3*per_task*len(VARIANTS),
        'verified_semantic_annotations':0,'prerequisite':str(prerequisite)}),flush=True)


def validate(root):
    p=read(root/'protocol.json')
    if p['source']!=source_identity(ROOT) or p['dependencies']!=dependency_identity() or p['resource_policy']!=shared_policy():
        raise ValueError('Frozen source/software/resource policy changed')
    if read(root/'manifest.json')['protocol_sha256']!=file_sha(root/'protocol.json'):raise ValueError('Protocol changed')
    for file,key in [('conditions.json','conditions_sha256'),('cohort.json','cohort_sha256'),('evidence.json','evidence_sha256')]:
        if file_sha(root/file)!=p[key]:raise ValueError('Frozen mechanism input changed')
    for f,digest in {**p['reference_hashes'],**p['prerequisite_hashes']}.items():
        if file_sha(f)!=digest:raise ValueError(f'Pinned reference changed: {f}')
    receipt=root/'prerequisite_receipt.json'
    if receipt.exists():
        for f,digest in read(receipt)['hashes'].items():
            if file_sha(f)!=digest:raise ValueError(f'Completed prerequisite changed: {f}')
    return p


def check(root,p,run,variant,phase,task):
    n=8 if phase=='smoke' else p['per_task']
    checked=parent.context.native.check_eval(run,variant,phase,tasks=(task,),counts={task:p['per_task']},prefix='context_topk')
    _,actual=parent.sample_rows(run,f'context_topk_{task}',n,parent.context.METRICS[task])
    ref=p['reference_runs'][task]
    _,baseline=parent.sample_rows(Path(ref['run']),ref['task_name'],parent.context.COUNTS[task],parent.context.METRICS[task])
    true_traces=parent.trace_rows(ref['run'],task)
    prior_marker=Path(p['prerequisite'])/f'eval/full/shuffled/{task}/complete.json'
    prior_run=Path(read(prior_marker)['run'])
    _,wrong=parent.sample_rows(prior_run,f'context_topk_{task}',parent.context.COUNTS[task],parent.context.METRICS[task])
    wrong_traces=parent.trace_rows(prior_run,task)
    cohort=read(root/'cohort.json')['tasks'][task]['source_doc_ids']
    for i,trace in parent.trace_rows(run,task).items():
        original=cohort[i]
        if trace['mechanism_variant']!=variant or trace['source_doc_id']!=original:raise ValueError('Wrong mechanism request mapping')
        if trace['true_selected_indices']!=true_traces[original]['selected_indices'] or trace['wrong_question_selected_indices']!=wrong_traces[original]['selected_indices']:
            raise ValueError('Underlying fixed selector/visual features changed')
        if (actual[i]['input'],actual[i]['target'])!=(baseline[original]['input'],baseline[original]['target']):
            raise ValueError('LLM prompt/target changed')
        if trace['score_mechanism']['score_energy_closure_relative']>1e-8:raise ValueError('Score energy decomposition did not close')
        comparison=baseline[original] if trace['selected_indices']==trace['true_selected_indices'] else wrong[original] if trace['selected_indices']==trace['wrong_question_selected_indices'] else None
        if comparison is not None and actual[i]['filtered_resps']!=comparison['filtered_resps']:
            raise ValueError('Identical selected tokens produced different answers')
        if trace['evidence_coverage']['true']['verified'] or trace['semantic_mechanism_claimed']:
            raise ValueError('This unannotated run must not claim verified semantic grounding')
    return checked


def report(root,p):
    records=[]
    cohort=read(root/'cohort.json')
    for task in TASKS:
        complete={v:root/f'eval/full/{v}/{task}/complete.json' for v in VARIANTS}
        complete={v:f for v,f in complete.items() if f.exists()}
        if 'observe' not in complete:continue
        tables={v:parent.sample_rows(Path(read(f)['run']),f'context_topk_{task}',p['per_task'],parent.context.METRICS[task])[1] for v,f in complete.items()}
        observed_run=Path(read(complete['observe'])['run'])
        traces=parent.trace_rows(observed_run,task)
        eligible=[i for i,t in traces.items() if t['swap_plan']['eligible']]
        ids=cohort['tasks'][task]['image_fingerprints']
        score=lambda variant:[float(r[parent.context.METRICS[task]]) for r in tables[variant]]
        ref=p['reference_runs'][task]
        _,baseline=parent.sample_rows(Path(ref['run']),ref['task_name'],parent.context.COUNTS[task],parent.context.METRICS[task])
        true_score=[float(baseline[i][parent.context.METRICS[task]]) for i in cohort['tasks'][task]['source_doc_ids']]
        record={'task':task,'images':p['per_task'],'complete_variants':list(complete),'eligible_mediation_images':len(eligible),
            'semantic_evidence_status':'pending_review','scores':{v:sum(score(v))/p['per_task'] for v in complete},
            'true_score':sum(true_score)/p['per_task'],
            'mean_common_shift_energy_fraction':sum(t['score_mechanism']['common_shift_energy_fraction'] for t in traces.values())/len(traces),
            'mean_rank_spearman':None,'mean_swapped_tokens':sum(t['score_mechanism']['swapped_token_count'] for t in traces.values())/len(traces),
            'contrasts':{}}
        correlations=[t['score_mechanism']['rank_spearman'] for t in traces.values() if t['score_mechanism']['rank_spearman'] is not None]
        if correlations:record['mean_rank_spearman']=sum(correlations)/len(correlations)
        if all(v in tables for v in VARIANTS) and eligible:
            scores={v:score(v) for v in VARIANTS}
            zeros=[0.]*len(eligible)
            effects={
                'restored_target_vs_control_addition':[(scores['restore_tt'][i]+scores['restore_tc'][i]-scores['restore_ct'][i]-scores['restore_cc'][i])/2 for i in eligible],
                'removed_target_vs_control_distractor':[(scores['restore_tt'][i]+scores['restore_ct'][i]-scores['restore_tc'][i]-scores['restore_cc'][i])/2 for i in eligible],
                'reverse_target_vs_control_removal':[(scores['remove_tt'][i]+scores['remove_tc'][i]-scores['remove_ct'][i]-scores['remove_cc'][i])/2 for i in eligible]}
            record['contrasts']={name:parent.context.clustered_difference(values,zeros,[ids[i] for i in eligible]) for name,values in effects.items()}
        write_text(root/f'cases/{task}_mechanisms.jsonl',''.join(json.dumps({'subset_doc_id':i,'source_doc_id':t['source_doc_id'],
            'image_fingerprint':ids[i],'score_mechanism':t['score_mechanism'],'condition_path':t['condition_path'],
            'spatial_bins':t['spatial_bins'],'swap_plan':t['swap_plan'],'evidence_coverage':t['evidence_coverage']},ensure_ascii=False)+'\n' for i,t in sorted(traces.items())))
        records.append(record)
    write_json(root/'summary.json',{'tasks':records,'semantic_evidence_status':'pending_review','limits':p['limits']})
    lines=['# Selector mechanisms: candidate-token causality, semantic grounding pending','',
        '| Task | Images | Complete variants | Eligible swap images | Common score-shift energy | Mean exchanged tokens |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['task']} | {r['images']} | {len(r['complete_variants'])}/9 | {r['eligible_mediation_images']} | {100*r['mean_common_shift_energy_fraction']:.2f}% | {r['mean_swapped_tokens']:.2f} |")
    for r in records:
        for name,d in r['contrasts'].items():lines.append(f"\n{r['task']}/{name}: {100*d['delta']:+.2f} pp [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}].")
    write_text(root/'report.md','\n'.join(lines+['','No automatic candidate is labeled as verified semantic evidence.','',*p['limits']])+'\n')


def review_report(root,annotation_path):
    """Append verified coverage without changing a frozen GPU run.

    This geometrical review is not an evidence-guided causal intervention.
    A different annotation hash creates a separate extension, never an overwrite.
    """
    validate(root)
    document=read(annotation_path)
    if document.get('format')!='selector_evidence_annotations_v1' or not document.get('rows'):
        raise ValueError('A nonempty verified annotation export is required')
    cohort=read(root/'cohort.json')['tasks']
    known={f'{task}:{source}':image for task,entry in cohort.items()
        for source,image in zip(entry['source_doc_ids'],entry['image_fingerprints'])}
    annotations=document['rows']
    for key,annotation in annotations.items():
        if key not in known:raise ValueError(f'Annotation outside frozen cohort: {key}')
        evidence_coverage(annotation,[],height=1,width=1,image_fingerprint=known[key])
    digest=file_sha(annotation_path)
    out=root/'semantic_extensions'/digest
    if out.exists():raise ValueError('Preserve the existing annotation extension')
    records=[]
    for task,entry in cohort.items():
        marker=root/f'eval/full/observe/{task}/complete.json'
        if not marker.exists():raise ValueError(f'Observe results not complete: {task}')
        run=Path(read(marker)['run'])
        traces=parent.trace_rows(run,task)
        for i,trace in traces.items():
            source=entry['source_doc_ids'][i];key=f'{task}:{source}'
            annotation=annotations.get(key)
            if annotation is None:continue
            height,width=trace['native_grid']
            coverage={name:evidence_coverage(annotation,indices,height=height,width=width,
                image_fingerprint=known[key]) for name,indices in
                [('true',trace['true_selected_indices']),('wrong_question',trace['wrong_question_selected_indices'])]}
            records.append({'key':key,'image_fingerprint':known[key],'native_grid':[height,width],
                'evidence_coverage':coverage,'semantic_causality_established':False})
    write_json(out/'annotations.json',document)
    write_json(out/'coverage.json',{'annotations_sha256':digest,'protocol_sha256':file_sha(root/'protocol.json'),
        'verified_images':len(records),'unreviewed_images':len(known)-len(records),'rows':records,
        'limits':['Independent labels establish geometric evidence coverage, not unique semantic causation.',
            'Candidate-token swaps remain score-based; this extension does not relabel them as ground-truth evidence rescue.']})
    print(json.dumps({'extension':str(out),'verified_images':len(records),'unreviewed_images':len(known)-len(records)}),flush=True)


def pipeline(root,gpu):
    p=validate(root);manifest=read(root/'manifest.json')
    prerequisite=Path(p['prerequisite'])
    def stage(name,status='running'):
        manifest.update(stage=name,status=status,gpu=gpu,updated_utc=parent.context.native.now());write_json(root/'manifest.json',manifest)
    try:
        stage('waiting_for_question_condition_full','queued')
        while not (prerequisite/'full_complete.json').exists():
            if read(prerequisite/'manifest.json')['status']=='failed':raise RuntimeError('Question-condition prerequisite failed; no mechanism GPU job launched')
            time.sleep(20)
        if read(prerequisite/'manifest.json')['status']!='complete':raise ValueError('Prerequisite is not fully complete')
        receipt=root/'prerequisite_receipt.json'
        if not receipt.exists():
            pins={str(prerequisite/'full_complete.json'):file_sha(prerequisite/'full_complete.json')}
            for task in TASKS:
                marker=prerequisite/f'eval/full/shuffled/{task}/complete.json'
                run=Path(read(marker)['run'])
                samples,_=parent.sample_rows(run,f'context_topk_{task}',parent.context.COUNTS[task],parent.context.METRICS[task])
                for filename in (marker,samples,run/'compression_runtime.requests.jsonl'):
                    pins[str(filename)]=file_sha(filename)
            write_json(receipt,{'hashes':pins,'created_utc':parent.context.native.now()})
        validate(root)
        def evaluate(variant,phase,task):
            parent_dir=root/f'eval/{phase}/{variant}/{task}';marker=parent_dir/'complete.json'
            if marker.exists():
                saved=read(marker)
                if saved['source']!=p['source'] or file_sha(saved['config'])!=saved['config_sha256']:raise ValueError('Completed cell changed')
                check(root,p,Path(saved['run']),variant,phase,task);return
            while available_gpus([gpu])!=[gpu]:stage('waiting_for_gpu','queued');time.sleep(20)
            validate(root);stage(f'{phase}:{variant}:{task}')
            run=parent_dir/f'attempt_{time.time_ns()}'
            filename=root/'configs'/f'{phase}_{variant}_{task}_{time.time_ns()}.yaml'
            write_text(filename,yaml.safe_dump(config(root,p,variant,phase,task,run),sort_keys=False))
            parent.context.native.execute([sys.executable,'-m','mllm_comp.cli.main','eval','--config',str(filename)],
                parent_dir/f'launch_{time.time_ns()}.log',[gpu],{'MLLM_SELECTOR_MECHANISM_COHORT':str(root/'cohort.json'),
                    'MLLM_TOPK_PERFORMANCE':str(run/'request_performance.jsonl'),
                    'HF_HOME':str(root/'cache/huggingface'),'HF_DATASETS_CACHE':str(root/'cache/datasets'),'XDG_CACHE_HOME':str(root/'cache/xdg')})
            checked=check(root,p,run,variant,phase,task)
            write_json(marker,{'run':str(run),'config':str(filename),'config_sha256':file_sha(filename),
                'source':p['source'],'requests':checked['requests'],'completed_utc':parent.context.native.now()})
            if phase=='full':
                manifest['completed_cells']=sorted(set(manifest['completed_cells'])|{f'{variant}/{task}'})
                report(root,p);write_json(root/'manifest.json',manifest)
        for phase in ('smoke','full'):
            for variant in VARIANTS:
                for task in TASKS:evaluate(variant,phase,task)
            write_json(root/f'{phase}_complete.json',{'status':'complete','requests':len(VARIANTS)*3*(8 if phase=='smoke' else p['per_task']),
                'semantic_evidence_status':'pending_review','time':parent.context.native.now()})
        stage('automatic_diagnostics_complete');manifest.update(status='complete',finished_utc=parent.context.native.now())
        write_json(root/'manifest.json',manifest)
    except BaseException as exc:
        manifest.update(status='failed',error=str(exc),updated_utc=parent.context.native.now());write_json(root/'manifest.json',manifest)
        write_json(root/f'failure_{time.time_ns()}.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','pipeline','review-report'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--prerequisite',type=Path)
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--per-task',type=int,default=250);parser.add_argument('--gpu',type=int,choices=(0,1),default=1)
    args=parser.parse_args();root=owned_output(args.root)
    if args.command=='prepare':
        if args.prerequisite is None:parser.error('--prerequisite is required')
        prepare(root,args.prerequisite.resolve(),args.per_task)
    elif args.command=='review-report':
        if args.annotations is None:parser.error('--annotations is required')
        review_report(root,args.annotations.resolve())
    else:
        with (root/'runner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pipeline(root,args.gpu)
