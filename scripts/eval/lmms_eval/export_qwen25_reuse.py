"""Export small prediction evidence and native scores, never models or images."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.eval.lmms_eval.portable_results import contract, sample_signature
from scripts.eval.lmms_eval.qwen25_portable_queue import METHODS, score_name, sha

ALIASES = {"R1_weighted_fps": "importance_fps", "T2_importance_pool": "importance_fps_pool",
    "frozen_visual": "visual_selector", "trained_selector": "visual_selector", "image_context": "image_context_selector",
    "question_context": "question_context_selector", "feature_anchor_adapter_k128": "old_mlp"}
COUNTS = {"chartqa":2500,"ocrbench":1000,"ai2d":3088,"mmstar":1500,"pope":9000,"okvqa":5046,
    "gqa":12578,"textvqa":5000,"scienceqa_img":2017,"docvqa":5349,
    "refcoco_bbox_rec_testA":5657,"refcoco_bbox_rec_testB":5095,
    "refcoco+_bbox_rec_testA":5726,"refcoco+_bbox_rec_testB":4889,"refcocog_bbox_rec_test":9602}


def task_name(name):
    if "refcoco" in name:
        split = "testA" if "testA" in name else "testB" if "testB" in name else "test"
        prefix = "refcocog" if "refcocog" in name else "refcoco+" if "plus" in name or "+" in name else "refcoco"
        return f"{prefix}_bbox_rec_{split}"
    for token in ("scienceqa", "ocrbench", "chartqa", "textvqa", "docvqa", "mmstar", "okvqa", "ok_vqa", "ai2d", "pope", "gqa"):
        if token in name.lower():
            return {"scienceqa":"scienceqa_img", "ok_vqa":"okvqa"}.get(token,token)


def candidates(roots):
    for root in roots:
        summary = root / 'summary.json'
        if summary.is_file():
            for record in json.loads(summary.read_text()).get('results', []):
                if record.get('run') and record.get('config') and record.get('method'):
                    for samples in sorted(Path(record['run']).glob('**/*_samples_*.jsonl')):
                        yield record['method'], samples, Path(record['config']), str(summary)
        # Include references already audited by the original selective runner.
        protocol = root / "protocol.json"
        if protocol.exists():
            for ref in json.loads(protocol.read_text()).get("references", []):
                samples = ref.get("samples")
                configs = [Path(p) for p in ref.get("artifacts_sha256", {}) if p.endswith('.yaml') and '/outputs/' in p]
                if samples and configs:
                    yield ref["method"], Path(samples), configs[0], "audited_reference"
        for marker in sorted(root.glob("**/complete.json")):
            if not any(part in {"full", "full_selective"} for part in marker.parts) or any(part in {"source", "smoke"} for part in marker.parts):
                continue
            record = json.loads(marker.read_text())
            if not record.get("run") or not record.get("config") or record.get("status") in {"failed", "incomplete"}:
                continue
            parts = marker.parts
            pivot = next(i for i,p in enumerate(parts) if p in {"full", "full_selective"})
            method = record.get("method", parts[pivot+1])
            for samples in sorted(Path(record["run"]).glob("**/*_samples_*.jsonl")):
                yield method, samples, Path(record["config"]), str(marker)


def main():
    import yaml
    from types import SimpleNamespace
    from lmms_eval.utils import load_yaml_config
    from lmms_eval.api.registry import get_aggregation
    from scripts.eval.lmms_eval.p6000_dataset_smoke import specifications, model_config
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-run',type=Path,action='append',required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--methods',nargs='+',default=['dense','importance_fps','importance_fps_pool'],choices=METHODS)
    args=parser.parse_args()
    identity=json.loads(args.identity.read_text())
    specs={name:(template,pattern) for name,template,pattern in specifications()}
    cells, ignored, cached_hashes={},[],{}
    for original, samples, config_path, provenance in candidates(args.source_run):
        method=ALIASES.get(original,original)
        task=task_name(samples.name)
        if method not in args.methods or task not in COUNTS or (method,task) in cells:
            continue
        config=yaml.safe_load(config_path.read_text())
        checkpoint=config['backbone'].get('model_path')
        if not checkpoint or Path(checkpoint).resolve()!=Path(identity['checkpoint']).resolve():
            continue
        rows=sorted((json.loads(line) for line in samples.read_text().splitlines() if line.strip()),key=lambda r:r['doc_id'])
        if len(rows)!=COUNTS[task] or [r['doc_id'] for r in rows]!=list(range(COUNTS[task])):
            ignored.append({'method':method,'task':task,'reason':'not_full_split'});continue
        template,pattern=specs[task]
        old_template=Path(config['eval']['args']['include_path'])
        native_name=samples.name.split('_samples_',1)[1].removesuffix('.jsonl')
        matches=[]
        for path in old_template.glob('*.yaml'):
            raw=yaml.safe_load(path.read_text().replace('!function ',''))
            if raw.get('task')==native_name: matches.append(path)
        if len(matches)!=1:
            ignored.append({'method':method,'task':task,'reason':'template_missing'});continue
        old_task=load_yaml_config(str(matches[0]))
        options = config['pipeline']['stages'][0]['config'] if config['pipeline']['stages'] else {}
        current_args=SimpleNamespace(model=checkpoint,method=method,identity_path=str(args.identity),
            method_checkpoint=options.get('adapter_path',options.get('checkpoint_path')),
            scorer_checkpoint=options.get('scorer_path'))
        old_contract=contract(config,matches[0])
        if old_contract!=contract(model_config(current_args),template):
            ignored.append({'method':method,'task':task,'reason':'protocol_mismatch','config':config_path.name})
            continue
        metric=score_name(task)
        if any(metric not in r for r in rows):
            ignored.append({'method':method,'task':task,'reason':'primary_metric_missing'});continue
        cfg=next(item for item in old_task['metric_list'] if item['metric']==metric)
        agg=cfg['aggregation']
        if isinstance(agg,str): agg=get_aggregation(agg)
        score=float(agg([r[metric] for r in rows]))
        sources=sorted(args.data_root.glob(pattern))
        if not sources and task=='mmstar':
            sources=[Path('/home/ZhouFengyu/fastv-repro/datasets/MMStar/mmstar.parquet')]
        files=[]
        for source in sources:
            if str(source) not in cached_hashes: cached_hashes[str(source)]=sha(source)
            relative=(str(source.relative_to(args.data_root)) if source.is_relative_to(args.data_root) else f'MMStar/{source.name}')
            files.append({'relative_path':relative,'bytes':source.stat().st_size,'sha256':cached_hashes[str(source)]})
        if not files: raise FileNotFoundError(f'No dataset sources for {task}')
        cells[method,task]={'method':method,'task':task,'metric':metric,'score':score,'samples':len(rows),
            'contract':old_contract,'dataset_files':files,
            'sample_signatures':{str(r['doc_id']):sample_signature(r) for r in rows},
            'predictions':[{'doc_id':r['doc_id'],'prediction':r.get('filtered_resps'),'score':r[metric]} for r in rows],
            'source':{'run':samples.parent.parent.name,'samples_sha256':sha(samples), 'config_sha256':sha(config_path),
                      'provenance':'audited_reference' if provenance=='audited_reference' else Path(provenance).parent.name}}
        print(f'EXPORTED {method}/{task}: {score} N={len(rows)}',flush=True)
    payload={'schema_version':1,'model_identity_sha256':identity['identity_sha256'],
        'cells':list(cells.values()),'not_exported':ignored}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if args.output.suffix=='.gz' else open
    with opener(args.output,'wt',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    print(f'Exported {len(cells)} full cells: {args.output} ({args.output.stat().st_size} bytes)')


if __name__=='__main__': main()
