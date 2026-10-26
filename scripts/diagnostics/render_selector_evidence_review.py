"""Export an answer/selection-blind local evidence-box review packet.

Original image bytes are embedded unchanged. Exported annotations only become
verified through the reviewer's explicit confirmation; no automatic GT labels.
"""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def render(root,per_task=100):
    import pyarrow.parquet as pq
    from PIL import Image
    from transformers import AutoTokenizer
    from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
    from mllm_comp.analysis.common.learned_topk_experiment import owned_output
    from mllm_comp.core.serialization import atomic_write_text
    root=owned_output(root)
    read=lambda p:json.loads(Path(p).read_text())
    p=read(root/'protocol.json')
    tasks=read(root/'review/blind_tasks.json')['tasks']
    original_cohort=read(Path(p['reference'])/'cohort.json')
    tokenizer=AutoTokenizer.from_pretrained(read(p['identity_path'])['checkpoint'],local_files_only=True,use_fast=False)
    packets=[]
    for task in ('chartqa','ocrbench','ai2d'):
        selected=sorted([r for r in tasks if r['task']==task],key=lambda r:hashlib.sha256(f"blind-review:{r['image_fingerprint']}".encode()).hexdigest())[:per_task]
        needed={r['source_doc_id']:r for r in selected}
        files=original_cohort[task].get('paths') or [original_cohort[task]['path']]
        offset=0
        for filename in files:
            for batch in pq.ParquetFile(filename).iter_batches(batch_size=16,columns=['image']):
                for doc in batch.to_pylist():
                    item=needed.get(offset);offset+=1
                    if item is None:continue
                    data=doc['image'];raw=data.get('bytes')
                    if raw is None:raw=Path(data['path']).read_bytes()
                    with Image.open(BytesIO(raw)) as image:
                        if media_fingerprint([image.convert('RGB')])!=item['image_fingerprint']:raise ValueError('Review image fingerprint mismatch')
                        mime=Image.MIME.get(image.format,'application/octet-stream')
                    packets.append({k:item[k] for k in ('task','source_doc_id','image_fingerprint')}|
                        {'question':tokenizer.decode(item['question_token_ids'],clean_up_tokenization_spaces=False),
                         'image':'data:'+mime+';base64,'+base64.b64encode(raw).decode()})
        if sum(r['task']==task for r in packets)!=len(selected):raise ValueError('Incomplete blind image packet')
    payload=json.dumps(packets,ensure_ascii=False).replace('<','\\u003c')
    template=(Path(__file__).with_name('selector_evidence_review.html')).read_text()
    output=root/'review/index.html'
    if output.exists():raise ValueError('Preserve the existing review packet')
    atomic_write_text(output,template.replace('__REVIEW_DATA__',payload))
    print(json.dumps({'output':str(output),'images':len(packets),'verified_annotations':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--per-task',type=int,default=100);args=parser.parse_args()
    if args.per_task<1:parser.error('Positive per-task count required')
    render(args.root,args.per_task)
