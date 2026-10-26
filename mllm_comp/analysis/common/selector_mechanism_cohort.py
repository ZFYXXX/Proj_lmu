"""One hash-selected question per independent image, without score filtering."""
import hashlib
from collections import defaultdict


def choose_images(condition_rows,*,per_task=250,seed=20261025):
    if isinstance(per_task,bool) or not isinstance(per_task,int) or per_task<1:raise ValueError('Positive cohort size required')
    groups=defaultdict(lambda:defaultdict(list))
    for key,row in condition_rows.items():groups[row['task']][row['image_fingerprint']].append((key,row))
    selected,seen={},set()
    def digest(text):return hashlib.sha256(f'{seed}:{text}'.encode()).hexdigest()
    for task in ('chartqa','ocrbench','ai2d'):
        images=sorted((image for image in groups[task] if image not in seen),key=lambda image:(digest(image),image))
        if len(images)<per_task:raise ValueError('Not enough independent images for the frozen cohort')
        rows=[]
        for image in images[:per_task]:
            _,row=min(groups[task][image],key=lambda item:(digest(item[0]),item[0]))
            rows.append(row);seen.add(image)
        selected[task]=sorted(rows,key=lambda r:r['doc_id'])
    return selected
