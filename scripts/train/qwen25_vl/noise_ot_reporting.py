"""Resolve completed evaluations without treating a schedule as a result index.

Pure orchestration helper: no model imports, no writes to historical results.
"""
import json
from pathlib import Path


def resolve_completed_plan(reference, plan, *, file_sha, expected_source):
    reference = Path(reference)
    resolved = {**plan, 'cells': [dict(cell) for cell in plan['cells']]}
    for cell in resolved['cells']:
        if 'run' in cell:
            continue
        if cell.get('action') != 'evaluate':
            raise ValueError('Reference cell has neither a run nor an evaluation action')
        method, task = cell['method'], cell['dataset']
        if method not in {'frozen_visual', 'image_context', 'question_context'} or task not in {'chartqa', 'ocrbench', 'ai2d'}:
            raise ValueError('Unknown reference evaluation cell')
        marker = reference / 'eval/full_selective' / method / task / 'complete.json'
        if not marker.is_file():
            raise ValueError(f'Reference evaluation has no completion marker: {marker}')
        saved = json.loads(marker.read_text())
        if saved.get('method') != method or saved.get('dataset') != task or saved.get('requests') != cell['questions']:
            raise ValueError('Reference completion method/dataset/count mismatch')
        if saved.get('source') != expected_source:
            raise ValueError('Reference completion source mismatch')
        if saved.get('evaluation_plan_sha256') != file_sha(reference / 'evaluation_plan.json'):
            raise ValueError('Reference completion belongs to a different evaluation plan')
        if file_sha(saved['config']) != saved.get('config_sha256'):
            raise ValueError('Reference evaluation config changed')
        run = Path(saved['run'])
        if not run.is_dir():
            raise ValueError('Completed reference run is missing')
        cell.update(run=str(run), task_name=f'context_topk_{task}',
            completion_marker=str(marker), completion_marker_sha256=file_sha(marker))
    return resolved
