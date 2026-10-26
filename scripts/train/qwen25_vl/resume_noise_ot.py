"""Resume original hash-frozen model code with a separately pinned report fix.

Only enrich the in-memory evaluation-plan view with checksum-validated result
locations. Do not edit the original plan, execution worktree, weights, protocol,
or completed inference artifacts; do not repeat completed training/evaluations.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def load_reporting_helper(path):
    spec = importlib.util.spec_from_file_location('noise_ot_reporting_amendment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=1)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path('/home/ZhouFengyu')):
        raise ValueError('Only owner-local experiment outputs may be resumed')
    original = json.loads((root / 'protocol.json').read_text())
    source = Path(original['source']['source_root']).resolve()
    sys.path.insert(0, str(source))
    from scripts.train.qwen25_vl import run_noise_ot as runner
    if Path(runner.__file__).resolve() != source / 'scripts/train/qwen25_vl/run_noise_ot.py':
        raise ValueError('Resume did not import the original frozen runner')
    helper_path = Path(__file__).with_name('noise_ot_reporting.py').resolve()
    helper = load_reporting_helper(helper_path)
    original_read = runner.read
    reference = Path(original['reference']).resolve()
    plan_path = reference / 'evaluation_plan.json'
    expected_source = original_read(reference / 'protocol.json')['source']
    def resolved_read(path):
        data = original_read(path)
        if Path(path).resolve() == plan_path:
            return helper.resolve_completed_plan(reference, data, file_sha=runner.file_sha,
                expected_source=expected_source)
        return data
    runner.read = resolved_read
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = runner.validate(root)
        # Resolve all controls and successfully aggregate the already-completed
        # ChartQA cell BEFORE taking GPU resources or launching another job.
        runner.report(root, protocol)
        if args.report_only:
            print((root / 'report.md').read_text(), flush=True)
            return
        attempt = root / 'resume_attempts' / str(time.time_ns())
        manifest = original_read(root / 'manifest.json')
        runner.write_json(attempt / 'manifest_before.json', manifest)
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(), helper_path)}
        amendment = {'status': 'validated', 'created_utc': runner.context.native.now(),
            'source': protocol['source'], 'protocol_sha256': runner.file_sha(root / 'protocol.json'),
            'orchestration_files_sha256': hashes,
            'fix': 'Read completed run paths from validated completion markers, not action=evaluate schedule cells',
            'scope': 'Report/orchestration only; original frozen model/training/eval source unchanged',
            'gpu': args.gpu, 'completed_before': manifest.get('completed_cells', [])}
        runner.write_json(attempt / 'orchestration.json', amendment)
        manifest.pop('error', None)
        manifest.update(status='resuming', resume_amendment=str(attempt / 'orchestration.json'),
            updated_utc=runner.context.native.now())
        runner.write_json(root / 'manifest.json', manifest)
        runner.pipeline(root, args.gpu)


if __name__ == '__main__':
    main()
