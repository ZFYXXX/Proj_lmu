import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('first_exit', [0, 7])
def test_chain_order_failure_and_same_allocation(tmp_path, first_exit):
    root = Path(__file__).resolve().parents[2]
    script = root / 'scripts/eval/lmms_eval/lmu_qwen25_factorial_then_remaining.sbatch'
    scripts = tmp_path / 'scripts/eval/lmms_eval'
    scripts.mkdir(parents=True)
    names = ['lmu_qwen25_local_factorial.sbatch', 'lmu_qwen25_keep15_remaining.sbatch']
    for index, name in enumerate(names):
        (scripts / name).write_text(
            'set -eu\n'
            'test "$CUDA_VISIBLE_DEVICES" = "0,1,2,3"\n'
            'test "$SLURM_JOB_ID" = "test-job"\n'
            'test "$FALLBACK_REPO" = "$KEEP15_REPO"\n'
            f'echo {index} >> order.txt\n'
            f'exit {first_exit if index == 0 else 0}\n')
    env = dict(os.environ, SLURM_JOB_ID='test-job', CUDA_VISIBLE_DEVICES='0,1,2,3',
               CHAIN_REPO=str(tmp_path))
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == first_exit
    assert (tmp_path / 'order.txt').read_text().splitlines() == (['0', '1'] if first_exit == 0 else ['0'])
    assert ('CHAIN_COMPLETE' in result.stdout) == (first_exit == 0)
