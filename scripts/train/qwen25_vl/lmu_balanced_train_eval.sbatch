#!/usr/bin/env bash
#SBATCH --job-name=28-q25-balanced
#SBATCH --account=hk-project-p0024767
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=23:50:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/slurm/%x_%j_output.log
#SBATCH --error=logs/slurm/%x_%j_error.log
#SBATCH --mail-type="NONE"

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?Submit from the project root}"
cd "$REPO"
ENV=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/zfy-mllm-comp
PY="$ENV/bin/python"
WS=/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace
DATA_ROOT=/hkfs/work/workspace/scratch/tum_yvc3016-comp/mllm_dataset/lmms_lab_dataset
MODEL_PATH="$WS/ckpts/ckpts-lmm/Qwen2.5-VL-7B-Instruct"
CKPT_ROOT="$WS/ckpts/ckpts-lmm/Proj"
RUN_ID="${RUN_ID:-qwen25_lmunext_balanced2560_r1_v1}"
BUNDLE_PATH="${BUNDLE_PATH:-$WS/inputs/qwen25_balanced2560_20261104}"
RUN_ROOT="$WS/outputs/Fengyu/$RUN_ID"
TRAIN_RUN="$RUN_ROOT/train"
TRAIN_PROFILE="$RUN_ROOT/lmu_profile.json"
TRAIN_CONFIG="$RUN_ROOT/training_config.json"
IDENTITY="$RUN_ROOT/model_identity.json"
EVAL_RUN="$WS/outputs/Fengyu/${RUN_ID}_eval"
CACHE_DIR="$WS/Fengyu/cache/$RUN_ID"
RESULT_BUNDLE="$REPO/artifacts/eval/qwen25_p6000_reuse_20261103.json.gz"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/slurm "$RUN_ROOT" "$CACHE_DIR"

# Fetch the fixed private transfer bundle only inside this allocated job.
if [[ ! -f "$BUNDLE_PATH/balanced_manifest.json" ]]; then
    [[ -n "${PROJ_TRANSFER_TOKEN:-}" ]] || { echo "Missing training bundle; export PROJ_TRANSFER_TOKEN before sbatch" >&2; exit 2; }
    HF_HUB_OFFLINE=0 "$PY" - "$BUNDLE_PATH" <<'PY'
import hashlib,os,sys,tarfile,tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download
target=Path(sys.argv[1]).resolve()
if target.exists():
    raise SystemExit(f'Incomplete training bundle already exists: {target}; inspect it before retrying.')
target.parent.mkdir(parents=True,exist_ok=True)
archive=Path(hf_hub_download('ZFFFY/Proj-training','qwen25_balanced2560_20261104.tar.gz',
    repo_type='dataset',revision='9fd5bad414a0c96320db928db1c710e4fe4b2264',token=os.environ['PROJ_TRANSFER_TOKEN'],
    local_dir=str(target.parent/'.hf_training_transfer')))
digest=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
if digest.hexdigest()!='40ee09ed6b6b6fed657d1789d41812d69c77895cf21d7da777e5d8be15941689':
    raise SystemExit('Downloaded training archive checksum mismatch')
with tempfile.TemporaryDirectory(prefix='.balanced_bundle_',dir=target.parent) as directory:
    stage=Path(directory).resolve()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            if not (member.isfile() or member.isdir()) or not (stage/member.name).resolve().is_relative_to(stage):
                raise SystemExit('Unsupported training archive member')
        tar.extractall(stage)
    extracted=stage/'qwen25_balanced2560_20261104'
    for name in ('balanced_manifest.json','balanced_qa.parquet','images','bundle_manifest.json'):
        if not (extracted/name).exists(): raise SystemExit(f'Incomplete archive: {name}')
    extracted.rename(target)
print(f'Downloaded and verified fixed training bundle: {target}')
PY
fi
unset PROJ_TRANSFER_TOKEN

for path in "$PY" "$DATA_ROOT" "$MODEL_PATH/config.json" \
    "$CKPT_ROOT/model_identity.json" "$CKPT_ROOT/checkpoints/old_mlp" \
    "$CKPT_ROOT/checkpoints/image_context_selector" "$BUNDLE_PATH/balanced_manifest.json" \
    "$RESULT_BUNDLE"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
[[ -e "$TRAIN_RUN/workflow.json" ]] && { echo "Training run already exists: $TRAIN_RUN" >&2; exit 2; }
[[ ! -e "$EVAL_RUN" ]] || { echo "Evaluation run already exists: $EVAL_RUN" >&2; exit 2; }

# Verify that this node sees all four allocated GPUs before any model load.
[[ "${CUDA_VISIBLE_DEVICES:-}" != "" ]] || { echo "Slurm did not set CUDA_VISIBLE_DEVICES" >&2; exit 2; }
[[ $(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l) -eq 4 ]] || { echo "Expected four allocated GPUs: $CUDA_VISIBLE_DEVICES" >&2; exit 2; }
nvidia-smi

# Bind the old projector/scorer to the LMU-local copy of the exact same base model.
"$PY" - "$CKPT_ROOT/model_identity.json" "$MODEL_PATH" "$RUN_ROOT" <<'PY'
import sys
from pathlib import Path
from scripts.eval.lmms_eval.qwen25_portable_queue import resolve_identity
resolve_identity(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
PY

# Confirm bundle integrity and rewrite its relative paths for this workspace.
"$PY" "$REPO/scripts/train/qwen25_vl/package_balanced_lmu_data.py" relocate --bundle "$BUNDLE_PATH"
"$PY" "$REPO/scripts/train/qwen25_vl/prepare_lmu_balanced_run.py" \
    --repo "$REPO" --bundle "$BUNDLE_PATH" --model "$MODEL_PATH" --checkpoint-root "$CKPT_ROOT" \
    --identity "$IDENTITY" --output-root "$WS/outputs/Fengyu" --cache-dir "$CACHE_DIR" \
    --profile-output "$TRAIN_PROFILE" --training-output "$TRAIN_CONFIG" --run-id "$RUN_ID"

# Fail before optimization if any requested benchmark source is absent.
"$PY" - "$REPO/configs/workflow/experiments/r1_balanced_full_eval.json" "$DATA_ROOT" <<'PY'
import json,sys
from pathlib import Path
tasks=json.loads(Path(sys.argv[1]).read_text())['method_tasks']['importance_fps']
catalog=json.loads(Path('configs/workflow/tasks.json').read_text())['tasks']
root=Path(sys.argv[2])
missing=[]
for task in tasks:
    pattern=catalog[task]['pattern']
    found=list(root.glob(pattern))
    if task == 'mmstar' and not found:
        found=list(root.glob('MMStar/data/*.parquet'))
    if not found:
        missing.append((task,pattern))
if missing:
    raise SystemExit('Missing evaluation datasets before training: '+repr(missing))
print(f'All {len(tasks)} requested evaluation task sources are present.')
PY

# One-GPU optimization is intentional; all four A100s are used by the resident eval workers below.
"$PY" "$REPO/scripts/workflow.py" train --profile "$TRAIN_PROFILE" \
    --training-config "$TRAIN_CONFIG" --run-id "$RUN_ID" --run-dir "$TRAIN_RUN"

CHECKPOINT=$("$PY" - "$TRAIN_RUN/arms/R1_balanced_data_scratch/train/latest.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
print((p.parent/d['checkpoint']).resolve())
PY
)
[[ -s "$CHECKPOINT/projector.safetensors" ]] || { echo "Missing trained checkpoint: $CHECKPOINT" >&2; exit 2; }

# Continue with the complete benchmark list, reusing the existing dense references.
export OLD_MLP="$CHECKPOINT"
export RUN_DIR="$EVAL_RUN"
export CACHE_DIR="$WS/Fengyu/cache/qwen25_r1_balanced_eval"
export EXPERIMENT_CONFIG="$REPO/configs/workflow/experiments/r1_balanced_full_eval.json"
export METHODS=importance_fps
export TASKS="gqa okvqa scienceqa_img ai2d mmstar mmmu pope hallusionbench textvqa docvqa ocrbench refcoco_bbox_rec_testA refcoco_bbox_rec_testB refcoco+_bbox_rec_testA refcoco+_bbox_rec_testB refcocog_bbox_rec_test"
export RESULT_BUNDLE="$RESULT_BUNDLE"
export SEED_RUN_DIRS="$WS/outputs/qwen25_remaining_dynamic_testdev_20261101:$WS/outputs/qwen25_remaining_benchmarks_20261031"
bash "$REPO/scripts/eval/lmms_eval/lmu_qwen25_resident_queue.sbatch"
