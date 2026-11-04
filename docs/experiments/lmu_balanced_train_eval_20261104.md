# LMU: balanced Qwen2.5-VL training followed by five-category evaluation

This is a portable, single-job workflow for the answer-independent 2,560-image cohort frozen on P6000. The cohort has four training source domains (OCR/document/chart 35%, general 25%, grounding 20%, science 20%); the evaluation covers the five benchmark categories General, Reasoning, Hallusion, OCR, and Grounding. Reported category averages are calculated from task results, not separate datasets. This suite intentionally follows the user's listed datasets and does not add ChartQA.

Training runs one arm only: `R1_balanced_data_scratch` (importance-guided FPS, mean cluster pooling, zero-residual MLP initialization, 320 optimizer steps, global batch 8). The current trainer is single-GPU. After training, the same Slurm allocation uses all four A100s as resident evaluation workers with a shared chunk queue. It evaluates the trained `importance_fps` method; existing dense reference cells are reused from the configured P6000/LMU result sources. MMMU remains single-image-only, and HallusionBench prediction generation is retained for its configured offline scoring.

## Prepare the data bundle on P6000

Run from the repository root with the project's Python environment. This creates a portable directory plus a compressed archive; it does not upload anything.

```bash
PY=/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python
SRC=/home/ZhouFengyu/fastv-repro/worktrees/mlp-next/outputs/training_data/next_subset2560_20261104_v2
OUT=/home/ZhouFengyu/fastv-repro/worktrees/mlp-next/outputs/lmu_upload/qwen25_balanced2560_20261104
ARCHIVE=/home/ZhouFengyu/fastv-repro/worktrees/mlp-next/outputs/lmu_upload/qwen25_balanced2560_20261104.tar.gz
mkdir -p "$(dirname "$OUT")"
"$PY" scripts/train/qwen25_vl/package_balanced_lmu_data.py package \
  --source "$SRC" --output "$OUT" --archive "$ARCHIVE"
```

The frozen archive is transferred through the private dataset repository `ZFFFY/Proj-training` (not GitHub). If the bundle is absent, the batch job downloads it into workspace storage after receiving a compute-node allocation. It checks the pinned archive SHA256, extracts to `/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/inputs/qwen25_balanced2560_20261104`, verifies the bundle files, and relocates its paths before training. An existing partial extraction is preserved for inspection.

## Run on LMU

The repository code is synced by the user. In the project root on LMU, make the Slurm output directory before submitting (Slurm opens those files before the job starts):

```bash
mkdir -p logs/slurm
read -rsp "HF token: " PROJ_TRANSFER_TOKEN
echo
export PROJ_TRANSFER_TOKEN
sbatch scripts/train/qwen25_vl/lmu_balanced_train_eval.sbatch
unset PROJ_TRANSFER_TOKEN
```

Use a token with read access to `ZFFFY/Proj-training`. No token is stored in source or command history. Skip the prompt/export if the complete training bundle already exists on LMU. This job-specific variable is removed after the download stage, before training/evaluation subprocesses start.

The published LMU branch includes the required 2.4 MB baseline reuse file `artifacts/eval/qwen25_p6000_reuse_20261103.json.gz`. It carries dense references for several suite tasks that are not in the old LMU run folders. Training images remain a separate workspace data transfer.

The job uses account `hk-project-p0024767`, partition `accelerated`, one node, four A100 GPUs, 32 CPUs, 23:50 wall time, job-name prefix `28-`, offline logging, and no email. It writes the trained checkpoint and run logs under `$WS/outputs/Fengyu/` and the evaluation run under a sibling `_eval` directory. The job fails fast if required datasets, model/checkpoint paths, bundle integrity, or the four-GPU allocation are missing.

The training data itself is not committed to Git. The prepared bundle is about 674 MB before archive compression; keep it in workspace storage, not `/home`.
