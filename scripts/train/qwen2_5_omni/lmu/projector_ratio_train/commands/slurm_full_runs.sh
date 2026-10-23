#!/usr/bin/env bash
# 38 independent, fully expanded sbatch commands: 19 x 3B, then 19 x 7B.
# Run this file to submit every job, or copy one complete mkdir + sbatch block.
# Each job uses 4 GPUs allocated by Slurm and trains for the configured 1 epoch.
set -euo pipefail

# 01. 3B | Segmentwise Top-K | 10% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-topk-k10
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-segmentwise-topk-keep10-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train segmentwise_topk 3b 10 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 02. 3B | Segmentwise Top-K | 25% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-topk-k25
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-segmentwise-topk-keep25-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train segmentwise_topk 3b 25 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 03. 3B | Segmentwise Top-K | 50% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-topk-k50
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-segmentwise-topk-keep50-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train segmentwise_topk 3b 50 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 04. 3B | Segmentwise Top-K | 75% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-topk-k75
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-segmentwise-topk-keep75-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train segmentwise_topk 3b 75 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 05. 3B | LTBM | 10% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-ltbm-k10
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-ltbm-keep10-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train ltbm 3b 10 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 06. 3B | LTBM | 25% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-ltbm-k25
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-ltbm-keep25-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train ltbm 3b 25 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 07. 3B | LTBM | 50% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-ltbm-k50
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-ltbm-keep50-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train ltbm 3b 50 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 08. 3B | LTBM | 75% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-ltbm-k75
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-ltbm-keep75-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train ltbm 3b 75 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 09. 3B | SA-MAP | 10% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-samap-k10
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-sa-map-keep10-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train sa_map 3b 10 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 10. 3B | SA-MAP | 25% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-samap-k25
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-sa-map-keep25-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train sa_map 3b 25 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 11. 3B | SA-MAP | 50% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-samap-k50
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-sa-map-keep50-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train sa_map 3b 50 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 12. 3B | UnSeg | natural segmentation
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-unseg-natural
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-unseg-natural-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train unseg 3b natural \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 13. 3B | UniSamp | 10% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-unisamp-k10
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-unisamp-keep10-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train unisamp 3b 10 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 14. 3B | UniSamp | 25% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-unisamp-k25
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-unisamp-keep25-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train unisamp 3b 25 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 15. 3B | UniSamp | 50% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-unisamp-k50
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-unisamp-keep50-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train unisamp 3b 50 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 16. 3B | UniAvg | 10% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-uniavg-k10
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-uniavg-keep10-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train uniavg 3b 10 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 17. 3B | UniAvg | 25% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-uniavg-k25
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-uniavg-keep25-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train uniavg 3b 25 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 18. 3B | UniAvg | 50% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-uniavg-k50
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-uniavg-keep50-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train uniavg 3b 50 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 19. 3B | FastSLM HTA | default HTA
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-3b-hta-default
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-3B
export MODEL_PATH_3B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-3b-projector-fastslm-hta-default-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train fastslm_hta 3b default \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 20. 7B | Segmentwise Top-K | 10% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-topk-k10
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-segmentwise-topk-keep10-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train segmentwise_topk 7b 10 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 21. 7B | Segmentwise Top-K | 25% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-topk-k25
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-segmentwise-topk-keep25-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train segmentwise_topk 7b 25 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 22. 7B | Segmentwise Top-K | 50% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-topk-k50
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-segmentwise-topk-keep50-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train segmentwise_topk 7b 50 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 23. 7B | Segmentwise Top-K | 75% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-topk-k75
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-segmentwise-topk-keep75-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train segmentwise_topk 7b 75 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 24. 7B | LTBM | 10% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-ltbm-k10
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-ltbm-keep10-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train ltbm 7b 10 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 25. 7B | LTBM | 25% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-ltbm-k25
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-ltbm-keep25-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train ltbm 7b 25 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 26. 7B | LTBM | 50% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-ltbm-k50
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-ltbm-keep50-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train ltbm 7b 50 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 27. 7B | LTBM | 75% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-ltbm-k75
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-ltbm-keep75-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train ltbm 7b 75 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 28. 7B | SA-MAP | 10% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-samap-k10
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-sa-map-keep10-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train sa_map 7b 10 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 29. 7B | SA-MAP | 25% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-samap-k25
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-sa-map-keep25-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train sa_map 7b 25 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 30. 7B | SA-MAP | 50% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-samap-k50
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-sa-map-keep50-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train sa_map 7b 50 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 31. 7B | UnSeg | natural segmentation
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-unseg-natural
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-unseg-natural-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train unseg 7b natural \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 32. 7B | UniSamp | 10% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-unisamp-k10
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-unisamp-keep10-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train unisamp 7b 10 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 33. 7B | UniSamp | 25% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-unisamp-k25
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-unisamp-keep25-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train unisamp 7b 25 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 34. 7B | UniSamp | 50% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-unisamp-k50
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-unisamp-keep50-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train unisamp 7b 50 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 35. 7B | UniAvg | 10% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-uniavg-k10
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-uniavg-keep10-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train uniavg 7b 10 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 36. 7B | UniAvg | 25% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-uniavg-k25
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-uniavg-keep25-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train uniavg 7b 25 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 37. 7B | UniAvg | 50% retention
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-uniavg-k50
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-uniavg-keep50-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train uniavg 7b 50 \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

# 38. 7B | FastSLM HTA | default HTA
mkdir -p "/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm" &&
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=hkn0632,hkn0409
#SBATCH --job-name=26-pj-7b-hta-default
#SBATCH --account=hk-project-p0024767
#SBATCH --export=ALL
#SBATCH --output=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform/logs/lmu/projector_ratio_lora/slurm/slurm-%x-%j.out

set -uo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export MODEL_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B
export MODEL_PATH_7B="$MODEL_PATH"
export DATASET_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl
export ROOT_IMAGE_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT_BASE=29502
export MASTER_PORT="$MASTER_PORT_BASE"
export WANDB_MODE=disabled
export RUN_TAG="lmu-7b-projector-fastslm-hta-default-train-${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)"
export LOG_ROOT="$PROJECT_ROOT/logs/lmu/projector_ratio_lora"
export RESULT_ROOT="$PROJECT_ROOT/runs/lmu/projector_ratio_lora"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
echo "job=$SLURM_JOB_ID node=$(hostname) python=$PYTHON_BIN run=$RUN_TAG"

bash "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/run_one.sh" \
  train fastslm_hta 7b default \
  2>&1 | tee "$LOG_ROOT/slurm/${RUN_TAG}.log"

train_status=${PIPESTATUS[0]}
echo "training exit status=$train_status"
exit "$train_status"
SBATCH

