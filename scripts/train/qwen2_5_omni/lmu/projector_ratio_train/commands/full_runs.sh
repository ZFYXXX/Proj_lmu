#!/usr/bin/env bash
set -euo pipefail

command_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${command_dir}/../server_env.sh"
TRAIN_DIR="${PROJECT_ROOT}/scripts/train/qwen2_5_omni/lmu/projector_ratio_train"
cd "${PROJECT_ROOT}"

# Run all 38 settings sequentially, or copy an individual command.
# Smoke commands set --max_steps 1; full commands train for one epoch.

# 3B: 18 compression settings + FastSLM HTA.
bash "${TRAIN_DIR}/run_one.sh" train segmentwise_topk 3b 10
bash "${TRAIN_DIR}/run_one.sh" train segmentwise_topk 3b 25
bash "${TRAIN_DIR}/run_one.sh" train segmentwise_topk 3b 50
bash "${TRAIN_DIR}/run_one.sh" train segmentwise_topk 3b 75

bash "${TRAIN_DIR}/run_one.sh" train ltbm 3b 10
bash "${TRAIN_DIR}/run_one.sh" train ltbm 3b 25
bash "${TRAIN_DIR}/run_one.sh" train ltbm 3b 50
bash "${TRAIN_DIR}/run_one.sh" train ltbm 3b 75

bash "${TRAIN_DIR}/run_one.sh" train sa_map 3b 10
bash "${TRAIN_DIR}/run_one.sh" train sa_map 3b 25
bash "${TRAIN_DIR}/run_one.sh" train sa_map 3b 50

bash "${TRAIN_DIR}/run_one.sh" train unseg 3b natural

bash "${TRAIN_DIR}/run_one.sh" train unisamp 3b 10
bash "${TRAIN_DIR}/run_one.sh" train unisamp 3b 25
bash "${TRAIN_DIR}/run_one.sh" train unisamp 3b 50

bash "${TRAIN_DIR}/run_one.sh" train uniavg 3b 10
bash "${TRAIN_DIR}/run_one.sh" train uniavg 3b 25
bash "${TRAIN_DIR}/run_one.sh" train uniavg 3b 50

bash "${TRAIN_DIR}/run_one.sh" train fastslm_hta 3b default

# 7B: 18 compression settings + FastSLM HTA.
bash "${TRAIN_DIR}/run_one.sh" train segmentwise_topk 7b 10
bash "${TRAIN_DIR}/run_one.sh" train segmentwise_topk 7b 25
bash "${TRAIN_DIR}/run_one.sh" train segmentwise_topk 7b 50
bash "${TRAIN_DIR}/run_one.sh" train segmentwise_topk 7b 75

bash "${TRAIN_DIR}/run_one.sh" train ltbm 7b 10
bash "${TRAIN_DIR}/run_one.sh" train ltbm 7b 25
bash "${TRAIN_DIR}/run_one.sh" train ltbm 7b 50
bash "${TRAIN_DIR}/run_one.sh" train ltbm 7b 75

bash "${TRAIN_DIR}/run_one.sh" train sa_map 7b 10
bash "${TRAIN_DIR}/run_one.sh" train sa_map 7b 25
bash "${TRAIN_DIR}/run_one.sh" train sa_map 7b 50

bash "${TRAIN_DIR}/run_one.sh" train unseg 7b natural

bash "${TRAIN_DIR}/run_one.sh" train unisamp 7b 10
bash "${TRAIN_DIR}/run_one.sh" train unisamp 7b 25
bash "${TRAIN_DIR}/run_one.sh" train unisamp 7b 50

bash "${TRAIN_DIR}/run_one.sh" train uniavg 7b 10
bash "${TRAIN_DIR}/run_one.sh" train uniavg 7b 25
bash "${TRAIN_DIR}/run_one.sh" train uniavg 7b 50

bash "${TRAIN_DIR}/run_one.sh" train fastslm_hta 7b default
