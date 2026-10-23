#!/usr/bin/env bash
set -euo pipefail

command_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${command_dir}/../server_env.sh"
TRAIN_DIR="${PROJECT_ROOT}/scripts/train/qwen2_5_omni/lmu/encoder_ratio_train"
cd "${PROJECT_ROOT}"

# Each line is one complete training run. Run this file to execute all 24 sequentially,
# or copy an individual line after sourcing server_env.sh and setting TRAIN_DIR.
bash "${TRAIN_DIR}/run_one.sh" train encoder_topk 3b 10
bash "${TRAIN_DIR}/run_one.sh" train encoder_topk 3b 25
bash "${TRAIN_DIR}/run_one.sh" train encoder_topk 3b 50
bash "${TRAIN_DIR}/run_one.sh" train encoder_topk 3b 75
bash "${TRAIN_DIR}/run_one.sh" train encoder_topk 7b 10
bash "${TRAIN_DIR}/run_one.sh" train encoder_topk 7b 25
bash "${TRAIN_DIR}/run_one.sh" train encoder_topk 7b 50
bash "${TRAIN_DIR}/run_one.sh" train encoder_topk 7b 75

bash "${TRAIN_DIR}/run_one.sh" train atome 3b 10
bash "${TRAIN_DIR}/run_one.sh" train atome 3b 25
bash "${TRAIN_DIR}/run_one.sh" train atome 3b 50
bash "${TRAIN_DIR}/run_one.sh" train atome 3b 75
bash "${TRAIN_DIR}/run_one.sh" train atome 7b 10
bash "${TRAIN_DIR}/run_one.sh" train atome 7b 25
bash "${TRAIN_DIR}/run_one.sh" train atome 7b 50
bash "${TRAIN_DIR}/run_one.sh" train atome 7b 75

bash "${TRAIN_DIR}/run_one.sh" train progressive_fixed 3b 10
bash "${TRAIN_DIR}/run_one.sh" train progressive_fixed 3b 25
bash "${TRAIN_DIR}/run_one.sh" train progressive_fixed 3b 50
bash "${TRAIN_DIR}/run_one.sh" train progressive_fixed 3b 75
bash "${TRAIN_DIR}/run_one.sh" train progressive_fixed 7b 10
bash "${TRAIN_DIR}/run_one.sh" train progressive_fixed 7b 25
bash "${TRAIN_DIR}/run_one.sh" train progressive_fixed 7b 50
bash "${TRAIN_DIR}/run_one.sh" train progressive_fixed 7b 75
