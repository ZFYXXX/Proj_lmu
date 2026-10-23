#!/usr/bin/env bash
set -euo pipefail

command_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${command_dir}/../server_env.sh"
TRAIN_DIR="${PROJECT_ROOT}/scripts/train/qwen2_5_omni/lmu/encoder_ratio_train"
cd "${PROJECT_ROOT}"

# Each line trains one step and checks the real compressed LoRA forward/backward.
# Run this file to test all 24 sequentially, or copy an individual line.
bash "${TRAIN_DIR}/run_one.sh" smoke encoder_topk 3b 10
bash "${TRAIN_DIR}/run_one.sh" smoke encoder_topk 3b 25
bash "${TRAIN_DIR}/run_one.sh" smoke encoder_topk 3b 50
bash "${TRAIN_DIR}/run_one.sh" smoke encoder_topk 3b 75
bash "${TRAIN_DIR}/run_one.sh" smoke encoder_topk 7b 10
bash "${TRAIN_DIR}/run_one.sh" smoke encoder_topk 7b 25
bash "${TRAIN_DIR}/run_one.sh" smoke encoder_topk 7b 50
bash "${TRAIN_DIR}/run_one.sh" smoke encoder_topk 7b 75

bash "${TRAIN_DIR}/run_one.sh" smoke atome 3b 10
bash "${TRAIN_DIR}/run_one.sh" smoke atome 3b 25
bash "${TRAIN_DIR}/run_one.sh" smoke atome 3b 50
bash "${TRAIN_DIR}/run_one.sh" smoke atome 3b 75
bash "${TRAIN_DIR}/run_one.sh" smoke atome 7b 10
bash "${TRAIN_DIR}/run_one.sh" smoke atome 7b 25
bash "${TRAIN_DIR}/run_one.sh" smoke atome 7b 50
bash "${TRAIN_DIR}/run_one.sh" smoke atome 7b 75

bash "${TRAIN_DIR}/run_one.sh" smoke progressive_fixed 3b 10
bash "${TRAIN_DIR}/run_one.sh" smoke progressive_fixed 3b 25
bash "${TRAIN_DIR}/run_one.sh" smoke progressive_fixed 3b 50
bash "${TRAIN_DIR}/run_one.sh" smoke progressive_fixed 3b 75
bash "${TRAIN_DIR}/run_one.sh" smoke progressive_fixed 7b 10
bash "${TRAIN_DIR}/run_one.sh" smoke progressive_fixed 7b 25
bash "${TRAIN_DIR}/run_one.sh" smoke progressive_fixed 7b 50
bash "${TRAIN_DIR}/run_one.sh" smoke progressive_fixed 7b 75
