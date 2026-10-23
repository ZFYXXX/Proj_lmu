#!/usr/bin/env bash
# 在 Bash 中 source 本文件；现有环境变量优先于下面的服务器默认值。
# 所有默认服务器路径来自本项目已有运行文档；按实际机器修改。
export AUDIO_PROBES_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd -- "$AUDIO_PROBES_DIR/../../.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B}"
export DATASET_ROOT="${DATASET_ROOT:-/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset}"
export MMSUFINAL_DATASET_DIR="${MMSUFINAL_DATASET_DIR:-$DATASET_ROOT/mmsu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# 下面一项留空时，使用服务器 mmsufinal task YAML 中已经配置的数据路径。
# 如果 task YAML 仍含本机 /Users/... 路径，取消注释并填入服务器真实路径。
# export MMSU_JSONL=/absolute/server/path/AudioMCQevaluator/MMSU.jsonl

# 每次完整运行使用新输出目录；也可在 source 前自行 export ANALYSIS_RUN_DIR。
export ANALYSIS_RUN_DIR="${ANALYSIS_RUN_DIR:-$PROJECT_ROOT/runs/analysis/audio_four_probes/omni_full_$(date +%Y%m%d_%H%M%S)}"

# 独立 Taylor 入口默认取 CUDA_VISIBLE_DEVICES 的第一张卡，输出到 ANALYSIS_RUN_DIR/hierarchical_taylor。
# 可按需指定一张卡和独立的新输出目录：
# export TAYLOR_GPU=0
# export TAYLOR_RUN_DIR=/absolute/server/path/runs/taylor_full_run
