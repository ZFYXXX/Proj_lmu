# 7B LoRA：六种 Projector 方法的十二组终端命令

更新时间：2026-09-19。每组命令独立设置环境，在拥有 GPU 0–3 的计算节点运行；六种方法顺序执行。先通过对应的小样本测试，再运行完整评估。

当前 Segmentwise Top-K、LTBM、SA-MAP 为 active；UniAvg、UniSamp、UnSeg 为 cpu_contract_verified，开发运行已启用。UnSeg preflight 已修复。远端 checkpoint、五任务评估尚未由本地测试证明；SA-MAP 的 threshold=0.80 仍可能因分组数量不足而失败。

SA-MAP 已移除 75% 档位，下面第 6、12 组现在只运行 10%、25%、50%。若只运行 50%，将这两组中的 `run_sa_map.sh` 替换为 `run_sa_map_keep50.sh`，并给 `RUN_TAG` 加上 `keep50` 便于区分；完成提示相应改为“50% 保留率”。

小样本参数 --sample_limit 1 对任务组中的每个子任务生效；不等于整组仅一条数据。它会执行真实模型和 MMAU-Pro evaluator。--debug 用于核对实际压缩报告。

远端需同步 mllm_comp/、本 projector_ratio_sweep_final/ 目录以及上一层 run_mmau_pro_stage.sh。若运行新增 pytest 验证，同步 tests/。本地 mmsufinal/mmarfinal task YAML 的 JSONL 路径含 /Users/...，同步这些任务前应改成服务器上的实际路径；保留服务器已验证的数据路径。AudioMarathon task 包须位于 AUDIOMARATHON_TASK_DIR。

## 1. Segmentwise Top-K：小样本测试

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="smoke-7b-lora-projector-segmentwise-topk-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval_smoke

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_segmentwise_topk.sh" \
  --num_processes 4 --use_cache --sample_limit 1 --debug \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：Segmentwise Top-K 五个任务、四档保留率（10%、25%、75%、50%）的小样本测试。\n'
)
```

## 2. UniAvg：小样本测试

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="smoke-7b-lora-projector-uniavg-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval_smoke

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_uniavg.sh" \
  --num_processes 4 --use_cache --sample_limit 1 --debug \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：UniAvg 五个任务、三档保留率（10%、25%、50%）的小样本测试。\n'
)
```

## 3. UniSamp：小样本测试

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="smoke-7b-lora-projector-unisamp-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval_smoke

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_unisamp.sh" \
  --num_processes 4 --use_cache --sample_limit 1 --debug \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：UniSamp 五个任务、三档保留率（10%、25%、50%）的小样本测试。\n'
)
```

## 4. UnSeg：小样本测试

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="smoke-7b-lora-projector-unseg-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval_smoke

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_unseg.sh" \
  --num_processes 4 --use_cache --sample_limit 1 --debug \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：UnSeg 五个任务、自然分段的小样本测试。\n'
)
```

## 5. LTBM：小样本测试

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="smoke-7b-lora-projector-ltbm-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval_smoke

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_ltbm.sh" \
  --num_processes 4 --use_cache --sample_limit 1 --debug \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：LTBM 五个任务、四档保留率（10%、25%、75%、50%）的小样本测试。\n'
)
```

## 6. SA-MAP：小样本测试

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="smoke-7b-lora-projector-sa-map-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval_smoke

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_sa_map.sh" \
  --num_processes 4 --use_cache --sample_limit 1 --debug \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：SA-MAP 五个任务、三档保留率（10%、25%、50%）的小样本测试。\n'
)
```

## 7. Segmentwise Top-K：完整评估

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="eval-7b-lora-projector-segmentwise-topk-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_segmentwise_topk.sh" \
  --num_processes 4 --use_cache \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：Segmentwise Top-K 五个任务、四档保留率（10%、25%、75%、50%）的完整评估。\n'
)
```

## 8. UniAvg：完整评估

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="eval-7b-lora-projector-uniavg-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_uniavg.sh" \
  --num_processes 4 --use_cache \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：UniAvg 五个任务、三档保留率（10%、25%、50%）的完整评估。\n'
)
```

## 9. UniSamp：完整评估

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="eval-7b-lora-projector-unisamp-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_unisamp.sh" \
  --num_processes 4 --use_cache \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：UniSamp 五个任务、三档保留率（10%、25%、50%）的完整评估。\n'
)
```

## 10. UnSeg：完整评估

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="eval-7b-lora-projector-unseg-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_unseg.sh" \
  --num_processes 4 --use_cache \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：UnSeg 五个任务、自然分段的完整评估。\n'
)
```

## 11. LTBM：完整评估

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="eval-7b-lora-projector-ltbm-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_ltbm.sh" \
  --num_processes 4 --use_cache \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：LTBM 五个任务、四档保留率（10%、25%、75%、50%）的完整评估。\n'
)
```

## 12. SA-MAP：完整评估

```bash
(
set -euo pipefail

export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export EVAL_NUM_PROCESSES=4
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export AUDIOMARATHON_ROOT="$DATASET_ROOT/AudioMarathon"
export AUDIOMARATHON_TASK_DIR="$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/audiomarathon2"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

export MMAU_PRO_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export MMAU_PRO_NUM_GPUS=4
export MMAU_PRO_QWEN_JUDGE_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/NV-Embed-v2
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY=true
export MMAU_PRO_EVALUATOR_DEVICE=auto
export MMAU_PRO_EVALUATOR_DTYPE=auto
export MMAU_PRO_NVEMBED_DEVICE_MAP=balanced
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS=512
export MMAU_PRO_JUDGE_TEMPERATURE=0.1

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="eval-7b-lora-projector-sa-map-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_eval

cd "$PROJECT_ROOT"
mkdir -p logs/lmu-eval
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("cuDNN:", torch.backends.cudnn.version())'

bash "$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/run_sa_map.sh" \
  --num_processes 4 --use_cache \
  2>&1 | tee "$PROJECT_ROOT/logs/lmu-eval/${RUN_TAG}.log"

printf '完成：SA-MAP 五个任务、三档保留率（10%、25%、50%）的完整评估。\n'
)
```
