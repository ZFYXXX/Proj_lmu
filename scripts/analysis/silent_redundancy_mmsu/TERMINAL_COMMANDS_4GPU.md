# MMSU-Final Silent Redundancy：四卡服务器命令

下面的命令适配 `/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform` 和现有 `lbw-comp-platform` 环境，可直接粘贴到终端。

## 推荐：先运行 8 题 smoke test

```bash
export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"
export PYTHONUNBUFFERED=1

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_PROCESSES=4
export MAIN_PROCESS_PORT=29649

export MMSUFINAL_DATASET_DIR=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/mmsu
unset MMSU_JSONL MMSUFINAL_JSONL

export CHECKPOINT_PATH=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B

export RUN_DIR="$PROJECT_ROOT/runs/analysis/silent_redundancy_mmsufinal_7b/smoke_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="$RUN_DIR"
export ANALYSIS_RUN_DIR="$RUN_DIR"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$RUN_DIR")"

bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh --limit 8 \
    2>&1 | tee "$RUN_DIR.log"
```

Smoke test 会执行全部四阶段，但只取前 8 题，共进行 `8 × 20 = 160` 次生成。检查以下内容后再启动全量实验：

- 日志中没有 Transformers 版本、FlashAttention 或数据路径错误；
- `$RUN_DIR/observe/coverage.json` 的 `complete` 为 `true`；
- 19 个 `$RUN_DIR/conditions/*/coverage.json` 均为 `true`；
- `$RUN_DIR/report/observational/` 和 `$RUN_DIR/report/final/` 已生成图表与 CSV。

## 完整 5,000 题实验

环境设置与上面相同，只需使用新的输出目录，并去掉 `--limit 8`：

```bash
export RUN_DIR="$PROJECT_ROOT/runs/analysis/silent_redundancy_mmsufinal_7b/full_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="$RUN_DIR"
export ANALYSIS_RUN_DIR="$RUN_DIR"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$RUN_DIR")"

bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh \
    2>&1 | tee "$RUN_DIR.log"
```

完整套件包含每题 1 次无压缩观察、15 次等预算 pruning 和 4 次因果删除，共 `5,000 × 20 = 100,000` 次生成。

## 分阶段运行与恢复

必须保持相同的 `RUN_DIR`、`CHECKPOINT_PATH` 和样本规模：

```bash
bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh --phase observe \
    2>&1 | tee -a "$RUN_DIR.log"

bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh --phase compare \
    2>&1 | tee -a "$RUN_DIR.log"

bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh --phase causal \
    2>&1 | tee -a "$RUN_DIR.log"

bash scripts/analysis/silent_redundancy_mmsu/run_mmsufinal_4gpu.sh --phase report \
    2>&1 | tee -a "$RUN_DIR.log"
```

已完成且 coverage 完整的条件会自动跳过。若某个条件目录存在但不完整，入口不会覆盖它；先保留该目录和日志用于诊断，再将该目录改名后重跑相应阶段。

`MAIN_PROCESS_PORT` 在本套件中作为端口基数。完整四阶段会顺序使用 29649–29668；同一节点上如果还有其他 Accelerate 作业，应换成一段未占用的端口，例如 `export MAIN_PROCESS_PORT=30100`。
