# 7B：全部 encoder 层、全部 head 的 MMSU 观测命令

更新：2026-09-22。本命令使用 **observe**，在完整 mmsufinal 的 5,000 条上每条只生成一次答案，共 **5,000 次生成**。保留全部音频 token，不做逐条件重复推理，也不划分校准集。随后从已保存统计生成全库可视化，以及 Perception / Reasoning 下六子类的报告。

本页保留历史文件名；下面的 checkpoint 来自已完成五层运行保存的实际配置，是基础 `Qwen2.5-Omni-7B`。代码不会自动加载 LoRA。需要分析 LoRA 时，将 `CHECKPOINT_PATH` 改为准确的完整 merged checkpoint，不能只指向 adapter 目录。

服务器已有 suite 作业时，应先在原运行终端用 Ctrl+C 停止它，再启动下方命令。同步新代码不会让已运行的进程自动切换模式。

当前配置与任务对应关系：

- 平台评估 YAML：`scripts/analysis/audio_sink/eval_mmsufinal.yaml`，明确 `tasks: [mmsufinal]`、`pipeline.stages: []`、`compression_mode: none`。
- 实际任务 YAML：`third_party/frameworks/lmms-eval/lmms_eval/tasks/mmsufinal/mmsufinal.yaml`，继续提供数据、prompt、生成参数和 scorer。
- 本次有效配置：`$ANALYSIS_RUN_DIR/eval.resolved.yaml`。入口默认和一键脚本均使用上述专用 YAML，不再加载通用的 `configs/eval/lmms_eval/lmu/eval_aud.yaml`。
- 观察配置：`$ANALYSIS_RUN_DIR/observation.json`，记录 `mode: observe`、实际观察层、案例索引和重建块大小。此入口不创建 suite_plan，不运行 suite_report。

下面沿用已有运行记录的服务器项目、Python 环境、数据目录和 checkpoint 路径。服务器必须保留已经跑通的 task JSONL 配置：`mmsufinal.yaml` 的 `dataset_kwargs.data_files.test` 应指向服务器上真实的 `MMSU.jsonl`。本地 task 文件包含 `/Users/...` 路径，不能直接覆盖服务器的数据路径。`MMSUFINAL_DATASET_DIR` 只选择 Parquet/音频缓存目录，不会覆盖 JSONL 路径。

按实际架构观察全部层和全部 head；Qwen2.5-Omni-7B 为 **L0–L31 × H0–H19 = 640 个组合**。全部有效 queries 和全部 5,000 条题目均参与统计。doc `0,1666,3333` 保存每层第 0 个 chunk 的详细表示图和全部 head attention 矩阵；选层和案例不读取测试答案。K/V cosine 仍是拼接 heads 后的表示相似度。

保留：K/V/hidden 相邻及远距相似度、incoming attention 的集中/熵/位置、head×毫秒距离、虚拟 Global/Segmentwise TopK 时间覆盖，以及 Mel/attention 对齐、Gram 矩阵和 PCA。它们用于描述 sink 候选和 merge 候选，不给出消融、低权重因果贡献或高频偏好的结论。

```bash
(
set -euo pipefail

export PROJECT_ROOT=/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/Bowen/MLLM-Comp-Platform
export DATASET_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset
export PYTHON_BIN=/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10
export PATH="${PYTHON_BIN%/*}:$PATH"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_PROCESSES=4
export MAIN_PROCESS_PORT=29649
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export CHECKPOINT_PATH=/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/ckpts/ckpts-omni/Qwen2.5-Omni-7B

export ANALYSIS_RUN_DIR="$PROJECT_ROOT/runs/analysis/audio_sink/mmsufinal_7b_all_layers_observe_$(date +%Y%m%d_%H%M%S)"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/runs/analysis/audio_sink"

bash "$PROJECT_ROOT/scripts/analysis/audio_sink/run_mmsufinal_observe.sh" \
  2>&1 | tee "${ANALYSIS_RUN_DIR}.log"

printf '结果目录：%s\n' "$ANALYSIS_RUN_DIR"
)
```

一键入口读取 checkpoint 的 config.json 确定全部层；随后用专用 `eval_mmsufinal.yaml` 执行 observe，再运行 `report` 和 `group_report`。实际层/head 覆盖记录在 `runtime.rank*.json`。进度条每完成一条题目就更新；生成 5,000 次答案后评估结束，绘图不再次加载模型。

query-block=128；每 rank 只在首条音频检查 shadow A@V，一共最多 4 条，没有额外生成。虚拟 TopK 保留比例 10%、时间段数 10，仅生成标记索引。主报告 bootstrap 按 audio_path 聚类；六组报告确认路径唯一后按音频记录重采样，1,000 次、seed 42。

输出位置：

- `figures/`：全 32 层相似度、32×20 attention 集中图、全部 head×毫秒距离（每页最多四层，共八页）、Mel 时间对齐、每层表示矩阵/PCA、每层所有 head 的单 chunk attention；PNG/PDF。
- `six_group_analysis/tables.md`：两大类均值和六子类的八列表格；同目录有逐样本/逐层/逐 head CSV 及汇总 JSON。
- `six_group_analysis/figures/`：八类图，包括六组各自完整的 32×20 mass、peak r、块首峰值热图。
- `summary.json`：MMSU 准确率、观察层的相似度/集中/距离统计、bootstrap 区间和重建误差。
- `samples.rank*.jsonl`：逐题预测和观察统计，每完成一题即写入。
- `coverage.json`：完整 5,000 条覆盖检查。
- `${ANALYSIS_RUN_DIR}.log`：完整运行日志，位于结果目录旁边。

旧 suite 的 ATTENTION_FAMILIES、ATTENTION_LAYERS 和 SINK_HEADS 不参与此入口。全层版仍只有 5,000 次生成，但相较五层版，诊断层数为 6.4 倍，统计与绘图量增加；总耗时不会简单按该比例缩放。旧五层结果不能离线补成全层结果。21–23 不随本命令运行，完整默认 suite 需要 56,560,000 次生成。

若只需重画六组报告，在同一项目环境中，将 ANALYSIS_RUN_DIR 设为实际已存在的结果目录，再执行：

```bash
"$PYTHON_BIN" -m scripts.analysis.audio_sink.group_report \
  --run-dir "$ANALYSIS_RUN_DIR" --dataset-dir "$MMSUFINAL_DATASET_DIR"
```

本次只做静态源码和语法检查，没有在本地运行模型、测试、绘图或测速，也没有提交服务器任务，不承诺与原始评估完全同速。
