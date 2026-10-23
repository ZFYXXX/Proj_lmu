# mmsufinal 四项分析与独立 Taylor：运行命令

执行代码、YAML、SH 和本文都在 `scripts/analysis/audio_four_probes/`。独立 Taylor 入口见第 7 节。
将这个目录放在服务器上同一版本的 `MLLM-Comp-Platform/scripts/analysis/` 下即可；
运行仍依赖项目的 `mllm_comp` 和 `third_party/frameworks/lmms-eval`，不是脱离项目的独立 Python 包。

两份 SH 都只运行 `mmsufinal`；这里的 MMSU 就是 `mmsufinal`。四项分析默认覆盖全部 5,000 条；
四个 Probe、全部 Thinker LLM 层、全部 query heads、全部层分别删除 Attention/FFN update。
原始音频不做压缩或截断。每条另生成一次完整答案，用原 task scorer 判分并做正确/错误分组。
输出首 Token 的分析语义、无标签数据、消融作用范围见 [README.zh.md](README.zh.md)。

## 1. 配置环境

在已获得 GPU 的服务器 **Bash** 终端执行。下面 `PROJECT_ROOT` 沿用项目已有服务器路径；
如果部署位置不同，先改第一行。`env.example.sh` 中的 Python 和 checkpoint 路径也来自已有运行文档。

```bash
export PROJECT_ROOT=/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/Bowen/MLLM-Comp-Platform
source "$PROJECT_ROOT/scripts/analysis/audio_four_probes/env.example.sh"
```

检查或按实际位置设置以下数据变量。未设置 `MMSU_JSONL` 时使用原生 mmsufinal task YAML 中的路径；
**如果原生 YAML 仍包含 `/Users/...`，必须通过这些变量指定真实服务器路径。**

| 变量 | 对应内容 |
| --- | --- |
| `MMSU_JSONL` | `MMSU.jsonl` 文件 |
| `MMSUFINAL_DATASET_DIR` | 含 `data/*.parquet` 的 MMSU 根目录 |

`MMSUFINAL_DATASET_DIR` 不替代 `MMSU_JSONL`。入口只在内存中改数据路径，保留原任务的对话、生成参数和 scorer。
使用完整 base/merged checkpoint；不能将 `CHECKPOINT_PATH` 指向仅含 LoRA adapter 的目录。
3B、7B 的层数/head 数自动从 checkpoint 读取，不需要修改 YAML。

依赖使用现有项目环境：Python 3.10.20、Transformers 4.57.3、`datasets`（现有 3.6.x），
以及 PyTorch、qwen-omni-utils、soundfile、tqdm、numpy、matplotlib 和 vendored lmms-eval 的依赖。

## 2. 查看完整执行计划

```bash
bash "$AUDIO_PROBES_DIR/run.sh" --dry-run
```

这一步只读取配置和 checkpoint 的 `config.json`，打印层数、head 数、任务、GPU 分片和预计 forward 次数。
不会加载模型或创建运行目录，也不等价于数据加载或 GPU 验证。

## 3. 首次部署的 GPU 小规模验证

```bash
ANALYSIS_RUN_DIR="$PROJECT_ROOT/runs/analysis/audio_four_probes/smoke_$(date +%Y%m%d_%H%M%S)" \
  bash "$AUDIO_PROBES_DIR/run.sh" --sample-limit 4 --layers 0,1
```

这一步取 mmsufinal 的 4 条样本，第 0、1 层做全部四项分析和两种删除，并真实生成图表。
`--layers` 会同时限制因果层，避免小规模验证意外执行全层删除。样本上限至少等于 GPU 数；
一张卡可设 `CUDA_VISIBLE_DEVICES=0` 和 `--sample-limit 1`。

## 4. 完整运行 mmsufinal 四项分析

```bash
export ANALYSIS_RUN_DIR="$PROJECT_ROOT/runs/analysis/audio_four_probes/omni_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname -- "$ANALYSIS_RUN_DIR")"
set -o pipefail
bash "$AUDIO_PROBES_DIR/run.sh" 2>&1 | tee "${ANALYSIS_RUN_DIR}.log"
```

不传 `--sample-limit`、`--layers`、`--causal-layers` 或 `--probes` 即为四项分析 YAML 的完整默认实验。
mmsufinal 按样本分给所选 GPU。无需指定 benchmark，也可以显式传入 `mmsufinal`；不接受 `all` 或 MMAU 任务。
每卡一个完整模型副本，无跨卡模型切分；不要同时把两个作业分配到同一组 GPU。

7B 默认每条 99 次分析 forward，mmsufinal 全量共约 **495,000 次 forward + 5,000 次完整答案生成**。
3B 默认每条 111 次分析 forward。Attention 保存一层矩阵仍有序列长度平方的显存开销；
全层 representation/residual 也保留多层 hidden。GPU 小规模验证只能确认所选短样本路径，不能保证最长音频不会 OOM。

## 5. 显式指定 mmsufinal 与缩小范围

```bash
# 显式指定 mmsufinal，与默认选择相同。
ANALYSIS_RUN_DIR="$PROJECT_ROOT/runs/analysis/audio_four_probes/mmsu_$(date +%Y%m%d_%H%M%S)" \
  bash "$AUDIO_PROBES_DIR/run.sh" mmsufinal

# 可选：只运行一个 Probe；这不属于默认四项完整实验。
ANALYSIS_RUN_DIR="$PROJECT_ROOT/runs/analysis/audio_four_probes/prediction_$(date +%Y%m%d_%H%M%S)" \
  bash "$AUDIO_PROBES_DIR/run.sh" --probes prediction_dynamics

# 可选：用更多显存减少 attention forward 次数。
ANALYSIS_RUN_DIR="$PROJECT_ROOT/runs/analysis/audio_four_probes/batch2_$(date +%Y%m%d_%H%M%S)" \
  bash "$AUDIO_PROBES_DIR/run.sh" --attention-layer-batch-size 2
```

每次实际运行必须使用新目录。已有目录会被拒绝；当前不提供中途续跑功能。
进程失败、缺样本、缺层、缺 head 或缺干预时不会报告全量成功。

## 6. 查看进度和重新绘图

```bash
tail -f "$ANALYSIS_RUN_DIR/mmsufinal/shard-000/worker.log"
```

每卡有独立日志；每条样本完成其生成和全部分析后更新进度。完成后的主要结果：

```text
$ANALYSIS_RUN_DIR/mmsufinal/report/README.zh.md
$ANALYSIS_RUN_DIR/mmsufinal/report/summary.json
$ANALYSIS_RUN_DIR/mmsufinal/report/figures/
```

只从已有结果重建报告，不加载模型：

```bash
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m scripts.analysis.audio_four_probes.report --run-dir "$ANALYSIS_RUN_DIR/mmsufinal"
```

加 `--no-plots` 可只生成 JSON/CSV。

## 7. 单独运行 hierarchical_taylor

复用第 1 节的环境、checkpoint 和数据变量；使用本目录新增的 `run_hierarchical_taylor.sh`。
本实验只运行 **`mmsufinal` 全部 5,000 条**，全部 LLM 层、heads 和 FFN 神经元。
这里的 MMSU 就是 `mmsufinal`，YAML 明确设置 `tasks: [mmsufinal]`。数据只需 `MMSU_JSONL` 和
`MMSUFINAL_DATASET_DIR`，Taylor 入口不读取或检查 MMAU 数据路径。

使用一张 GPU 汇总原生 Taylor 分数和排名。`TAYLOR_GPU` 可显式指定；
省略时取 `CUDA_VISIBLE_DEVICES` 的第一个 ID，未设置时为 0。无需指定 benchmark，也可以显式传入 `mmsufinal`。

```bash
# 在 Bash 终端加载与四项分析相同的环境。
export PROJECT_ROOT=/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/Bowen/MLLM-Comp-Platform
source "$PROJECT_ROOT/scripts/analysis/audio_four_probes/env.example.sh"
export TAYLOR_GPU=0

# 只查看计划，不加载权重、不创建运行目录。
bash "$AUDIO_PROBES_DIR/run_hierarchical_taylor.sh" --dry-run

# 首次部署：mmsufinal 1 条样本，捕获第 0、1 层，真实执行 forward/backward。
TAYLOR_RUN_DIR="$PROJECT_ROOT/runs/analysis/audio_four_probes/taylor_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash "$AUDIO_PROBES_DIR/run_hierarchical_taylor.sh" --sample-limit 1 --layers 0,1

# 全量：mmsufinal 全部 5,000 条、全部层，独立输出目录。
export TAYLOR_RUN_DIR="$PROJECT_ROOT/runs/analysis/audio_four_probes/taylor_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname -- "$TAYLOR_RUN_DIR")"
set -o pipefail
bash "$AUDIO_PROBES_DIR/run_hierarchical_taylor.sh" 2>&1 | tee "${TAYLOR_RUN_DIR}.log"
```

省略 `TAYLOR_RUN_DIR` 时，结果放到 `$ANALYSIS_RUN_DIR/hierarchical_taylor/`。
每次实际运行需要新目录。运行中的进度和完成后的主要结果：

```bash
tail -f "$TAYLOR_RUN_DIR/mmsufinal/shard-000/worker.log"
```

```text
$TAYLOR_RUN_DIR/mmsufinal/shard-000/analysis/hierarchical_taylor/hierarchical_taylor.task_scores.json
$TAYLOR_RUN_DIR/mmsufinal/shard-000/analysis/hierarchical_taylor/hierarchical_taylor.task_ranks.json
```

显式指定 `mmsufinal` 或调整捕获批大小：

```bash
TAYLOR_RUN_DIR="$PROJECT_ROOT/runs/analysis/audio_four_probes/taylor_mmsu_$(date +%Y%m%d_%H%M%S)" \
  bash "$AUDIO_PROBES_DIR/run_hierarchical_taylor.sh" mmsufinal

# 可选：一次捕获 2 层，减少重复 forward/backward，增加显存开销。
TAYLOR_RUN_DIR="$PROJECT_ROOT/runs/analysis/audio_four_probes/taylor_batch2_$(date +%Y%m%d_%H%M%S)" \
  bash "$AUDIO_PROBES_DIR/run_hierarchical_taylor.sh" --taylor-layer-batch-size 2
```

默认 7B 每条 32 次分析 forward + 32 次 backward，5,000 条共约 160,000 次 forward + 160,000 次 backward，
另有 5,000 次原始答案生成。`--layers` 只限制捕获层，forward/backward 仍经过完整模型。
不进行参数训练更新。默认关闭额外因果删除验证及逐样本神经元 raw 保存，任务级神经元分数完整保存。
Taylor 直接输出原生 scores/ranks JSON；第 6 节的四项分析重绘命令不适用于 Taylor。
