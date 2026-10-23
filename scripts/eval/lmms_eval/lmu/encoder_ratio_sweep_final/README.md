# Audio Encoder 三方法、四档保留率：指定五任务版

三个 SH 均按 **keep10 → keep25 → keep75 → keep50** 执行。每档依次运行：

`audiomarathon2 → mmaunew3 → mmsufinal → mmarfinal → mmau_pro`

因此每个 SH 默认启动 20 次 benchmark 评估，三个 SH 共 60 次；任务组内部的子任务由 lmms-eval 展开。某次评估失败时脚本立即停止。

## 文件

- `run_encoder_topk.sh`：Encoder Attention TopK。
- `run_atome.sh`：A-ToMe fixed-ratio 相邻合并。
- `run_progressive_fixed.sh`：Progressive Pruning fixed 模式。
- `common.bash`：三个入口共用的模型解析、LoRA 合并和任务调度。
- `configs/<method>/keep<ratio>/<benchmark>.yaml`：60 份完整平台配置，可分别检查和运行。

新目录放在项目的 `scripts/eval/lmms_eval/lmu/encoder_ratio_sweep_final/`。原有算法和配置不需改动；运行时复用上一层的 `run_mmau_pro_stage.sh` 及项目内 lmms-eval。复制到服务器时保持这个目录位置。


## 任务名与运行前提

本目录使用这五个 task/group：`mmsufinal`、`mmarfinal`、`audiomarathon2`、`mmaunew3`、`mmau_pro`。用户所称的 `mmaupro` 对应本地已注册名称 `mmau_pro`。

`mmsufinal` 复用项目已有的 MMSU 最终版任务，读取 AudioMCQ 的 `MMSU.jsonl`，使用现有 MMSU 字母答案协议。请确认服务器已有 `lmms_eval/tasks/mmsufinal/`，且任务 YAML 的 `dataset_kwargs.data_files.test` 指向正确 JSONL；设置 `MMSUFINAL_DATASET_DIR` 指向包含 `data/*.parquet` 和 `_audio_cache/` 的 MMSU 数据目录。`--plan_only` 不验证任务数据是否可读。

`mmarfinal` 已有任务定义，读取 AudioMCQ 版 MMAR JSONL，并从原 MMAR Parquet 目录补充统计元数据。请确认服务器 `lmms_eval/tasks/mmarfinal/mmarfinal.yaml` 的 `dataset_kwargs.data_files.test` 指向正确 JSONL，并设置 `MMARFINAL_DATASET_DIR` 指向包含 `data/` 和 `_audio_cache/` 的 MMAR 数据目录。本次只新增评估配置与启动入口，不修改任务包的数据源或评分方法。

## 参数含义

keep10/25/75/50 指**目标最终音频 token 保留率**。实际值受音频长度、chunk、逐层取整和 pooler 影响，并非每条样本都精确达到目标。

| 目标最终保留率 | TopK 每层 keep_ratio | A-ToMe 每层 merge_ratio | Progressive 每阶段 pruning_rate | 单个初始 100-token chunk 的最终音频 token 保留率 |
| --- | --- | --- | --- | --- |
| 10% | 0.470 | 0.220 | 0.530 | 10% |
| 25% | 0.640 | 0.140 | 0.360 | 24% |
| 75% | 0.915 | 0.035 | 0.085 | 74% |
| 50% | 0.800 | 0.075 | 0.200 | 50% |

- TopK：零基层 `[3, 6, 9]`，attention residual 后、FFN 前剪枝，每个 chunk 最少保留 2 个 token。
- A-ToMe：`criterion.policy: fixed_ratio`，`first_layer: 2`、`layer_interval: 3`；32 层 encoder 对应 `[2,5,8,11,14,17,20,23,26,29]`。使用 key cosine 对不相交相邻 pair 排序，选中的 hidden states 取均值。不是 threshold 模式。
- Progressive：`schedule.policy: fixed`，零基层 `[6,12,18]`，完整 block 后剪枝，每个 chunk 最少保留 2 个 token；配置中不保留原 linear 模式的首末 rate。

### 10% 档的计算依据

这些参数作用于每个选中的 encoder 层，不应直接把三者的参数都填成 0.10。忽略取整时，TopK 三层的总保留率约为 `keep_ratio^3`，Progressive 三阶段约为 `(1-pruning_rate)^3`，A-ToMe 十层约为 `(1-merge_ratio)^10`。对应 10% 的连续估算分别是 0.46416、0.53584、0.20567；本档按与既有表格相同的初始 100-token chunk、结合实际取整规则选取 0.470、0.530、0.220。

- TopK `keep_ratio: 0.47`：`100 → 47 → 22 → 10`。
- Progressive fixed `pruning_rate: 0.53`：每阶段保留 47%，同样为 `100 → 47 → 22 → 10`。
- A-ToMe `merge_ratio: 0.22`：`100 → 78 → 61 → 48 → 38 → 30 → 24 → 19 → 15 → 12 → 10`。

三者均经 stride-2 pooler 得到 5 个音频 token，相比未压缩的 50 个为 10%。这只是长度预算核验，不是实际 benchmark 校准：例如单个初始 50-token chunk，TopK / Progressive 最终为 2/25=8%，A-ToMe 为 3/25=12%。尾部 chunk、最小保留数、多 chunk 汇总后再池化都会影响最终比例，应以 debug 中每条音频实际输出数为准。

普通四组任务保留现有 LMU 配置的 `generation.use_cache: true`；MMAU-Pro 保留 `generation: {}` 和 `force_simple: true`，由任务定义提供生成协议。只对被评估的 Qwen 模型启用压缩，judge 和 embedding evaluator 沿用原流程。

## 运行

在服务器上已有的 Qwen2.5-Omni 评估环境中运行。先替换下面的绝对路径：

```bash
cd /absolute/path/to/MLLM-Comp-Platform
export SWEEP_DIR="$PWD/scripts/eval/lmms_eval/lmu/encoder_ratio_sweep_final"

export PYTHON_BIN=python
export MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-or-merged-checkpoint
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RUN_TAG="encoder-sweep-$(date +%Y%m%d_%H%M%S)"

export AUDIOMARATHON_TASK_DIR=/absolute/path/to/audiomarathon2
export AUDIOMARATHON_ROOT=/absolute/path/to/datasets/AudioMarathon
export MMSUFINAL_DATASET_DIR=/absolute/path/to/mmsu
export MMARFINAL_DATASET_DIR=/absolute/path/to/mmar
export MMAU_PRO_QWEN_JUDGE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct
export MMAU_PRO_NVEMBED_MODEL=/absolute/path/to/NV-Embed-v2

bash "$SWEEP_DIR/run_encoder_topk.sh"
bash "$SWEEP_DIR/run_atome.sh"
bash "$SWEEP_DIR/run_progressive_fixed.sh"
```

每条 SH 都可单独运行。使用同一组 GPU 时顺序执行这些命令。

完整模型也可用 `CHECKPOINT_PATH` 指定。若评估 LoRA adapter，清空 `MODEL_PATH` 和 `CHECKPOINT_PATH`，改为：

```bash
unset MODEL_PATH CHECKPOINT_PATH
export ADAPTER_PATH=/absolute/path/to/adapter/checkpoint-N
# 可选：export MERGED_MODEL_PATH=/absolute/path/to/merged-model
bash "$SWEEP_DIR/run_encoder_topk.sh"
```

每个 SH 在开始扫比例之前解析模型；默认将 LoRA 导出到 `${ADAPTER_PATH}-merged`，该目录已有 `config.json` 时复用。需要重新合并时使用一个新的 `MERGED_MODEL_PATH`。

输出默认位于：

```text
<project>/runs/lmu/eval/<RUN_TAG>/audio_encoder_sweep_final/
  encoder_topk/keep10/audiomarathon2/
  encoder_topk/keep10/...
  encoder_topk/keep25/audiomarathon2/
  encoder_topk/keep25/mmaunew3/
  ...
  encoder_topk/keep75/...
  encoder_topk/keep50/...
  atome/...
  progressive_fixed/...
```

每个 benchmark 都有独立 `manifest.json` 和评估产物。可设置 `OUTPUT_DIR` 替换上述 `audio_encoder_sweep_final` 这一层及其之前的路径；脚本始终继续追加 `<method>/keep<ratio>/<benchmark>`，避免三种方法混写。新实验请使用新的 RUN_TAG 或 OUTPUT_DIR。

## 先检查，再小样本运行

```bash
# 检查全部 20 次平台启动计划；不加载模型、不合并 LoRA、不运行 judge。
bash "$SWEEP_DIR/run_encoder_topk.sh" --plan_only

# 单 GPU，先对 MMAR 的一个样本逐档验证；按 10 → 25 → 75 → 50 执行。
CUDA_VISIBLE_DEVICES=0 EVAL_NUM_PROCESSES=1 BENCHMARKS=mmarfinal \
  bash "$SWEEP_DIR/run_encoder_topk.sh" --sample_limit 1 --debug
```

其他两个 SH 使用相同参数。`--dry_run` 等同于 `--plan_only`。计划模式只验证平台配置与原生 generation 启动参数，不验证数据可读性或 MMAU-Pro evaluator 环境。实际包含 MMAU-Pro 的运行会在开始前要求两个 evaluator 模型变量。

`BENCHMARKS` 是空格分隔的任务组名，可选集合为上面的五组；例如 `BENCHMARKS="mmarfinal mmsufinal"`。比例顺序始终固定。`--sample_limit 1` 对任务组的每个子任务分别限制样本，并不代表整组总共只有一个样本。

`--debug` 产出的 `debug_diagnostic*.jsonl` 应显示正确 plugin、active layers、逐层 token 减少，以及 `kept_audio_tokens < original_audio_tokens`。正式评估默认不开 debug；效率测量可用 `--profile`，并与相同模型和生成协议的 baseline 比较。

## 其他设置

- 进程数优先级：`EVAL_NUM_PROCESSES` → `--num_processes` → 可见 GPU 数。脚本保证实际启动使用同一个解析后的进程数。
- MMAU-Pro 可单独设置 `MMAU_PRO_CUDA_VISIBLE_DEVICES`、`MMAU_PRO_NUM_GPUS`；未设置时继承普通评估的 GPU 和进程数。
- 默认普通评估端口：TopK 从 29634 开始、A-ToMe 从 29644 开始、Progressive 从 29654 开始；MMAU-Pro 分别为 29734、29744、29754。可用 `AUDIOMARATHON_MAIN_PROCESS_PORT`、`MMAUNEW3_MAIN_PROCESS_PORT`、`MMSUFINAL_MAIN_PROCESS_PORT`、`MMARFINAL_MAIN_PROCESS_PORT`、`MMAU_PRO_MAIN_PROCESS_PORT` 覆盖。
- `--debug`、`--profile`、`--sample_limit`、生成长度等平台 eval 参数会传给每次评估。配置、任务、模型、输出路径和压缩身份由 sweep 管理，不通过同名 CLI 参数覆盖；使用上面列出的环境变量。
- AudioMarathon2 任务包默认从项目上两层的 `audiomarathon2/` 查找；音频默认从该位置的 `datasets/AudioMarathon` 查找。`DATASET_ROOT` 可替换 datasets 根目录，显式设置 `AUDIOMARATHON_ROOT` 优先。
- 新配置使用任务名 `audiomarathon2`、`mmaunew3`、`mmsufinal`、`mmarfinal`、`mmau_pro`。其余任务的数据路径仍由服务器上的 lmms-eval task YAML 和任务代码提供。
