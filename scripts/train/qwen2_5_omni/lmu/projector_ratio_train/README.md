# Qwen2.5-Omni Projector 压缩 LoRA 训练

本目录参照相邻的 `encoder_ratio_train/`，提供七种 Projector-Level 方法的 3B/7B LoRA 训练入口。每条命令运行一个模型、一个方法和一个设置，共 **38 份独立 YAML**：每个模型 18 个普通压缩设置，加 1 个 FastSLM HTA 设置。

| 方法 / 入口 | 设置参数 | 每个模型的配置数 | 压缩位置 |
| --- | --- | ---: | --- |
| Segmentwise Top-K / `run_segmentwise_topk.sh` | `10`、`25`、`50`、`75` | 4 | projector 前 |
| LTBM / `run_ltbm.sh` | `10`、`25`、`50`、`75` | 4 | projector 前 |
| SA-MAP / `run_sa_map.sh` | `10`、`25`、`50` | 3 | projector 后、LLM 前 |
| UnSeg / `run_unseg.sh` | `natural` | 1 | projector 前 |
| UniSamp / `run_unisamp.sh` | `10`、`25`、`50` | 3 | projector 前 |
| UniAvg / `run_uniavg.sh` | `10`、`25`、`50` | 3 | projector 前 |
| FastSLM HTA / `run_fastslm_hta.sh` | `default` | 1 | projector 前 |

## 训练参数

所有配置均为 `execution_phase: forward`、`compression_mode: compressed`，采用参考目录的 LoRA 参数：rank 8、alpha 32、学习率 **2e-5**、每卡 batch 8、梯度累积 4、1 epoch、BF16、FlashAttention、gradient checkpointing 和 ZeRO-2。LLM LoRA 可训练，Audio Encoder 与原生 projector 冻结。3B 的 `save_total_limit` 为 4，7B 为 8；每 500 步评估和保存。

FastSLM 额外配置 `modules_to_save: [_mllm_comp_fastslm_hta_audio]`，使可学习的 HTA 模块全参数训练并随 checkpoint 保存；其 LLM 仍使用上述 LoRA 参数。HTA 从新初始化权重开始，默认 `window_tokens: 750`、`output_query_tokens: 50`、`keep_ratio: 1/15`，名义保留率约 **6.67%**。这是一组压缩感知 SFT 配置，未组织原论文的三阶段训练数据。

## 压缩参数

- Segmentwise Top-K：`num_segments: 10`、`allocation: balanced_exact`，`keep_ratio` 分别为 0.10、0.25、0.50、0.75。
- LTBM：`window_size: 8`，四个 `keep_ratio` 与上面相同。
- SA-MAP：沿用现有评估设置，`similarity_threshold: 0.80`、`attention_power: 1.0`；只提供 10%、25%、50%。实际运行仍受现有算法的分组数量约束，须通过真实单步测试确认数据可运行。
- UniSamp / UniAvg：整数 `compression_factor` 分别为 10、4、2，对应保留 10%、25%、50%。没有 75% 配置，因为该比例不对应整数步长。
- UnSeg：`budget_policy: natural_peaks`、`boundary_detector: strict_local_peak`，不设置 `keep_ratio`。约 32–33% 是先前记录中的观察值，不是本训练的固定比例或保证。

百分比均相对压缩入口处的音频 token 数，表示保留率。每条音频的整数取整会影响实际比例；UniSamp 输出通常按 `ceil(N/K)`，UniAvg 按 `max(1, floor(N/K))` 计算。所有配置只启用一个 projector stage。

## 启动

服务器模型、数据、Python 和 GPU 默认设置集中在 [`server_env.sh`](server_env.sh)，路径沿用 `encoder_ratio_train/server_env.sh`。模型默认从原始 3B/7B 权重开始，首次运行前确认服务器文件存在。可在 `source` 前设置同名环境变量覆盖默认值；`ROOT_IMAGE_DIR` 原样传给训练进程。

```bash
export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
source "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train/server_env.sh"
TRAIN_DIR="$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/projector_ratio_train"
cd "$PROJECT_ROOT"

# 配置与启动计划检查，不加载模型；run_one.sh 会保留检查日志。
bash "$TRAIN_DIR/run_one.sh" smoke segmentwise_topk 3b 10 --plan_only

# 一个真实单步测试。
bash "$TRAIN_DIR/run_one.sh" smoke ltbm 7b 25

# 自然分段的正式训练。
bash "$TRAIN_DIR/run_one.sh" train unseg 3b natural

# FastSLM HTA + 7B LLM LoRA 的正式训练。
bash "$TRAIN_DIR/run_one.sh" train fastslm_hta 7b default
```

38 条逐条可复制的正式训练命令在 [`commands/full_runs.sh`](commands/full_runs.sh)，38 条 `--max_steps 1` 单步测试命令在 [`commands/smoke_tests.sh`](commands/smoke_tests.sh)。也可直接执行这两个文件，按 3B、7B 顺序运行全部组合；任何一组失败即停止。

```bash
bash "$TRAIN_DIR/commands/smoke_tests.sh"
bash "$TRAIN_DIR/commands/full_runs.sh"
```

完整的 **38 段 Slurm 正式训练提交命令**在 [`commands/slurm_full_runs.sh`](commands/slurm_full_runs.sh)，按 3B 19 组、7B 19 组排列。每段都是独立可复制的 `mkdir` + `sbatch <<'SBATCH'`，含全部环境变量；也可在服务器执行下面的命令，一次提交 38 个独立作业，由 Slurm 调度：

```bash
bash "$TRAIN_DIR/commands/slurm_full_runs.sh"
```

Slurm 命令使用用户指定的 `accelerated` 分区、账户 `hk-project-p0024767`、单节点 4 GPU、32 CPU、24 小时时限，排除 `hkn0632,hkn0409`，数据集为 `/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/AudioMCQ_570k_swift_latest.jsonl`。提交前创建 `logs/lmu/projector_ratio_lora/slurm/`，GPU 编号沿用 Slurm 分配；`MASTER_PORT_BASE=29502` 配合现有组合偏移。各作业的 `RUN_TAG` 含配置名称与 `SLURM_JOB_ID`，训练结束后立即读取 `PIPESTATUS[0]` 并作为作业退出状态。

各方法脚本也可直接调用，例如 `bash "$TRAIN_DIR/run_uniavg.sh" 3b 25 --plan_only`。直接调用时自行设置 `DATASET_PATH`，以及 `MODEL_PATH` 或对应的 `MODEL_PATH_3B` / `MODEL_PATH_7B`。参数 `natural` 仅用于 UnSeg，`default` 仅用于 FastSLM HTA，其他方法使用表中的数字比例。

## 输出与恢复

`run_one.sh` 使用 `tee` 保存完整训练输出，并把 smoke 与正式训练放在不同目录：

```text
日志：<PROJECT_ROOT>/logs/lmu/projector_ratio_lora/<RUN_TAG>/<train|smoke>/<method>/<setting>/<3b|7b>_lora.log
结果：<PROJECT_ROOT>/runs/lmu/projector_ratio_lora/<RUN_TAG>/<train|smoke>/<method>/<setting>/<3b|7b>_lora/
```

`<setting>` 为 `keep10`、`keep25`、`keep50`、`keep75`、`natural` 或 `default`。每个组合都有独立输出目录和 Manifest。可用 `LOG_ROOT`、`RESULT_ROOT` 修改前缀；直接调用方法脚本时，`OUTPUT_DIR` 是各方法输出的公共根目录。

只有恢复同一方法、设置、模型的训练时，才传入 `--resume /path/to/checkpoint-N`；新 run 从其他完整模型权重初始化时，设置 `MODEL_PATH_3B` 或 `MODEL_PATH_7B`。旧的全参数 SFT checkpoint 不会自动用于新 LoRA run。

端口以 `MASTER_PORT_BASE` 为基数，默认 29540，为 38 个组合分别添加偏移，默认有效端口位于 29540–29589。若只提供 `MASTER_PORT`，它也作为基数。多个终端并行运行时仍需分配不重叠的 `CUDA_VISIBLE_DEVICES`；同一组合并行运行还需不同端口基数、`RUN_TAG` 或输出根目录。

本目录新增启动脚本和配置，复用平台已有压缩算法。2026-10-01 本地检查通过：38 份 YAML 的方法参数与 LoRA 参数校验、12 个 shell 文件的语法检查、38 组正式训练与 38 组 smoke 的真实 CLI 启动计划、独立输出路径与端口、无效组合拒绝及训练失败状态通过 `tee` 的传递。配置与计划检查使用本地 Python 3.13 / PyYAML 6.0.3；服务器环境仍沿用参考目录的 Python 3.10。

以上检查未加载真实模型或启动 GPU 训练，不代表 forward/backward、显存占用、checkpoint 保存或训练收敛已验证；先运行对应 smoke 命令再启动正式训练。
