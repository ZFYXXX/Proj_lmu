# Qwen2.5-Omni Encoder 压缩 LoRA 训练

本目录提供 Encoder Attention TopK、A-ToMe 和 Progressive Pruning 的 3B/7B LoRA 训练配置。每个方法有一个 `run_*.sh`，每条命令只运行一个模型和一个目标保留率，共有 24 份独立 YAML。

`10/25/50/75` 表示**最终音频 token 的目标保留率**，不是每层直接保留或删除的比例。参数沿用 `scripts/eval/lmms_eval/lmu/encoder_ratio_sweep_final/` 的四档设置。短音频、chunk、逐层取整和 pooler 会改变实际保留率。

| 目标保留率 | Encoder TopK 每层 `keep_ratio` | A-ToMe 每层 `merge_ratio` | Progressive 每阶段 `pruning_rate` |
| --- | ---: | ---: | ---: |
| 10% | 0.470 | 0.220 | 0.530 |
| 25% | 0.640 | 0.140 | 0.360 |
| 50% | 0.800 | 0.075 | 0.200 |
| 75% | 0.915 | 0.035 | 0.085 |

Progressive 使用 `policy: fixed`，对应评估目录的 `progressive_fixed` 方法。三种方法的训练阶段均为 `execution_phase: forward`、`compression_mode: compressed`。LoRA 参数按本次实验要求固定为 rank 8、alpha 32、学习率 `2e-5`、每卡 batch 8、梯度累积 4、ZeRO-2。3B 保存上限为 4，7B 为 8。

## 服务器上的 24 条正式命令与 24 条单步测试命令

服务器路径和你提供的环境变量集中在 [`server_env.sh`](server_env.sh)。7B 默认从原始 `Qwen2.5-Omni-7B` 启动；3B 路径按同一权重目录下的 `Qwen2.5-Omni-3B` 填写，首次运行前请确认该目录存在。`ROOT_IMAGE_DIR` 会原样传给训练进程。`MASTER_PORT=29504` 作为端口基数，运行器为 24 个组合加上不同偏移。

```bash
export PROJECT_ROOT=/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform
source "$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/encoder_ratio_train/server_env.sh"
TRAIN_DIR="$PROJECT_ROOT/scripts/train/qwen2_5_omni/lmu/encoder_ratio_train"
cd "$PROJECT_ROOT"

# 单个正式训练：3B Encoder TopK，目标保留 10%
bash "$TRAIN_DIR/run_one.sh" train encoder_topk 3b 10

# 单个真实单步测试：7B A-ToMe，目标保留 25%
bash "$TRAIN_DIR/run_one.sh" smoke atome 7b 25

# 只检查启动配置，不加载模型
bash "$TRAIN_DIR/run_one.sh" smoke progressive_fixed 3b 75 --plan_only
```

24 条**逐条可复制**的正式命令在 [`commands/full_runs.sh`](commands/full_runs.sh)，24 条 `--max_steps 1` 单步测试命令在 [`commands/smoke_tests.sh`](commands/smoke_tests.sh)。这两个文件也可以直接用 `bash` 执行，分别按顺序运行全部 24 组。单条命令只占用本次分配的 GPU；若在多个终端并行启动，请给各终端分配不重叠的 `CUDA_VISIBLE_DEVICES`。

`run_one.sh` 调用对应方法的 `run_*.sh`，并使用 `tee` 保留完整训练输出。默认路径分开：

```text
日志：<PROJECT_ROOT>/logs/lmu/encoder_ratio_lora/<RUN_TAG>/<train|smoke>/<method>/keep<ratio>/<3b|7b>_lora.log
结果：<PROJECT_ROOT>/runs/lmu/encoder_ratio_lora/<RUN_TAG>/<train|smoke>/<method>/keep<ratio>/<3b|7b>_lora/
```

可分别设置 `LOG_ROOT` 和 `RESULT_ROOT` 改变这两个目录的前缀。`smoke` 与 `train` 的结果目录相互独立。只有同一个方法、比例及模型的 LoRA 训练中断后，才把该 run 的 checkpoint 作为 `--resume /path/to/checkpoint-N` 传给单条 `train` 命令；给出的 7B 全参数 SFT `checkpoint-1100` 不会被自动用于这些新 LoRA run。若希望新 run 从其他完整模型权重初始化，使用对应的 `MODEL_PATH_3B` 或 `MODEL_PATH_7B`。

也可直接调用三个方法脚本：`run_encoder_topk.sh`、`run_atome.sh` 和 `run_progressive_fixed.sh`，分别接受 `<3b|7b> <10|25|50|75>` 与后续训练 CLI 参数。直接调用时需自行设置模型、数据、输出环境变量。

已有真实训练证据只覆盖部分 3B 全参数单步路径；本目录的 3B/7B LoRA 四档组合尚需实际运行验证。
