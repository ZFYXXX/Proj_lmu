# LMU/KIT 四卡无压缩训练基线

这四个入口用于 Qwen2.5-Omni 音频无压缩基线：3B/7B 分别运行 full SFT 与 LoRA。
它们必须在一个已经取得 4 张 GPU 的 Slurm allocation 内执行。

## 固定训练口径

- 4 张 GPU，`per_device_train_batch_size=1`，`gradient_accumulation_steps=32`；
- 全局 batch size = `1 x 4 x 32 = 128`；
- bfloat16、FlashAttention、gradient checkpointing、最大长度 4096；
- full SFT 使用 ZeRO-3，LoRA 使用 ZeRO-2；
- 冻结视觉/音频塔与 aligner，只训练 Thinker 的 LLM 或其 LoRA adapter；
- `compression_mode=none`，不启用任何 token compression stage；
- 默认 W&B offline；可用 `WANDB_MODE=online` 覆盖。

四个 shell 都是独立的薄入口：设置运行环境后直接调用各自的 YAML，不依赖公共
launcher。脚本不会申请 Slurm allocation，也不会检查 GPU 是否已被其他进程占用。

## allocation 内的公共设置

```bash
cd /absolute/path/to/MLLM-Comp-Platform
export DATASET_PATH=/absolute/path/to/audio_train.jsonl
export PYTHON_BIN=/absolute/path/to/env/bin/python
export RUN_TAG=baseline-$(date +%Y%m%d-%H%M%S)
```

Hugging Face 模型默认从已登录账号的缓存解析；也可以对每条命令设置
`MODEL_PATH=/absolute/path/to/downloaded/model`。Token 不应写入脚本。

## 先做单步显存定标

两个 tmux/allocation 同时最多各跑一组，不能在同一个四卡 allocation 内并行两组。

```bash
WANDB_MODE=disabled bash scripts/train/qwen2_5_omni/lmu/train_aud_3b_lora.sh --max_steps 1
WANDB_MODE=disabled bash scripts/train/qwen2_5_omni/lmu/train_aud_3b_sft.sh --max_steps 1
WANDB_MODE=disabled bash scripts/train/qwen2_5_omni/lmu/train_aud_7b_lora.sh --max_steps 1
WANDB_MODE=disabled bash scripts/train/qwen2_5_omni/lmu/train_aud_7b_sft.sh --max_steps 1
```

单步通过后，使用新的 `RUN_TAG` 并移除 `--max_steps 1` 即为完整 1 epoch 训练。
每个 run 会保存 Swift checkpoint/log 与平台 Manifest。用新的 `RUN_TAG` 或显式
`OUTPUT_DIR` 区分重复运行，避免覆盖已有结果。
