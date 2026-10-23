# 可运行训练示例

本目录保存可以直接交给 `mllm-comp train` 的最小训练示例。模型、数据集和输出路径
由配套 shell 脚本设置为环境变量；
迁移到其他机器时应覆盖环境变量，不要把机器路径写回 YAML。

配套脚本保留 `MLLM_COMP_TRAIN_RUN_DIR` 协议，它是精确输出目录，每次实验需换新值。
新建普通配方可以省略 `output`，使用平台自动生成的运行目录。`train.resume` 指定恢复
输入，不复用旧实验目录；实际 checkpoint 与训练日志路径以本次结果中的后端输出为准。

平台公共字段见[配置字段参考](../../docs/reference/config.zh.md)；`train.args` 中的
训练参数由 ms-swift 拥有。

- `qwen25_omni_sft.yaml`：Qwen2.5-Omni-3B Thinker 的一步 LoRA SFT；配套入口为
  `examples/train/train_qwen25_omni_sft.sh`。
- `qwen3_vl_sft.yaml`：Qwen3-VL-4B `language_model` 的一步 LoRA SFT；配套入口为
  `examples/train/train_qwen3_vl_sft.sh`。

两个脚本默认使用 `data/text_sft_smoke.jsonl`，目的是低成本验证生产训练链路：

```text
mllm-comp train
  -> 构造原生 swift sft argv/environment
  -> ms-swift 加载模型、模板和数据集
  -> 执行一步 LoRA SFT
  -> 保存 checkpoint/logging.jsonl/trainer_state.json
  -> 平台收集原生产物并写入 Manifest
```

先检查实际命令而不启动训练：

```bash
bash examples/train/train_qwen25_omni_sft.sh --plan_only
bash examples/train/train_qwen3_vl_sft.sh --plan_only
```

确认计划后执行默认的一步训练：

```bash
bash examples/train/train_qwen25_omni_sft.sh
bash examples/train/train_qwen3_vl_sft.sh
```

常用覆盖方式：

```bash
MLLM_COMP_TRAIN_DATASET=/path/to/train.jsonl \
MLLM_COMP_TRAIN_RUN_DIR=/path/to/output \
bash examples/train/train_qwen25_omni_sft.sh --max_steps 10
```

该最小数据集只有文本，不证明 image/video/audio 数据模板或训练质量正确。Qwen2.5-Omni
当前只声明 Thinker 训练，脚本固定关闭 audio output，不支持 Talker 训练；Qwen3-VL 当前
只声明 `language_model` 训练。完整的 Qwen2.5-Omni 分模态训练配置位于
`configs/train/qwen2_5_omni/`。
