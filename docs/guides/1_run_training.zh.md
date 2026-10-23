# 运行训练

> 类型：Guide
>
> 文档层级：1 / 平台使用者
>
> 目标结果：通过 ms-swift 完成一次已有能力范围内的训练，并获得可复核的 checkpoint 与 Manifest
>
> 适用范围：当前已声明的 baseline SFT、LoRA 与 compression-aware training 路径
>
> 当前状态：Active

`mllm-comp train` 负责验证平台配置、构造 ms-swift 原生 `swift sft` 启动参数、激活
可选 compression runtime，并在训练结束后收集原生产物。模型 template、dataset
解析、optimizer 和 trainer 生命周期仍由固定版本的 ms-swift 负责。

## 1. 最小训练

两个教学入口默认使用 `data/text_sft_smoke.jsonl`，执行一步 LoRA SFT：

```bash
QWEN25_OMNI_MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B \
bash examples/train/train_qwen25_omni_sft.sh --plan_only

QWEN25_OMNI_MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B \
bash examples/train/train_qwen25_omni_sft.sh
```

Qwen3-VL 对应入口为 `examples/train/train_qwen3_vl_sft.sh`。`--plan_only` 只用于检查
argv、环境和预期产物；第二条真实命令才会加载模型并执行训练。

默认文本 fixture 只能证明生产训练入口和 checkpoint 生命周期，不证明 image、audio、
video 数据模板或训练质量。选择正式配置前先阅读对应状态页。

## 2. 完整流程

### 选择支持范围

先区分四种结论：

| 状态 | 可以说明什么 |
| --- | --- |
| inference active | 推理路径已经激活，不能推导训练可用 |
| `train_step_verified` | 一步真实压缩、loss、gradient 与 optimizer 已验证 |
| checkpoint reload evidence | 保存并重载后仍满足相同合同 |
| 完整训练/质量证据 | 指定数据和协议下的训练收敛与评测结果 |

ImageLLM 的 72 个 `train_step_verified` 单元和 Qwen2.5-Omni 分层证据不是同一个支持
声明。精确范围分别见[ImageLLM 状态](../support/imagellm/status.zh.md)和
[Qwen2.5-Omni 状态](../support/qwen25_omni/status.zh.md)。

### 选择配置和数据

- 最小 baseline：`examples/train/`；
- Qwen2.5-Omni 按输入模态选择的训练：`configs/train/qwen2_5_omni/`；
- ImageLLM 历史一步训练：`configs/train/imagellm/`；
- 新增三基座 DOP-V：[`configs/comparison/`](../../configs/comparison/)，其他新增方法按[接入页](../support/imagellm/comparison_methods.zh.md)构造 stage；
- Qwen2.5-VL / Qwen3-VL 图像分支 Attention/FFN 替换：[Qwen2.5-VL SFT 配方](../../configs/modality_split/qwen25_vl_sft.yaml)，合同和其他配方见[分模态说明](../../mllm_comp/modality_split/README.md)；
- 双基座 MWOP 头、路径、FFN 遮罩：[MWOP（Mask 形式）](fixed_mask.zh.md)；
- 正式 shell 入口：`scripts/train/qwen2_5_omni/`。

新增 Bifocal 一步 LoRA 配方见[独立状态与配置](../support/imagellm/status.zh.md#bifocal-接入)。
这些新增入口不继承历史 ImageLLM 的 72 个训练证据，真实 GPU Swift 训练/恢复需分别验证。

Qwen2.5-Omni 当前训练范围是 Thinker，Talker 训练未激活。训练数据必须符合当前
ms-swift template 和模态路径约定；不要用“文件可以解析”替代一次真实 batch 运行。

常见覆盖：

```bash
MLLM_COMP_TRAIN_DATASET=/absolute/path/to/train.jsonl \
MLLM_COMP_TRAIN_RUN_DIR=/absolute/path/to/output \
bash examples/train/train_qwen25_omni_sft.sh --max_steps 10
```

Qwen2.5-Omni 音频压缩训练使用相应正式入口，例如：

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B \
DATASET_PATH=/absolute/path/to/train_audio.jsonl \
OUTPUT_DIR=/absolute/path/to/segmentwise_output \
bash scripts/train/qwen2_5_omni/train_aud_sft_segmentwise_topk.sh
```

### 从一步训练扩大到正式训练

先完成一个 batch 的 forward/backward、finite non-zero gradient 和 optimizer step，
再扩大 `max_steps`、数据和 GPU 数。结构或动态选择方法还要验证 gradient checkpoint
replay 不会重复选择或破坏布局。

正式训练必须使用独立输出目录。恢复训练时只复用明确的 checkpoint，不要把新的运行
覆盖到一次失败或协议不同的目录中。

## 3. 输出与验收

根据训练器和配置，一次通过的训练至少应包含：

- ms-swift 原生日志与 `logging.jsonl`；
- `trainer_state.json`；
- checkpoint 或明确的保存记录；
- 平台 Manifest；
- effective dataset、model、launcher 与 train args；
- compressed 训练中的 stage、token/structure evidence 和适用 invariant。

“loss 有限”不是完整验收。声明 `train_step_verified` 时还要有真实压缩、非零有限梯度、
optimizer step、布局修复和适用的 checkpoint replay 证据；声明完整可恢复训练时还要
完成 checkpoint reload。

## 4. 能力边界

- 配置存在或 `--plan_only` 成功不等于训练已验证；
- 一步 smoke 不证明收敛和质量；
- LoRA、全参数和 compression-aware training 是不同支持面；
- video/audio-video 与各模态 LoRA 不能由 image/audio baseline 外推；
- profile 和逐层 debug 会改变资源使用，不应混入正式吞吐结论。

新增训练能力属于[开发压缩方法](2_dev_compression_method.zh.md)或
[开发 Backbone](2_dev_backbone.zh.md)的工作；已有训练失败按
[运行时排错](3_troubleshoot_runtime.zh.md)处理。
