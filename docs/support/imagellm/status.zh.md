# ImageLLM 多算法当前状态

> 更新日期：2026-10-03；同步 Bifocal、14 组对比方法及各自验证边界。以下历史真实模型矩阵证据来自 2026-08-19，不表示已在当前提交全量重跑。
> 权威来源：公共多模态 registry；历史图像声明在 `mllm_comp/compressors/multimodal_registry.py`，新增对比方法声明在 `mllm_comp/compressors/comparison/registry.py`

## 当前结论

当前图像 Token 目录包括原 25 个算法、26 个 variant、75 个三基座 binding，仅接入
Qwen2.5-VL 的 Bifocal，以及本轮新增的 7 个 Token algorithm ID、21 个三基座 binding。
合计 33 个图像 Token algorithm ID、34 个 variant、97 个 binding。新增项没有真实 checkpoint 证据。
MWOP（Mask 形式）与分模态 Attention/FFN 通过各自的 Operation 路径接入，不计入这里的 Token 方法矩阵。
另有本轮新增的 10 个 Operation ID、30 个三基座 binding。

[OV 对比方法接入](comparison_methods.zh.md) 共 14 组、17 个 ID、51 个 binding，覆盖
LLaVA-OneVision、Qwen2.5-VL、Qwen3-VL。已通过原生小模型 CPU 训练、缓存、生成及公开入口检查；
SupportUnit 保持 `declared`。Twig 训练权重、Short-LVLM 外部 SCP 校准及真实模型质量/速度边界见该页。

以下表格仅描述原 25 个算法的历史矩阵：除 D2Pruner 外，24 个算法均有真实 checkpoint
推理证据，并在每个基座上选择一个精确 phase 完成了压缩感知一步训练验证，共 72 个
`train_step_verified` 单元。新增方法不继承这些证据。

| 能力 | 状态 |
|---|---|
| inference SupportUnit | 78 active：36 forward、42 generate |
| 一步压缩感知训练 | 72 active 且 `train_step_verified`，每个基座 24 个 |
| 完整训练闭环 | 未声明；所有单元的 `train_verified` 仍为 false |
| D2Pruner | 6 个 forward/generate 单元均 declared；可加载外部校准 prior，未计入真实模型矩阵 |

`train_step_verified` 只表示一个真实 optimizer step 已通过。它不表示完整训练、收敛、checkpoint reload、训练后评测或 selector 可学习。

## 历史验证矩阵与 phase

下表记录原验证矩阵使用的 SupportUnit phase。当前训练配置使用 `forward`；生成时的
prefill 与增量 decode 是不同阶段，不能将表中的 `generate` 理解为在 decode 中重复压缩。

| 类别 | 算法 | 一步训练使用的 phase |
|---|---|---|
| forward | AIM、DivPrune、GAP、MMTok、PruMerge、SAINT、ToMe、VisionZip、VisPruner、ZOO-Prune | forward |
| forward + generate | PACT、PyramidDrop | forward；generate 仍只有推理证据 |
| generate | BTP、DART、FastV、FEATHER、FinePruner、HalfV、IVC-Prune、Nüwa、RESTORE、SGL、SparseVLM、TransPrune | generate 的无 KV-cache teacher forcing |
| declared | D2Pruner | 无 |

training-free 是运行方式，不是算法能力上限。同一 kernel、variant 和 binding 同时服务预训练权重上的压缩 eval/generate 与压缩生效的 training forward，没有第二份训练实现。

## Bifocal 接入

`bifocal.image.qwen25_vl` 在视觉 merger 之后、LLM scatter 之前执行图像压缩，
支持 `forward` 和 `generate` 的执行合同。它保留文本，使用显式稀疏 M-RoPE，并同步
图像占位符、mask 和 labels；当前 binding 限定单图，不声明 LLaVA-OneVision 或 Qwen3-VL 支持。

- 评测配置：[Qwen2.5-VL 14 benchmark Bifocal](../../../configs/eval/lmms_eval/qwen25_vl_14bench_bifocal.yaml)。
- 训练配置：[Qwen2.5-VL Bifocal 一步 LoRA](../../../configs/train/imagellm/qwen25_vl_bifocal_train_step.yaml)。
- 现有 CPU 测试覆盖 kernel、梯度、精确 binding、执行桥及配置/Swift 启动计划，见
  [Bifocal 测试](../../../tests/unit/test_bifocal_image_kernel.py)。

两个 phase 的 SupportUnit 均为 `declared`、没有已登记的真实模型 evidence；配置和测试
不证明完整 benchmark、真实 Swift 训练/恢复或质量/效率收益。它不属于下文 72 个历史训练案例，
也不由只生成 active 配方的 `render_imagellm_config` 提供；研究运行使用上述专用配置。

## 图像 Operation 入口

- [MWOP（Mask 形式）](../../guides/fixed_mask.zh.md)：LLaVA-OneVision / Qwen2.5-VL 的头、路径、FFN
  遮罩及清单生成；129 项 CPU 验证，精确 `prepare` 单元仍为 `declared`。
- [分模态模块替换](../../../mllm_comp/modality_split/README.md)：Qwen2.5-VL / Qwen3-VL
  图像分支 Attention/FFN 及训练恢复入口；精确 `.image_llm` 单元仍为 `declared`。

- [新增对比 Operation](comparison_methods.zh.md)：DOP/DOP-V、ShortV、Skip-Vision、RedundancyLens、YOPO、AdaLLaVA、Short-LVLM、Structural-Pruning 的三基座路径。

各类 Operation 的真实 GPU 训练/质量边界分别维护，不由本页图像 Token 的历史证据推导。

## 一步训练证据

三个固定 checkpoint：

- Qwen2.5-VL：`Qwen2.5-VL-3B-Instruct`
- Qwen3-VL：`Qwen3-VL-2B-Instruct`
- LLaVA-OneVision：`llava-onevision-qwen2-0.5b-ov`

包含 Qwen2.5-Omni 在内的四基座代表验证权重统一记录在
[`docs/progress.md`](../../progress.md#代表验证权重)；Omni 仍使用既有独立实现，
不属于本节的通用 ImageLLM 三基座矩阵。

固定数据为 `EITNLP-Img-Post-dataset/train_10k.jsonl` 的首个非空样本。每个案例均为 batch size 1、梯度累积 1、一步 SGD，并验证：

- LayoutDelta 真实缩短或合并 token；
- finite loss、finite non-zero gradient 与参数更新；
- labels、mask、position 和布局同步修复；
- `use_cache=False` teacher forcing；
- reentrant gradient-checkpoint 重放产生相同压缩布局；
- 选择过程可在 `no_grad` 下执行，但保留或合并后的 hidden states 未 detach。

证据文件位于 `outputs/imagellm/train_step/*_train_step_20260819.json`。它们分别对应精确基座矩阵，不用于宣称质量或速度收益。
此外，ToMe 已分别通过三个基座的公开 `mllm-comp train` 一步入口，均产生
token-reducing forward/backward 证据、checkpoint 和 Manifest。

## D2Pruner 边界

D2Pruner 已有 kernel、variant、三个 binding 和外部 prior 加载路径。研究运行需在
`stage.config.bias_prior_path` 指定实际校准文件；框架目前不提供 prior 校准流程或
随包分发的 prior。

加载发生在安装压缩 hook 之前。文件需要包含 `bias` 张量、`model_id`（或 `model_path`）
和 `scorer_layer`；模型身份必须与 runtime 一致，评分层为 `prune_after_layer - 1`。
缺少文件、身份或层不匹配会明确报错，不使用 uniform prior 代替。

原生小模型已覆盖 prior 加载、位置、训练梯度和缓存路径；这些测试不证明校准质量，
也不替代真实 checkpoint 支持证据。六个 SupportUnit 仍为 declared，不计入 72 个
真实训练案例。`render_imagellm_config` 只生成已有验证配方，仍排除 D2Pruner；
这不等于公共运行入口禁止用户提供 prior 的合法研究配置。

## 配置入口

以下脚本面向历史 active 矩阵，可为三个基座生成 24 个方法的 eval 配置和 23 个方法的 Swift train 配置。
SGL 仅有 generate Binding，其原生 teacher-forcing 一步证据不代表已接通 Swift 的
forward 训练入口；生成器会明确拒绝 SGL 的 train 请求。

```bash
python -m scripts.train.render_imagellm_config \
  --algorithm fastv \
  --backbone qwen3_vl \
  --mode train \
  --model_path /path/to/Qwen3-VL-2B-Instruct \
  --dataset /path/to/train.jsonl \
  --output_dir outputs/qwen3_fastv_train \
  --output /tmp/qwen3_fastv_train.yaml

mllm-comp train --config /tmp/qwen3_fastv_train.yaml
```

将 `--mode` 改为 `eval` 即生成 training-free 配置。仓库还保留一份可直接运行的 [Qwen2.5-VL ToMe 一步训练示例](../../../configs/train/imagellm/qwen25_vl_tome_train_step.yaml)。
LLaVA-OneVision 的训练配置会自动使用仓库内 `third_party/backbones/LLaVA-NeXT`，
满足原版 checkpoint 的 Swift loader 要求；同时只关闭 vision-tower checkpointing，
保留 language-model gradient checkpointing，避免视觉 ToMe callback 被重放。

## 不应外推的能力

- D2Pruner 的校准质量与真实 checkpoint 正式支持；
- 完整训练、收敛、checkpoint reload 或训练后质量；
- batch size 大于 1、多图联合压缩、video/audio ImageLLM；
- 质量无损、端到端加速或显存收益；
- Qwen2.5-Omni 的 AudioLLM 证据不属于本页；其 14 个方法现已复用同一多模态 v2
  contract/session，但 Qwen-specific executor 与支持状态见独立状态页。
