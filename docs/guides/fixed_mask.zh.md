# MWOP（Mask 形式）：LLaVA-OneVision / Qwen2.5-VL

本文描述 **MWOP 的 Mask 形式**：通过固定遮罩禁用选定的头、注意力路径与 FFN 中间激活。
框架中的算法标识为 `fixed_mask`，模态变体为 `fixed_mask.image_text`；这些是 MWOP Mask
实现使用的代码与配置名称，不表示另一个研究方法。
它使用现有模型加载、分析、ms-swift 恢复训练和 lmms-eval 评测入口，保留参数名称、形状和原生 checkpoint 权重格式。
本实现不包含可学习门控、物理剪枝、专用加速 kernel 或专项测速，也不移植 ShortV、RedundancyLens、YOPO、DOP_V。

## 遮罩语义

所有层号和组件编号均从 0 开始，清单中的编号表示**禁用**，不是保留。

| 配置键 | 禁用内容 |
| --- | --- |
| `heads` | 指定查询头在 O 投影前的完整输出；覆盖有效 prefill token |
| `v2v` | 图像 query → 图像 key 的注意力边 |
| `t2v` | 图像区域之后的文本 query → 图像 key 的注意力边 |
| `t2t` | 图像区域之后的文本 query → 同一区域文本 key 的注意力边 |
| `ffn_all` | 所有有效 token 的指定 FFN 中间激活 |
| `ffn_vision` | 图像 token 的指定 FFN 中间激活 |
| `ffn_text` | 所有非图像有效 token 的指定 FFN 中间激活，包括图像之前的文本 |

路径方向采用 query → key；路径文本范围与 FFN 文本范围不同。GQA 模型按查询头编号，不修改 KV 头。
路径遮罩在 softmax 前施加，保留因果约束；某个 query/head 的全部 key 被屏蔽时，其注意力输出为零。
头遮罩位于 `o_proj` 输入，FFN 遮罩位于 `down_proj` 输入，因而恢复训练时也覆盖这些投影上的 LoRA 分支。

当前使用 `mask_scope: prefill_only`：生成时只屏蔽 prefill，后续有 KV cache 的 decode 不施加新 Mask。
teacher forcing 的完整训练序列作为一次 prefill，包含答案 token。
验证范围为单样本、图像输入、eager/SDPA；不支持视频、packing、padding-free、FlashAttention 或融合替换这些投影的内核。
LLaVA 使用原生多模态展开后的图像位置，包含 AnyRes 图像块和图像换行 token。

这是稠密计算上的功能屏蔽。统计的禁用比例不是参数体积、显存或延迟减少比例，路径 Mask 还可能增加内存开销。

## 从已有 Taylor 分析生成清单

输入是现有 `hierarchical_taylor.task_scores.json`，支持其扁平头/路径评分与紧凑 FFN 向量。

```bash
python -m mllm_comp.fixed_mask.generate --scores runs/analysis/hierarchical_taylor.task_scores.json --ratios v2v=0.4,t2v=0.6,ffn_vision=0.4 --allocation layer --selection taylor --output masks/taylor.json
python -m mllm_comp.fixed_mask.generate --scores runs/analysis/hierarchical_taylor.task_scores.json --ratios v2v=0.4,t2v=0.6,ffn_vision=0.4 --allocation layer --selection random --seed 42 --output masks/random.json
```

`layer` 对每层分别使用 `floor(候选数 × 比例)`；`global` 对同一种组件跨层分配预算。
Taylor 在各预算池内对每个任务做平均并列百分位排名，再取跨任务均值最低的组件。
不同组件类型分别排名；默认使用 `normalized_taylor`，可显式切换 `--family direct_removal_taylor`。
整头可指定 `--head-group all|vision|text`，任务可用 `--tasks task_a task_b` 限定。
随机对照从相同候选集合、相同预算池抽样；同一 seed 可复现。
候选集合仅包含分析实际测量的组件；不同任务缺失候选或含非有限数值时会报错。

## 评测与恢复训练

起点配置：

- [Qwen2.5-VL](../../configs/fixed_mask/qwen25_vl.yaml)
- [LLaVA-OneVision](../../configs/fixed_mask/llava_onevision.yaml)

示例中的少量编号仅用于展示配置，不是推荐压缩清单。将 stage 的 `config` 整体替换为生成的 JSON，例如：

```python
import json
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/fixed_mask/qwen25_vl.yaml").read_text(encoding="utf-8"))
cfg["pipeline"]["stages"][0]["config"] = json.loads(Path("masks/taylor.json").read_text(encoding="utf-8"))
Path("mask_experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
```

设置相应模型路径、`MLLM_COMP_RUN_DIR`；训练还需设置 `MLLM_COMP_TRAIN_DATASET`。
分别执行 `mllm-comp eval --config mask_experiment.yaml --plan_only` 和
`mllm-comp train --config mask_experiment.yaml --plan_only` 检查后，去掉 `--plan_only` 执行。
训练示例采用 LoRA；全参数恢复可改为 `tuner_type: full`，删除 LoRA 专用字段。
Mask 不引入新参数，梯度继续流向未屏蔽的原生计算路径。

可追加已有 `zoo_prune.image` 或 `pyramid_drop.image` stage；训练用 `execution_phase: forward`，生成用相应 binding 已支持的 generate 配置。
这两个 token 方法择一使用，沿用现有 binding 的互斥约束。
Mask stage 始终为 `prepare`，先安装后执行 token 压缩。
ZOO-Prune 保留后的图像位置和 PyramidDrop 每次删 token 后的位置都会同步给 Mask，并保留原始文本区域身份。
PyramidDrop 当前组合支持层间边界；其 token 评分仍采用框架现有的独立评分实现。
其他方法以及模态分支替换尚未作为 Mask 组合接入。

## 保存、重载和验证边界

完整模型的 `save_pretrained` 保存 `fixed_mask.json`，同时在模型 config 中记录 Mask。
ms-swift 原生保存钩子也给 LoRA checkpoint 保存该文件。
框架从 checkpoint 评测或恢复训练时，检查配置中的 Mask 与已保存清单一致；缺失 stage 或不同清单会报错。
LoRA 评测加载 adapter 指定的原始基座，合并 adapter 后安装相同 Mask。

直接使用原生 Transformers / PEFT API 时，需要显式安装 Mask：

```python
from mllm_comp.fixed_mask.runtime import install_fixed_mask, save_fixed_mask, load_fixed_mask

session = install_fixed_mask(model, mask_spec, backbone="qwen25_vl")
# PEFT 包装后直接 save_pretrained 只保存 adapter，额外调用：
save_fixed_mask(peft_model, "adapter_checkpoint")
# 新加载的原生模型上恢复 Mask，再包装或加载 adapter：
session = load_fixed_mask(reloaded_model, "adapter_checkpoint")
```

Mask 不是 Transformers 内置配置功能；仅调用其 `from_pretrained` 不会自动安装执行钩子。

本次 CPU 测试覆盖原生小模型的空 Mask 等价性、全头/FFN 屏蔽、路径因果性、全屏蔽行、
梯度检查点两种重算模式、ZOO/PyramidDrop 组合、原生生成、AnyRes、LoRA 更新、完整/adapter 权重重载和 Swift 模型保存边界。
未在真实大模型 GPU 上执行完整 ms-swift CLI 训练或质量基准，注册表的支持证据保持 declared，不能据此宣称压缩质量或速度提升。

本地集中验证为 129 项通过。测试环境使用独立的 Python 3.10.18、PyTorch 2.8.0 CPU、Transformers 4.57.3 和 PEFT 0.19.1；项目要求的 Python 3.10.20 及 GPU 依赖声明未改动。
