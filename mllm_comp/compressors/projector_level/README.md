# Projector-Level 压缩

Projector-Level 泛指模态 Encoder 输出到 LLM 输入之间的压缩区间，不限于 projector
模块内部。本目录主要维护 Qwen2.5-Omni 音频实现；下表列出三个 active 方法：

| Plugin | 执行点 | 核心操作 |
| --- | --- | --- |
| `segmentwise_topk_audio` | `ln_post` 后、`proj` 前 | 分段 attention Top-K pruning |
| `local_temporal_bipartite_merging_audio` | `ln_post` 后、`proj` 前 | 局部奇偶二分匹配与逐轮等权 merging |
| `similarity_attention_merge_prune_audio` | `proj` 后、Thinker 前 | G-SAM merging 后接 attention-weighted DPP pruning |

```text
Qwen2.5-Omni audio encoder
  -> avg_pooler -> ln_post
  -> [Segmentwise Top-K 或 LTBM]
  -> audio_tower.proj
  -> [SA-MAP]
  -> Thinker LLM
```

三个方法复用同一 plan、placeholder/RoPE legalizer 和 controller，支持评估 `generate` 与训练 `forward/backward`；离散选择不引入可训练参数，保留下来的 Token 或合并值仍保持 autograd 路径。

Segmentwise Top-K 支持最后一层 encoder incoming-attention score、分段精确预算、Global Top-K（`num_segments=1`）、uniform/random baseline 与时间顺序恢复。Qwen2.5-Omni 的 encoder 原生按 chunk 做注意力，importance 在每个原生 attention chunk 内重算，再映射到池化后的 Token；这与论文所用 Whisper encoder 的实现背景不同，实验报告必须明确标注。

当前实现绑定 Transformers 4.57.3，作用域为 Qwen2.5-Omni Thinker audio。Projector stage 可按物理合同与 Encoder、LLM Token stage 及 prepare 阶段的 Operation 组合；同一互斥运行槽内仍至多一个 stage。可训练 HTA 是 `target` 候选，其精确训练与恢复证据见[候选方法状态](../../../docs/support/qwen25_omni/audio_compression/candidate_methods.zh.md)。Talker 和 vision 不在本目录范围内。

本目录另有 HTA、UniAvg、UniSamp、UnSeg 四个 target 候选；数值、真实模型与恢复证据见
[候选方法状态](../../../docs/support/qwen25_omni/audio_compression/candidate_methods.zh.md)。
图像 Projector 方法由 ImageLLM 与 comparison 实现维护，见[图像状态](../../../docs/support/imagellm/status.zh.md)
及[新增对比方法](../../../docs/support/imagellm/comparison_methods.zh.md)。

配置位于 `configs/eval/lmms_eval/` 与 `configs/train/qwen2_5_omni/`。当前 Omni 执行族的 target site 和共享职责见 [`Qwen2.5-Omni 音频压缩方法扩展指南`](../../../docs/support/qwen25_omni/audio_compression/extension.zh.md)；Segmentwise 重构后的 Strategy/Controller 分工、代码阅读顺序和最小 Debug 见 [`Projector-Level 架构与 Debug 手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/projector_level.zh.md)。正式新增通用方法以 [`开发压缩方法指南`](../../../docs/guides/2_dev_compression_method.zh.md) 为准。
