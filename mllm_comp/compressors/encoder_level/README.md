# Encoder-Level 压缩

Qwen2.5-Omni 当前激活三个 Encoder-Level 音频压缩方法：

| Plugin | 执行点 | 默认策略 |
| --- | --- | --- |
| `adjacent_token_merging_audio` | MHSA residual 后、FFN 前 | 从第 2 层起每隔 3 层，按 attention-key 相似度合并相邻 Token |
| `encoder_attention_topk_audio` | MHSA residual 后、FFN 前 | 在第 4、7、10 个 block 按 mean received attention 保留 Top-K |
| `progressive_attention_pruning_audio` | 完整 block 后、下一层前 | 从第 6 层起，在选定层按固定或线性递增 rate pruning |

Controller 会在每个启用层后重建 ragged attention metadata，并将变长操作限制在各个 FlashAttention chunk 内。每层输出长度由 plan 预先确定，最终每段至少保留两个 encoder token，以满足 Qwen stride-2 audio pooler；池化后的长度继续传播到 LLM placeholder 和多模态 RoPE。

三个方法共用评估 `generate` 与训练 `forward` 路径，并支持 gradient-checkpoint backward recomputation。离散选择必须可重复，重算不会重复写 telemetry；压缩后的 tensor 保持到 `input_features` 的梯度路径。

运行模式与算法正交：`default` 不注册详细采集，`debug` 记录逐层 Token 数和压缩细节，`profile` 额外记录 scoring/merging overhead。

配置位于 `configs/eval/lmms_eval/` 与 `configs/train/qwen2_5_omni/`。当前 Omni 执行族的长度规划和执行点见 [`Qwen2.5-Omni 音频压缩方法扩展指南`](../../../docs/support/qwen25_omni/audio_compression/extension.zh.md)；A-ToME 重构后的 Strategy/Controller 分工、代码阅读顺序和最小 Debug 见 [`Encoder-Level 架构与 Debug 手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/encoder_level.zh.md)。正式新增通用方法以 [`开发压缩方法指南`](../../../docs/guides/2_dev_compression_method.zh.md) 为准。
