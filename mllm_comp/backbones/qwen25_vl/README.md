# Qwen2.5-VL Backbone

状态：active。该 Backbone 最初承担 Phase 1c 的迁移验证；当前已完成
text/image/video → text、visual span、M-RoPE、按需 text trace 与真实 checkpoint
验证，并已进入统一 capability catalog。

该 adapter 使用 Transformers 原生实现。`image_grid_thw`、`video_grid_thw`、
`second_per_grid_ts`、M-RoPE position、visual span、mask、position 和 label
属于原生 Backbone 契约；ImageLLM v2 provider bridge 在此基础上负责已验证的 image
SupportUnit 布局修复。

独立的 video bridge 固定在 Transformers 4.57.3 的 post-merger video feature
边界，目前只接受 batch 1、单视频且不混合 image。它保留原始
`video_grid_thw` 用于视觉编码和原生 M-RoPE 构造，把压缩特征散射到源序代表
placeholder，再在 LLM 边界共同压缩 placeholder、attention mask、IGNORE_INDEX
label、cache position，并从原生三轴位置中收集稀疏锚点；不把不规则结果伪装成新
的稠密 grid。prefill 后发布的 rope delta 按实际压缩长度精确重算，generate decode
输入按同一删除量修复。text/image 调用保持原样，image 和 video provider 不在同一
模型实例上同时打补丁。

上述 facade/legalizer 契约本身不激活算法 SupportUnit。九个独立 video 变体已有各自的
真实 checkpoint 推理与最小训练/重载证据，范围见[VideoLLM 状态](../../../docs/support/video_compression/status.zh.md)。

图像历史验证矩阵、Bifocal 见[ImageLLM 状态](../../../docs/support/imagellm/status.zh.md)。
本基座也接入[14 组新增对比方法](../../../docs/support/imagellm/comparison_methods.zh.md)、
[MWOP（Mask 形式）](../../../docs/guides/fixed_mask.zh.md)和
[分模态模块替换](../../modality_split/README.md)；这些新增精确单元保持 `declared`，
不继承既有图像或视频方法的验证证据。
