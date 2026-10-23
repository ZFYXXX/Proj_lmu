# Qwen2.5-Omni Backbone

阶段：Phase 1a 原生 backbone 已完成；CPU 合同测试与真实 checkpoint 全模态/语音 GPU smoke 均已通过，catalog 状态为 `ACTIVE`。

该 adapter 使用 Transformers 的原生 Qwen2.5-Omni 实现；初始契约来自 4.52.3，现已在 4.57.3 下完成全模态/Talker 回归。`create_qwen25_omni_backbone` 组合 processor、model、state builder、native validator、compression legalizer 和 runner。大型 processor/model 只存在于命令生命周期的 `BackboneRuntime`。

原生范围：

- 输入：text、image、audio、video，以及 `use_audio_in_video`；
- 输出：Thinker text tokens/text，及可选 Talker + Token2Wav waveform；
- `GenerationResult` 分别表达 text 和 speech，语音输出当前 batch size 固定为 1；
- `TraceArtifacts` 按需通过 hook 捕获 `thinker_reply_part`、Talker codec tokens 和模型中间量，默认不保留；Omni 只接受显式 `thinker_*`/`talker_*`/`token2wav_*` trace 字段并拒绝通用 `text_*` 字段；
- `ForwardState` 同时保留 Transformers 原生 kwargs 和平台规范化字段；
- image/video/audio 占位 token span、grid、feature mask、speaker、waveform 和采样率由 native validator 检查；
- 当前验证的 Transformers 4.57.3 复合 `generate()` 是同步接口，不模拟不存在的公开流式接口。
- 4.57 默认 fast processor 会改变旧 checkpoint 的预处理语义，因此 adapter 默认 `use_fast=false`；模型加载把弃用的 `torch_dtype` 转换为 `dtype`。

压缩 `Legalizer` 在没有 token action 时为恒等路径；当前已激活八种受支持的音频压缩所需 placeholder、mask、labels、M-RoPE 和 span 修复。六个 `target` 候选也具有精确的 Audio Variant、Qwen Binding 与 declared SupportUnit，可复用相同执行族采集开发证据，但不会因此进入 active registry。vision grid 修复和任意多模态组合尚未激活。

平台把 LLM 完整序列互斥划分为正文（`system_prompt`、`instruction`、`response`）、消息模板（`sys_special`、`ins_special`、`resp_special`）、模态内容（`audio`、`image`、`video`）、模态边界（`audio_special`、`image_special`、`video_special`）、`global_special`、`pad` 和 `unknown`。当已启用方法声明需要 `semantic_instruction_mask` 时，processor 默认路径和 Swift bridge 即使在 default 模式也会生成运行时语义标签供算法使用；这不等于保存详细调试快照。`runtime_mode=debug` 才额外收集 processor、Legalizer、各层与生成结果的 Token 统计，所有 snapshot 与逐层记录均满足 `sum(token_counts.values()) == total_tokens`，且 `token_counts` 不保存冗余 `total`。评估端依据 chat-template 字符区间与 tokenizer offsets 分类，并让标签随 Legalizer 同步；训练端若只有 `labels` 而没有原始角色消息，只分类可证明的 response、模态、控制边界和 pad，其余 prompt 保留为 unknown。profile 只负责效率测量。

需要沿真实 pipeline 人工复核时，先参见 [`Qwen2.5-Omni Backbone Pipeline 调试手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/pipeline.zh.md)。四个执行族的职责划分、代码阅读顺序与最小 Debug 分别见 [`Projector-Level`](../../../docs/support/qwen25_omni/audio_compression/debugging/projector_level.zh.md)、[`Encoder-Level`](../../../docs/support/qwen25_omni/audio_compression/debugging/encoder_level.zh.md)、[`LLM-Token / Audio-Pyramid`](../../../docs/support/qwen25_omni/audio_compression/debugging/llm_token_audio_pyramid.zh.md)和 [`LLM-Operation / SPADE`](../../../docs/support/qwen25_omni/audio_compression/debugging/llm_operation_spade.zh.md)。八种 active 方法及四个执行族的统一接入边界见 [`Qwen2.5-Omni 压缩方法扩展指南`](../../../docs/support/qwen25_omni/audio_compression/extension.zh.md)，六个 target 候选见 [`候选方法状态`](../../../docs/support/qwen25_omni/audio_compression/candidate_methods.zh.md)。
