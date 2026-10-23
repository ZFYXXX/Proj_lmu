# Qwen2.5-Omni 文档

> 类型：Support index
> 适用范围：Qwen2.5-Omni baseline、AudioLLM v2 压缩与 Thinker/LLM 音频分析

- [当前训练、压缩、评测与分析状态](status.zh.md)
- [近期音频研究与比例实验入口](status.zh.md#2026-10-02-音频实验接入)：配置和脚本已接入，真实运行证据分别记录。
- [七 Probe 运行指南与 Qwen 音频示例](../../guides/1_run_analysis.zh.md)
- [AudioLLM 压缩扩展](audio_compression/extension.zh.md)
- [八方法算法与验证边界](audio_compression/eight_methods_contract.zh.md)
- [八方法开发验证步骤](audio_compression/eight_methods_runbook.zh.md)
- [六个 target 候选方法与激活门禁](audio_compression/candidate_methods.zh.md)
- [FastSLM HTA 权重与运行边界](audio_compression/fastslm_hta.zh.md)
- [HeadRouter 方法映射与运行边界](audio_compression/headrouter.zh.md)
- [FastAdaSP 方法映射与运行边界](audio_compression/fastadasp.zh.md)
- Pipeline 与架构/Debug：[完整 pipeline](audio_compression/debugging/pipeline.zh.md)、[Encoder-level](audio_compression/debugging/encoder_level.zh.md)、[Projector-level](audio_compression/debugging/projector_level.zh.md)、[LLM Token / Audio-Pyramid](audio_compression/debugging/llm_token_audio_pyramid.zh.md)、[LLM Operation / SPADE](audio_compression/debugging/llm_operation_spade.zh.md)

Qwen2.5-Omni 当前有八个 active 音频压缩方法和六个 target 候选方法；两类都不能作为
image/video 支持声明，候选配置存在也不等于正式支持。
