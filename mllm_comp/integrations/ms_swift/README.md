# ms-swift integration

状态：active（train）。

训练采用 Native CLI First：平台构造并直接启动 `swift sft`，不再导入或调用
`sft_main()`。Swift 完整拥有参数解析、模型和 template 加载、Trainer、日志、进度、
分布式/DeepSpeed、resume、checkpoint 和 `logging.jsonl`；平台只在进程成功结束后
收集这些原生产物并生成 Manifest。

Qwen2.5-Omni 使用 Swift 原生 `qwen2_5_omni` model/template，固定
`ENABLE_AUDIO_OUTPUT=0`，当前只声明 Thinker 训练边界。Qwen3-VL 使用
`qwen3_vl` model/template 和 `language_model` 边界，不包含 Talker/audio-output。
Qwen2.5-VL 使用 `qwen2_5_vl`，LLaVA-OneVision 使用 `llava_onevision_qwen`，
二者同样声明 `language_model` 训练边界。能力查询从 adapter 的基座策略生成。

`experiment.seed` 是原生训练 seed 的默认值；`train.args.seed` 和最后的
`native_argv --seed` 可显式覆盖。实际调用参数进入 backend `argv`。

Qwen2.5-Omni 的 `max_pixels`、`use_audio_in_video`、`sampling_rate`、
`video_min_pixels`、`video_max_pixels`、`fps`、`fps_min_frames`、
`fps_max_frames` 和 `frame_factor` 可按原生名称写入 `backbone.config`。
适配器只映射显式配置的值，所有字段都转为对应的 Qwen 原生环境变量。
启动前会清除这些环境变量的父进程残留，使未配置字段继续
使用上游默认值。其余 Swift 训练参数仍直接写入 `train.args`。

Qwen2.5-Omni 通过 Swift external plugin 在模型加载后、optimizer 创建前安装
CompressionSession；原生 template 继续负责数据准备，模型 bridge 驱动当前 Token 和
Operation 执行路径。动态训练需收集实际 forward 和对应 loss backward 证据。
支持将结构 Operation 与动态 Token stage 组合，安装依赖和有效结构句柄由 Session 管理。
八个已有 active 方法为：
`segmentwise_topk_audio`、`local_temporal_bipartite_merging_audio`、
`similarity_attention_merge_prune_audio`、`adjacent_token_merging_audio`、
`encoder_attention_topk_audio`、`progressive_attention_pruning_audio`、
`audio_pyramid_attention_pruning` 或 `structured_llm_layer_pruning`。前三个动态
Token 执行族必须使用 `execution_phase: forward`；结构层剪枝使用
`execution_phase: prepare`，并在 runner/optimizer 创建前改写模型。

另有六个 `target` 候选可按精确合同进入训练，其 SupportUnit 保持 `declared`，
不会进入 `ACTIVE_COMPRESSORS`。候选的数值与原生小模型证据、HTA 在选定组合下的
真实联合训练和完整 Swift 恢复，以及其余验证范围，统一见
[`候选方法状态`](../../../docs/support/qwen25_omni/audio_compression/candidate_methods.zh.md)。

`train.runtime_mode` 默认为 `default`，不注册效率或逐层调试采集；`--profile` 只采集
forward 时间和 CUDA 峰值显存；`--debug` 只记录 processor、legalizer、Audio Encoder
各层、Projector 和 LLM 各层的分类 Token 数量。动态 Token 训练仍拒绝 packing/padding-free；
Omni 纯结构删层不受这个动态布局限制。三个视觉基座的图像 Token stage 在训练时使用
`execution_phase: forward`；新增 comparison Operation、MWOP Mask 和分模态替换使用
`prepare`，在模型加载后、optimizer 创建前安装，再作用于训练前向。组合按各方法合同
判断：新增 comparison 使用互斥运行槽；MWOP 仅开放文档列出的有限组合；分模态 stage
由专用联合编译路径处理。LLaVA ToMe 还需关闭 `vit_gradient_checkpointing`，AdaLLaVA
训练需关闭 `gradient_checkpointing`。Talker 训练尚未声明。

[新增对比方法](../../../docs/support/imagellm/comparison_methods.zh.md)共有 51 个三基座
binding；Swift 桥分别记录 Token 减少和 Operation 实际执行，不能以 Token 数不变判断
Operation 未生效，也不能伪造 Token 减少证据。新增参数的原生完整 checkpoint 保存恢复
已有 CPU 测试，PEFT/Swift adapter-only 保存恢复及真实 Swift 全链路仍待验证。

阅读实现时，从 `adapter.py` 的启动和证据收集进入 `training_plugin.py` 的模型安装路径；
人类可读的诊断整理在 `diagnostics.py`，原生训练生命周期和记录格式保持独立。

Segmentwise Top-K 与 A-ToMe 已完成真实 Swift 单步训练、checkpoint 和 Manifest
验收；LTBM、SA-MAP、Encoder TopK、Progressive Pruning 与 Audio-Pyramid 已完成真实
Qwen2.5-Omni-3B forward/backward 和 Swift bridge/launch-plan 回归。SPADE
Operation-Only 还完成了结构 checkpoint、哈希 Manifest、reload 与重载后生成。
上述 active 集合中后六种方法尚未各自完成完整 Swift run、收敛和质量实验。

真实训练的 template bridge、plan/legalizer、方法执行点、loss backward、evidence 与
Manifest 验收入口见
[`Projector-Level 架构与 Debug 手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/projector_level.zh.md)、
[`Encoder-Level 架构与 Debug 手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/encoder_level.zh.md)、
[`LLM-Token / Audio-Pyramid 架构与 Debug 手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/llm_token_audio_pyramid.zh.md)、
[`LLM-Operation / SPADE 架构与 Debug 手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/llm_operation_spade.zh.md)
和 [`Omni 八方法 Runbook`](../../../docs/support/qwen25_omni/audio_compression/eight_methods_runbook.zh.md)。
所有八个 active 方法共用的注册与扩展步骤见
[`Qwen2.5-Omni 音频压缩方法扩展指南`](../../../docs/support/qwen25_omni/audio_compression/extension.zh.md)。
