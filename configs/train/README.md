# 可运行训练配置

本目录保存可直接交给 `mllm-comp train` 的项目训练配置。配置必须包含模型来源和训练数据，并且运行前不能残留未解析的环境变量。这里保留配套脚本的 `MLLM_COMP_TRAIN_RUN_DIR` 输出协议；该变量解析出的目录是精确位置，每次实验需提供新目录。新建普通配方可以省略 `output`，使用平台自动生成的运行目录。

`qwen2_5_omni/` 中包含八份分模态 SFT/LoRA 配置，以及十四份压缩训练配置：八种 active 方法和六种 target 候选。每份配置都在 `scripts/train/qwen2_5_omni/` 下提供对应的薄入口，也可直接交给 `mllm-comp train`。Qwen2.5-Omni 原生模态控制放在 `backbone.config`，Swift 训练参数放在 `train.args`；两者都只保存该实验相对上游默认值需要显式指定的内容。target 配置只能证明候选入口可显式验证，不构成正式训练支持。

基线与八个 active 压缩方法的训练证据见[Omni 状态](../../docs/support/qwen25_omni/status.zh.md)。六个候选保留 `target`/`declared` 状态；其已有数值测试、HTA 选定组合的真实联合训练与完整恢复，以及剩余验证范围，统一见[候选方法状态](../../docs/support/qwen25_omni/audio_compression/candidate_methods.zh.md)。video/audio-video 与四份常规模态 LoRA 配置仍只声明为“契约通过”。方法扩展见[Omni 执行族参考](../../docs/support/qwen25_omni/audio_compression/extension.zh.md)。

本目录不是全部训练能力的清单。图像压缩训练配方位于 [`imagellm/`](imagellm/)，
最小 baseline 示例位于 [`examples/train/`](../../examples/train/)，
三基座 DOP-V 双用途配置位于 [`configs/comparison/`](../comparison/)。新增对比方法的
Token stage 在训练时使用 `forward`，Operation stage 使用 `prepare`；原生 CPU 测试不代表
真实 Swift 训练闭环通过，校准与保存恢复限制见[对比方法说明](../../docs/support/imagellm/comparison_methods.zh.md)。
