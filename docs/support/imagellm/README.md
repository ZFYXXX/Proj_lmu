# ImageLLM 多算法文档

> 适用范围：LLaVA-OneVision、Qwen2.5-VL、Qwen3-VL 图像压缩；各方法的基座覆盖分别声明

- [当前支持状态](status.zh.md)：历史三基座矩阵、D2Pruner prior，以及新增 Bifocal 和对比方法的独立声明。
- [OV 对比方法接入](comparison_methods.zh.md)：新增 14 组方法的三基座实现、配置、CPU 验证及校准/权重限制。
- [验证手册](validation_runbook.zh.md)：复跑历史 72 个一步训练案例，以及新增能力的独立验证入口；历史矩阵不含 Bifocal 和新增对比方法。
- [MWOP（Mask 形式）](../../guides/fixed_mask.zh.md)：LLaVA-OneVision / Qwen2.5-VL 的 `fixed_mask` Operation 路径。
- [分模态模块替换](../../../mllm_comp/modality_split/README.md)：Qwen2.5-VL / Qwen3-VL 图像分支 Attention/FFN，不继承 Token 方法的训练证据。
- [开发压缩方法](../../guides/2_dev_compression_method.zh.md)：kernel、variant、binding 与证据接入规则。
- [平台架构](../../platform/architecture.zh.md)：ImageLLM 与 Qwen2.5-Omni AudioLLM
  共用的 v2 四层能力模型，以及各自 Binding executor 的物理边界。
