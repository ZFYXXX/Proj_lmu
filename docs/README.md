# 文档索引

文档按“任务 Guide、配置参考、稳定平台规范、具体能力状态、历史设计与未来工作”组织。先根据
当前角色选择 Guide，不要从文件名猜测实现状态。实际执行合同决定能否运行；
精确 SupportUnit 和对应状态页说明已有验证证据。

## Guide

从 [Guide 索引](guides/README.md) 按四类角色和当前任务选择文档：

- `1_`：平台使用者，配置和运行已有能力；
- `2_`：压缩研究人员，增加或接入新的能力；
- `3_`：平台维护者，验证、激活和排查精确能力；
- `4_`：Agent，通过程序接口开展实验迭代、方法开发与研究能力改进。

Guide 目录的 `README.md` 是统一入口。数字前缀只表达目标读者，不表示实现优先级、
项目里程碑或同层文档的严格阅读顺序。

## 配置参考

- [配置字段参考](reference/config.zh.md)：公共 YAML 字段、默认值、含义和 CLI 对应关系。
- [压缩位置参考](reference/compression_positions.zh.md)：各方法的应用位置、评分来源、预算及模型限制。
- [机制与方法状态参考](reference/compression_execution.zh.md)：信号采集、状态更新、缓存、旁路和 checkpoint 重算。

## 平台规范

- [统一压缩框架架构](platform/architecture.zh.md)：长期稳定的多模态、多基座、多阶段抽象。
- [顶层命令设计](platform/top_level_commands.zh.md)：`analyze`、`eval`、`train` 的路由和执行模式。
- [核心概念](platform/core_concepts.zh.md)：state、action、plan、legalizer、pipeline 和 Manifest。
- [接口规范](platform/interface_spec.zh.md)：稳定接口、生命周期、不变量和激活规则。
- [外部框架接入](platform/framework_integration.zh.md)：lmms-eval 与 ms-swift 的职责和参数边界。
- [量化联合接入设计](platform/quantization_integration.zh.md)：内部与外部量化共同训练、推理和恢复的设计；量化执行器尚未实现。

`platform/` 回答“当前稳定契约是什么”，`reference/` 回答“字段、位置和机制如何工作”，
`guides/` 回答“怎样完成一个任务”。不要在 Guide 中复制完整接口定义，也不要用
架构文档替代可运行步骤。

## 具体能力与验证

- [压缩方法支持范围](support/compression_methods.zh.md)：方法分类、实际能力发现入口和验证层级。
- [Qwen2.5-Omni](support/qwen25_omni/README.md)：baseline、音频方法、扩展和专项调试。
- [ImageLLM 多算法](support/imagellm/README.md)：三基座历史验证矩阵，以及新增 Bifocal 和对比方法的独立边界。
- [OV 对比方法接入](support/imagellm/comparison_methods.zh.md)：14 组新增实现、51 个三基座 binding、配置、CPU 验证与待完成的真实模型验证。
- [VideoLLM 九算法](support/video_compression/status.zh.md)：九个独立 video 变体在三个基座上的推理、最小训练与证据边界。
- [Qwen3-VL](support/qwen3_vl/README.md)：原生基座和 operation replacement 调试入口。
- [Qwen 分模态模块替换](../mllm_comp/modality_split/README.md)：Qwen2.5-VL / Qwen3-VL 图像与文本分支、缓存、训练配置及保存恢复。
- [MWOP（Mask 形式）](guides/fixed_mask.zh.md)：LLaVA-OneVision / Qwen2.5-VL 头、路径、FFN 遮罩及清单生成；代码标识为 `fixed_mask`，CPU 验证与真实 GPU 验证分别声明。
- [RefCOCO 定位评分](guides/1_run_evaluation.zh.md#refcoco-定位评分)：Qwen 两种坐标协议与评测可比性。
- [外部框架人工验收](support/integration/validation_runbook.zh.md)：在当前开发工作区复核 lmms-eval/ms-swift。

正式支持范围以对应 `status.zh.md` 和代码中的精确 SupportUnit 为准。满足执行合同的
研究配置可以运行；算法文件、配置存在、`--plan_only` 或 baseline 通过都不等于正式支持。

## 项目状态、历史与未来

- [项目进度](progress.md)：当前里程碑、近期主线合入和下一步工作；历史验证计数不代表在当前提交全量重跑。
- [平台设计 v0.3](archive/platform_design_v0.3.zh.md)：早期总体设计，仅用于理解设计来源。
- [Omni v2 收敛记录](archive/omni_compression_v2_convergence.zh.md)：已完成的音频压缩与分析迁移。
- [具身扩展草案](future/embodied_design_notes.zh.md)：尚未激活的未来方向。
- `todo/`：尚未完成的设计决策和收敛任务，不是当前运行接口。

## 维护约定

- Guide 文件名前缀只表达读者层级，不表达优先级；不增加 `1_1_` 等二级序号。
- Guide 目录和任务入口统一维护在 [Guide 索引](guides/README.md)。
- Guide 围绕任务结果组织，通常包含“准备、最小闭环、完整流程、验收与边界”；配置说明、
  产物解读和常见问题按需添加，不要求固定段数。
- 稳定架构与接口写在 `platform/`，公共配置字段写在 `reference/`；当前实现状态写在
  `progress.md` 和 `support/*/status.zh.md`。
- 专项人工步骤放在 `support/*/validation_runbook.zh.md`，模型细节放在 `support/*/debugging/`。
- 历史方案移入 `archive/`，不得继续作为当前接口依据。
- 运行日志、个人服务器绝对路径和一次性报告不写入长期文档。
- future/target 能力不得写成 active；支持声明必须精确到算法、variant、基座、模态和 phase。
