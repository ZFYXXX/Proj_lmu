# 按任务使用框架

| 你要完成的任务 | 从这里开始 | 完成后得到什么 |
| --- | --- | --- |
| 运行已有压缩方法 | [运行压缩实验](1_run_compression_experiment.zh.md) | CLI/Python 计划、实验结果、比较与恢复 |
| 替换一个评分步骤或增加算法 | [开发压缩方法](2_dev_compression_method.zh.md) | 一个方法文件、一份配方，复用现有执行机制 |
| 理解失败或修改运行机制 | [排查运行时问题](3_troubleshoot_runtime.zh.md) | 从报错到实际责任代码及最小回归 |
| Agent 自动研究与能力改进 | [Agent 研究入口](4_agent_research.zh.md) | 现有程序接口、实验反馈与 auto-research / RSI 的目标边界 |

## 针对具体任务继续阅读

- 配置与命令：[CLI 与配置](1_cli_and_config.zh.md)。
- 单独运行：[分析](1_run_analysis.zh.md)、[评测](1_run_evaluation.zh.md)、[训练](1_run_training.zh.md)。
- 新合入的图像能力：[14 组 OV 对比方法及三基座适配](../support/imagellm/comparison_methods.zh.md)、[Bifocal](../support/imagellm/status.zh.md#bifocal-接入)、[MWOP（Mask 形式）](fixed_mask.zh.md)、[Qwen 分模态模块替换](../../mllm_comp/modality_split/README.md)。各自保留独立的验证范围。
- 音频比例实验与研究脚本：[Omni 实验入口](../support/qwen25_omni/status.zh.md#2026-10-02-音频实验接入)。
- 新增模型或任务：[Backbone](2_dev_backbone.zh.md)、[Analysis 方法](2_dev_analysis_method.zh.md)、[Evaluation Task](2_dev_evaluation_task.zh.md)。
- 发布支持声明：[验证和激活能力](3_verify_capabilities.zh.md)。这是声明支持范围的步骤，不是试验新算法的准入门槛。

CLI、Python 和 Agent 使用同一配置、`run_experiment` 与结果格式。首次使用不要求读完所有接口定义；各指南在需要深入时链接对应参考。
