# 可运行配置

`configs/` 只保存可以交给真实命令的完整配置：

- `eval/lmms_eval/`：Qwen2.5-Omni、Qwen2.5-VL、Qwen3-VL 与 LLaVA-OneVision 的 lmms-eval baseline、压缩探针和 Operation replacement 配置；
- `fixed_mask/`：双基座 MWOP（Mask 形式）配方、清单生成与有限组合，见[Mask 指南](../docs/guides/fixed_mask.zh.md)；
- `modality_split/`：Qwen2.5-VL / Qwen3-VL 图像/文本分支替换、训练和恢复配方，见[分模态说明](../mllm_comp/modality_split/README.md)；
- `comparison/`：三种视觉基座的 DOP-V 双用途训练/评测配置；新增对比方法的参数、执行阶段和校准限制见[接入说明](../docs/support/imagellm/comparison_methods.zh.md)；
- `train/imagellm/`：历史图像方法一步训练与新增 Bifocal 配方，证据见[图像状态](../docs/support/imagellm/status.zh.md)；
- `train/qwen2_5_omni/`：八个原生模态 SFT/LoRA 配置，以及覆盖四个执行族、八种 active 方法和六种 target 候选方法的音频压缩训练配置。

所有配置都必须声明本次运行的 `backbone`、任务与必要数据。普通配方省略 `output`，每次实验默认写入 `runs/<experiment.name>/<run_id>/`。显式 `output.run_dir` 是精确目录，省略 `manifest_path` 时采用该目录下的 `manifest.json`；非空目录或已有 Manifest 会被拒绝。训练环境变量、论文 sweep 和诊断配置保留其显式输出协议，重复运行时需提供新目录或新标签。

模型特有且随实验变化的参数写在该配置的 `backbone.config` 中；模型固定能力由 `mllm_comp.registry.catalog` 和对应 backbone 实现声明。`target` 配置用于显式候选验证，不表示对应能力已进入 active registry。CLI 与 Python 入口、有效配置及结构化结果见[运行压缩实验](../docs/guides/1_run_compression_experiment.zh.md)。

项目不提供 YAML `include`、`extends` 或合并机制，也不维护单独的 `configs/backbones/` 片段。这里接受少量字段重复，以保持每份配置可独立理解和执行。

可运行的 analysis 示例位于 `examples/analysis/`；compression definition 示例位于 `examples/compression/`；小型训练与评测示例位于 `examples/train/` 和 `examples/eval/`。
