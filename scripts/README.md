# Script entry points

`scripts/` 保存项目维护的运行入口、研究实验与诊断工具：

- `train/`：与 `configs/train/` 对应的训练入口；
- `eval/`：与 `configs/eval/` 对应的评测入口；
- `analysis/`：七 Probe 机制分析矩阵、音频声学/时间/类别/冗余/sink 等研究实验及结果后处理；
- `diagnostics/`：可重复使用的框架契约与 GPU smoke test 工具。

标准 train/eval 薄入口由 YAML 拥有实验参数，Shell 负责机器环境变量与 CLI 启动。
专项研究脚本另有自己的参数和产物协议，按所在目录的 README 使用；脚本存在不表示实验
已完成或方法已激活。`examples/` 保存最小配置和用法说明。

三基座新增对比方法使用公共 CLI 和 [`configs/comparison/`](../configs/comparison/)，
不要求每个 binding 配套一份 Shell。源码目录中的 `lmu/` 是实验配方目录，不是共享工作区
根目录下那个独立 `lmu` Git 仓库；部署仍按当前 Dev lane 规则执行。

单方法教学示例使用 `analysis/run_analysis_method.sh`。它按方法名选择
`examples/analysis/methods/qwen25_vl_<method>.yaml`，并为每次运行创建独立目录：

```bash
bash scripts/analysis/run_analysis_method.sh representation_dynamics
bash scripts/analysis/run_analysis_method.sh prediction_dynamics
bash scripts/analysis/run_analysis_method.sh attention_flow
bash scripts/analysis/run_analysis_method.sh residual_stream_decomposition
bash scripts/analysis/run_analysis_method.sh layer_importance
bash scripts/analysis/run_analysis_method.sh late_injection_early_exit
bash scripts/analysis/run_analysis_method.sh hierarchical_taylor
```

四个 active Backbone 的完整七 Probe 分析共用
`analysis/run_analysis_matrix.sh`。该脚本按 Backbone 选择
`examples/analysis/*_analysis_matrix.yaml`，创建带时间戳的新输出目录，并允许用
`--probe hierarchical_taylor` 等 CLI 参数缩小运行范围：

```bash
bash scripts/analysis/run_analysis_matrix.sh qwen25_vl
bash scripts/analysis/run_analysis_matrix.sh qwen3_vl --probe hierarchical_taylor
```

Qwen2.5-Omni 的双卡图像和音频评测入口分别为 `eval/lmms_eval/qwen25_omni_2gpu.sh` 与 `eval/lmms_eval/qwen25_omni_2gpu_aud.sh`；它们使用 Accelerate 启动两个 worker，但不会进行模型切分。

`diagnostics/` 保存可复用的环境检查、模型与压缩链路诊断；已经闭环的一次性 benchmark 差分脚本不长期保存。

八种 Qwen2.5-Omni 压缩方法的开发验证入口为
`diagnostics/run_qwen25_omni_eight_method_validation.sh`，复用已有 pytest 和
`mllm-comp` CLI。各命令独立运行，不依赖已保存的 PASS 状态或前置完整回归；
`check segmentwise|atome|llm|extensions|all` 可组合对应检查。
日志和实际报告写入 `VALIDATION_ROOT`；已有报告会拒绝覆盖，代码、模型、数据或生成设置
变化时应使用新目录。用法见[八方法开发验证 Runbook](../docs/support/qwen25_omni/audio_compression/eight_methods_runbook.zh.md)。
