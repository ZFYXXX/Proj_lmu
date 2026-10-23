# 测试

| 目录 | 覆盖范围 |
| --- | --- |
| `unit/` | 算法数值、公开输入和产物、状态与绑定生命周期、关键失败路径 |
| `contract/` | 已安装的 lmms-eval、ms-swift 与平台插件接口 |
| `integration/` | 原生小模型组合，以及显式启用的真实 checkpoint smoke |

修改时先运行直接覆盖变化的现有测试；只有新增行为或已发现的缺陷缺少覆盖时才加用例。
每组测试应能说明它会发现什么错误，并有一个主要承担位置。共享 validator 的相同坏值、
内部字段和类名镜像、脚本源码字符串、反复构造完整报告后只翻转一个字段，不作为独立覆盖。
数学、模型布局、梯度、缓存、保存与恢复的实际差异单独保留；新增方法不要求复制全部方法×基座矩阵。

算法测试用独立数值预期；原生小模型负责布局、梯度、prefill/decode 和缓存行为；
真实 checkpoint 脚本负责目标权重上的端到端运行。合成报告仅为实际消费它的解析、展示或
失败处理提供最小输入，不代替算法或模型证据。已有原生覆盖时，不再复制整套假模型执行循环。
固定回归期望值应来自可复现命令和固定样本，只保存最小、与机器路径无关的必要结果。

视频训练诊断的公共流程集中在 `scripts/diagnostics/video_training_checkpoint_common.py`：
准备训练输入、更新选定参数、检查后保存、在新模型上重载并执行无标签推理，以及保存失败证据。
`test_video_training_workflow.py` 用真实 autograd、AdamW 和 safetensors 验证 3/5 步的闭环。
方法脚本保留自己的信号、布局、标签和缓存检查；跨基座一致的纯方法检查放在
`scripts/diagnostics/video_method_checks.py`。这里的重载推理是 `forward`，不生成增量 decode。
新增诊断时直接复用相同的步骤；只有模型或方法的实际行为不同，才保留独立流程。

配置或解析器变化时，`test_compression_definitions.py` 对每份正式配置只加载一次，检查
运行输入和启用的压缩定义；`examples/compression/` 中不完整的 schema 示例只检查压缩定义。
分析示例由 `test_analysis_contracts.py` 检查可解析性，原生训练参数由对应 Swift 示例测试承担。
这些检查不固定每份 YAML 的所有默认值或模型 revision。

需要完整单元与接口检查时：

```bash
python -m pytest tests/unit tests/contract -q
```

依赖未安装时，相关测试会跳过；跳过不构成验证通过。小模型、双设备和真实 checkpoint
测试的选择方式见 [集成测试](integration/README.md)，支持声明所需证据见
[验证和激活能力](../docs/guides/3_verify_capabilities.zh.md)。

## 新增图像对比方法

`tests/unit/test_comparison_*.py` 同时包含 registry/公开配置检查、独立算法数值和原生
小模型测试；目录名不表示它们全部无需模型依赖。按改动选择其中的 registry、projector、
decoder、operations、policies、qwen 文件；完整原生测试需要固定 PyTorch/Transformers
与 vendored LLaVA 环境。2026-10-03 的 378 项结果是相关回归集合，不是全仓库测试数；
精确覆盖和未完成的真实模型验证见[对比方法说明](../docs/support/imagellm/comparison_methods.zh.md)。

## 临时文件

测试通过 pytest 标准 `tmp_path` 获得每个用例独立的目录，由 pytest 管理命名和保留策略。
不在 `tests/fixtures/` 或历史任务目录中手工创建运行产物。

默认使用系统临时目录。受限环境需要将临时文件放在工作区内时，可以指定 pytest 的
临时根目录；`.cache/` 已被 Git 忽略。

```bash
mkdir -p .cache/pytest
PYTEST_DEBUG_TEMPROOT="$PWD/.cache/pytest" python -m pytest tests/unit tests/contract -q
```

PowerShell：

```powershell
New-Item -ItemType Directory -Force .cache/pytest | Out-Null
$env:PYTEST_DEBUG_TEMPROOT = (Resolve-Path .cache/pytest).Path
python -m pytest tests/unit tests/contract -q
```

如果 Windows 受限进程无法访问 pytest 创建的用户专属目录，请在正常用户终端或已配置的
远端测试环境运行；仅修改临时根目录不会改变 pytest 的访问权限设置。
