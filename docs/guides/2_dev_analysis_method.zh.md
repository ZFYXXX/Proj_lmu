# 开发 Analysis 方法

> 类型：Guide
>
> 文档层级：2 / 压缩研究人员
>
> 目标结果：实现一个共享样本循环、可清理、可聚合并能写入 Manifest 的新 Probe
>
> 适用范围：新增 analysis 方法或为已有 Probe 扩展精确 backbone 语义
>
> 当前状态：Active

Analysis Probe 观察或干预模型内部语义，但不拥有 dataset 循环、模型加载、输出目录
生命周期或 Manifest。所有 Probe 共享 `analysis/runtime.py` 的 lmms-eval 样本迭代，并
通过 semantic binding 访问 backbone；使用方式见[运行分析](1_run_analysis.zh.md)。

## 1. 扩展目标与准备

实现前写清：

- 研究问题和计算定义；
- 需要的模型语义节点；
- pre/post compression phase；
- 样本级结果和聚合方式；
- 是否需要干预、额外 forward 或 backward；
- 能够验证的 backbone；
- 产物中必须记录的因果对照或守恒条件。

如果现有 `AnalysisBackboneBinding` 无法表达目标语义，应先扩展公共 binding 和 contract，
不要在 Probe 中按模型类名定位私有层。

## 2. 最小扩展

将方法内置时需要接入以下文件；本地原型可先只提供自己的 Probe 模块，使用第 3 节的 Python 调试入口：

```text
mllm_comp/analysis/probes/<method>.py
mllm_comp/analysis/probes/__init__.py
mllm_comp/registry/catalog.py
examples/analysis/methods/qwen25_vl_<method>.yaml
tests/unit/test_<method>.py
```

Probe provider 至少实现严格 config 校验和 `open(...)`。`open` 返回 session，session
生命周期固定为：

```text
open(backbone, runtime, config, phase, output_dir)
  -> process(sample)  # 对共享迭代器中的每个样本调用
  -> finish()         # 返回一个 AnalysisRecord
  -> close()          # 成功或失败都必须清理
```

参考 `representation_dynamics.py`，不要在 Probe 中自建 lmms-eval adapter、重复加载模型
或自行决定总输出目录。

需要标准答案的 teacher-forcing 输入时，在实际 provider 类上声明
`requires_teacher_forcing = True`，当前 `HierarchicalTaylor` 使用此能力。runtime 汇总所选
Probe 的需求后，让共享样本迭代器为每个样本准备一次该输入；默认不需要。普通观察 Probe 仍可
读取 `target_text` / `target_source`，但不会因此触发第二次多模态输入准备。

## 3. 完整实现流程

### Config 与 definition

新增字段应有明确类型、默认值、非法组合和边界。模糊布尔值、未知键、重复逻辑名称、
空层集合和不支持的 metric 应在加载模型前失败。CLI override 只覆盖已有稳定语义，不
为单个 Probe 创建新的顶层命令。

### Semantic binding

使用 `resolve_analysis_binding()` 和公共 selector 获取 decoder layers、attention、FFN、
norm、lm head、input embeddings、DeepStack 等语义。需要新 handle 时同时补：

- binding 数据结构；
- 每个目标 backbone 的解析；
- 缺失语义的明确错误；
- `tests/unit/test_analysis_bindings.py` contract。

### Hook、干预和清理

使用 `HookSession` 管理 forward/pre-forward hook。临时模型修改必须可逆，并在 sample、
finish 或异常路径后清理。干预方法应记录替换位置、对照语义和适用 mask，而不是只保存
一个最终分数。

### 流式聚合与产物

优先使用 `StreamingAnalysisRecords`，避免把所有样本 tensor 留在内存。区分：

- sample artifact；
- task aggregate；
- overall aggregate；
- skipped sample 和原因；
- binding、layout、hook 与数值稳定性证据。

`finish()` 必须返回合法 `AnalysisRecord`，至少包含非空 name/method、phase、summary、
artifact paths 和 sample count。没有成功样本时应失败，不得生成看似完成的空记录。

采用 aggregate 加可选 raw 文件的 Probe，可调用
`analysis.artifacts.write_streaming_analysis_result()` 发布产物并返回记录，参考
`representation_dynamics.py` 的 `finish()`。方法负责计算 aggregate 和专属 summary；
公共函数填写成功/跳过计数、保留原始样本的流式写入，并统一记录身份。`filename_stem`
明确指定文件名前缀，不随逻辑 Probe 别名变化。需要额外的 aggregate metadata 时传入
`aggregate_metadata`；Taylor 的 scores/ranks 等独立产物继续使用底层 artifact writer。

### 注册与示例

Analysis catalog 的状态是 `active`、`target`、`future`，没有 `inactive` 或 `declared`。
`active` 条目的 provider 进入 `registry/analyses.py` 的运行注册表；`target` 和 `future`
用于记录候选或后续方向，runtime 会拒绝执行。它们不是压缩 SupportUnit 的证据状态。

尚未注册的本地 Probe 可先用已有 Python 调试入口，复用同一模型加载、样本循环与清理：

```python
from mllm_comp.analysis.definition import AnalysisDefinition
from mllm_comp.analysis.runtime import execute_analysis
from mllm_comp.config.loader import load_config
from my_probe import MyProbe  # 已实现 validate_config / open 及 session 生命周期

config = load_config("examples/analysis/methods/qwen25_vl_representation_dynamics.yaml")
config.output.run_dir = "outputs/local_probe_trial"  # 使用新的空目录
config.output.manifest_path = "outputs/local_probe_trial/manifest.json"
probe = {"name": "local_probe", "method": "my_probe", "layers": [0]}
MyProbe().validate_config(probe)
definition = AnalysisDefinition(
    compression_mode="none", phase="pre_compression",
    tasks=list(config.analysis.tasks), sample_limit=1, runtime_batch_size=1,
    probes=[probe],
)
execution = execute_analysis(config, definition, provider_resolver=lambda name: MyProbe)
print(execution.records)
```

此入口把未入 catalog 的方法记为 `external`，返回记录并写 Probe 自己的产物；不自动写
`run_experiment` 的 `result.json` 或 Manifest，也不经过其 definition 校验，本例因此显式
调用新 Probe 的 validator。它不会绕过已有 `target` / `future` 条目的限制。

要从统一的 CLI 和 `run_experiment(config, task="analyze")` 使用新方法，在本地代码中
导入 provider，并在 catalog 添加 `CapabilityKind.ANALYSIS`、`CapabilityStatus.ACTIVE`
的 `CapabilitySpec`，填写新方法的 name 与 provider，
用 `metadata["backbones"]` 限定实际支持基座。新进程会从 catalog 构造运行注册表，无需
手工修改 `ACTIVE_ANALYSES`。提供单样本、单层 YAML 后即可沿统一入口运行和记录结果；
当前没有 Analysis 版 `pipeline.extensions`，该字段只加载压缩方法注册函数。

## 4. 测试、激活与边界

最低证据包括：

1. 计算定义的确定性单测；
2. strict config 和 definition 回归；
3. session lifecycle、异常清理和输出隔离；
4. semantic binding 与 token group/layout invariant；
5. artifact envelope 和 `AnalysisRecord`；
6. 至少一个真实 checkpoint 单样本；
7. 每个拟激活 backbone 的独立证据。

需要 backward、attention probability 或因果干预的方法，还要验证 finite values、梯度
路径、eager/SDPA 实现边界和对照实验。一个 backbone 通过后再逐格扩大支持矩阵，不
把“代码可复用”写成“全部模型已验证”。

完成后同步更新 `1_run_analysis.zh.md`、示例 README、catalog 和项目进度。统一验收
规则见[能力验证](3_verify_capabilities.zh.md)。
