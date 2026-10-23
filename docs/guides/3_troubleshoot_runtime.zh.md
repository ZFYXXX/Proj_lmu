# 排查运行时问题

> 类型：Guide
>
> 文档层级：3 / 平台维护者
>
> 目标结果：把失败定位到配置、依赖、模型、binding、外部框架、训练或产物层，并用最小回归确认修复
>
> 适用范围：`analyze`、`eval`、`train` 和 compression runtime
>
> 当前状态：Active

排错从最早失败边界开始，不先修改算法或添加兜底。保存原命令、完整 traceback、有效
配置、输出目录和运行环境；失败重跑使用新目录。模型专用细节继续查阅
`docs/support/*/debugging/`，本指南只维护跨模型诊断顺序。

## 先按修改任务找到责任代码

普通模型加载主线是：

```text
run_experiment → runtime/loader → activate_runtime_compression
  → CompressionDefinition → Dispatcher.resolve → Session.install
  → Binding.executor_factory → 对应 provider/controller → model.forward/generate
```

Swift 拥有训练循环，在其 model loader 返回后接入同一个压缩 session。每个 Binding
生成一个安装记录，Session 按依赖创建并安装 executor。`prepare` 可以改变模型结构，
所以 executor 安装前会重新读取物理句柄；这与配置解析时的校验目的不同。

| 要修改什么 | 首先阅读 | 何时需要深入 |
| --- | --- | --- |
| 数学、评分、选择、合并 | 所选 `compressors` 方法及 rule 消费点 | 现有信号或执行位置不足时，才进入对应 provider |
| 新方法复用现有机制 | [方法接口](2_dev_compression_method.zh.md#3-先确认执行机制能否复用)与[扩展示例](../../examples/compression_extensions/README.md) | 先确认输入、结果和状态兼容，再用 `compressors/extensions.py` 注册 |
| 图像方法的基座位置与回调声明 | `compressors/image/contracts/` 中相应基座文件 | 中央 registry 与适配器共用这份物理声明；实际布局处理仍在对应基座 |
| Qwen3 Attention/FFN 模块替换 | `compressors/operation_registry.py` 与 `backbones/qwen3_vl/operation_binding.py` | 算法负责规划和构造；Session 按 prepare 顺序安装与回滚 |
| 配置与计划、启动前错误 | `config/validators.py`、`experiments.py`、`pipeline/definition.py` | 先根据 `error.stage` 和字段路径定位 |
| 安装顺序、异常释放、结构修改后的 hook | `pipeline/runtime_session.py` | 沿所选 Binding 的 factory 查实际安装对象 |
| mask、labels、position 或旁路布局 | 对应基座 `provider_bridge.py` 或 Omni controller | 图像 LLM 请求阶段由 `decoder_lifecycle.py` 管理 |
| Swift 参数、训练加载、恢复 | `integrations/ms_swift/adapter.py`、`training_plugin.py`、`restoration.py` | optimizer/数据循环归原生 Swift；不在算法里另建 Trainer |
| decode KV 的读取、释放或 beam 重排 | DyCoKe controller 或 `qwen3_vl/attention_replacement.py` | 选择策略在 `compressors/video/dycoke.py`；区分物理缓存、稀疏读视图和递归历史 |

`ForwardState → CompressionPlan → Legalizer` 仍是保留的参考接口。排查实际压缩方法时，
先沿上面选中的 Binding 追踪，不需要先读参考执行器。

## 1. 最小复现

先把问题缩小到：

- 一个固定 checkpoint；
- 一个 task/sample 或一个 train batch；
- 一个 Probe 或一个 compression stage；
- 单进程、单 GPU（除非问题只在多卡出现）；
- 默认 runtime mode，不同时开启 profile/debug；
- 一个全新的输出目录。

记录：

```bash
python -V
python scripts/diagnostics/validate_framework_integrations.py
nvidia-smi
```

`validate_framework_integrations.py` 是依赖诊断，不是通用 preflight，也不加载模型。
`analyze/eval/train --plan_only` 均可检查计划；`output.effective_config` 显示本次实际配置。任何计划或
配置检查通过都不代表真实问题已经复现或修复。

## 2. 分层诊断

### A. 配置和 definition

症状通常是模型加载前出现 `ConfigError`、unknown key、inactive capability 或定义冲突。

检查：

- command 与 YAML 工作流是否一致；
- `compression_mode`、target 和 enabled stages；
- algorithm/variant/binding/placement/phase 精确匹配；
- CLI override 是否产生空值或非法组合；
- output/manifest 是否指向已有非空运行；
- task/Probe 名称是否被 catalog 激活。

不要增加第二套 schema-only 路径。真实命令和 runtime activation 必须共用 definition。

### B. 依赖和启动计划

如果 integration 未安装、版本错误或 launcher argv 不正确：

```bash
python scripts/diagnostics/validate_framework_integrations.py
mllm-comp eval --config /path/to/eval.yaml --plan_only
mllm-comp train --config /path/to/train.yaml --plan_only
```

检查当前 environment 的 editable package 是否指向当前 Dev lane。本项目不设固定 Review
工作区；并行验证其他版本时使用独立 Dev 工作树及其环境，不可指向另一 lane 的源码。
不可变 execution 环境不能使用 editable 安装。

### C. 原生 Backbone

先关闭 compression，在同一 checkpoint 和样本上运行 native baseline。若 baseline
失败，问题不应归因于 compressor。按顺序检查：

- processor 输出和媒体加载；
- model/device/dtype/attention implementation；
- `ForwardState` token spans；
- 原生 validator；
- runner forward/generate；
- prompt/chat 和 output parsing。

Backbone 专项入口见各目录 README 和 `tests/integration/test_*_gpu_smoke.py`。

### D. Compression binding 和 legalizer

Baseline 通过、compressed 失败时，先确认 kernel/Strategy 的确定性测试，再检查：

- hook/target site 是否命中一次且只命中应有阶段；
- source/result token 数与 budget；
- `LayoutDelta`/Alignment 语义；
- placeholder、grid、mask、position、M-RoPE、DeepStack、AnyRes；
- prefill/decode 和 KV cache；
- runtime metadata；
- uninstall/异常清理。

不要通过静默跳过 stage 或回退 baseline 让命令成功。目标位置不存在或 invariant 失败
必须明确报错。

### E. lmms-eval/ms-swift

Native 路径通过但外部框架失败时，比较原生框架最终 argv、环境和日志：

- lmms-eval model name、plugin spec、task、batch、seed、output path；
- chat/simple 路由和媒体请求；
- accelerate/torchrun 的 rank、端口和 GPU 可见性；
- ms-swift model type、template、dataset、target modules 和 output；
- 原生结果是否被平台正确收集。

不要修改 third-party 的有效 HPC 数据路径来“恢复上游默认值”。实际任务数据来源以目标
环境中的 task YAML 和最终 argv 为准。

### F. 训练

如果 forward 成功但 backward/step 失败，检查：

- loss 是否连接到目标值路径；
- hidden states 是否意外 detach；
- 离散选择是否允许训练且是否需要可学习参数；
- gradient checkpoint replay 是否重复做动态选择；
- labels、padding、position 和 teacher forcing；
- finite non-zero gradient；
- checkpoint 保存范围和 reload 契约。

一步训练通过但正式训练失败时，再检查数据规模、OOM、distributed state、resume 和长
序列边界，不回头重写已通过的算法 kernel。

### G. 输出和 Manifest

命令成功但没有有效结果时检查：

- output path 和原生框架实际产物；
- Manifest 中 task、metrics、stage、invariant 和 trace；
- skipped samples；
- baseline/compressed target；
- 是否复用了旧目录；
- 多进程是否由正确 rank 聚合和写文件。

空指标、空 `AnalysisRecord`、没有实际 compression metadata 或混合旧产物都不能算
成功运行。

## 3. 常见症状

| 症状 | 首先检查 |
| --- | --- |
| 研究方法没有 SupportUnit | 这本身不阻止运行；按实际错误检查 factory、phase、位置及输入要求 |
| Token 数变化后生成失败 | layout、placeholder、position、mask、grid |
| Prefill 成功、decode 失败 | KV cache、position、是否重复选择 |
| loss finite 但梯度为零 | detach、目标值路径、离散选择语义 |
| gradient checkpoint 时结果漂移 | replay 是否重新执行动态决策 |
| reload 后回到 baseline | 导出范围或 YAML 重应用契约 |
| 单卡成功、多卡失败 | local rank、端口、GPU 可见性、聚合输出 |
| 质量正常但 profile 变慢 | debug hook、同步点、稀疏后仍走 dense kernel |
| Manifest 存在但无指标 | 原生结果解析和 target 归类 |

## 4. 选择足够的回归

| 改动 | 首选检查 |
| --- | --- |
| 只改说明、移动展示代码 | 相关链接或已有展示行为；不新增测试正文匹配 |
| 修改算法/rule | 该方法的小型确定性输入；变换必须有预期数值、索引或分组 |
| 修改研究方法注册 | `test_researcher_extension_contract.py` 与该执行族的一个原生接入案例 |
| 修改安装、回滚和依赖顺序 | `test_runtime_compression_session.py`；影响训练时加已有 Swift session 案例 |
| 修改训练状态或恢复 | `test_experiment_restore_cpu.py`，并按变更选择原生后端生命周期 |
| 修改真实布局/cache | 对应模型、位置或 cache 案例；确实影响设备行为时再加 GPU |

选用已有测试中的对应案例，不要求每次运行整行全部文件。研究方法可有自己的算法
测试并复用现有执行族回归；不因新增一个名字复制所有基座与模态的测试矩阵。

公共函数的成功、失败和清理行为只测一处，各入口保留参数及委托关系检查。不同算法的
独立数值预期、不同基座的布局/位置/梯度/cache 行为仍分别测试；调用名字不同但实际是
同一函数，不需要重复整组输入。模型测试使用统一环境中的 PyTorch 和原生小模型，
无需为缺少 PyTorch 另造一套假 `nn`。

视频诊断入口复用 `video_checkpoint_common.py` 的加载、baseline/压缩比较和 GPU 报告，
训练入口复用 `video_training_checkpoint_common.py` 的更新、保存、重载及证据函数。
入口保留方法检查和基座输入的实际差异；历史失败批次留在运行产物中，不写进新运行的代码。

## 5. 修复验收

修复后只重跑能够证明原失败消失且不破坏相邻合同的最小集合：

1. 原最小复现；
2. 直接覆盖改动的 unit/contract test；
3. 适用的真实 checkpoint 最小运行；
4. 涉及 integration 时的原生框架单样本或一步训练；
5. 涉及 reload/multi-stage/multi-GPU 时的对应生命周期回归。

记录命令、exit status、关键输出和新的 artifact。没有修改相关代码或条件时，不重复
运行同一昂贵检查。达到激活标准后再按[能力验证](3_verify_capabilities.zh.md)更新状态。
