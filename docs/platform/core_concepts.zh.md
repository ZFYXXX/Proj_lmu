# MLLM-Comp-Platform 核心概念

> 类型：Reference / Tutorial
> 当前状态：Active
> 正式契约：[接口规范](interface_spec.zh.md)

这是一份教程式说明。正式字段和验收条件以 `interface_spec.zh.md` 为准。
第一次阅读可先按[自己的任务选择入口](#15-推荐阅读代码顺序)，无需先读完所有类和模型适配代码。

## 1. 一句话主线

```text
config
  -> run_experiment (shared by Python and CLI)
  -> independent run directory / effective config / definition validation
  -> lmms-eval / ms-swift backend or analysis runtime
  -> load BackboneRuntime through BackboneFacade
  -> install Binding executors in CompressionSession
  -> backend calls model forward / generate
     -> executor runs compression and backbone repairs model state
  -> result.json / trace / manifest / cleanup
```

职责分工：

1. config 描述实验，`run_experiment` 管理一次独立运行，definition 校验解析方法与执行要求；
2. 外部训练/评测后端拥有数据循环，通过平台接入加载模型和安装压缩执行器；
3. 动态压缩在真实模型调用期间决定保留、合并等动作，由基座适配修复相应状态；
4. 静态 Operation 在安装阶段改写模型，先于 runner 或 optimizer 使用模型；
5. `result.json` 记录成功或失败，trace 和 manifest 记录实际执行与结果。

以上是内置多模态 v2 路径。Qwen3-VL Attention/FFN replacement 当前仍由显式
Operation binding 接入同一个 session。`ForwardState → CompressionPlan → Legalizer`
是保留的参考执行接口，见第 8、9 节；真实 eval/train 不要求所有方法先经过该接口。

## 2. 为什么需要统一抽象

多模态模型在 processor 输出、模态 token 位置、mask、position、label 和内部模块命名上差异很大。如果每个压缩算法直接修改具体模型 tensor，它会同时承担算法、模型适配和状态修复，难以复用和排查。

平台把问题拆开：

- 模型差异留在 backbone；
- 算法意图留在 compressor；
- 安装与依赖顺序留在 session，张量布局和模型状态修复留在基座 executor/bridge；
- 实验事实留在 trace/manifest。

### 结构位置与生成阶段

描述压缩时，分别说明“在模型哪里执行”和“在什么时候执行”：

| 维度 | 术语 | 含义 |
| --- | --- | --- |
| 模型结构位置 | Encoder / Projector / LLM-level | 所作用的模块或层；LLM 的 Transformer 层也常被原生模型称为 Decoder 层 |
| 生成阶段 | prefill | 处理输入上下文；缓存生成时建立对应的 KV 状态 |
| 生成阶段 | decode（增量解码） | 利用已有上下文状态继续生成后续 Token |
| 直接前向 | 训练前向 / 推理或分析 forward | 直接计算给定序列；训练中的 teacher forcing 不称为 decode |

自然语言中用“LLM 层”或“LLM-level”表示结构位置，不能用单独的“Decoder”代指增量 decode。
`forward()` 是调用形式，prefill、decode 和训练都可能调用它；Stage 的 `generate` 则覆盖生成请求，
不自动表示压缩决策在 decode 执行。各方法须另外说明决策发生于 prefill、decode 或两者。
prefill 中随 Token 压缩进行的 KV 同步修复，也不能作为已实现 decode 阶段 KV 压缩的依据。
引用既有类名、文件名、`target_site` 和枚举值时保留代码原名，并按上述维度解释。

### 状态属于谁

模型参数、一次请求的决定和生成 cache 有不同寿命：模型模块持有可训练参数；执行器
持有配置和安装句柄；请求上下文持有本次信号、布局和待重放决定；生成历史跟随实际
cache。由权重计算出的代理还要在权重更新时失效，beam 重排还要同步对应的递归历史。
实际实现与边界见[架构中的状态归属表](architecture.zh.md#3-stage-与运行时)。
因此，重用同一个 Python 模型对象，不表示可以重用上一次请求的决定或更新前的派生值。

## 3. 组合式 `BackboneFacade`

facade 是模型接入的统一入口，但不是单体大类：

```text
BackboneFacade
  ├─ processor_adapter
  ├─ model_adapter
  ├─ state_builder
  ├─ native_validator
  ├─ legalizer
  ├─ runner
  └─ operation_applier (optional)
```

processor/model adapter 共享同一 capability profile。加载后的大型 model 和 processor 实例由一次运行中的 `BackboneRuntime` 持有，不存进 facade。拥有 runtime 的调用方负责关闭它；`close()` 卸载压缩和调试会话，并释放其模型、processor 引用。

这样可以单独测试状态构造和 legalization，也允许 Qwen2.5-Omni、Qwen3-VL 与 Qwen2.5-VL 复用部分组件。

## 4. `ForwardState`

`ForwardState` 是 compressor 和 pipeline 看到的统一模型状态，包含：

- token carrier：`input_ids` 或 `inputs_embeds`；
- 派生字段：`attention_mask`、`position_ids`、`labels`；
- media tensor 和 grid；
- `modality_token_spans`；
- sample id、hidden state 和 stage history。

### `modality_token_spans`

span 表示某个 sample 的某类模态在序列中的半开区间：

```text
sample 0: image tokens occupy [35, 611)
```

Qwen2.5-Omni 原生接入要求 image、video 和 audio span；Qwen3-VL 要求 image/video span，并允许时间戳对齐后同一 video 产生多个不连续 span。scene/3D 类型仍为后续扩展预留。

关键规则：

- StateBuilder 创建 span；
- compressor 读取 span；
- legalizer 在 token 数变化后修复 span；
- invariant 检查顺序、边界和重叠。

### 原生输入输出与压缩状态修复不是一件事

`ProcessorAdapter.prepare_inputs` 把对话和媒体转换成 `ProcessedBatch`，`StateBuilder` 负责字段归一化，`native_validator` 检查基座原生契约。Qwen2.5-Omni 覆盖 image/video/audio、`use_audio_in_video`、Thinker text、Talker codec 和 Token2Wav waveform；Qwen3-VL 覆盖 text/image/video 和 text output，并显式拒绝 audio/Talker。

compression `Legalizer` 只处理 token action 造成的 token/grid/mask/position/span 变化。基座支持音频或 Talker，不代表 audio compression 已经实现。

`GenerationResult` 分开保存 `TextGenerationOutput` 与可选 `SpeechGenerationOutput`。通用 text logits/hidden states/attentions/KV cache 默认不保留；Qwen2.5-Omni 额外支持 Thinker/Talker、`thinker_reply_part` 与 codec trace。只有 `TraceRequest` 明确请求时才通过 hook 写入 `TraceArtifacts`。

## 5. `TokenAction`

`TokenAction` 表达“希望如何变换 token”，例如：

- keep；
- drop；
- merge；
- pool；
- reorder；
- reweight。

参考接口中的 fixed visual token drop 可以产生：

```python
TokenAction(op="keep", target_indices=...)
```

action 本身不修改 `attention_mask`、`position_ids`、`labels` 或 span。

## 6. `Legalizer`

legalizer 把 action 应用到 state，并修复所有派生字段。

删除 image token 后，至少可能影响：

- token carrier 的 sequence length；
- attention mask；
- labels；
- modality span；
- Qwen M-RoPE position。

因此 legalizer 必须是 backbone-specific。通用 pipeline 负责调用它，但不能假设所有模型都采用相同 position layout。

legalizer 返回新 `ForwardState`；不原地修改输入 state。

## 7. `OperationAction`

方法声明与执行动作使用同一对象语义：`AlgorithmDescriptor.method` 为
`operation_compression` 时，`operations` 使用 `OperationKind`，例如
`frozenset({OperationKind.PRUNE})`；其 Variant 显式声明 `token_unit=None`。
Token 方法仍使用 `TokenOperation` 和对应 Token 单位。注册表核对两者，纯结构操作
不再借用 Token 删除来描述。

OperationAction 描述静态计算路径变化，而不是 token 变化，例如：

- skip layer/attention/FFN；
- mask attention route；
- prune head/channel；
- replace module。

route 使用 target query 到 source key/value 的方向：

```text
v2v  v2t  t2v  t2t
```

当前已激活的 operation compression 有两个基座限定子集。Qwen2.5-Omni Thinker LLM 支持完整层结构剪枝：
`op="prune"`、`target_kind="layer"`、`target_scope="llm"`、
`apply_mode="structural_rewrite"`、`shape_effect="remove_module"`。它在 runner 和
optimizer 创建之前由 backbone-specific `OperationApplier` 执行，并要求导出可重载的
Transformers checkpoint。

Qwen3-VL LLM 支持 prepare-time、保持 hidden-state I/O shape 的 Attention/FFN replacement：
`op="replace"`、`target_kind="attention" | "ffn"`、`target_scope="llm"`、
`apply_mode="structural_rewrite"`、`shape_effect="preserve_io_shape"`。精确策略由 catalog
声明；它们在原始 checkpoint 加载后按 YAML 安装，session 卸载
时恢复原模块，reload 时重新应用配置而不是导出独立 replacement checkpoint。

除这两个精确子集外，Attention/FFN pruning、head/channel/expert、route mask、跨基座
replacement、任意通用组合与 Talker operation 仍只是接口词汇，不是已激活能力。

## 8. `CompressionPlan`

plan 是 compressor 和 pipeline 的交接物，说明：

- stage、placement 和 execution phase；
- 目标 modality 和 token role；
- method 和 object family；
- token actions；
- legalization policy；
- operation actions。

通用 token plan 的核心规则：

```text
compressor 只交计划
pipeline 执行计划
```

这条规则描述 `CompressionPlan`/legalizer reference contract。ImageLLM 与
Qwen2.5-Omni AudioLLM 都先解析 Algorithm/Variant/Binding，再由
`CompressionSession` 管理 Binding executor；Qwen 的动态 Token controller 和
prepare-time OperationApplier 是各自 Binding 的物理执行器。SupportUnit 单独记录精确身份的
验证状态；合法研究方法可以没有 SupportUnit，也不会因此自动获得正式支持声明。

`method="quantization"` 仍只有保留的配置词汇，尚无可执行的算法契约。Token 与
Operation 操作声明的区分不代表量化接入或任意跨类型组合已经实现。

## 9. `CompressionPipeline` 与真实 runtime

调用参考 `CompressionPipeline.apply` 时，一个 Token stage 的执行顺序为：

1. 解析 active plugin；
2. plugin 生成 plan；
3. legalizer 应用 token action；
4. 执行通用和 backbone-specific invariant；
5. 写 stage metadata；
6. 更新 `state.stage_history`。

在调用通用 token pipeline 时，它负责写自己执行结果的 stage history；integration 不得
绕过执行器写入。真实多模态路径由 `CompressionSession` 统一管理 provider activation、
依赖与逐阶段 runtime metadata；Qwen controller 在 Binding 内返回合法 state 和执行记录。

`CompressionPipeline.apply` 已实现 token plan、legalization、通用 invariant、metadata 和
stage history 的 reference 执行，并明确拒绝 `operation_actions`。真实 eval/train 由
`activate_runtime_compression()` 和 `RuntimeCompressionDispatcher` 投影 v2 合同并创建
session；已激活的 Qwen LLM Operation 使用 Binding 内的 prepare-time `OperationApplier`，
因为结构改写必须发生在 runner/optimizer 之前。runtime/integration 调用方再把 report、
trace、训练 evidence 与 Manifest 串入各自生命周期。definition validator 保存已解析的
v2 合同投影，同一次 runtime activation 将它交给 Dispatcher 和视频执行器复用；运行时
仍检查请求 phase、模型 structural handles、执行器及物理合同。历史支持证据不作为研究运行准入。直接调用 Dispatcher 或视频
执行器时仍执行对应的定义或配置校验。这些检查不代替真实 planner、
legalizer、provider/operation apply 或 forward 验证。

参考执行器保持 `from mllm_comp.pipeline.runner import CompressionPipeline` 的导入
接口及 `apply` 行为，用于检验 token plan、调用方提供的 legalizer 和状态不变量。
接入真实压缩方法应复用已有 Binding/session 路径；参考执行器不承担模型 hook 生命周期。

通用 token legalization 和统一 checkpoint bundle 仍属于未来设计方向，当前不提供
只会抛出“未实现”异常的占位 API。实际 token 修复由各基座的 legalizer/bridge 负责；
已支持的结构剪枝导出使用 `mllm_comp.export.operation_export.export_operation_actions`。
`native_source` 仍按 catalog 标为 FUTURE，不因占位整理而升级或删除。

## 10. `StructuralHandles`

operation compression 需要稳定的语义路径，例如：

```text
llm.layer[0]
llm.layer[0].attn
llm.layer[0].ffn
vision_encoder
audio_encoder
projector
```

plugin 使用这些路径，而不是猜测不同源码树里的 Python 属性名。

## 11. Analysis 与 Eval

lmms-eval wrapper 负责：

- task document；
- prompt；
- image/media；
- task metadata。

平台负责：

- processor/state；
- compression-aware execution；
- mechanism probe；
- artifact、metric 和 manifest。

analysis 与 eval 共用样本构造，避免同一任务出现两套 preprocess 逻辑。

原生 backbone 激活先验证全模态输入、text/speech 输出和按需内部 trace；压缩 probe 和 eval task 在真实压缩里程碑中接入。

## 12. Manifest

manifest 是实验账本，回答：

- 使用了什么 config 和 override；
- 使用了哪个模型、revision 和环境；
- 哪个 stage 执行了什么压缩；
- token 数和 invariant 如何变化；
- analysis/eval 产物在哪里；
- 是否可以复现。

当前 schema v2 的主记录是
`compression_stages: list[CompressionStageRecord]`，逐阶段统计写入
`per_stage_stats`；单 compressor 字段只保留为历史 artifact 的序列化投影，不参与 Qwen
runtime 选择。

Python 与 CLI 共用 `run_experiment()`：analyze、eval、train 在实际执行前写入
`effective_config.json` 和状态为 `running` 的 `result.json`；成功后写本次最终 Manifest，
再把结果状态更新为 `succeeded`。执行失败写结构化失败结果，不伪造成功 Manifest；
`plan_only=True` 不写运行文件。每次运行有独立目录，已有运行产物不会被覆盖。

Manifest 保存该任务实际返回的指标、checkpoint、analysis 记录及可获得的压缩运行元数据。
它与详细 trace、原生训练日志和导出产物互相引用；字段缺失不等于已完成相应验证。
这套运行记录已经可用，但不能替代任意算法的训练恢复验证，也不表示已有通用量化或
所有跨模态组合的 checkpoint bundle。使用入口见[运行实验指南](../guides/1_run_compression_experiment.zh.md)。

## 13. 原生 backbone 调用示例

```text
text / image / audio / video sample
  -> Qwen processor
  -> ProcessedBatch keeps provider-native inputs
  -> StateBuilder creates image/video/audio spans
  -> native validator checks grids/features/placeholders
  -> Thinker generates text
  -> optional Talker generates codec tokens
  -> Token2Wav generates waveform
  -> opt-in hooks record cross-stage artifacts
  -> smoke report records environment and outputs
```

压缩通过选中的 Binding executor 接入这条原生链路；具体训练、结构恢复和模态组合范围以
方法契约及对应证据为准。原生基座的输入输出能力不会自动成为压缩或量化支持。

## 14. 常见问题

### 为什么 compressor 不直接改 tensor？

否则 mask、position、label 和 span 修复会散落在每个算法中，无法跨 backbone 复用。

### 什么情况下研究方法会被拒绝？

缺少可执行工厂、请求了未声明的 phase、位置或模态，或者模型结构不满足 Binding 时会失败。
已有执行机制且合同合法的方法可以运行；没有历史 SupportUnit 证据时，运行记录保留这一事实，
不会把一次研究运行当成正式能力晋级。

### integration 或 third_party 能写算法吗？

不能。wrapper 只连接外部框架；可复用算法属于 `compressors`、`pipeline` 或 `analysis`。


## 15. 推荐阅读代码顺序

| 当前任务 | 从哪里开始 | 何时继续往下读 |
| --- | --- | --- |
| 压缩使用者：运行、比较、恢复实验 | [运行实验指南](../guides/1_run_compression_experiment.zh.md) → 对应 YAML → `experiment_tools.discover_capabilities` / `experiments.run_experiment` → 结果与 `compare_runs` | 配置被拒绝时查看 `config/schema.py`、`pipeline/definition.py`；异常再沿所选 Binding 查实际调用。 |
| 压缩研究者：修改规则或新增方法 | [压缩扩展指南](../guides/2_dev_compression_method.zh.md) → 所选 kernel、Spec 和规则消费点 → 对应数值测试 | 需要新信号、位置或状态时，再读 `contracts.py`、Binding 的 factory 与对应 provider；已有机制可复用时调用 `compressors/extensions.py::register_method`，从 `examples/compression_extensions` 的完整方法开始。 |
| 框架维护者：修接线和生命周期 | `pipeline/definition.py` / `v2_validation.py` → `runtime_session.py` → 所选 executor / provider | 用实际 tensor、布局和 cache 调用定位责任，再查 `run_manifest.py`、`core/manifest.py` 的记录；分析流程从 `analysis/runtime.py` 进入。 |
| Agent：自动迭代实验与改进研究能力 | [Agent 研究入口](../guides/4_agent_research.zh.md) → 能力发现、计划、运行、比较与恢复 | 修改方法时复用研究者入口；修改执行机制时复用维护者入口。auto-research / RSI 的自动编排仍属后续研究范围。 |

Agent 使用相同的 `discover_capabilities`、`run_experiment(..., plan_only=True)`、正式运行及
`compare_runs`。短期自动尝试配置时读取 `effective_config.json`、`result.json`、trace 和
Manifest；需要修改方法时沿研究者路径增加实现及相应证据。规划结果和单次运行成功
分别表示配置可解析、该次执行完成，均不会自动升级支持状态。

需要研究参考 plan/legalizer 接口时，再阅读 `core/state.py`、`core/actions.py`、
`core/plan.py` 和 `pipeline/runner.py`。
