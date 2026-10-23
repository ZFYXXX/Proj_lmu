# Token 压缩数量合同与运行时决策重构

> 类型：通用架构 TODO / 决策记录
> 状态：Deferred；已有数据依赖的预计算路径，通用数量合同与布局传播尚未实施
> 记录日期：2026-08-12
> 最后核对：2026-09-08
> 启动条件：现有执行路径无法满足新的数量决策或跨 stage 布局需求时，再启动通用化

## 1. 为什么需要这项重构

当前 Qwen2.5-Omni 已有两种确定输出数量的路径。预算驱动方法可以在执行前计算精确数量：

- A-ToME 用 `merge_ratio` 确定每层 merge 数；
- Encoder Attention Top-K 用 `keep_ratio` 确定保留数；
- Progressive Pruning 用 pruning schedule 确定删除数；
- Segmentwise、LTBM、SA-MAP 最终都有精确预算；
- Audio-Pyramid 每个 stage 也有精确 keep ratio。

数据依赖方法则可以先执行预计算，再准备 LLM 输入布局：

- Encoder 的初始 Plan 允许 `keep_count=None`；Controller 复用已安装的 Encoder
  执行循环，以保留 RNG 的无梯度预计算得到实际数量，再更新 Plan 并调用 Legalizer。
- Projector 路径按需捕获所选边界的真实 hidden states，交给 Plugin 生成依赖数据的
  Plan；正式 forward 重新计算带梯度的特征，并重放已确定的计划。

因此，“所有 Plugin 都必须在看到数据之前给出精确数量”已不符合实现。当前要求是：
在 Qwen 消费 Audio placeholder 和 M-RoPE 布局之前确定实际数量；预算方法继续校验
planned/actual 一致，预计算路径还需保持计划重放和梯度语义。

实现见 [Encoder prepare_state](../../mllm_comp/backbones/qwen25_omni/encoder_compression.py)
和 [Projector prepare_state](../../mllm_comp/backbones/qwen25_omni/audio_compression.py)；
已有 [原生小模型回归](../../tests/integration/test_audio_custom_rules_cpu.py)覆盖自然数量、
计划重放与 checkpoint backward。这不代表通用 cardinality 接口已经实现，也不改变
各方法的 active/target 状态。

尚未解决的是跨 Binding 的统一数量合同、运行时布局结果和跨 stage 传播。例如纯阈值方法：

```text
merge every pair whose cosine similarity > 0.8
```

超过阈值的 pair 数依赖当前样本和当前层 hidden states，不能只靠输入长度和配置预测。
现有预计算解决了部分 Qwen 执行边界的需要，但还没有统一表达“何时确定数量、如何传播
实际布局”的公开合同。后续若需要单次 forward 内决策或跨 stage 组合，应在这些已有
路径上扩展，不能伪造 `keep_count`。

> `CompressionPlan` 应表达决策协议，不能默认等于已经完成的精确布局决策。

以下第 2–6 节保留尚未实施的通用化设计，不是当前字段或支持能力清单。

## 2. 已确定的设计原则

### 2.1 区分三类输出数量合同

Token Stage 至少需要表达以下三种 cardinality：

| 模式 | 请求前已知内容 | 运行时责任 | 典型方法 |
| --- | --- | --- | --- |
| `exact` | 精确输出数或可确定的逐层长度 | 决定具体索引，并兑现精确长度 | Top-K、固定 keep/merge ratio |
| `bounded` | 最小/最大输出数或最大压缩量 | 根据内容决定区间内的实际长度 | threshold + 最大 merge 数 |
| `runtime` | 决策规则，可能只有输入长度 | 完全根据当前样本产生实际长度 | 纯 threshold merge/drop |

具体类名在实现时根据现有接口确定，不预先承诺；但合同必须能明确回答：

- 是否承诺精确数量；
- 若不精确，允许的最小值和最大值是什么；
- 数量在 request、sample、layer、prefill 还是其他粒度确定；
- 下游何时可以读取最终布局。

### 2.2 Plan、Decision 与 LayoutDelta 分工

三类对象的语义必须分开：

```text
CompressionPlan
  = 执行前意图、规则、预算/边界和保护策略

Runtime CompressionDecision
  = 看到真实 hidden states 后得到的 selected indices、groups、scores 和实际数量

LayoutDelta
  = 决策执行后 source layout → result layout 的事实映射
```

`exact` 方法允许 Plan 携带精确数量，但仍由运行时 Decision 记录具体选择；
`bounded/runtime` 方法不得在 Plan 中伪装成 exact。现有 `core/layout.py` 的
`LayoutDelta` 应成为运行时事实，而不是只存在于 kernel 返回值或 manifest 空字段中。

### 2.3 Legalizer 按决策时机拆分

不能再假设所有布局修复都发生在 forward 前。通用流程需要区分：

1. **pre-decision preparation**：检查输入、保护 Token、准备 hook 和原始坐标；
2. **post-decision legalization**：根据实际 `LayoutDelta` 修复 placeholder、mask、
   position、M-RoPE、labels 和 media spans；
3. **post-stage propagation**：把实际布局交给后续 Stage，并支持多阶段 compose。

`exact` 方法可以保留提前 legalization 的优化路径，但其结果必须与运行时
`LayoutDelta` 一致。`bounded/runtime` 方法必须在实际决策后、下游消费者前完成修复。

### 2.4 不把 mask-first 当作已经完成短序列压缩

shape-preserving mask 可以作为动态阈值方法的参考实现，用于验证选择语义和梯度；但如果
仍执行完整 dense kernel，就不能报告真实 Token 长度、FLOPs 或延迟已经下降。框架应允许：

```text
dynamic decision → mask reference
dynamic decision → compact materialization
```

两者共享 Decision 语义，不强制共用执行器，也不能混淆效率声明。

## 3. 目标执行流程

### 3.1 `exact` 方法

```text
Plan(exact cardinality)
  → 可选提前 Legalize
  → forward Strategy decision
  → actual LayoutDelta
  → 检查 actual == planned
  → 下游模块
```

现有 Omni 预算驱动方法属于此类，迁移时应保持算法和长度结果不变；数据依赖的
预计算路径不能仅因最终 Plan 已有精确数量，就被归类为预算驱动方法。

### 3.2 `bounded` / `runtime` 方法

```text
Plan(policy + bounds)
  → pre-decision preparation
  → forward Strategy decision
  → actual cardinality + LayoutDelta
  → post-decision Legalizer
  → 更新当前 ForwardState / modality layout
  → 下游 Projector 或 LLM
```

对于 Encoder-Level 方法，Backbone Binding 必须提供一个安全边界，使实际 Encoder 输出
产生后、被 Projector/placeholder scatter/M-RoPE 消费前能够完成 legalization。若 provider
结构没有这个边界，应明确拒绝单次 forward 内决策的组合。现有 Qwen 预计算路径仍是
可用实现，但应单独记录预计算开销，不能把它当作已实现单次 forward 内的布局修复。

### 3.3 多阶段方法

每个 Stage 读取上一个 Stage 的实际布局：

```text
Layout⁰
  → Decision¹ / Delta¹
  → Layout¹
  → Decision² / Delta²
  → Layout²
```

第二阶段不得重新按原始 Token 索引决策。最终审计记录应能从各阶段实际
`LayoutDelta` compose 得到最终布局。

## 4. 与现有框架对象的关系

| 当前对象 | 需要修正的语义 |
| --- | --- |
| `CompressionPlan` | 增加/关联 cardinality contract；不再默认精确输出布局 |
| `TokenAction` | 表达执行意图或预先已知选择，不能同时冒充所有运行时结果 |
| `EncoderTokenStrategy` 等 Strategy | 返回结构化 Decision，而不只返回 tensor、length 和 debug dict |
| `LayoutDelta` | 成为每次真实 token mutation 的必需事实记录，并进入 State/Session/Manifest |
| `ForwardState` | 持有每个 modality/media unit 的当前布局，而不只保留原始 spans |
| Backbone Binding / Controller | 声明支持 `exact`、`bounded` 或 `runtime` 中哪些模式，以及安全 legalize 边界 |
| Legalizer | 支持 pre-decision 与 post-decision 生命周期 |
| SupportUnit | 按 backbone × modality × placement × phase × cardinality capability 精确激活 |
| Runtime/Manifest evidence | 记录 planned contract 与 actual decision，不能只记录预计压缩率 |

Operation 压缩的静态/动态/混合执行由
[`Operation 压缩执行模型 TODO`](operation_compression_execution_model.zh.md)负责。本文件只处理
Token 布局和输出数量何时确定，不把两类重构混成一个 executor。

## 5. 推荐实施顺序

### M0：冻结现有预算与预计算基线

- 完成 A-ToME、Segmentwise-TopK、Audio-Pyramid 的人工语义复核；
- 保持 active 方法的 O/E/T 自动回归和已有自定义规则小模型回归通过；
- 保留 Encoder/Projector 自然数量、预计算 RNG、计划重放与梯度检查；
- 记录现有方法的逐 stage planned/actual 数量和布局不变量；
- 重构期间不重新解释算法公式或默认参数。

### M1：增加中立的 cardinality contract

- 让 Plan 明确声明 `exact`、`bounded` 或 `runtime`；
- 为 bounded/runtime 表达真实上下界和决策粒度；
- 为现有预算驱动方法声明 `exact`，为数据依赖路径记录其实际决策时机，行为保持不变；
- registry 和 Binding 对不支持的模式 fail closed。

### M2：统一运行时 Decision 结果

- 定义 Strategy 的结构化运行时结果；
- 至少包含实际数量、selected indices/merge groups、`LayoutDelta` 和必要 metadata；
- 让 Controller/Session 收集实际结果，不再从 Plan 推测真实布局；
- 多 media unit 各自产生独立 delta，禁止跨 unit 合并。

### M3：接通 post-decision legalization

- 在通用 lifecycle 中区分 preparation 与 legalization；
- 以已有 Projector 预计算为基线，在实际需求要求时验证单次 forward 内的动态长度修复；
- 再处理 Encoder-Level 的 pooler、placeholder 和 M-RoPE 边界；
- 将实际布局写入 `ForwardState`、Session metadata 和 manifest stage record。

### M4：迁移现有 exact 方法

- 用 Segmentwise-TopK 迁移 Projector exact 路径；
- 用 A-ToME 迁移 Encoder 逐层 exact 路径；
- 用 Audio-Pyramid 迁移 LLM prefill 多阶段 exact 路径；
- planned/actual equality 继续作为 exact 方法的强断言。

### M5：增加一个最小纯阈值 contract vertical

- 使用小型、确定性的 threshold merge/drop kernel 验证 `runtime` cardinality；
- 至少构造两个样本，使相同阈值得到不同输出长度；
- 验证 actual `LayoutDelta`、post-decision legalization 和下游 forward；
- 该切片用于证明框架能力，不要求论文新颖性、benchmark 或发布状态。

### M6：收敛集成和文档

- eval/train bridge 消费实际 Decision/LayoutDelta；
- 新增方法指南分别展示 exact 与 runtime 示例；
- 删除所有把 Plan 数量无条件当作实际数量的通用路径；
- Qwen Omni Descriptor 路径已随
  [`Omni v2 收敛历史记录`](../archive/omni_compression_v2_convergence.zh.md)删除；后续 cardinality
  扩展只修改 v2 contract/session，不恢复双轨。

## 6. 最小验收标准

### Exact 路径

- 现有方法的 O/E/T 行为和结果不退化；
- Plan 精确数量、Strategy 实际数量与 `LayoutDelta.result_count` 一致；
- 多 audio unit、chunk、pooler、placeholder 和 position 不变量保持成立。

### Bounded / Runtime 路径

- 相同配置可因样本内容产生不同合法输出长度；
- Plan 不虚报精确数量，实际结果满足声明的 bounds；
- 每次 mutation 都产生可组合的 `LayoutDelta`；
- placeholder、mask、position、labels 和 modality spans 使用实际结果修复；
- 多样本和多 media unit 不串流；
- 下游 forward、必要的 backward 和声明支持的 phase 可运行。

### 研究人员扩展

- 新增纯阈值方法只新增 kernel/algorithm、variant、必要 Binding 和 SupportUnit；
- exact/bounded/runtime 的数量规则只定义一次，由 Plan 与运行时共同消费，不要求研究人员
  在 Plugin 和 Strategy 中维护两份长度逻辑；
- 不需要在共享 runtime 中加入论文方法名判断；
- Binding 不支持动态 cardinality 时给出明确能力错误；
- mask reference 与 compact execution 的能力和效率报告分开。

具体研究者接口与开发运行/支持晋级的拆分见
[`低成本算法扩展 TODO`](low_cost_compression_algorithm_extension.zh.md)。

## 7. 暂不纳入本轮重构

- 修改已验证方法的论文语义、默认参数或质量结论；
- 为所有 backbone 一次实现所有 cardinality 模式；
- 将已有两遍 Encoder 预计算推广为所有方法的默认路径；现有数据依赖路径继续保留，
  并如实计入预计算成本；
- 把 padded/masked dense 张量直接宣称为真实短序列加速；
- 同时解决 Operation、Token、weight、channel 和 expert 的全部动态压缩；
- 为尚无真实调用者的旧接口建立长期兼容层。

## 8. 预计涉及的代码区域

启动实现时优先复核：

- `mllm_comp/core/plan.py`
- `mllm_comp/core/actions.py`
- `mllm_comp/core/layout.py`
- `mllm_comp/core/state.py`
- `mllm_comp/compressors/strategies.py`
- `mllm_comp/compressors/contracts.py`
- `mllm_comp/compressors/multimodal_registry.py`
- `mllm_comp/pipeline/runtime_session.py`
- 各 backbone 的 compression Binding、Controller 与 legalizer
- `mllm_comp/run_manifest.py`

实现前应以预算驱动和数据依赖的现有 Token 路径重新核对这些接口，只做满足 exact/runtime 两条真实
vertical 所需的最小重构；本 TODO 固定语义方向，不预先固定最终类名。
