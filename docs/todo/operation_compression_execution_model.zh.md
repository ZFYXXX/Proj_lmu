# Operation 压缩统一决策与分离执行重构

> 类型：架构 TODO / 决策记录
> 状态：Deferred；本文的通用决策/执行重构未启动，主线已有方法专用动态与 Mask 实现
> 记录日期：2026-08-12；范围核对：2026-10-03
> 启动条件：在实际方法需要统一 Operation 决策合同且范围明确后另行启动

## 1. 前置工作与启动顺序

本 TODO 最初要求先依次完成：

1. A-ToME：Encoder Token 路径；
2. Segmentwise-TopK：Projector Token 路径；
3. Audio-Pyramid（`audio_pyramid_attention_pruning`，即本轮讨论中的
   Audio-PDrop）：LLM Token 的 prefill/decode 路径。

三项验证和 SPADE 静态 v2 迁移现已完成；本节保留启动顺序的设计背景。动态/混合
Operation 仍需以 SPADE Operation-Only 为静态基线，不能回退旧调度。

本重构应在已完成的
[`Omni v2 收敛`](../archive/omni_compression_v2_convergence.zh.md)基础上，扩展通用 Operation
决策与执行契约；不要重新迁移 Qwen 静态路径或新增与 v2 Binding 并行的 `llm_operation` 调度。

## 2. 当前状态与问题

当前 Omni `llm_operation` 只激活了一条静态结构路径：模型完整加载后，在
runner/optimizer 前根据显式或预计算层号删除完整 Thinker LLM layer，再导出可重载
checkpoint。

现有基础并非需要推倒重写：

- `OperationAction` 已表达 target、operation、dependency、shape effect，以及
  `runtime_gate`、`static_mask`、`structural_rewrite` 三类 apply mode；
- `CompressionPlan` 已能携带 Operation actions 和组合策略；
- `CompressionSession` 已提供有序安装、卸载和多阶段依赖生命周期；
- `Qwen25OmniOperationApplier` 已验证静态完整层删除、重编号、配置更新和导出重载。

上述 Omni 执行路径仍仅限静态删层：

- 当前 Qwen `structured_llm_layer_pruning` Binding 只声明
  `execution_phase: prepare`；
- `LLMOperationStrategy` 只在已有 model/structural handles 上规划静态 actions；
- Qwen OperationApplier 只接受 `llm + layer + prune + structural_rewrite +
  remove_module`；
- 对该 Omni 静态删层 applier，`runtime_gate` 与 `static_mask` 仍没有通用执行支持；
- 当前 `execution_phase` 混合了“何时安装组件”和“何时作出压缩决策”两个概念。

此外，[Qwen3-VL Attention/FFN replacement](../support/qwen3_vl/debugging.zh.md#operation-实际激活参考) 已有
prepare-time 模块替换执行路径。它不经过 Omni 删层 applier，不能用上述限制概括整个
框架。主线后来接入了[MWOP Mask](../guides/fixed_mask.zh.md)、
[Qwen 分模态替换](../../mllm_comp/modality_split/README.md)与
[14 组对比方法](../support/imagellm/comparison_methods.zh.md)，其中包括方法专用的动态调度、
mask 和跳过运算；这些实现不等于本文提出的通用决策模型或任意混合组合已经完成。
本文其余部分保留为未来设计，不作为当前配置合同，也不涉及 refactor 分支。

## 3. 已确定的架构原则

本 TODO 采用以下决定：

> 统一决策接口，不统一执行机制；mask-first 可以，但必须从一开始保留静态物化路径。

数学上，静态选择可以视为恒定的动态 mask；工程上，静态结构重写与动态门控具有
不同的参数加载、optimizer、计算图、KV cache、导出和部署语义，不能强制共用一个
executor。

### 3.1 三类 Operation

| 类型 | 决策时机 | 执行方式 | 主要资源语义 |
| --- | --- | --- | --- |
| 静态 | 运行前由配置、校准或离线分析确定 | static mask 或 structural rewrite | 可物化、缩小 checkpoint，并排除被删 optimizer 参数 |
| 动态 | 依赖 request、sample、hidden state 或生成状态 | runtime gate、skip call 或动态 route | 通常保留全部候选参数，主要减少实际执行计算 |
| 混合 | prepare 静态准备，运行时完成最终选择 | 静态候选结构 + runtime gate，可选最终物化 | 同时包含准备态、运行态和导出态 |

### 3.2 mask 的定位

mask/gate 是重要的决策载体和参考执行方式，但不是所有 Operation 的唯一公共表示：

- 对 layer/head/channel/expert 的 keep/skip/prune，hard mask 或 selected indices
  可以表达选择结果；
- 可训练动态方法还可能需要连续 gate scores；
- replace、模块替换和其他结构变换需要额外 payload，不能只靠 mask 表达；
- 计算完整模块后再把输出乘零只验证语义，并未减少该模块的计算；
- whole-layer 动态压缩必须真正绕过 `layer.forward()`；
- head/channel mask 若仍进入原 dense kernel，也不能宣称获得实际 FLOPs 收益。

因此，mask-first 的含义是先建立可逆、可对照的参考语义，而不是把所有最终执行都
实现成张量乘法。

## 4. 目标模型

建议把算法决策与基座执行分成以下结构：

```text
Operation algorithm / selector
  -> OperationDecision（统一语义）
       |- semantic target
       |- decision timing and granularity
       |- selected indices / hard mask / gate scores
       |- requested application mode
       `- optional replacement payload
  -> backbone binding
       |- StaticOperationApplier
       |- RuntimeOperationController
       `- HybridOperationExecutor（组合前两者）
```

统一的是“选中了什么、为什么选择、决策对谁有效”；分离的是“如何在具体 backbone
上执行”。同一选择结果应能在适用时分别交给 mask/gate reference executor 和
structural materializer。

### 4.1 必须拆开的生命周期轴

不能继续只靠一个 `execution_phase` 同时描述所有语义。重构时至少要区分：

- setup/materialization timing：model 构造前、model load 后但运行前；
- decision timing：静态、training forward、prefill、decode；
- decision granularity：checkpoint、run、request、sample、token；
- application mode：structural rewrite、skip call、tensor/weight mask、route、replace；
- export policy：仅运行时、可固化、必须导出。

具体字段名在实现时确定，但这些语义不能再次折叠到单一 phase 字段中。

## 5. 推荐实施顺序

### M0：冻结现有静态基线

- 保留 SPADE Operation-Only 的 prepare-time structural rewrite；
- 现有 generate/backward/export/reload 行为不得退化；
- 不先把 SPADE 改写成动态 mask。

### M1：统一决策契约

- 将 `OperationAction` 从“静态 action”校正为中立的 Operation 意图；
- 明确 decision timing、granularity、application mode 与 payload；
- 让 Variant/Binding 声明自己真正支持的组合，不再对所有
  `llm_operation` 一律强制 `prepare`；
- 保持算法选择逻辑与 backbone module path/安装细节分离。

### M2：最小动态参考路径

- 先实现 request 级、hard-mask 的完整 LLM layer skip；
- prepare 只安装 controller，实际选择和 skip 发生在 forward/generate 生命周期；
- executor 必须证明被跳过 layer 没有被调用，而非调用后把输出清零；
- 首版要求一个 request 内 prefill/decode 使用一致的 layer 决策，避免 KV cache
  语义尚未定义时引入逐 token 变拓扑。

### M3：静态物化与等价性

- 将同一 hard decision 交给现有 structural materializer；
- 对可比较方法验证 runtime-gate 结果与物化模型结果一致；
- 验证参数数量、optimizer 成员、state dict、保存和 reload；
- 明确首次从原始 checkpoint 转换仍会加载完整模型，导出后再次加载才跳过被删参数。

### M4：混合式路径

- 支持 prepare 安装候选结构或缩小候选范围；
- runtime 根据输入产生最终 gate；
- 若方法允许，支持 soft gate 训练、hard decision 固化和静态 checkpoint 物化；
- 报告中分开记录准备结果、运行时决策和最终物化结果。

### M5：扩展 target 与 backbone

先完成 layer 级闭环，再按真实研究需求扩展 attention/head、FFN/channel、expert、
projector 和其他 backbone。不得仅因 schema 已列出 target 就宣称对应执行能力存在。

## 6. 最小验收标准

### 静态 Operation

- 决策在运行前确定，结构重写发生在 runner/optimizer 前；
- 被删参数不进入最终模型和 optimizer；
- module/config/state dict 连续且一致；
- 导出 checkpoint 只包含物化后的结构，并能直接 reload；
- 当前 SPADE Operation-Only 回归保持通过。

### 动态 Operation

- mask/gate 的作用域和生命周期明确；
- 被跳过模块确实未执行；
- 非目标 sample/token/module 不受影响；
- training forward、prefill、decode 和 KV cache 行为符合声明；
- 动态方法不得把“输出乘零”报告为实际计算缩减。

### 混合式 Operation

- prepare 结果与 runtime decision 分开记录；
- soft/hard gate 转换可复现；
- 声明支持物化时，动态 hard decision 与静态模型结果可对照；
- 多阶段依赖、安装顺序和冲突规则明确。

### 研究人员扩展

- 新增同类算法只增加 selector/strategy、Algorithm/Variant、必要的 backbone
  binding 配置与测试；
- 同一 Operation target 与 application mode 的 selector 应复用已有 executor capability，
  不为每篇论文复制 runtime controller 或 structural applier；
- 公共 runtime 不增加论文方法名分支；
- support 状态按 backbone × target × operation mode × lifecycle 精确声明；
- 未实现的 executor 或 target 必须保持未激活，而不是由 mask 字段推断为支持。

如何把这些内部对象收敛成一个低成本研究者入口，见
[`低成本算法扩展 TODO`](low_cost_compression_algorithm_extension.zh.md)。

## 7. 暂不纳入本轮重构

- 论文新颖性、目标任务质量或完整训练收敛；
- 一开始同时实现 layer/head/channel/expert 的所有组合；
- 为首次原始 checkpoint 加载实现通用参数跳读；
- 逐 decode token 改变 layer 拓扑；
- 仅为统一接口而把已正确工作的静态 SPADE 路径退化成 dense mask。

## 8. 预计涉及的代码区域

重构开始时优先复核，而不是立即全部修改：

- `mllm_comp/core/actions.py`
- `mllm_comp/core/plan.py`
- `mllm_comp/pipeline/stage.py`
- `mllm_comp/compressors/strategies.py`
- `mllm_comp/compressors/audio_multimodal_registry.py`
- `mllm_comp/pipeline/runtime_session.py`
- `mllm_comp/pipeline/operation_apply.py`
- `mllm_comp/backbones/qwen25_omni/v2_compression.py`

启动本 TODO 时，应先对照所选动态或混合方法与已有静态删层、模块替换的实际边界，
再确定最小补丁；本文件只固定架构方向，不预先承诺具体类名或迁移实现。
