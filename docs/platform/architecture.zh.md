# 统一多模态压缩框架：架构说明

> 类型：Architecture
> 适用范围：多模态、多基座、多阶段、多类型压缩的稳定抽象
> 当前状态：Active
> 当前实现进度：[项目进度](../progress.md)；ImageLLM 多算法的精确状态见[状态页](../support/imagellm/status.zh.md)

## 1. 目标与边界

本项目提供可扩展的多模态压缩研究框架，而不是为某个模型或某篇论文维护一次性实现。框架沿四个相互独立的维度扩展：

- 多模态：image、video、audio，以及未来的联合音视频；
- 多基座：算法内核只实现一次，基座适配层负责真实插入点和布局修复；
- 多阶段：input、encoder、广义 projector、LLM 等阶段可按依赖组合；
- 多类型：Token 压缩、已实现的 Operation 与 KV cache 机制，以及后续量化接入；当前量化边界见[量化接入设计](quantization_integration.zh.md)。

这里的“Projector”是广义阶段名：所有发生在 Encoder 输出之后、LLM 输入之前的压缩都属于 `projector_level`，不要求模型中存在名为 projector 的独立模块。

本页只描述长期稳定的架构契约。某个算法是否已经在某个基座、模态和执行阶段可用，必须查询对应的 SupportUnit 和状态页，不能从架构设计自动推导。

框架面向四类角色：压缩使用者运行与比较实验；压缩研究人员开发方法；平台维护者
维护模型接入与生命周期；Agent 通过程序接口迭代实验、开发方法并评估研究能力改进。
Agent 是独立的目标使用者，可执行前三类角色的任务；它需要机器可读的能力发现、
计划、结果和错误反馈，并复用同一套 CLI/Python 实现。
[按任务阅读](core_concepts.zh.md#15-推荐阅读代码顺序)；[Agent 入口与 auto-research / RSI 目标](../guides/4_agent_research.zh.md)。

## 2. 四层能力模型

```text
AlgorithmDescriptor
  └─ ModalityVariantDescriptor
       └─ BackbonePlacementBinding
            └─ SupportUnit(algorithm, variant, backbone, modality, phase)
```

### 2.1 Algorithm：共享数学内核

`AlgorithmDescriptor` 描述算法名称、版本、压缩类型、逻辑阶段、token 操作和内核工厂。它不包含具体基座类名，也不负责模型布局修复。

算法内核应该在同一模态语义下跨基座复用。例如 ImageLLM 参考切片中的 ToMe、VisionZip 和 PyramidDrop 各只有一份共享实现：

- `mllm_comp/compressors/image/tome.py`
- `mllm_comp/compressors/image/visionzip.py`
- `mllm_comp/compressors/image/pyramid_drop.py`

### 2.2 Variant：模态语义

`ModalityVariantDescriptor` 显式描述：

- 被压缩的 `target_modalities`；
- 参与打分但不删除的 `conditioning_modalities`；
- 必须保留的 `preserved_modalities`；
- token unit、跨流 coupling 和 alignment 规则。

image、video、audio 和联合音视频必须分别注册 variant。共享视觉或声学 Encoder 不等于共享支持声明。

### 2.3 Binding：基座物理插入点

`BackbonePlacementBinding` 把一个 variant 接到某个基座的精确位置，并声明：

- placement 和 execution phase；
- 所需 structural handles；
- executor/provider factory；
- 布局、mask、position、placeholder、label 和 cache 修复；
- 与其他 binding 的依赖和冲突。

数学算法不能复制到各个基座适配器中。适配器只处理该基座的数据流、生命周期和结构不变量。

三个图像基座的物理声明按模型放在 `compressors/image/contracts/`，适配器与中央
registry 共用其 variant、回调位置和原生 metadata。中央 Binding 补充逻辑目标、
逐目标句柄和执行工厂，不再手写另一份物理位置表。LLaVA 的回调时机、能力与修复要求
仍属于该基座声明；相同字段不意味着不同模型的实际执行可以合并。

### 2.4 SupportUnit：精确能力声明

支持能力的最小身份是：

```text
(algorithm, variant, backbone, target_modality, phase)
```

注册表只接受显式单元，不对算法集合、基座集合和模态集合做笛卡尔积。以下事实都不能单独推出“已支持”：

- 算法内核已经存在；
- 基座 baseline 已经运行；
- binding 声明了某个插入点；
- 同一方法在另一模态或另一 phase 已经通过。

`forward`、`generate` 和 `prepare` 是执行 phase；训练支持另由对应的训练证据描述，
不能从一次推理 forward 或 generate 的成功推出。

## 3. Stage 与运行时

一个 stage 描述“用哪个算法变体，在什么 placement 和 phase，压缩哪些模态”。运行时按以下顺序工作：

1. 校验 algorithm、variant、binding、backbone、modality 和 phase 是否精确匹配；
2. 核对实际 executor 和执行约束，读取精确 SupportUnit 的证据状态；没有证据的合法研究配置也可执行；
3. 按依赖顺序完成 `prepare` 结构变换，再安装请求阶段的 provider activation；
4. 执行模型并收集布局变化和逐阶段统计；
5. 成功结束后按逆序显式释放 activation。

运行时会对动态 hook 做事务式安装：失败时连同已安装项按逆序执行 best-effort rollback，正常结束时也按逆序释放。`rollback_safe` 只描述 executor 是否能保证恢复；对不可逆的静态结构改写，即使调用清理接口也仍须把 runtime 视为不可复用并重新加载 checkpoint。

定义解析确定契约，dispatcher 使用解析后的 Binding 执行槽计算安装依赖；已有 `CompressionDefinition` 时直接复用其 projections。v2 binding 的 `executor_factory` 在轮到该项安装时才调用，返回一个实现 `install`、`uninstall`、`runtime_metadata` 的执行器。此时重新读取实际模型结构，避免后续 Token hook 持有已被 Operation 替换的层。`prepare` 使用自身阶段，动态项仍须匹配本次 `forward` 或 `generate` 请求；依赖环和真实冲突继续拒绝。这一顺序本身不构成任意组合的正确性证据。

`prepare` 也用于安装在后续 forward/generate 中工作的 Operation，而不只表示静态
改权重。新增 comparison、MWOP Mask 与分模态替换都复用公共 Session，但各自保留
决策、布局、缓存和保存恢复约束；不能将 comparison Operation 路由到分模态联合编译器。
具体当前实现见[压缩方法支持范围](../support/compression_methods.zh.md)。

多阶段组合不是把若干 hook 任意叠加。每个 stage 都必须声明依赖、冲突和布局输入条件，并在相邻阶段之间保持 token 语义一致。

Qwen3 的 Attention/FFN replacement 使用同一条定义和安装路径，由
`compressors/operation_registry.py` 注册；不再由 Session 内置方法名分派到另一套入口。
Operation 的 planner、模块构造和回滚保留各自职责，Token、KV 和量化也继续按真实
物理行为区分。量化尚未有可执行实现，不能从这条公共安装路径推导量化已可组合。

状态按实际使用范围归属，而不是全部放在 activation 上：

| 状态 | 当前持有者与失效时机 |
| --- | --- |
| 模型参数和可学习组件 | 实际模型模块；执行器在安装时注册，沿原 optimizer、state dict 和方法恢复路径使用。方法配置与注册句柄由 activation 持有。 |
| 一次 forward / generate 的信号与离散决定 | 图像 `Method.new_call()`、`ImageDecoderLifecycle` 和 provider 的请求上下文；重算重放已有决定。LLaMA-VID 等信号提供者也使用请求上下文，不把上一请求的特征当新输入。 |
| 从可变权重派生的缓存 | ZOO 的 activation 内低秩代理按实际 weight 版本失效；无版本计数器的 inference tensor 每次重建，full proxy 直接引用原模块。权重地址不变不能表示其数值未变。 |
| 随生成延续的 KV 和递归历史 | 随实际 cache 对象存活。DyCoKe activation 仅弱引用仍存活的 cache controller；Qwen3 LoLCATs/NHA 的递归历史按层挂在原生 cache 上，`reorder_cache(beam_idx)` 同时重排 KV 和历史张量。 |

这些是已实现机制的所有权边界，不是所有方法都支持多请求并发或 beam search 的声明。
具体限制仍由所选 Binding 和 provider 校验；更换 cache、更新权重与开始新请求是不同事件。

图像 LLM 选择的调用链为 `Binding.strategy_factory` → Method → `new_call()` →
provider 请求信号 → Call 选择 → 原生布局修复。FastV、PyramidDrop、BTP、HalfV、
TransPrune、DART、D²Pruner、FEATHER、IVC-Prune、FinePruner、SparseVLM 和 PACT 使用这条路径。
Method 保存配置与不可变算法资源，Call 保存本次调用的信号和决策历史。
选择与合并的输出顺序、checkpoint 重放由公共机制读取 `LayoutDelta`；Qwen3 的
DeepStack、Qwen 的 M-RoPE 和 LLaVA 的 AnyRes/newline 仍在各自 provider 处理。
PACT 的固定评分来源和聚类质量由 Call 持有，原生投影由 provider 提供；LLaVA 的既有
直接回调复用同一方法与布局修复。RESTORE 的 `RestoreAttentionState` 统一持有校准用的
位置和 Token 质量，处理首层质量、decode 追加及掩码合成；provider 的请求上下文管理
其生命周期，并提供原生位置和因果掩码，诊断记录不承担执行状态。SGL 的 activation
继续管理 guide 生命周期，并将其 `consume_scores` 交给 `SGLDecoderMethod`；target 的
选择、布局和重算复用同一 Method/Call 路径。Nüwa 的复合 activation 管理第一阶段
视觉聚合，第二阶段由 `NuwaDecoderMethod` 持有固定评分并执行选择；原生位置、
DeepStack 和 checkpoint 重放复用同一基座机制。复合注册只向 decoder 站点传递
decoder 选择对象，视觉站点继续接收聚合请求。
执行复用不改变各方法的支持状态。

## 4. 布局与跨模态对齐

`LayoutDelta` 是 token 布局变化的统一结果，至少表达：

- source/result token 数；
- kept、dropped、protected indices；
- merge groups 和 `old_to_new` 映射；
- media unit、token unit 和可选坐标。

连续 stage 的 delta 只有在 token unit 和计数兼容时才能 compose。Provider 还必须同步模型特有的 placeholder、mask、position、label、grid、M-RoPE 或其他分支布局。

多流方法应为每个流分别维护 `LayoutDelta`，再用 `AlignmentDelta` 表达 preserve、cascade、lockstep 或 joint 关系。分别开启两个单模态方法不等于联合音视频压缩。

## 5. 参考 ImageLLM 三阶段

当前参考设计把三个图像算法放在三个逻辑阶段：

| 逻辑阶段 | 算法 | 图像变体 |
|---|---|---|
| Encoder | ToMe | `tome.image.encoder_dense_boundary` |
| 广义 Projector | VisionZip | `visionzip.image` |
| LLM | PyramidDrop | `pyramid_drop.image` |

这些名称描述算法和阶段，不代表所有基座或 phase 都已激活。精确矩阵和限制见 [ImageLLM 多算法状态](../support/imagellm/status.zh.md)。

### 5.1 ToMe 的基座边界

ToMe 在视觉 Encoder 层内进行 token merge，并输出 `LayoutDelta`。某些基座的后续结构仍要求稠密网格，因此 binding 可以在 family-specific boundary 恢复 dense view：

- Qwen2.5-VL 需要保护 window layout 和 visual merger；
- LLaVA-OneVision 需要保护 AnyRes pack；
- Qwen3-VL 还要保护 DeepStack 与最终 merger。

当前实现只在这些结构消费者前临时恢复 dense view；结构消费完成后，会按同一 ToMe merge group 再做永久输出压缩。因此已验证切片既减少后续 Encoder 层的计算，也让进入 LLM 的图像 token 实际减少。

### 5.2 VisionZip 的广义 Projector 边界

VisionZip 在 Encoder 之后、LLM splice/scatter 之前选择并合并视觉 token。不同基座必须分别修复：

- LLaVA-OneVision：AnyRes pack、image newline 和 crop metadata；
- Qwen2.5-VL：placeholder 与稀疏 M-RoPE；
- Qwen3-VL：主分支和全部 DeepStack 分支的同步布局。

### 5.3 PyramidDrop 的 LLM 边界

PyramidDrop 在 LLM 层循环内按层分阶段删除图像 token；这些层内阶段与生成的 prefill / decode 不同。当前选择在 prefill 或直接前向中执行，增量 decode 不重复选择。Binding 必须原子更新下一层使用的 hidden states、mask、position 和媒体索引；普通 module output hook 不足以表达这一语义。使用 KV cache 的生成支持是独立能力面，不能由无 cache 的 forward 结果外推。

## 6. Qwen2.5-Omni AudioLLM 与后续模态

Qwen2.5-Omni 音频方法与 ImageLLM 使用同一四层能力模型。Audio Algorithm/
Variant 由多模态 registry 登记，Qwen Binding 再选择 Projector Token、Encoder Token、
LLM Token 或 LLM Operation executor；controller 和 OperationApplier 只承担 Qwen 物理
执行与不变量修复。方法数量、当前执行位置和验证范围由 `mllm-comp discover` 查询，
结构重构或一次研究运行不会隐式晋级已有支持声明。

后续 video、audio 和联合音视频扩展应各自新增 variant、binding 和 SupportUnit，并维护对应的时间、空间或跨流对齐信息。

## 7. 关键代码入口

- 能力契约：`mllm_comp/compressors/contracts.py`
- 多模态注册：`mllm_comp/compressors/multimodal_registry.py`
- AudioLLM 注册：`mllm_comp/compressors/audio_multimodal_registry.py`
- 支持状态：`mllm_comp/core/support.py`
- 布局与对齐：`mllm_comp/core/layout.py`
- 共享校验：`mllm_comp/pipeline/v2_validation.py`
- 多阶段运行时：`mllm_comp/pipeline/runtime_session.py`
- Qwen AudioLLM Binding executor：`mllm_comp/backbones/qwen25_omni/v2_compression.py`
- Provider bridges：`mllm_comp/backbones/{llava_onevision,qwen25_vl,qwen3_vl}/provider_bridge.py`
- Manifest：`mllm_comp/core/manifest.py`、`mllm_comp/run_manifest.py`

新增压缩方法的具体步骤见[开发指南](../guides/2_dev_compression_method.zh.md)。
