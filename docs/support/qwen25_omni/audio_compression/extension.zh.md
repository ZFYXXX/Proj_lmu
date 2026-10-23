# Qwen2.5-Omni 音频压缩 v2 执行族参考

> 类型：Backbone-specific Reference
>
> 适用范围：Qwen2.5-Omni AudioLLM 的方法合同与四个执行族
>
> 当前状态：多模态 v2 当前路径

本文不是第二份“如何新增压缩方法”教程。唯一端到端教程是
[开发压缩方法指南](../../../guides/2_dev_compression_method.zh.md)；本文只记录 Qwen
Binding 的 target site、`executor_family`、生命周期和不变量。能力状态与真实
运行证据统一见 [Qwen2.5-Omni 状态](../status.zh.md)。

## 1. 方法身份、执行族与支持证据

Qwen 音频方法使用精确身份：

```text
AlgorithmDescriptor
  -> Audio ModalityVariantDescriptor
  -> Qwen2.5-Omni BackbonePlacementBinding
```

SupportUnit 按 algorithm、variant、backbone、modality、phase 记录验证状态与证据。
它不参与执行器选择，也不是合法研究运行的前置条件。没有支持记录的扩展方法可以在
已实现的机制上运行，但不会因此获得 active 状态。当前精确能力应查询注册表：

```bash
mllm-comp discover --backbone qwen25_omni --modality audio
```

配置中的 `plugin` 是算法标识；解析后的 Qwen v2 Stage 中它必须与 `algorithm` 相同。
Projector 采用广义定义：Audio Encoder 输出之后、LLM 输入之前的压缩都属于此阶段。

| `executor_family` | 负责的物理边界 |
| --- | --- |
| `projector_token` | Audio Encoder 输出、Projector 前或后 |
| `encoder_token` | Audio Encoder 层边界或 MHSA residual 后、FFN 前 |
| `llm_token` | Thinker decoder 层边界或 Attention 后，prefill 选择与布局重放 |
| `llm_operation` | `prepare` 时改写 Thinker LLM 结构 |

各方法原始位置、可移动位置与真实验证范围分别由 Binding、位置解析器和支持记录描述。
候选研究路径的具体限制见[候选方法说明](candidate_methods.zh.md)。

## 2. 唯一运行路径

```text
v2 YAML stage
  -> definition 校验 algorithm / variant / binding
  -> executor / interface / phase / physical contract validation
  -> RuntimeCompressionDispatcher / CompressionSession
  -> Qwen Binding executor factory
  -> projector_token | encoder_token | llm_token | llm_operation
  -> runtime metadata / Manifest
```

内置方法的声明集中在 `mllm_comp/compressors/audio_multimodal_registry.py`，由
`builtin_multimodal_compression_contracts()` 汇总。外部研究方法通过
`pipeline.extensions: ["package.module:register"]` 显式注册到本次独立 registry；
注册函数接收 registry，不扫描目录或修改全局内置表。Qwen executor factory 位于
`mllm_comp/backbones/qwen25_omni/v2_compression.py`。

普通配置可只写算法或精确 Binding 以及必要的 phase、方法参数；配置解析只补齐唯一
候选的身份、模态和位置族。多个 Variant、位置族或 phase 都合法时必须显式选择，
不会按候选顺序猜测。展开后的配置与显式完整配置经过同一 definition 校验。

Algorithm 自带的 Spec 与 strategy factory 仍用于归一化方法参数和复用已验证算法
语义；它们是 v2 Algorithm/Binding 的内部实现，不是另一套注册或 runtime 路径。
新增研究方法不需要在 catalog、lmms-eval、ms-swift 或共享 session 中增加论文方法名分支。

`generate`、训练 `forward`、prefill、decode 和 `prepare` 的通用关系见
[通用指南的阶段与状态说明](../../../guides/2_dev_compression_method.zh.md#结构位置请求阶段和状态)。
Qwen 的具体约束是：

- Projector/Encoder 方法在首次处理 prompt 时运行；decode 不重新执行 Audio Encoder；
- LLM Token 方法只在 prefill 的指定层选择/合并，decode 复用已建立的短 KV 布局；
- LLM Operation 在 model load 后、runner/optimizer 前静态物化；
- 当前没有 Audio Encoder 专用 Analysis Probe，共享 Probe 观察的是 Thinker/LLM 音频 Token；
- Talker 和 Token2Wav 不属于本压缩路径。

## 3. Projector Token executor

可复用 target site：

| target site | 输入张量 | hook |
| --- | --- | --- |
| `audio_encoder_output_pre_projector` | pooler 与 `ln_post` 后 | projector pre-hook |
| `audio_projector_output_post_projector` | `audio_tower.proj` 后 | projector forward-hook |

Algorithm/strategy 负责 score、分段/合并规则和 exact budget；
`Qwen25OmniAudioCompressionController` 负责每个 audio unit 的独立切分、attention 捕获、
长度核对、placeholder/M-RoPE legalization、autograd 与 telemetry。新增同类方法不应在
`audio_compression.py` 增加算法名判断。

Qwen 音频 Projector 前后保留同一时间顺序和 Token 数，特征维度可能不同。七种现有
Projector 方法都可用 `operation.target_site` 选择上述两个位置；默认位置保持原样。
Attention 评分继续使用原 Encoder 层来源。FastSLM HTA 根据实际边界宽度构建可训练模块，
旧 checkpoint 的不匹配形状会明确报错。UnSeg 的无梯度预计算与实际压缩读取同一位置的
特征，按自然峰值规划 placeholder 数量，不引入固定预算。

与 Encoder 压缩联用时，UnSeg 预计算在已准备的上游请求中运行，并沿用 provider 的
批量音频打包。Attention 规则使用评分位置的实际 chunk/audio 布局；若捕获后还有
Encoder 压缩，仅保留原评分层的归一化输入，沿用该压缩的 replay 映射，在最终布局
上用原评分模块计算一次重要性。单个私有布局回调在 checkpoint 返回后通知，不在
反向重算时改写请求状态；标准配方未联用上游压缩时保持原捕获路径。

LTBM 可复用 Encoder/LLM 的同一层适配接口。它从应用边界特征计算局部 cosine，
`requires_layer_input=False`，没有独立的 Q/K 评分层。原 Projector 内核与 replay 共用
逐轮等权均值的值计算；层适配只保存每轮 source→destination 决定，供重算与待用评分源
映射。LLM 的绝对位置跟随实际 destination 的沿袭，Encoder 的预算仍逐 chunk 计算。

Segmentwise 的 uniform/random 分支复用同一适配器，按已解析的 spec 判断是否需要
外部评分。选择规则返回的索引直接成为 replay 计划，供重算和下游评分源对齐；
原 Projector 的选择、分段和报告保持原样。LLM 值变换接口收到 `audio_unit_index`
用于保留逐音频随机种子，Encoder 则使用逐 chunk 编号。
有层预算适配器时，Encoder 的规划上下文设置 `defer_token_budget`，允许原插件先提供
未缩短的音频 span，再由适配器按 pooler 前 chunk 计算最终预算；Segmentwise 不会先用
投影后的长度做 `paper_floor` 校验。请求不保留这一临时预算的 requested-count 字段。
默认 Encoder Attention 分支已经接入 Encoder：规则按实际 spec 声明可用的 layer families，
Encoder 适配器提供固定评分层编号与 Q/K probe，再把逐 chunk 分数交回原 Segmentwise
选择方法。controller 将规则阶段编号与实际评分层分开，捕获、归一化、评分、可用性校验
和报告使用同一映射；不改变旧三种 Encoder 规则的默认评分层。
SA-MAP 复用这两个层执行族，固定倒数第二层评分源，按实际位置保留合并和剪枝两项
重要性。`sa_map_audio_importance()` 共用原 Q/K 及 attention reduction；Projector/LLM
使用真实 pooler 的两帧聚合与 odd-query 排除，Encoder 使用每个 chunk 的未池化分数。
层内评分以两列 Tensor 传递，列顺序为 merge、prune；标准 Projector 的 Mapping 接口
保持。LLM 固定源组件同步这两列的上游布局，与单分数参考共用原生命周期。
规则的 replay-plan factory 接收当前 `scores`，只保存原分组、合并重要性和最终选择；
`merge_sa_map_groups()` 供原 kernel 与重算共用全部分组的加权值路径，然后执行原选择，
保持非重入 checkpoint 的计算图。重算不重新分组或运行 DPP，也不把合并改为等权均值。
LLM 的默认 Segmentwise 分支通过 `AudioAttentionReference` 使用真实 Encoder 参考。
该组件只管理跨模块评分捕获及请求状态，压缩仍由原 LLM controller 和策略执行。
它复用 Encoder 布局通知、链入一个 Projector 通知并在卸载时恢复原回调；原 Q/K
评分经 pooler 转为逐音频分数，后续选择/合并通过规则的 `map_reference()` 同步。
六种有索引或分段/合并对应关系的 Projector 规则已接入。HTA 的非线性、多头、多尺度
查询网络没有定义固定 Encoder-token 重要性转移，捕获评分源后经过非 identity HTA 的
组合会明确拒绝；在评分源之前运行 HTA 或 identity 则可联用。该约束针对固定参考的
继承，不限制 HTA 自身的应用位置。LLM 重复应用只按绝对位置读取参考，源特征和分数表在请求结束后释放，
重算沿用整数选择计划。配置需要 `audio_encoder` 与 `llm` handle，运行记录包含
`encoder_score_layer_zero_based`，与 LLM 自身的 `score_layers_zero_based` 分开。

HTA 同样可配置 Encoder/LLM 层边界，复用 TemporalResampling 接口和现有查询内核。
三个 controller 共用注册/刷新函数，Projector 保留原模块名，层内模块分别追加
`_encoder_token`/`_llm_token`，按实际宽度在 optimizer 前构建；trainer 替换后跟随已注册
child。Encoder 按 chunk 执行，LLM 按 audio unit 分窗，并以窗口内等距原位置作查询锚点。
查询锚点只用于 M-RoPE，不是旧 Token 身份。HTA 的 checkpoint replay 保留同一学习网络
计算图。冻结基座训练 HTA 时，Encoder 在包含可训练模块的 checkpoint 前启用隐藏输入
梯度，LLM 在原生层 checkpoint 之前通过模型输入 hook 建立梯度入口；不解除基座参数
冻结，卸载时移除该 hook。独立参考映射与 replay 分开，不能把一次网络重算当成旧重要性的线性转移。

## 4. Encoder Token executor

可复用 target site：

- `audio_encoder_layer_post_mhsa_pre_ffn`：原生 MHSA residual 后压缩，再对短序列运行 FFN；
- `audio_encoder_layer_output_post_block`：完整 block 后压缩，并把新 `cu_seqlens` 交给下一层。

Algorithm/strategy 负责 active layers、逐层 output length、identity 条件和
`compress()` 与 `replay()`；有评分的规则额外提供 `probe()` 获得评分上下文，`compress()` 对给定边界的
特征执行选择或合并，并返回特征、chunk 长度、报告与重放计划；`replay()` 只应用计划。
`Qwen25OmniEncoderCompressionController` 负责 Attention、残差、FFN、变长 encoder
loop、chunk 边界、pooler 最小长度、gradient-checkpoint replay 和 report。
三个现有规则都可通过 `operation.target_site` 选择上述两个 site，默认位置不变。
`operation.layers` 单独配置应用层；完整层 site 的 `layer_edge: before` 归一为上一层
输出，层 0 表示首层前的初始输入。原 active layers 仍决定评分层和逐阶段预算。
应用位置不能早于所需评分层输入。延后应用期间，待用评分特征随其他压缩的选择／合并
同步映射，使用后释放；checkpoint 仅重放保存的决定，不重复评分或保留完整评分特征。
UniSamp/UniAvg 也可使用上述 Encoder site 与显式 `operation.layers`。它们声明
`requires_layer_input=False`，直接复用原采样／池化规则，按每个原生 chunk 独立计算，
各层重复使用原压缩因子。接口只适配 chunk 预算和 pooler 后的 placeholder 数量；
不足以产生一个 pooler 输出的配置在准备阶段拒绝，不改写规则的预算。
UnSeg 复用同一接口，未知 chunk 输出长度为 `None`。准备阶段启用当前 controller 的
请求，在相同 Encoder 循环中无梯度执行自然分段；根据实际 chunk 长度确定 pooler 和
placeholder 数量，再进行 legalization。预计算保存并恢复 RNG，使用原生音频打包方式，
在输出数量确定后校验；正式前向重算自然分段并检查计划数量。checkpoint 闭包保存
实际 chunk 长度和规则原分段，反向重放均值计算，不重新检测峰值。上游布局通知仍
只传递规则计划，供下游评分参考映射；无需第二套预规划 executor。
这里描述当前 v2 协议；扩展方法应注册同一 Algorithm/Variant/Binding 合同，并复用实际执行族。

每个 audio unit 最终至少保留 pooler 所需的 Encoder Token。运行时长度必须与 plan
一致；`cu_seqlens`、feature lengths、projector 输出、placeholder 和 M-RoPE 数量必须
共同收敛到同一布局。

## 5. LLM Token executor

Audio-Pyramid 在配置的 decoder layer 后，按 instruction query 对 audio key 的
received attention 逐级保留音频 Token，并恢复时间顺序。FastAdaSP 与 HeadRouter
复用 `llm_token` 执行族，各自保留算法语义和独立的验证记录。

Algorithm/strategy 提供 stage layers、输出长度和方法 metadata；有评分来源的规则额外
提供音频打分与索引选择，值变换提供压缩及重放。UniSamp/UniAvg 可通过显式
`operation.layers` 与 LLM target site 复用此执行族，`requires_layer_input=False`，
不捕获层输入、不构造无用分数；原 Projector 策略和内核继续承担采样／池化计算。
LLM 接口适配每个 audio unit 的多阶段预算、绝对位置与重放，标准 Projector 配方不变。
UnSeg 也使用此接口，`output_length()` 返回 `None` 表示预算由当前边界特征决定。
`merge_audio_tokens()` 复用原自然峰值分段和均值，controller 从输出读取实际数量，
并按各段首 Token 的绝对位置修复后续布局；重算只重放原 segments。
LLM 计划保留未知数量（`null`），不伪造固定比例或执行 Projector preflight。
原 Projector 模式继续使用其真实位置预规划；Encoder 预规划见上一节。

`Qwen25OmniLLMTokenCompressionController` 负责：

```text
Prefill layer i
  -> 保存原评分层输入；初始边界可在第 0 层 Attention 前压缩
  -> 使用该层入口布局适配 causal mask / M-RoPE / cache_position
  -> 原生 Attention/残差；若选择层内位置，在 FFN 前压缩
  -> 原生 FFN/残差；若选择层间位置，在完整层后压缩
  -> 保存下一层入口的绝对位置与语义布局

Decode token t
  -> 不重新选择、不再次压缩
  -> 按每层 compact prompt KV 长度适配 mask
  -> 原生 DynamicCache 追加当前 Token
```

位置策略为 `preserve_absolute`：保留 Token 延续原始三轴 M-RoPE，后续文本位置不因物理
compact 而改写。训练时选择可在 `no_grad` 中执行，但 hidden gather 不能 detach；
gradient checkpoint backward 只重放选择布局或保存的加权合并计划，不重复打分和写 report。

`operation.layers` 控制应用层，`operation.layer_edge=before` 将完整层 site 归一为上一层
输出，层 0 对应初始输入；不另设层入口 executor。原 stage layers 保留评分和逐阶段预算
语义，`LayerPosition` 只表达实际边界。执行器分别保存 layer 调用输入、Attention 实际输入
和压缩结果布局，使初始压缩、checkpoint 重算和逐层 KV 长度保持一致。
评分输入按源层捕获并按当前绝对位置映射；HeadRouter 始终使用初始嵌入及原 probe 层权重。
应用位置早于所需源层输入时拒绝，不能虚构未来特征。配置示例见主运行指南。

Analysis 会读取每层入口的有效语义布局。若某方法在 layer 8 出口压缩，观察压缩后表示
应选择 layer 9；不能把 layer 8 的压缩前 Attention/FFN 宽度与压缩后 layer output
放进同一记录。

## 6. LLM Operation executor

`structured_llm_layer_pruning` 提供完整层删除；SPADE Operation-Only profile 的层选择
来自显式配置或预计算结果，不包含 WLI 自动选层、teacher、蒸馏、head/FFN pruning
或 Talker。

Qwen OperationApplier 统一完成 ModuleList 重写、`layer_idx` 重排、
`num_hidden_layers` 更新、参数量统计和 old-to-new mapping。静态 head/FFN channel
裁剪具有不同形状耦合，必须先扩展对应 target-kind contract；运行时动态跳层、gating
或路由则需要 forward/generate 生命周期，不能伪装成 `prepare` 静态改写。

`prepare` 发生在原始 checkpoint 加载后，因此首次改写不会降低原模型加载峰值。只有
另行导出的结构 checkpoint 才能在后续加载时减少构造层数。最小功能 smoke 可以只验证
结构改写、generate 和 backward，不要求导出大型 3B checkpoint；宣称 checkpoint
reload 支持时仍必须使用已有的独立导出/重载证据。

## 7. 新增同类 Qwen 方法

先遵循通用指南建立 Algorithm 与 Audio Variant，再完成以下 Qwen-specific 工作：

1. 在研究模块的 `register(registry)` 中声明 Algorithm、Audio Variant 和 Qwen Binding，并用 `pipeline.extensions` 加载；合入内置方法时再集中整理到 `audio_multimodal_registry.py`。
2. 复用已有执行族：Binding 的 `executor_family`、`strategy_factory`、`exclusive_runtime_slot` 和 `prepare_after_slots` 是显式执行字段；目标点和 handles 由 `targets` 声明，所需请求上下文由 `metadata.required_runtime_context` 描述。不要把执行控制重新藏进 metadata。
3. 声明方法 Spec、Variant 默认参数与校验，以及实际可消费的 `supported_rules`；配置可省略能够唯一推导的身份。
4. 验证方法数值与重放、合同投影、执行器生命周期，并按声明的训练、生成和恢复范围运行必要的真实模型检查。
5. 正式声称支持时，为已验证的精确身份建立或晋级 SupportUnit；这一步不阻止此前的合法研究运行。

只有新方法需要新的物理位置、状态或结构形状时才修改共享 executor。新增方法不需要
修改 `RuntimeCompressionDispatcher`、lmms-eval、ms-swift 或创建 Descriptor。

## 8. 代表性配置与验证入口

| 执行族 | 代表方法 | eval 配置 | 主要测试 |
| --- | --- | --- | --- |
| Projector Token | Segmentwise Top-K | `configs/eval/lmms_eval/qwen25_omni_aud_segmentwise_topk.yaml` | `test_qwen25_omni_audio_compression_{cpu,gpu_smoke}.py` |
| Encoder Token | A-ToME | `configs/eval/lmms_eval/qwen25_omni_aud_atome.yaml` | `test_qwen25_omni_encoder_compression_{cpu,gpu_smoke}.py` |
| LLM Token | Audio-Pyramid | `configs/eval/lmms_eval/qwen25_omni_aud_audio_pyramid.yaml` | `test_qwen25_omni_llm_{token_cpu,level_compression_gpu_smoke}.py` |
| LLM Operation | SPADE Operation-Only | `configs/eval/lmms_eval/qwen25_omni_spade_operation_only.yaml` | `test_operation_compression.py` 与 LLM-Level integration |

原有验证配方的算法身份和 adaptation 边界见
[八方法契约](eight_methods_contract.zh.md)；当前支持状态以 discover 和对应证据为准。四类专项 Debug 手册分别是
[Projector](debugging/projector_level.zh.md)、[Encoder](debugging/encoder_level.zh.md)、
[LLM Token](debugging/llm_token_audio_pyramid.zh.md) 和
[LLM Operation](debugging/llm_operation_spade.zh.md)。

除通用 O/E/T 外，Qwen 验证还必须确认：

- Algorithm/Variant/Binding 身份精确一致，运行记录不把缺失证据写成正式支持；
- 当前 provider 结构与 Binding target site 一致；
- Encoder/Projector 的 chunk、pooler、placeholder、mask 与 M-RoPE 合法；
- LLM Token 的 prefill/decode、absolute M-RoPE、KV cache、labels 与 layer-entry layout 合法；
- Operation 的结构层号连续、配置同步，声明 reload 时具有独立导出/重载证据；
- eval/train/analysis 的 Manifest 能追溯实际身份、已有 SupportUnit 证据和可获得的 runtime metadata；没有支持记录时如实保留。

历史迁移原因与删除旧路径的完成条件保存在
[`Omni 压缩与分析收敛记录`](../../../archive/omni_compression_v2_convergence.zh.md)。
该文件是历史记录，不再定义当前接入流程。
