# 压缩机制与方法状态参考

本文说明已有执行机制怎样保留评分来源、算法状态、模型布局及 checkpoint 重算。
新增方法的操作步骤见[研究者指南](../guides/2_dev_compression_method.zh.md)；
实际应用位置的配置语法见[压缩位置参考](compression_positions.zh.md)。
这些规则限定当前实现，不把单个子步骤或探索配置扩大为完整方法的新能力。

## 音频执行族

Qwen2.5-Omni 的 14 个音频方法与 ImageLLM 一样先解析完整四层身份。Audio Variant、
Qwen Binding 和 SupportUnit 位于
`mllm_comp/compressors/audio_multimodal_registry.py`；通用 registry 由
`builtin_multimodal_compression_contracts()` 汇总。运行时再由
`mllm_comp/backbones/qwen25_omni/v2_compression.py` 把精确 Binding 分派给四个
Qwen executor_family：

| executor_family | 物理职责 | 代表方法 |
| --- | --- | --- |
| `projector_token` | Audio Encoder 输出到 LLM splice 之间的选择、合并或池化 | Segmentwise Top-K |
| `encoder_token` | Audio Encoder 层内/层间变长执行 | A-ToME |
| `llm_token` | Thinker prefill 层间音频 Token 压缩及 KV/position/labels 修复 | Audio-Pyramid |
| `llm_operation` | runner/optimizer 前的静态 Thinker 层结构改写 | SPADE Operation-Only |

新增同类 Qwen 方法遵循以下边界：

- 算法 plugin、配置 Spec 和可复用数学语义归入 Algorithm；
- audio target、conditioning/preserved modality 与时间/Token 对齐归入 Audio Variant；
- executor_family、target site、所需 structural handles 与 Qwen runtime context 归入 Binding；
- `forward`、`generate` 或 `prepare` 的状态与证据归入精确 SupportUnit；
- placeholder、chunk、pooler、M-RoPE、mask、prefill/decode、KV cache、labels 和结构重编号留在 Qwen executor/legalizer；
- runtime、lmms-eval 和 ms-swift 不增加论文方法名分支。

可配置位置由 Binding 的 `position_resolver` 解析为实际 Binding 副本，通用 v2 校验、
执行分派和 Manifest 使用同一解析器。标准注册表不随单次探索配置改变。
Encoder 策略通过 `probe` 提供评分上下文、通过 `compress` 对给定边界的特征执行
选择或合并；Attention、残差、FFN 和 chunk/mask 更新由 Encoder executor 控制。

## 图像 Encoder 与永久输出合并

图像 ToMe 使用同一解析方式，`operation.targets.encoder_merge` 只控制名为
`encoder_merge` 的 BindingTarget，其余结构恢复和最终输出合并步骤保留。三个 provider
在原执行循环内选择残差后、完整层输出或起始边界。起始压缩使用首层输入评分；
Qwen2.5-VL 与 Qwen3-VL 共用已有的比例 Attention 计算，Qwen2.5-VL 的默认路径
仍调用原生 Attention。LLaVA 的裸回调继续采用默认全层边界。
`criterion.score_layer` 只指定 encoder_merge 的层入口 Key 来源。各 provider 在原循环
采样，使用 ToMe 已有质量与 LayoutDelta 映射参考；Qwen2.5 的层末密集恢复和 Qwen3／
LLaVA 的跨层紧凑传播分别遵循各自布局，不能以更改应用层重新采样代替固定来源。

独立的永久输出合并通过 `tome_output_site` 解析 `operation.targets.permanent_output_merge.target_site`。三基座复用原 output ToMe 内核、单位初始质量、预算、固定 Key 及原保序规则：Qwen 保留完整 raw 空间组，原循环按配置在原 merger 前或后调用同一个输出函数；Qwen3 对主分支和 DeepStack 复用同一组映射。LLaVA 在原 packed 回调中复用 raw AnyRes 对应和内容投影/newline 恢复，完整 projected QK 参考仍由原流程计算，其成本如实记录。结构恢复只广播 Encoder 已有映射，位置受原网格消费者约束。此次接线保留原 FORWARD 无 cache 生命周期，未增加平行 executor 或生成实现。

LLaVA 公开回调联用时只允许一个 packed reducer 位于 Projector 前：连续两个前置操作需要跟踪第二次 raw 输入及评分来源的布局，不能重复使用第一份原始映射。该组合会明确报错；原单前置操作与默认后置 ToMe 的顺序保持。普通实验配置仍遵循已有方法间的冲突校验。

## 图像 LLM 层内压缩的来源、应用与重算

此处 LLM 层表示结构位置。下述共享图像 Token 选择在生成的 prefill 或直接前向中执行；
训练前向单独描述，生成的增量 decode 沿用选择后的状态，不重复选择。
代码中的 `Decoder*`、`decoder_*` 保留原名，含义见[术语约定](../platform/core_concepts.zh.md#结构位置与生成阶段)。

图像 FastV 的 `operation.targets.decoder_prune_boundary` 配置实际 LLM 层间位置，
三个基座都复用原 LLM 层循环和布局修复。`config.prune_layer > 0` 保持原规则的
第 `prune_layer - 1` 层输入 Q/K 来源，`prune_layer=0` 保持原随机选择；应用层改变时
不移动来源。来源可在同一个层入口先采样，不能读取后续层。栈末边界在最后一层与
最终 norm 之间压缩；Qwen3-VL 保持该层 DeepStack 注入之后的顺序。缓存仍采用
compact-prefix，只压缩应用边界之前已经形成的缓存，decode 不重复选择。
FFN 前的位置通过共享的 `decoder_post_attention_boundary` 按基座原有顺序调用
Attention、归一化和 MLP，在 Attention 残差合并后插入压缩；模块及完整层输出的
hook 保持实际调用语义，并沿用设备 hook。每次调用拥有自己的 checkpoint
闭包；首次前向只保存离散选择，前向与重算使用相同的值重放计算图，避免重复
评分、更新布局或缓存。该边界的 compact-prefix 包含当前 Attention 已形成的 KV，
Qwen3-VL 在层调用后、注入前以可微索引同步当前及后续 DeepStack 分支。

FastV、PyramidDrop、BTP、HalfV、TransPrune、DART、D²Pruner、FEATHER、IVC-Prune、FinePruner、SparseVLM 和 PACT 的方法对象、固定评分来源、逐次前向状态及 kernel 调用集中在
`compressors/image/decoder_selection.py`；参数的 Spec 定义仍在各算法文件中。位置规划与注册执行共用 `decoder_config.resolve_decoder_method_config`：
Variant 默认值依次被旧式显式参数、`stage.config`、所选 Binding 的参数覆盖；规划读取参数，不改写 stage。
适配器直接读取方法的 Spec 和位置，三个 provider 按 `application_positions` 决定是否应用选择。
Q/K 仅在方法请求时采集；Qwen3 的位置、mask、cache 与 DeepStack 在 provider 内更新，
LLaVA 的注册方法和公开回调共用 newline、布局与 cache 修复。公开回调仍保留自身的阶段记录语义。
这些方法的 `LayoutDelta.source_layout` 与 `result_layout`
统一用 `application_position: {boundary, index}` 表示实际应用位置；评分来源仍由
`score_layers` 单独记录，读取旧运行产物无需重写历史记录。

纯选择可以返回 `ImageTokenSelection`，其索引对应当前完整序列中的图像位置。
`execute_decoder_selection` 用共享物化代码保留文本和受保护位置，构造可微 gather 与
`LayoutDelta`；已有完整结果继续交给原布局处理。FastV 共用物化实现，并保留公开的
`FastVResult` 类型。该接口不改变合并值、请求历史或 checkpoint 重放的语义。

DART 与 FastV 共用 `decoder_token_operation` / `decoder_token_position` 以及上述
provider 边界。DART 在固定层入口捕获 Key，并在该完整层完成后捕获归一化 hidden states，选择仍调用原 kernel；
位置解析额外检查完整层来源已存在。固定评分张量在应用时对齐实际设备，保留原
归一化、pivot、配额和剪枝规则。公开工厂与 Manifest 使用最终 binding config 解析位置，
等价边界继续适用原记录，改变位置只标记历史证据的适用范围，不改写历史证据。

D²Pruner 也复用上述边界，和 FastV 共用固定层入口 Q/K 捕获；已有 Attention reduction
得到 image score，仍交给原 D²Pruner kernel 做 prior 去偏及 graph/pivot/MIS 选择。
prior 的模型身份和 scorer_layer 校验继续绑定评分来源，不能按应用层改写。

IVC-Prune 沿用原完整层输出的 V 投影和该层 RoPE 坐标，采样位置与 application_position
独立。Call 通过 `output_feature_layers` 请求完整层观测、保存并消费这份信号；
三个 provider 交付原生特征，默认及探索位置共用选择与修复路径，重算只重放索引。
Qwen3 在 FFN 前同步当前及后续 DeepStack 分支、完整层后保留注入顺序，不能因旧实现
曾限定 DeepStack 后而拒绝已有映射能力支持的较早位置。原 FEATHER 调度不受此调整影响。

SparseVLM 的 Call 通过 `initialize` 从初始嵌入选取一次文本评分者，将原评分层入口与实际应用边界分开。
每个来源的原生 RoPE Q/K 和全 Key 因果分母保持；前阶段与后来源位于同一入口时，先应用前阶段，再捕获后来源。
提前捕获的后来源分数按 `LayoutDelta` 合并列质量，丢弃列不再参与后续选择。
`decoder_layout_replay` 根据 `LayoutDelta` 固定选择索引或合并成员，FFN 前的
checkpoint 重算重放同一决定；后续 DeepStack 分支也沿用相同成员与聚合规则。
该工具也由 PACT 使用。SparseVLM 保持 `compact_prefix` 和连续位置，PACT 保持
各层独立缓存；provider 依据方法的实际缓存和位置策略处理 prefill 与 decode。
运行记录同时保存评分来源和实际应用位置，identity 阶段也保留实际位置；历史证据的适用范围沿用现有判断。

PACT 的 Method/Call 复用同一应用边界与各层独立缓存。完成原评分层后，provider 用
下一层的原生投影提供 RoPE 前 Q/K 和旋转后的聚类 Key，Call 保存 EUTI 分数与 Key。
实际应用位置只消费这份来源；FFN 前的 checkpoint 重算按固定聚类成员重放均值，
不重新评分或聚类。Qwen3 剩余 DeepStack 分支使用同一成员映射，并在重算决定之外保留分支梯度。
聚类质量与比例 Attention 属于本次 Call，在当前 prefill / 训练前向的后续层应用；
增量 decode 沿用已有缓存，未新增跨调用的质量状态。三个基座复用 `pact_attention_mask`，
LLaVA 的直接回调继续保留原 Q/K 请求，并复用共享方法与布局修复。
运行记录区分 `score_layers` 与 `application_positions`，等价层间表达继续复用原配方证据。

HalfV 和 PyramidDrop 复用相同多阶段选择、布局更新和层内重放能力。
FinePruner 的 Method/Call 也使用这些执行边界：IAC 在应用特征上做原 LSH 首代表选择，ARP 通过 `capture_input` 请求固定层的原生 Attention，逐头分数由 Call 持有。
如果 IAC 发生在某个 ARP 来源之后，Call 将已捕获的逐头 Attention 按保留的图像索引同步映射；不能重新归一化这些概率，或改成其他层的评分。ARP 消费后释放分数，provider 不再维护 FinePruner 专属状态或调度分支。
Qwen 的连续位置和各层 decode 位置从实际首次应用边界推导；Qwen3 的当前及后续 DeepStack 分支复用共享选择映射。原 IAC/ARP 配方层计数保留为规则阶段身份，布局与运行记录另记 `application_position`。
判断跨阶段能力时检查完整操作的输入依赖。完整 FinePruner 的 ARP 需要 LLM Attention，不将其 IAC 子步骤拆成新算法来声称整套方法可以在 Encoder/Projector 执行。
TransPrune 复用同一选择和缓存路径，在原完整评分层捕获 TTV 历史快照与下一层权重计算的 IGA；延迟应用时只消费该阶段快照。
其方法对象持有历史、IGA 和选择结果。`begin_transition()` 仅在需要时请求原生 Attention/FFN
规范化输入与模块输出，基座完成本层后用 `record_transition()` 交回观测；临时 hook 由基座在退出时清理。
层内裁剪后，Attention 的 TTV 先在原完整域计算再选取幸存项，FFN 的 TTV 使用实际缩短后的域。
`score_layers` 表示 IGA 来源，TTV 累积层仍由方法配置的 `accumulating_layers` 指定。
若在正在积累 TTV 的层内压缩，Attention 和 FFN 分别按自身实际输入计算原公式及 softmax，之后才把 Attention 的视觉分数映射到 FFN 的保留 Token 上相加。先裁剪 Attention 特征再重新归一化会改变原规则。
各次选择同时映射全局 TTV 历史和待应用阶段的快照；最后评分来源之后不再收集没有消费者的 TTV。
HalfV 的两种 Attention 评分及 IVR 完整层特征固定在原来源，PyramidDrop 固定在完整层输出后使用下一层权重评分。
提前取得的后阶段分数按实际保留的视觉索引映射；回调中的原层参数保留阶段身份，`application_positions` 决定物理执行位置。
PyramidDrop 的 `layer_local` 缓存策略按已经执行 Attention 的层检查缓存，允许层间和残差后 FFN 前压缩，且不重写已有 KV。
Qwen3-VL 较早压缩复用剩余 DeepStack 分支的索引映射，记录中的 DeepStack 状态按实际应用位置计算。
LLaVA 正式 PyramidDrop 执行器只调用实际阶段，省去原来的非阶段空回调。
裸公开回调仍可用 `register_callback(PYRAMID_DROP_SITE, callback)` 注册：声明
`callback._mllm_algorithm = "pyramid_drop"`，不附带内部 `_mllm_pyramid_schedule`，
便会收到原逐层 `LlavaDecoderLayerRequest` 和原始 Q/K；可用 `_mllm_prune_layers`
限制来源层。原 FastV 裸回调的 `_mllm_algorithm = "fastv"`、`_mllm_prune_layer`
和可选 `_mllm_score_source` 同样保留。上述公开形式在注册边界转成显式请求适配对象，
正式方法与裸回调共用原生捕获和应用入口。

三个基座的正式 FastV、PyramidDrop、BTP、HalfV、TransPrune 都通过
`apply_decoder_selection` 执行；Qwen2.5 的 `decoder_layer_loop` 仅保留为语义站点名。
已删除无消费者的逐方法直接包装及请求/返回结构。LLaVA 的 FastV、PyramidDrop
和 PACT 直接请求适配仍有公开回调消费者，继续保留；新增正式方法复用公共执行入口。

BTP 的原 callback 层参数和 `schedule` 保留规则阶段的身份，用于匹配原预算；
`application_positions` 单独决定实际物理边界。provider 从固定来源层入口捕获分数，
在各实际位置调用同一 kernel，并通过既有缓存、mask、位置及 DeepStack 映射继续执行。
同层入口的前序应用必须先于后序阶段的来源采样；不能因为把评分统一移到循环开头，
就改变连续阶段原本看到的压缩布局。FinalWipe 不采样 Attention，也不构造未来来源。

Nüwa 的复合 activation 保留第一阶段视觉聚合；`NuwaDecoderMethod` 和 Call 管理
第二阶段的固定评分与选择，三个 provider 不再分别维护 Nüwa 分数和应用分支。
released Attention 按原 `stage2_after_layer-1` 的实际层输入捕获，使用原 last-prompt
查询和完整可见键域；random/cosine 分支不采样这份来源。分数在选择后释放，
重算只重放选择，增量 decode 不重复压缩。
`decoder_target_site` 单独描述第二阶段；复合注册把视觉聚合和 decoder 选择分别
传给对应站点。第二阶段记录在 `source_layout.stage` 标明 `stage2`，并保留
`application_position`、固定 `score_layer` 和原 RPME 位置描述。
第一阶段聚合权重和 RPME 保持原实现；Qwen3 剩余 DeepStack 分支、LLaVA newline
及缓存按同一布局修复，层内压缩沿用已有 checkpoint 重放机制。

FEATHER 复用相同 decoder 执行循环与层内残差函数。解析出的两个 application_positions
与原 spec.stage_layers 按顺序关联，kernel 仍按原阶段计算预算；实际位置单独写入记录。
两次无 RoPE 评分由 Call 使用请求的完整层输出计算，保留原规则回调的张量和模块上下文。
对已捕获而尚未消费的下一轮分数，按本轮 selected_visual_indices 做映射；尚未捕获的
来源直接观察后续当前特征。重算仅重放选择，不重复运行阶段或消费评分。

SGL 的 activation 在实际 target 深度确定后构造 `SGLDecoderMethod`，注入现有 guide 的
`consume_scores`。Call 在应用位置消费一次分数并调用原 kernel；三个基座复用既有
布局修复、compact-prefix 缓存和 checkpoint 重放。guide 的加载与
attach/stage/consume/finish/detach 仍由 `SGLGuideController` 及 activation 管理。
应用重算只重放已选索引，不再次消费 guide 分数；增量 decode 不重新选择。
`prune_layer_fraction` 仅保留默认层规则，实际应用位置由 target 深度和显式配置解析。
这不扩展原 GENERATE 支持阶段；训练兼容验证仍使用已有 GENERATE activation 的
teacher-forced 入口。

RESTORE 的视觉聚合沿用原 Projector 路径；训练前向和 prefill 根据聚合结果建立
`RestoreAttentionState`，其中普通层使用实际聚类质量，首层按 `first_layer_size`
选择均匀或聚类质量。增量 decode 只追加原生位置与单位质量，所有层共用校准和
掩码合成，不重复视觉聚合。该状态由各 provider 的请求上下文持有，独立于诊断记录。
Qwen 的 M-RoPE 轴与 LLaVA 的真实 AnyRes 代表位置及文本续接仍由各基座处理；
RESTORE 保持原生物理 KV 存储，没有额外的 KV 裁剪。

图像方法的参数名与默认值由算法 Spec 提供。内置注册读取 Spec 的默认字段，保留
D²Pruner 资源路径的既有省略语义，以及 FEATHER/IVC 的单图限制；旧直接入口复用
Method 的 `legacy_config_keys`，不再手写另一份参数白名单。参数优先级仍为旧分区、
`stage.config`、binding 配置依次覆盖，未识别的显式参数继续报错。复用已有执行器的
研究方法通过所选 binding 和自己的 variant 解析配置，无需加入旧入口的内置方法表。

## 音频方法跨执行族复用

A-ToMe、Encoder Top-K/Progressive 的 Projector／LLM 适配共用原 Encoder 规则的 compress/replay，
分别复用两个控制器的布局和生命周期。LLM 保留 mask、位置、labels、checkpoint 和 cache
处理；Projector 在一个投影边界逐轮选择，并将累计索引交给下游布局观察者。每个固定 Encoder
评分层复用现有 AudioAttentionReference；多个来源按顺序安装、逆序卸载，保持上游
布局观察链，Projector 自己的来源在安装时只观察上游阶段。计划按实际音频跨度和
应用族转换数量、位置策略及生命周期，不把原 Encoder 预算当作执行结果。

A-ToMe 的参考保存池化 Key 向量。Projector 逐轮用方法的 replay 映射后续来源，并将本次
压缩返回的原始重放计划交给下游参考映射；适配器不从 `selected_pair_starts` 等诊断字段
重建算法计划。计划只在本次布局观察链中使用，所有观察者执行后从公开记录移除。
独立策略可以使用自己的计划类型，无需继承 A-ToMe 或 Encoder Top-K 的实现类。
解析器检查执行器实际消费的 `compress`、`replay`、预算、代表索引和评分来源接口。
LLM 在正常前向合并后按保存的 replay 更新各 Key 来源的绝对
位置映射，checkpoint 重算复用原值图，不再次捕获或更新来源。标量注意力来源保留
原选择语义，不能把合并后的 Key 简化为只保留代表 Token 的 Key。

改变应用位置不应同时隐式改变评分来源。LLM executor 同样使用解析后的边界决定
在完整层输出或 Attention 残差后执行，并维护训练重算、labels 和缓存。

音频 LLM 的首次前向与 checkpoint 重算共用 `_run_compressed_layer`。层内应用直接
复用 `_decoder_boundary.py` 的 `decoder_post_attention_forward`，执行原生
Attention→压缩→FFN；不临时将 MLP 替换为零输出，也不在压缩前多执行一次 FFN
归一化。这个数值实现也由图像、视频的 `decoder_post_attention_boundary` 调用，
checkpoint 和决策重放仍由各自调用方管理；音频路径保留外层 Omni checkpoint
以及 Accelerate 的整层设备/卸载钩子。选择与合并的数学、源分数和重放计划继续属于方法。
布局适配只返回需要改变的原生 kwargs；首层压缩使用原始 RoPE，不预先构建随后丢弃的
mask/位置副本。增量 decode 只改写已有缓存对应的 mask，不接收未使用的 hidden 参数。

变长样本直接使用 PyTorch `pad_sequence` 补齐。prepare 阶段沿用 tokenizer 的左右
padding；LLM 选择/合并后在右侧补零值和 `-1` 位置。合并后的序列按有序音频区间
拼接未压缩文本及合并结果，不再逐 Token 拷贝文本；正常前向和计划重放共用该过程。

Stage 的完整身份由同一配置解析器展开；方法、基座和任务能唯一确定的字段可以省略，
有多种合法选择时显式填写。执行校验使用实际合同，不按历史支持证据设置研究运行门禁。
当前验证状态从 `discover` 查询；移动位置或修改规则不会自动继承原配方的证据。

详细 target site、运行语义和验证入口见
[`Qwen2.5-Omni 执行族接入参考`](../support/qwen25_omni/audio_compression/extension.zh.md)。
迁移的历史动机和完成条件保存在
[`Omni 压缩与分析收敛记录`](../archive/omni_compression_v2_convergence.zh.md)，但不再
作为当前接入步骤。

## 视频 Projector 与 decode KV 视图

VQToken、FastVID 和 VisionZip 共用 `backbones/video_projector.py` 的
`VideoProjectorExecutor`。Binding 的 `strategy_factory()` 创建方法对象；执行器调用
`method(payload, config=..., rules=...)`，需要重放 Qwen3 DeepStack 时再调用
`method.replay(features, result, config=..., rules=...)`。内置 kernel 也走这一接口。
算法负责决定和数值重放，执行器负责视频桥、回调、信号捕获的安装与释放、布局记录和
旁路应用。FastVID/VisionZip 仍保留各自真实信号来源及诊断；自定义方法或规则不继承
原论文配方的验证结论。[Stride 视频示例](../../examples/compression_extensions/stride_video.py)
无需自定义 executor，即可替换决定保留哪些 Token 的算法。

DyCoKe 的 Token 与 decode KV 联合机制还需要每个 cache 独立的策略。
`backbones/dycoke_video.py` 负责 prefill 布局、原生 Query/RoPE 捕获、完整 DynamicCache
写入、单 Token decode 读取和卸载；`compressors/video/dycoke.py` 负责 Token 压缩、
重放、视觉 Attention 评分与历史决策。`new_decode_policy(config=..., rules=...)`
为每个新 cache 创建策略，`plan_view(...)` 返回本步读取位置。teacher-forced 前向只
执行 Token 压缩，不创建 decode 历史；生成的 prefill 先完成布局与缓存准备，后续
增量 decode 才选择 KV 读取视图。

[Stride＋最近视觉 KV 示例](../../examples/compression_extensions/stride_recent_video.py)
同时替换这两种算法决定，沿用三个视频基座的物理接入。它保留全部非视觉 KV，按原位置
选择最近的视觉 KV，不计算 Attention 相似度，因此该记录为 `None`。这条机制保留
完整物理 KV，只减少实际读取的视图；实现物理缓存压缩仍需修改对应 cache 接口。

## 新增 comparison 方法的生命周期

新增方法共享 Algorithm/Variant/Binding/SupportUnit 与 Session，按各自模型边界安装。
Token stage 在训练时使用 `forward`、评测时使用 `generate`；Operation stage 两者均为
`prepare`。这里 `prepare` 是安装时机，不能据此认定其决策只运行一次，例如 AdaLLaVA
会在运行时调度，DOP-V 还执行视觉 Token reduction。

新增方法一次启用一个 comparison stage；OV 保留原生 AnyRes 几何，Qwen 保留 MRoPE，
Qwen3 的选择/聚合同步映射 DeepStack。重放应复用离散决定并保留值路径梯度；各方法的
checkpoint 重放证据分别维护，AdaLLaVA 训练当前要求关闭 gradient checkpointing。
Swift 分开记录 Token 变化与 Operation 执行，不以 Token 数减少代替运算压缩证据。

AdaLLaVA 与 Skip-Vision 的可学习参数有原生完整模型 sidecar 保存恢复测试，Swift/PEFT
adapter-only 路径仍需验证。所有新增精确单元保持 `declared`，范围、校准资产和配置见
[14 组对比方法](../support/imagellm/comparison_methods.zh.md)。

## Qwen3 Operation 的构造与安装

Attention/FFN 方法在各自 `compressors/llm_level/qwen3_vl_*_replacement.py` 中声明
`strategies`、生成计划并用 `build_replacement` 构造模块。数学参数和默认构造归方法，
已有 `factory` rule 可以替换构造结果。`qwen3_vl/operation_applier.py` 只消费该结果，
负责物理目标、替换顺序、模块/参数归属、重复安装检查和恢复；不再维护另一份策略白名单
或各算法构造分支。结构修改仍在 optimizer 创建前完成，沿用原 Session 与训练后端。

上述是 `.shared_llm` 的可逆模块替换；两插件的 `.image_llm` variant 在 Qwen2.5-VL /
Qwen3-VL 上由公共 pipeline 联合编译，分别维护图像与文本分支及保存恢复，见
[分模态说明](../../mllm_comp/modality_split/README.md)。MWOP 的 Mask 形式则是独立的
[头、路径与 FFN 遮罩](../guides/fixed_mask.zh.md)，不通过此模块替换路径执行。

## 方法的投影位置与旁路接入细节

下列说明供修改信号、值应用位置或原生接入时查阅。复用现有机制新增算法从
[扩展指南](../guides/2_dev_compression_method.zh.md)开始。

图像 VisionZip/PruMerge/GAP/DivPrune/AIM/VisPruner/MMTok/ZooPrune 和 Qwen3/LLaVA 的 SAINT packed 变体通过 `image_projector_sites` 解析原有 Projector targets；
三个 provider 按最终 sites 把原 kernel 应用在输入或输出 values 上。Qwen 前置 values 是原 LayerNorm 前
完整的空间 patch 组，Qwen3 main/DeepStack 保留共同布局决策；LLaVA 保留既有评分源与真实 AnyRes 布局。
评分与实际 values 分卡时对齐设备，不改变权重、聚合或选择公式。标准路径与探索路径共享这些实现，
resolver 保留默认 binding 和历史 evidence。配置例子、固定评分的计算成本及数学不适用的布局见
[压缩实验指南](../guides/1_run_compression_experiment.zh.md)。

Qwen2.5 SAINT 保留 Encoder 变体，使用 `saint_encoder_positions` 将各原步骤解析为实际边界。
`layer_indices` 继续选择原 norm/QKV Key 投影模块，`operation.layers/layer_edge` 决定应用位置；
模块选择与应用边界分别记录为 `key_projection_layers` 和 `application_positions`。
共用原手写视觉执行与裁剪过程，不新增执行器或视觉 checkpoint 重算机制。

MMTok/ZooPrune 保留原 projected 评分输入，按原 kernel 索引采样当前 values。
投影前所需的完整 projected reference 与实际压缩 values 分开传递，并记录其完整评分投影成本。
不把 raw 视觉维度代入原文本覆盖或投影空间多样性公式，也不复制选择内核。

Nüwa/RESTORE 的 `image_projector_sites` 只返回原 Projector targets，resolver 保留后续 decoder/辅助目标，
Nüwa 的 Stage1 与 Stage2 可同时按 target 配置。主分支的原 weights/membership 作用于各分支实际 values；
评分来源、Token 代表/大小以及后续位置和缓存处理保持原机制。
LLaVA 的这两法旧 callback 内部执行 raw 聚合再投影；显式 `post_vision_projector` 才切换到已投影值聚合。
沿同一 kernel 和回调分支实现，不添加另一套执行器；默认与历史 site 保留原数学，记录 callback 和 value_space 的区别。


视频 VQToken 通过 `video.positions` 解析原 `fixed_k_cluster` 步骤，再由现有视频桥的 `register_callback(..., projector_position="before")` 选择实际值边界；默认 `after` 保留旧调用。算法仍只接收 `VideoTokenPayload` 并返回原结果类型，不增加并行执行器或评分接口。Qwen 桥在原视觉调用内暂缓 merger，以完整 raw 空间组通过原排序与拆分，随后把压缩结果送入完整原 merger；临时方法在异常路径也恢复。LLaVA 使用原池化后的 raw 特征，原准备结果仅用于布局和换行参考，压缩后投影并复用原 splice。这个前置位置包括保持原池化 Token 域的语义，必须记录 `video_pre_vision_projector_pooled` 和完整准备投影的成本，不能声称池化与非线性投影等价交换。固定网格空间投影器的输入限制仍有效。


VisionZip/FastVID 视频位置继续复用原 kernel、signal provider 和视频桥。FastVID provider 的 `requires_projected_reference` 声明有实际用途：Qwen 前置时用已恢复的原 main merger 生成完整投影参考供现有 `enrich_payload` 计算帧均值，再恢复 callback 的 raw actual values；不新增 payload 字段或另一个评分执行器。VisionZip 的固定视觉层信号不需要完整投影参考。跨设备仅移动运算使用的评分张量，独立 raw 流仍保留原坐标与来源，不为布局验证复制整份 raw 特征。

DyTok/DyCoKe 的位置解析保留全部原 `Binding.targets` 及其顺序、角色和依赖。DyTok 辅助检测层由 `video_detection_layers` 解析，并传给原冻结检测器实际逐层 QK 捕获；检测模型准备时按自身深度校验，不能拿主模型深度代替。其 VisionZip 固定视觉信号继续复用原 provider，值聚合可作用于投影前后特征，不增加完整投影评分参考。DyCoKe 由 `video_decode_prune_layer` 解析首次稀疏层；原请求局部 cache 控制器随之改变评分层、挂接点和 prefill 已完成层集合，保留物理 cache、稀疏返回视图及单 Token decode 生命周期。二者均复用原 executor 和视频桥，不为辅助评分或 KV 操作创建 Token 层 hook。Manifest 沿用现有 `score_layers`、`application_positions` 和历史证据适用性判断，按 target 名区分辅助检测模型与主 decoder。

LLaMA-VID/LongVU 的自有 learned projector 由原 executor、共享 kernel 和原 Torch 模块接线，不能把其 LLM 宽输出再送入基座投影器。LLaMA-VID 的 `video_learned_projector_position` 将默认聚合解析到自有输出投影前；移动后仍用 raw QFormer/Key 取得同一 Attention，原 val/content projector 逐 Token 执行后按原权重聚合。LongVU 的 `video_longvu_frame_selection_site` 只移动原 DINO 筛选索引的应用，保留 combined target 与固定 SVA；全帧执行 SVA 后可在输出投影前或后筛选。均保持原参数名、注册/优化器时机及严格 state reload，不创建第二套模块或临时可学习参数。默认注入自定义模块沿用原调用；新位置要求模块明确接收所需位置参数，不能静默忽略。运行记录与 Manifest 沿用 `application_positions` 描述内部步骤，原视频回调 site 仍是默认兼容别名。

METok/PruneVid 共用 `backbones/video_decoder.py` 中的三基座原生循环。基座接入负责
Q/K 捕获、mask、位置、缓存和 DeepStack；`METokDecoderCall` 拥有事件预算、历史
分数映射与累计选择，`PruneVidDecoderCall` 拥有单次 VTP 的评分与选择。
`video_decoder_positions` 将原评分层与实际应用点分开。循环按方法阶段顺序交错采样
和应用，避免同层的后续阶段错误地看到压缩前布局；METok 最后一轮不请求评分。
LLaVA 的 METok 保留源位置、PruneVid 使用连续位置；Qwen 均保留原稀疏 M-RoPE。
层内调用 `decoder_post_attention_boundary`，同一 checkpoint 范围包住 Attention、
固定索引重放和 FFN。cache 使用 `compact_dynamic_cache_prefix` 按实际已写入层数
更新；待消费 DeepStack 在决策回调之外 gather，保留梯度。标签、累计位置及增量
decode 修复仍由原视频桥负责，诊断记录不参与算法状态或 checkpoint 重放。

METok/PruneVid 的 Projector 诊断记录只读取已生成的 DeepStack 分支形状，不再次执行
池化或聚合规则。分支数量、结果形状和共同 source groups 由现有 Qwen3 响应合同校验；
分支数值由原生测试中的独立双线性/均值预期验证。运行记录不再输出
`deepstack_projector_replay_exact` / `deepstack_source_group_replay_exact`，相应诊断命令
只报告实际检查的布局一致性。自定义规则在生成实际输出时执行，记录诊断不会重复调用。

DyCoKe、DyToK、FastVID、VisionZip、LLaMA-VID、LongVU 同样不输出
`deepstack_*replay_exact` 标记：比较对象身份或 source groups 不构成数值重放验证。
Qwen3 请求/响应合同统一检查分支形状、数量和 source groups；公共 splice 的
`shared_main_deepstack_layout` 记录布局结果，不再重复输出
`deepstack_source_groups_replayed`。算法记录保留分支数量和形状，诊断命令对账
callback 与实际 splice 的布局；分支数值和梯度由测试验证。
DyCoKe 继续要求缓存更新返回原始物理存储，并在其上构造稀疏读取视图；删除的长度
检查只是把同一个张量与刚从它读取的长度比较，不改变物理缓存或 decode 策略。

Projector 前置通过原 `register_callback` 的 `preserve_projected_values=True` 请求实际使用的完整评分参考，沿现有 `signals['projector_reference_features']` 传递；默认调用无需该参数。Qwen 与 FastVID 共用一次完整主投影参考计算，LLaVA 复用原准备输出，之后原计划分别重放双线性池化或分组均值到 raw 值。没有新增 payload 类型、伪 signal provider 或算法执行器。历史规则层字段保持兼容，新的 `application_position` 明确实际执行边界；Manifest 沿用 `score_layers` / `application_positions` 判断证据适用范围。
