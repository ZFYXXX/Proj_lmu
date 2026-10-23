# 压缩位置与评分来源

> 类型：Reference
>
> 适用范围：当前 Audio、Image、Video Token 方法的位置探索配置；实际执行仍受精确 binding 和 runtime 契约约束。

本页按模态集中记录应用位置、评分来源、预算、布局与缓存约束。常规配置、运行、恢复和比较见
[运行压缩实验](../guides/1_run_compression_experiment.zh.md)；公共字段见
[配置参考](config.zh.md)。新增方法接口见[开发压缩方法](../guides/2_dev_compression_method.zh.md)。

- [Audio](#audio)：Qwen2.5-Omni 的 Encoder、Projector 与 LLM 方法。
- [Image](#image)：三个图像基座的 Encoder、Projector 与 LLM 结构边界。
- [Video](#video)：视频 Projector、辅助模型、可学习模块、LLM 结构边界与增量 decode 的 KV 步骤。

位置和评分来源分别配置、分别记录。更改应用位置不自动更换原评分规则，也不能把历史位置的
质量、效率或训练证据直接用于新配置；各方法的具体可用范围如下。
结构位置与 prefill / decode 生成阶段分别描述，见[术语约定](../platform/core_concepts.zh.md#结构位置与生成阶段)。

## Audio

### LLM 与 Encoder 边界

位置探索沿用同一 v2 stage。Qwen2.5-Omni 的 AudioPyramid、FastAdaSP 和 HeadRouter
目前可通过 `operation.target_site` 选择以下 LLM 边界，`activation_scope.layers`
继续使用各方法已有的层号语义：

| target_site | 当前层的计算 |
| --- | --- |
| `llm_decoder_layer_output_audio_tokens` | 完整层执行后压缩，后续层使用压缩结果 |
| `llm_decoder_layer_post_attention_audio_tokens` | Attention 残差合并后压缩，当前 FFN 使用压缩结果 |

Omni 的 LLM 与 Encoder Token 方法可用 `operation.layers` 单独指定应用层，原有 `activation_scope.layers`
及方法的 schedule 层号仍控制评分来源和预算。应用层数量须与原规则的阶段数量相同。
完整层 site 的 `operation.layer_edge` 默认为 `after`；指定 `before` 时必须同时给出
`operation.layers`。例如层前 `[2]` 与层后 `[1]` 都解析为层间边界 `index=2`，只执行一次。
层前 `[0]` 解析为初始输入 `index=0`，首层 Attention 和 FFN 都使用压缩结果：

```yaml
# Audio-Pyramid 单阶段示例；其余 v2 身份和预算字段沿用原配置。
activation_scope:
  layers: [0]                 # 使用第 0 层输入及 Q/K 评分
  lifecycle: prefill_only
operation:
  type: keep
  target_site: llm_decoder_layer_output_audio_tokens
  layers: [0]                 # 应用位置
  layer_edge: before
```

Attention 残差后、FFN 前的 site 使用 `operation.layers` 指定当前层，不接受 `layer_edge`。
移动应用边界不会把评分层改成上一层：执行器保留原评分层输入；LLM 按当前绝对布局取值，
Encoder 按已经执行的选择或均值合并映射待用的评分特征。
需要尚未产生的未来层输入时明确拒绝；HeadRouter 使用初始嵌入和固定 probe 权重，保持
原校准来源，因此不受未来层 hidden state 的限制。运行记录分别列出 `score_layers_zero_based`
与 `application_positions`，后者的 `inter_layer` index 表示已完成的层数。

Encoder 的 A-ToMe、Encoder Attention Top-K 和 Progressive Attention Pruning 同样
支持 `audio_encoder_layer_post_mhsa_pre_ffn` 与 `audio_encoder_layer_output_post_block`。
两位置均保留各方法的层输入 Q/K 评分来源，只切换选择／合并作用的特征及后续计算。
完整层 site 同样接受 `layer_edge: before`：第 0 层前压缩作用于 CNN 和位置嵌入之后的
初始 Encoder 输入，不读取不存在的上一层。后续层入口与上一层输出使用同一执行边界。
基座统一维护残差、ragged chunk 布局、Attention mask 和 checkpoint 重算。

### Projector 边界

七种现有 Projector 方法可选择 `audio_encoder_output_pre_projector`（pooler 与
`ln_post` 后、投影前）或 `audio_projector_output_post_projector`（投影后）。
两处 Token 顺序和数量相同，特征维度可能不同。FastSLM HTA 按实际维度构建模块，
不兼容维度的旧权重会被拒绝；UnSeg 在实际选定位置预计算自然分段数量。

### UniSamp 与 UniAvg

UniSamp 与 UniAvg 还可在原配置上将 `placement` 改为 `llm_level`，使用上述任一 LLM
site，并显式给出 `operation.layers`。完整层 site 可使用 `layer_edge: before`。
这两种规则没有评分层，也不需要 instruction query；每次应用均对各 audio unit 独立
执行原有步长采样或窗口均值。多层应用重复使用原压缩因子，保留文本及其他模态。
均值池化保留原来的尾部丢弃规则，以各窗口首 Token 的绝对位置继续 M-RoPE；
采样沿用保留 Token 的原位置。配置后的实际 executor 决定阶段冲突，同一 LLM
executor 的重复压缩应放在一个 stage 的 `operation.layers` 中。
多个音频阶段复用现有依赖排序，按 Encoder → Projector → LLM 准备布局；配置列表的
排列不改变这一物理先后关系。显式 `depends_on` 与该顺序矛盾时，安装前报告依赖环。

UniSamp/UniAvg 同样可将 `placement` 改为 `encoder_level`，选择上述 Encoder site，
并显式设置 `operation.layers`；完整层 site 也支持 `layer_edge: before` 与第 0 层初始边界。
规则在每个原生 audio chunk 内独立执行，保留原有采样步长或窗口均值以及尾部规则。
最终 chunk 数量汇总后经过原生 pooler，再决定 LLM placeholder 数量。
如果预算不足以产生至少一个 pooler 输出，准备阶段会拒绝该配置。

### LTBM

LTBM 也支持上述 Encoder 和 LLM site，以及显式 `operation.layers`/`layer_edge`。
每次应用按原 `keep_ratio` 预算，在当前边界特征上执行局部二分合并；Encoder 中各
chunk 独立处理。多轮合并保留逐轮等权均值，checkpoint 重放各轮原决定。
LLM 合并结果沿用各目标 Token 的原绝对位置，记录为 `preserve_absolute_destination`。

### UnSeg

UnSeg 可将 `placement` 改为 `llm_level`，配置上述 LLM site 与显式 `operation.layers`，
完整层 site 同样支持 `layer_edge: before`。它在实际边界对每个 audio unit 检测相邻
Token 余弦差异的严格局部峰值，再对自然分段求均值；多层应用各自重新确定自然分段，
没有固定 keep ratio。LLM 规划时数量记为 `null`（待运行决定），不提前执行 Encoder；
运行报告记录真实数量。重算直接复用首次分段，不重新检测峰值；各输出沿用分段首
Token 的绝对位置。原 Projector 配方仍保留实际位置预规划与自然数量校验。
UnSeg 也支持上述 Encoder site 和层配置，每个原生 chunk 独立分段。准备阶段用当前
Encoder 执行循环进行一次保留 RNG 的无梯度预计算，确定各音频经过 pooler 后的占位数量；
正式前向重新计算实际特征及分段，校验数量与计划一致，checkpoint 只重放本次分段。
每个音频的最终 Encoder Token 少于 pooler 所需的两个时明确拒绝，不人为增加分段。

### SA-MAP

SA-MAP 也支持上述 Encoder/LLM site，使用 `operation.type: merge_prune` 和显式
`operation.layers`。评分仍来自 Encoder 倒数第二层的归一化输入；应用到 Encoder
时不能早于该输入可用的边界。合并重要性取各头最大注意力的 received 聚合，剪枝
重要性取各头平均注意力的对角线，两者保持分开。Encoder 按当前 chunk 计算，LLM
参考经过实际 pooler 及上游选择／合并映射；不会在未经过 pooler 的位置虚构两帧池化。
LLM 输出沿用所选分组首 Token 的绝对位置。分组阈值、注意力加权均值和 DPP 剪枝
保持原规则；分组少于请求预算时仍需提高 `similarity_threshold`，不会强行拆组。

### Segmentwise Top-K

Segmentwise Top-K 的 `criterion.score_source: uniform` 或 `random` 也可选择上述
Encoder/LLM site 与显式 `operation.layers`。每次应用重复原 `keep_ratio`、
`num_segments` 和 allocation 预算；`paper_floor` 即使 ratio 为 1 也可能因取整减少数量。
Encoder 按原生 chunk 分段并在 pooler 前规划；LLM 按 audio unit 分段，保留原绝对位置。
random 沿用 `random_seed + item_index`：Encoder 的 item 是当前请求中的 chunk，LLM 是
跨样本顺序编号的 audio unit。重算与后续评分源映射均重放已选索引，不重新抽样。
默认 `encoder_last_layer_attention` 同样支持 Encoder site。评分固定使用末 Encoder 层
输入的归一化及 Q/K，原分段选择规则在各 chunk 内执行；应用位置可为末层输入、
末层 Attention 残差后或末层完整输出。早于评分输入的位置明确拒绝；单层 Encoder 的
第 0 层初始边界可用。评分层不会随 `operation.layers` 改变。
默认 Attention 也支持上述 LLM site：当前请求必须实际执行 Audio Encoder，沿用其末层
Attention 分数及原 pooler 映射，在选定的 LLM 边界按各 audio unit 选择。重复应用按
保留的绝对位置读取同一份 Encoder 参考，不改用 LLM 的 Attention 分数。
前置 Encoder 压缩会同步原评分输入；前置 Segmentwise、UniSamp、UniAvg、LTBM、UnSeg
或 SA-MAP Projector 压缩会按实际选择、分段或加权合并同步参考。HTA 使用非线性、多头、
多尺度变换产生查询 Token，现有固定 Encoder-token 评分没有定义到这些查询的转移规则。
若 HTA 在该评分源捕获之后改变 Token，固定参考消费者会明确拒绝；HTA 在评分源之前
执行或实际为 identity 时可联用。运行记录另列实际 Encoder 评分层，
并在请求结束后释放参考，不依靠重新运行 Encoder 完成 checkpoint 重算。

### FastSLM HTA

FastSLM HTA 也支持上述 Encoder/LLM site、显式 `operation.layers` 与完整层的 `layer_edge`。
每次应用复用原查询网络和预算；Encoder 按原生 chunk，LLM 按 audio unit 分窗。
模块在 optimizer 建立前按实际边界宽度注册，同一模型可分别拥有 Encoder、Projector、
LLM 三个 HTA；各层应用复用所属 stage 的模块。checkpoint 重算重新执行同一可训练
网络，保留模块及输入梯度。冻结基座且只训练 HTA 时，基座接线在 checkpoint 输入处
建立必要的梯度入口，基座参数仍保持冻结。LLM 查询按窗口顺序编号，第 q 个查询的位置锚点为该窗口
起点加 `floor(q * 窗口长度 / 查询数)`，运行记录标为 `preserve_absolute_window_query_anchor`；
这只是 M-RoPE 位置，不表示查询对应某个原 Token 或继承其重要性。窗口长度以实际
边界的 Token 为单位，探索位置不沿用标准 Projector 的采样频率结论。原 HTA checkpoint
可在宽度兼容时复用；包含多个 HTA 的平台 checkpoint 必须有当前物理阶段的模块，
不能用另一个阶段的权重代替。

### Encoder Top-K、Progressive 与 A-ToMe 的跨阶段应用

音频 Encoder Top-K 与 Progressive 也可将应用位置设为上述 LLM 层边界。
`activation_scope.layers`（或原 cadence）继续选择 Encoder 评分层，`operation.layers`
选择 LLM 应用层，两者事件数量必须一致。保留原预算/线性剪枝顺序，将各评分层的
chunk 内平均接收注意力经过真实音频池化及上游布局映射，交给对应的 LLM 压缩事件。
例如来源 `[0, 2]` 可在 LLM 的 `layers: [0, 2], layer_edge: before` 分别执行。
Encoder 评分不读取 LLM 答案 Token；每个音频单元独立选择，原稳定 Top-K、时间顺序
和梯度重放规则不变。此处新位置仍使用 development runtime，不复用原位置的效果结论。

这两种规则也支持 `placement: projector_level`，目标可设为
`audio_encoder_output_pre_projector` 或 `audio_projector_output_post_projector`。
此时不填写 `operation.layers`／`layer_edge`；原 `activation_scope.layers` 或 cadence
仍决定评分层及各轮预算，在所选投影边界按 Encoder 层顺序逐轮执行，后续评分沿已选
索引对齐。每个音频单元按该边界的实际 Token 数计算预算，LLM 占位、位置与 labels
沿用 Projector 的紧凑布局处理。记录同时包含实际投影位置和每轮 Encoder 评分来源。

A-ToMe 也支持这些 Projector／LLM 位置，保持 `operation.type: merge` 和原
`schedule.merge_ratio`。`activation_scope.layers` 继续指定 Encoder Key 来源；Key
参考经过实际音频池化及上游布局变换，在每次合并后按同一配对取均值，再供后续事件
计算相邻 Key 余弦相似度。合并对象是应用边界处的 Token 值，原不相交相邻配对、稳定
选择和均值公式保持不变。LLM 使用合并对起点的绝对位置，Projector 使用紧凑占位；
第 0 层入口无需虚构前一层 Key，因为这些来源已在 Encoder 前向中产生。

## Image

### 新增对比方法的固定位置

14 组新增 comparison 实现按自己的 binding 声明位置：CDPruner、DPPruner、EADP、HiPrune
在 Projector 区间；VLM-Pruner、TwigVLM、CoreMatching 在 LLM Token 边界；其余为 LLM
Operation。DOP-V 同时含 Projector Token reduction，但整体仍以 `prepare` Operation
接入。下文历史方法的可移动位置与自定义规则不能自动套用到这些新增 binding。

OV 的 `tokens_per_crop` 是每个 crop 的 patch 预算，Qwen 是单图 post-merger 总预算；
最终 OV Token 数还受 unpad/newline 影响。精确参数、三基座映射与范围见
[对比方法说明](../support/imagellm/comparison_methods.zh.md)。

### ToMe：Encoder 与永久 Projector 步骤

图像 ToMe 的三个基座可以单独配置 `encoder_merge` 步骤：

```yaml
operation:
  type: merge
  targets:
    encoder_merge:
      target_site: visual_block_output
      layers: [1]
```

Qwen2.5-VL、Qwen3-VL 使用 `visual_block_output` 或
`visual_block_post_attention_residual_pre_mlp`；LLaVA-OneVision 对应使用 `vision_` 前缀。
旧 Qwen 配方里的 `vision_block_post_attention_residual_pre_mlp` 拼写仍有效。
`layers` 为零起始应用层号，默认在各层 Attention 残差后、FFN 前合并；层输出的
`layers: [1]` 与 `layers: [2], layer_edge: before` 等价，均在第 1 层 FFN 后合并。
`layers: [0], layer_edge: before` 表示首层输入；使用首层输入可计算的 Key 评分，
压缩后的 Token 进入首层 Attention 和 FFN，不虚构上一层上下文。

只改变合并步骤时，密集恢复、最终输出合并及 AnyRes/DeepStack 顺序继续保留。
ToMe 仍使用该层 Attention 输入的 key 做匹配；移动到层输出不改用 FFN 后重新投影的 key。
可用 `criterion: {score_layer: 0}` 固定使用第 0 层入口的 Key，搭配上述任意可用应用
边界。来源在该层压缩前捕获；后续合并按实际 Token 质量映射同一参考，不重新取
应用层 Key。来源必须在应用边界处可计算；`before k` 可使用第 k 层输入 Key，仍在同一边界
按原生 norm/投影计算，不运行未来的 Attention 或 FFN。Qwen2.5 如需密集窗口输入，
先沿已有结构恢复映射构造评分输入；该次压缩仍按原顺序执行。第 0 层入口只能使用第 0 层来源，不能
读取未来层。不设置 `score_layer` 时继续使用原有的应用层 Attention 输入 Key。
运行事件同时记录 `score_layer` 和 `application_position`，Manifest 保留历史引用，
将固定评分来源配置标记为探索配置；最终输出合并步骤的原有 Key 来源不受此设置影响。
LLaVA 已有 `config.layers`／`r_by_layer` 配方设置新应用层时，按原预算顺序映射到
新位置，应用层数量须与原预算序列一致。旧预算中零合并的层不要求固定来源已经可用。Qwen2.5-VL 保留每层恢复密集窗口的行为，
因此完整层输出合并后会恢复布局；首层输入压缩也在首层结束时恢复原窗口。
首层紧凑 Attention 使用所选代表 Token 的 RoPE 和累计 Token 权重；单层 Encoder
的后续步骤消费实际 Attention 的 Q/K 及恢复后的参考，不使用合并前旧 Q/K。
这与 Qwen3-VL 的跨层紧凑 Token 传播不同。

ToMe 的 `permanent_output_merge` 是使用独立 `output_r` 预算的第二次压缩，可单独选择 Projector 前后：

```yaml
operation:
  targets:
    encoder_merge:
      target_site: visual_block_output  # LLaVA 使用 vision_block_output
      layers: [1]
    permanent_output_merge:
      target_site: projector_input
config:
  r: 1
  output_r: 1
```

省略永久输出步骤的位置或填写 `projector_output`，继续使用原投影后配方；各基座旧物理位置名也有效。旧的顶层 `operation.layers`、`target_site` 等简写仍控制 Encoder，Projector 位置使用上面的步骤名配置。`criterion.score_layer` 继续只影响 Encoder；永久输出压缩保留其原评分来源和独立分组，不重放 Encoder 的旧合并结果。

Qwen2.5-VL 在完整 raw 2×2 空间组上聚合后执行原 LN/merger，只做一次原窗口逆排序；Qwen3-VL 的主分支和所有 DeepStack 分支共享一次输出分组，再各自投影。LLaVA 使用与原 AnyRes 布局对应的 raw 值，保留完整 projected packed 特征的第 0 层 decoder QK 评分参考，压缩后投影内容并恢复原 newline。因此 LLaVA 前置仍有完整评分投影成本；实际 resize/pooling 或固定网格投影器不满足既有 raw 布局映射条件时会明确拒绝。

`dense_restore` 等结构恢复步骤保持在原网格消费者之前；其显式位置只能填写原物理位置，不能随输出压缩移动。运行记录和 Manifest 分别记录 Encoder 与永久输出位置，并区分历史证据的适用范围。这三个现有 ToMe provider 继续支持 `forward` 且 `use_cache=False`，保留原来的 generate/cache 拒绝行为。

### FastV：层间与 FFN 前边界

图像 FastV 在三个基座上可将固定评分规则应用于不同 LLM 层间边界，例如：

```yaml
config:
  prune_layer: 2       # 保留第 1 层输入 Q/K 的原评分来源
  prune_ratio: 0.5
operation:
  targets:
    decoder_prune_boundary:
      target_site: llm_decoder_layer_output_image_tokens
      layers: [3]     # 第 3 层完整输出，即第 4 层入口
```

也可写 `layers: [4], layer_edge: before` 表达同一个边界；`before 0` 表示首层输入，
`before N` 表示 N 层模型的栈末输出。原 `llm_decoder_pre_prune_layer_image_tokens`
继续按层入口解释，不设置位置字段时保持原配方。评分来源必须已可计算：例如
`prune_layer: 1` 可在首层输入压缩，而 `prune_layer: 2` 不能提前到首层输入。
`prune_layer: 0` 仍为原有随机选择，可应用于上述层间位置。每次 prefill 只选择一次，
后续 decode 复用压缩后的缓存；Qwen3-VL 层输出保持在该层 DeepStack 注入之后。
FastV 也支持 `llm_decoder_layer_post_attention_image_tokens`，在 `layers: [k]` 指定的
第 k 层 Attention 残差合并后、FFN 前执行。当前 Attention 使用原 Token，当前 FFN
和后续层使用压缩 Token；此位置不接受 `layer_edge`。在缓存生成的 prefill 中，配套的 KV
布局修复包含当前 Attention 已写入的 KV；它不表示在后续增量 decode 中再次执行 KV 压缩决策。
Qwen3-VL 当前层的 DeepStack 分支随后按新布局注入。训练重算复用
本次选择，保持 Token 值的梯度，不重复评分或修改运行记录。

不设置新位置时保留标准配方默认值。改变位置不会改变评分规则、预算或启用第二条
运行路径；基座先校验实际接线能力，再使用相同算法执行。新位置通过 development runtime
运行，Manifest 记录实际位置并保留历史证据引用，同时标明这些证据不精确适用于新位置。
这类功能探索不要求重新激活支持单元；评价新位置的质量或效率仍需相应实验。

### DART

DART 也使用 `operation.targets.decoder_prune_boundary` 的上述层间／FFN 前位置。
`config.prune_after_layer=p` 保留原第 `p-1` 层入口的 rotary Key，以及该完整层输出
经 decoder norm 的相似度特征；Qwen3-VL 的完整输出包含该层 DeepStack 注入。
改变应用层不会移动这两项来源。因此最早可在层间边界 `p` 或第 `p` 层 FFN 前应用，
不能在来源层完成前使用这些评分。最后一层可作为来源，在栈末应用；栈末没有后续 FFN。
默认 `prune_after_layer`、保留预算及 pivot／allocation 规则保持不变。

### D²Pruner

D²Pruner 使用同一应用位置配置。`config.prune_after_layer=p` 固定第 `p-1` 层入口
Attention 评分；最早可以在层间边界 `p-1`（含起始 `0`）或第 `p-1` 层 FFN 前应用。
`bias_prior_path` 仍必须提供与原模型及评分层匹配的 prior；移动应用边界不会更改
prior 所属评分层，也不会触发重新校准。图的特征来自实际应用位置，评分去偏、
pivot/MIS 及保留比例规则保持原实现。

### SGL

SGL 的 guide 在 target 执行前提供分数，使用 `operation.targets.decoder_guide_prune_boundary`
或根 operation 的位置简写，目标模型可选择起始、
层间、FFN 前或栈末边界。原 `prune_layer_fraction` 仍决定未覆盖时的默认应用层，
不会改变 guide 的 Attention 来源或聚合规则；guide/target 的视觉 Token 对齐检查保留。
SGL 仍使用原 GENERATE 配方及 guide 生命周期。新位置配置按探索配置记录；运行前
只有层号比例而未加载目标深度时，不推断其与显式层号的历史证据等价性，运行时记录
实际解析的层号。

### IVC-Prune

IVC-Prune 也可选择层间、FFN 前和栈末边界。LLaVA 的操作目标名为
`native_rope_coordinate_prune`，两个 Qwen-VL 为 `native_mrope_coordinate_prune`；
也可用根 `operation` 的位置简写。`config.prune_layer=p` 保留原第 `p-1` 层的
RoPE 坐标及该完整层输出的 V 投影（Qwen3 含该层 DeepStack 注入），最早在层间 `p`
或第 `p` 层 FFN 前应用。移动应用位置不改坐标锚点、语义种子、前景评分或预算规则。
旧配置仍在第 `p` 层前压缩；不能在第 0 层入口使用尚未产生的完整层评分。

### FEATHER

FEATHER 的 `config.stage_layers=[p, q]` 继续定义两次原始完整层评分来源及对应预算。
通过 `operation.targets.two_stage_no_rope_prune`（或根 operation 简写）设置两次应用的
`layers`；可选上述层间或 FFN 前 target site，层间可包含栈末。应用层按顺序与两阶段
对应，每次都必须晚于自身评分层完成，不能省略、倒序或合并这两个阶段。
未覆盖时仍在 `p`、`q` 层前应用。若第二份分数在首轮压缩前就已产生，框架按首轮
保留索引更新它，不重新评分；原始跨度预算、注意力与均匀采样并集规则保持不变。

### Nüwa 第二阶段

Nüwa 的第二阶段通过 `operation.targets.stage2_last_prompt_attention_selection`
选择上述 LLM 层间、FFN 前或栈末位置；第一阶段视觉聚合及 Qwen3 的 DeepStack 聚合
目标仍保留原顺序。`config.stage2_after_layer=p` 在 released Attention 规则下固定
第 `p-1` 层入口的评分来源，最早可在同一入口或该层 FFN 前应用；默认仍在第 `p` 层前。
`random_stage2=true` 和 `stage2_score=paper_text_cosine` 不依赖这份 Attention，
可以选择首层入口；cosine 使用实际应用边界的图像与文本特征。`semantic_hidden_start/end`
始终指第一阶段的视觉 Encoder 来源，不是 LLM 评分层。位置变化不改变两阶段预算，
运行记录分别保存第二阶段实际位置和评分来源，历史证据仍按原配置确定适用范围。

### HalfV

HalfV 用 `operation.targets.decoder_ivr_ssr_boundaries` 的两个 `layers` 分别选择 IVR、SSR 应用边界。
原 `ivr_layer_index`、`ssr_layer_index` 和预算保持不变：IVR 需要原评分层入口的无 RoPE Attention 及该完整层的归一化输出；SSR 需要原评分层入口的 RoPE Attention。
因此 IVR 不能早于原完整评分层，SSR 可以在自身评分层的层间入口或 FFN 前应用。IVR 清空视觉 Token 时，SSR 按原规则跳过。

### FinePruner

FinePruner 用 `operation.targets.decoder_iac_arp_boundaries` 配置各次 IAC 和最后一次 ARP 的应用位置，按执行顺序提供 `layers`。
IAC 在所选位置的特征上执行原哈希分组和首代表选择；ARP 仍使用 `arp_center_after_layer`、`arp_layer_offsets` 指定的 LLM Attention，不能早于最后一个评分层输入可用的位置。
层间使用 `llm_decoder_layer_output_image_tokens` 与 `layer_edge: before/after`，层中使用 `llm_decoder_layer_post_attention_image_tokens`；IAC 可以放在第 0 层入口，ARP 可以放到末层输出。原阶段参数与预算不变，缓存仍采用 `layer_local`，剪枝后仍重建连续位置。
完整 FinePruner 包含依赖 LLM Attention 的 ARP，因此不能仅修改位置字段就将整套方法移到 Encoder 或 Projector；这不影响其兼容 LLM 位置的配置能力。

### TransPrune

TransPrune 用 `operation.targets.decoder_ttv_iga_boundaries` 配置各阶段的应用边界，顺序和数量与 `prune_after_layers` 一致；支持层间、FFN 前和栈末。
`accumulating_layers` 与 `prune_after_layers` 仍决定原 TTV/IGA 评分来源，每个应用边界必须在对应完整评分层之后。延迟应用不会把后续新增的统计算进当前阶段；中途剪枝会同步对齐已经捕获的统计。
原保留比例、连续位置和 `layer_local` 缓存策略不变。完整方法依赖 LLM Attention、FFN 转换统计和图像后的文本查询，不能直接放到缺少这些输入的 Encoder/Projector。

### SparseVLM

SparseVLM 用 `operation.targets.decoder_merge_boundaries` 配置各次选择、回收和合并的位置，
数量对应 `config.prune_layers`；层间、层输出与 FFN 前位置使用相同字段。
`prune_layers` 仍指定零基评分来源层，默认在这些层完整输出后应用；新位置最早可以在对应来源层入口，
来源为 0 时也允许第 0 层输入。初始文本评分者只选择一次，各阶段仍使用原来源层入口的 Attention。
如果后一步分数已提前取得，前一步合并后会按布局映射累加对应 Attention 列的概率质量。
选择预算、回收比例和聚类规则不变，实际应用处的特征参与均值合并，位置保持连续编号，已完成的 KV 前缀同步均值合并。

### PACT

PACT 用 `operation.targets.decoder_pact_boundary` 指定一个 LLM 应用位置，使用同样的
`target_site`、`layers` 和 `layer_edge` 字段，可选层间、Attention 残差后/FFN 前或栈末。
`reduction_layer` 仍指定原完整评分层（一基层号）；该层输出与下一层 Q/K 权重决定
EUTI 分数及聚类 Key，应用位置必须晚于该来源。延后应用时在新位置执行原聚类成员的均值聚合，
保持预算、剪枝及聚类规则、各层独立缓存与比例 Attention 开关；Qwen3 的后续 DeepStack 分支同步聚合。

### PyramidDrop

PyramidDrop 用 `operation.targets.decoder_stage_boundary` 配置各阶段应用位置，数量与原阶段数一致。
默认均匀调度（以及 Qwen 的 `config.layer_indices`）继续决定原完整评分层；评分仍使用下一层的 Q/K 投影和原指令查询。
应用可延后到层间、FFN 前或栈末，不能提前到完整评分层之前。它保留每层独立的 KV 长度；已计算 Attention 的缓存保持原长度，后续 Attention 使用压缩序列。
这些多阶段配置仍使用上述 `target_site`、`layers`、`layer_edge` 语法；历史证据适用范围与实际应用位置分别记录。

### BTP

BTP 使用 `operation.targets.decoder_balanced_pruning_boundaries` 配置每一阶段的
LLM 应用位置，`layers` 数量须与原阶段数一致，并保留最后的 FinalWipe。默认仍按
`config.prune_before_layers` 的原计数约定执行：值 `p` 对应零起始第 `p-2` 层的
入口 Attention 评分、完整层后应用。探索配置可将应用移到层间、FFN 前或栈末，
非最终阶段不得早于各自评分源；FinalWipe 无评分依赖。阶段预算、Attention 种子、
多样性与重平衡规则保持原配方。若前一阶段应用与后一阶段评分同处一个层入口，
框架先应用前一阶段，再采样后一阶段的输入；已提前捕获的分数则随保留索引同步。

### VisionZip、PruMerge、GAP 与 DivPrune 的 Projector 步骤

VisionZip、PruMerge、GAP、DivPrune 的图像 Projector 步骤可用
`operation.target_site: projector_input` 或 `projector_output` 选择投影前后；省略时保留原输出位置。
Projector 不接受 `layers` 或 `layer_edge`。Qwen3 可在 `operation.targets` 下分别指定主分支和
DeepStack 原目标的 `target_site`：VisionZip/DivPrune 为 `main_image_features` / `deepstack_image_features`，
PruMerge 为 `main_adaptive_merge` / `deepstack_adaptive_merge`，GAP 为
`main_grounding_aware_prune` / `deepstack_grounding_aware_prune`。未指定目标保持默认，所有分支仍采用同一布局决策。

Qwen 的投影前压缩单位是完整的 2×2 patch 组；压缩后的组重新交给原 merger，从其 LayerNorm 开始。
Qwen2.5 先还原窗口顺序，Qwen3 可混合主分支和 DeepStack 的前后位置。VisionZip、PruMerge、GAP
保持原 Attention/Key 来源和选择、聚合公式；DivPrune 在实际应用位置的特征上计算原余弦距离。
ToMe 共存时，投影前步骤执行后仍继续 ToMe 原投影输出步骤。

LLaVA 的前三种方法依赖完整已投影、已打包特征的原 LLM 第 0 层 QK 评分；投影前位置仍需这份完整参考，
因此不能由位置名称推断省去了该投影计算。实际压缩在视觉特征上进行，再投影内容 Token，并保留原 newline 值及梯度。
纯重排、unpad、nobase 布局可对应视觉输入；实际 resize/pooling 与非线性投影不交换，固定空间 Pooler
也不能接收任意缩短序列，这些输入的投影前配置会明确拒绝。DivPrune 原算法以受保护 Token 为初始距离中心，
而 newline 只有 LLM 空间表示，因此其投影前位置仅适用于没有 newline 的实际布局；默认投影后行为保持。

运行记录和 Manifest 保存最终物理位置，显式 `projector_output` 复用默认证据，改变位置保留历史引用并标记其适用范围。
位置配置沿用已有执行阶段能力。三个原生桥的纯 Projector 方法目前没有独立 generate 生命周期；历史注册表中
VisionZip/DivPrune 的 generate 声明不代表这一原生路径已经实现，本次不扩展该能力。

### AIM 与 VisPruner

AIM 和 VisPruner 同样使用上述 Projector 输入/输出配置。AIM 在实际主特征上运行原逐轮二分合并，
保持原均值计算与代表顺序；Qwen3 的 DeepStack 沿用既有按最终成员均值同步的规则，多轮时不应将其
描述为与主分支具有完全相同的聚合权重。VisPruner 保留原视觉特征与 Attention 作为选择来源，
将选择应用到实际输入或输出 values；LLaVA 保留原 per-crop 预算与 SigLIP 来源。
AIM 的 Qwen3 原目标名与 VisionZip 相同；VisPruner 为 `visual_attention_main_image_features` /
`deepstack_image_features`，Qwen2.5 为 `visual_attention_merged_image_features`，LLaVA 为
`visual_attention_crop_features`。这些位置配置不修改原算法参数或重新收集证据。

LLaVA VisPruner 需要与原 crop Token 一一对应的 SigLIP Attention。若联用 ToMe 的 Encoder 压缩，
只有 hidden 被恢复为 dense 而 Attention 仍为压缩布局时，不能把该 Attention 当作原 crop 评分；
该组合会明确拒绝。本次验证的联用保留 ToMe 的永久 Projector 输出步骤（Encoder `r=0`，`output_r>0`），
不增加评分重建或修改 per-crop 选择语义。

### SAINT

SAINT 保留各基座已有的不同适配阶段。Qwen2.5 的 `saint.image.encoder_progressive_prune` 用
`operation.targets.encoder_progressive_prune` 指定位置：`visual_block_output_progressive_patch_prune`
（或 `visual_block_output`）配合 `layers`、`layer_edge` 表示层间/完整层输出，
`visual_block_post_attention_residual_pre_mlp` 表示 Attention 残差后、FFN 前。
第 0 层入口写 `layers` 中的 `0` 与 `layer_edge: before`；栈末可写层数与 `before`。
位置数量必须等于原渐进步骤数。

`config.layer_indices` 仍决定各步骤采用哪个原 norm/QKV 模块及默认应用位置；`keep_ratios` 保持原预算。
在所选实际位置，用该步骤的原模块对当前特征计算 raw Key，不要求历史 Attention，也不要求投影模块
已参与前向。Manifest 的 `key_projection_layers` 专门记录这些模块编号，与 `application_positions` 区分。
原 source patch 顺序、顺序窗口重建、Token/group 两种选择规则和最终 merger 数量约束保留。
原 SAINT 视觉路径手动调用 Attention/FFN，本次未新增视觉 block checkpoint 重算机制。

LLaVA 和 Qwen3 的 SAINT 仍是 `saint.image.packed_feature_prune`，使用上面的 Projector 输入/输出语法，
目标名与 VisionZip 相同；分别保留原完整 packed decoder QK、最终 vision Key 评分来源。
改变位置不新增阶段或改变已有配方的评分规则、参数和证据结论。

### MMTok 与 ZooPrune

MMTok 和 ZooPrune 也使用 `projector_input` / `projector_output`。MMTok 保留原完整已投影特征、
视觉特征和文本嵌入的覆盖评分；ZooPrune 保留已投影特征的多样性评分，以及原 merger LayerNorm 后
MLP 输入的敏感度估计与代理。原选择索引应用到当前位置的特征，再交给原投影模块。
因此投影前配置仍需完整主投影作为评分来源，不能声称省去了这部分计算。
Qwen3 的主目标分别为 `multisource_main_image_features`、`projector_sensitivity_main_image_features`，
DeepStack 均为 `deepstack_image_features`，可分别设置前后位置，并沿用主分支的选择。
Qwen2.5 对应 `multisource_merged_image_features`、`projector_sensitivity_merged_image_features`；
LLaVA 对应 `multisource_anyres_features`、`projector_sensitivity_crop_features`，保留原逐 crop 预算、
映射后的 packed 索引和 newline。上述实际 AnyRes/pooling 限制仍适用。

### Nüwa 第一阶段与 RESTORE

Nüwa 第一阶段和 RESTORE 的视觉步骤也支持 Projector 前后位置。Nüwa 可同时配置两个原阶段：

```yaml
operation:
  targets:
    stage1_regional_visual_aggregation:
      target_site: projector_input
    stage2_last_prompt_attention_selection:
      target_site: llm_decoder_layer_post_attention_image_tokens
      layers: [2]
```

第二阶段仍受原评分来源的可用层约束。Qwen3 可另设 `stage1_deepstack_aggregation`，
与主分支共享原聚合权重。RESTORE 的视觉目标为 `vispruner_pristine_plus_distinctive_anchor_merge`，
Qwen3 另有 `shared_deepstack_distinctive_anchor_merge`，共享原成员关系、代表位置和 Token 大小。
RESTORE 的 Attention 校准是每个 decoder 层的 logits 偏置，保持原生命周期；它不是可移到 FFN 前的
Token 压缩步骤。Nüwa 的 RPME/第二阶段与 RESTORE 的校准/缓存机制继续有效。

LLaVA 的这两个方法有一项历史区别：旧 `post_anyres_pack_pre_llm_splice` 是回调名称，
内部实际先聚合 raw 视觉特征，再投影。省略位置或显式写旧 site 会保留这一行为。
`projector_input`（`pre_vision_projector`）通过公共接线投影聚合后的视觉值；
`projector_output`（`post_vision_projector`）则真正聚合已投影值，聚合后不再次投影。
两种数学分别是 `P(WV)` 和 `W P(V)`，非线性投影下可以不同，不能将显式 output 当作这两法的旧默认别名。
运行布局中的 `callback_site` 与 `value_space` 区分回调和实际值空间。
Nüwa 保持最终 SigLIP 特征及原 semantic/Attention/Key 来源，RESTORE 保持倒数第二层特征与 Attention；
默认参数、每 crop 预算、newline、代表位置与 Token 大小均不改变。前置所需的完整投影用于 AnyRes 打包参考，
仍按实际成本记录。上述 AnyRes/pooling 的真实结构限制继续适用。

## Video

### VQToken 的投影位置

`vqtoken.video` 在三个现有视频基座上支持 `operation.target_site: projector_input` 或 `projector_output`，也可用原步骤名配置：

```yaml
operation:
  targets:
    fixed_k_cluster:
      target_site: projector_input
```

省略位置或选择 `projector_output` 保持原默认配方。前置仍使用原 K=32、L2 归一化、全视频聚类与成员均值规则，规则直接作用于所选位置的特征，因此分组可能随位置改变；没有新增评分算法。Qwen3 的主分支和全部 DeepStack 分支同时切换位置，并沿用主分支分组与原分支归一化均值。

Qwen2.5/Qwen3 的前置输入是 LN 之前的完整 2×2 视觉特征组；压缩后的组经过原 LN 与投影器。源坐标仍以完整组为单位，保留 Qwen2.5 窗口还原和 Qwen3 DeepStack 对齐。运行位置分别记录为 `video_pre_vision_merger` 和 `video_pre_main_and_deepstack_mergers`。

LLaVA 视频保留原池化网格与换行 Token。记原视觉特征为 V、池化为 S、投影为 P、压缩为 C：默认执行 C(S(P(V)))，前置执行 P(C(S(V)))，记录为 `video_pre_vision_projector_pooled`。池化与非线性投影不交换，两种位置可以产生不同结果；前置使用池化后的原始视觉特征，不改成未池化网格的 Token 预算。当前仍执行原完整准备过程取得布局和换行参考，因此会保留一次完整投影计算，不能据前置配置宣称这部分计算已节省。要求完整固定网格的空间 PoolerProjector 不能接收压缩后的不规则行，前置会说明这一限制。

新位置作为探索配置记录；默认配置继续复用原支持与实验记录，历史 checkpoint 证据不被标为已验证前置效果。

### VisionZip 与 FastVID 的投影位置

视频 `visionzip.video` 和 `fastvid.video` 同样支持三个基座的 `projector_input` / `projector_output`，原步骤名分别为 `dominant_contextual_reduce` 和 `dyseg_stprune_dtm`；基座输入特征域、Qwen3 同步分支和 LLaVA 池化顺序与上述视频位置说明一致。默认预算与规则不变。VisionZip 保留原倒数第二层 Attention/hidden 评分来源、每帧 126+21 预算及目标值加其余成员均值的公式；前后位置只改变这些分组所聚合的值。FastVID 保留原分段评分信号，密度与后续聚合使用所选位置的特征。其 LLaVA 仍用原 SigLIP pooling head；Qwen 仍用原最后视觉层 Attention 和完整投影后的帧均值。因此 Qwen 的 FastVID 前置需要额外保留一次完整主分支投影作评分参考，运行记录单独计入该成本，DeepStack 无需这份完整评分投影。

### DyTok 与 DyCoKe 的独立步骤

两种方法都通过 `operation.targets` 分别配置已有步骤，省略某一步的配置会保留该步默认行为。Projector 步骤分别为 DyTok 的 `dynamic_visionzip_reduce` 和 DyCoKe 的 `ttm_visual_merge`，均支持上述三个视频基座的投影前后特征域。

DyTok 的 `detection_budget_provider.layers` 是**辅助检测模型**中实际用于计算帧重要性的 Attention 层号，默认 16–23；它不改变主模型的 decoder 层，也不是 Token 压缩边界。层号须非空、递增、唯一，且小于检测模型的真实深度。原检测、动态预算和 VisionZip 规则保持不变。

```yaml
operation:
  targets:
    detection_budget_provider:
      layers: [12, 16, 20]
    dynamic_visionzip_reduce:
      target_site: projector_input
```

DyCoKe 的 `decode_kv_view_prune.layers` 只填写一个层号，表示 decode **首次使用稀疏 KV 视图的层**；此前一层的完整 KV Attention 提供选择分数。从该层开始，后续层使用稀疏返回视图，物理 cache 仍完整。默认首次稀疏层为 3、评分层为 2；可配置范围是 1 至 decoder 最后一层。层 0 不能首次稀疏，因为没有前一层提供分数。此操作发生在 decode Attention 使用 KV 之前，不接受 FFN 前或层输出等 Token 边界。forward 阶段仍只执行可微的 TTM，不安装 decode 控制器。

```yaml
operation:
  targets:
    ttm_visual_merge:
      target_site: projector_input
    decode_kv_view_prune:
      layers: [1]  # 第 0 层评分，第 1 层起使用稀疏 KV 视图
```

运行记录分别写出实际 Projector 位置、检测层或 KV 评分/应用层。改变任一步骤位置后，原历史证据保留为引用，但不再作为新配置的质量或训练验证结论；显式填写原默认位置仍保持证据适用性。

### LLaMA-VID 与 LongVU 的自有投影模块

这两种方法的压缩发生在各自的可学习模块中，三个基座都沿用原视频回调。这里的 `projector_input` / `projector_output` 指**方法自己的输出投影器**；基座原投影及布局准备仍会执行，不应据此宣称省去了基座投影成本。省略配置、填写原视频回调 site，或填写 `projector_input`，均保留原配方。

LLaMA-VID 的 `qformer_context_content_pool` 默认在自有投影前聚合。可以改为逐 Token 投影后聚合：

```yaml
operation:
  targets:
    qformer_context_content_pool:
      target_site: projector_output
```

原 32 个 QFormer query、raw Key 和 Attention 权重不变，每帧仍输出 context、content 两个 Token。默认 context 是 Attention 加权均值后做线性投影，content 是均值后做非线性投影；改为输出位置后，两者先逐 Token 经过原投影器再按原权重聚合。context 在线性运算下理论等价，content 因非线性投影可产生不同值与梯度。Qwen3 DeepStack 保留原加权 context 和均值 content 映射。实际聚合位置记录为 `llamavid_pre_output_projectors` 或 `llamavid_post_output_projectors`。

LongVU 原步骤 `dino_frame_select_sva_fusion` 包含帧筛选和 SVA。SVA 的 query、Key/Value、残差与归一化依赖固定的视觉隐藏维度，不能穿过非线性输出投影器而保持原参数及规则；将整个步骤设为 `projector_output` 会明确报错。已有 DINO 帧筛选可以单独移动其**应用位置**：

```yaml
operation:
  targets:
    dino_frame_select_sva_fusion:
      frame_selection_site: post_output_projector
```

| `frame_selection_site` | 实际执行顺序 |
| --- | --- |
| `pre_sva`（默认） | 筛选帧 → 对选中帧执行 SVA → 输出投影 |
| `pre_output_projector` | 对全部帧执行 SVA → 筛选帧 → 输出投影 |
| `post_output_projector` | 对全部帧执行 SVA 和输出投影 → 筛选帧 |

三种位置使用同一份原 DINO 评分和所选帧索引；SVA 按帧独立，所选帧的数学规则与每帧 144 个 query 预算保持。后移筛选会增加参与 SVA 或输出投影的帧数，运行记录反映实际处理数量。`target_site` 可省略，或用 `projector_input` 明确 SVA 仍位于其输出投影前。Manifest 分别记录帧筛选和 SVA 的实际位置，改变帧筛选位置也会使历史证据不再精确适用于新配置。

### METok 与 PruneVid 的 Projector 和 decoder 步骤

METok 的 `event_pool`、PruneVid 的 `temporal_spatial_reduce` 均可设为 `projector_input` 或 `projector_output`，默认后置。分段、聚类和重要事件判断保留完整 projected 特征来源；前置按原计划处理 raw 值，再经过原投影器。METok 重放原双线性池化，PruneVid 重放原分组均值。前置仍保留完整投影参考的计算成本，Qwen3 使用同一计划处理主分支与 DeepStack。

LLM 中的评分层沿用已有 `config` 字段，实际应用位置由对应 `operation.targets` 单独配置：

| 方法 | 评分层字段及默认 | 应用步骤及默认 |
| --- | --- | --- |
| METok | `pruning_layers: [3, 10, 19]` 的前两项决定 Attention 评分层；第三项决定最后一步的默认位置 | `prefill_attention_prune`：在第 3、10、19 层入口依次应用三步 |
| PruneVid | `vtp_selected_layer: 10` | `vtp_prefill_prune`：第 10 层完整输出后应用 |

这两个已有层字段可以修改，其他预算和规则配置仍保持原规范。METok 保留三个递增非负层号及三步规则；最后一步删除全部剩余视觉 Token，不取 Attention 分数。显式覆盖其应用位置后，第三个名义层号无需作为模型中的评分层存在。

```yaml
config:
  pruning_layers: [0, 2, 19]
operation:
  targets:
    event_pool:
      target_site: projector_input
    prefill_attention_prune:
      target_site: llm_decoder_layer_post_attention_video_tokens
      layers: [0, 2, 4]
```

上例在第 0、2 层入口获取原 METok Attention 分数，在第 0、2、4 层的 Attention **残差后、FFN 前**依次应用三步；最后一步不等待第 19 层。实际模型须包含这三个应用层。

完整层边界使用 `llm_decoder_layer_output_video_tokens`，默认 `layer_edge: after`。例如 PruneVid 配置 `vtp_selected_layer: 0`，并将 `vtp_prefill_prune` 设为该 site、`layers: [0]`、`layer_edge: before`，表示在首层输入执行。`before k` 与上一层完整输出统一为层间边界；`before 0` 是起始输入，`before 层数` 是末端边界。层内 site 不接受 `layer_edge`。

有评分的步骤不能早于其评分层入口，METok 三步应用仍保持顺序。延后应用保留原来源分数，前一步删除 Token 后仅按保留索引更新已捕获分数，不重新归一化。Qwen3 会同步处理尚未注入的 DeepStack 分支；层内剪枝也会裁剪当前 Attention 已写入的 cache。运行记录的 `application_position` 表示实际边界，旧 `boundary_layer` / `selected_layer` 字段保留规则编号或评分来源含义。改变评分层或实际位置都按探索配置记录，默认与等价边界继续复用旧证据。
