# 各方法的自定义规则

本文是现有方法的规则目录。配置形式和函数加载方式见 [README](README.md)。规则名写在 `operation.rules`，未提供的步骤使用原实现。压缩位置仍由已有 `operation.targets` 配置决定；规则配置不增加该方法的基座、输入上下文或位置支持。

下面的 `N` 是当前步骤的候选 Token 数，`K` 是该步骤已解析的预算；不一定等于整个 stage 的最终预算。索引属于 request 明确提供的候选空间，不能把视觉局部索引与完整 LLM 序列位置混用。完整 request 的构造及可执行例子链接到对应实现和测试，避免另维护一套脱离代码的类型定义。 配对、分组和离散选择返回保存的决定，实际值聚合按同一决定在原执行或重算路径中调用；不能依赖调用次数维护跨请求状态。固定预算步骤必须保留其已解析的数量，允许动态分段数的步骤在表中单独说明。

## 图像

本节列出的规则覆盖 `llava_onevision`、`qwen25_vl`、`qwen3_vl` 上的原 25 个图像方法、26 个 variant、75 个 binding，不是当前全部图像能力的数量。ToMe 使用 `tome.image.encoder_dense_boundary`；SAINT 在 Qwen2.5-VL 使用 `saint.image.encoder_progressive_prune`，其余两个基座使用 `saint.image.packed_feature_prune`；其他 variant 为下表名称加 `.image`。

新增 [14 组对比方法](../../docs/support/imagellm/comparison_methods.zh.md)不自动开放本表的
`operation.rules` 或位置探索合同。Bifocal 同样不继承这些接口，其专用配置和支持状态见
[Bifocal 接入](../../docs/support/imagellm/status.zh.md#bifocal-接入)。MWOP（Mask 形式）和图像分支模块替换
属于独立 Operation 接口，也不计入本节规则矩阵。

| 方法 | 规则名 | 输入、输出与保留的语义 |
| --- | --- | --- |
| `fastv` | `scorer`, `selector` | 实际 hidden、视觉索引和原 Q/K 来源上下文；返回 `[N]` 分数或恰好 K 个视觉局部索引，必须保留保护项。详细合同见 README。 |
| `gap` | `scorer`, `selector` | 原优先级 `[N]`；选择 K 个局部索引，保护项计入预算。 |
| `divprune` | `scorer`, `selector` | scorer 返回 `[N,N]` 距离矩阵；原 max-min 过程使用它，selector 可直接替换最终 K 个索引。 |
| `vispruner` | `scorer`, `selector` | 视觉特征及原 Attention 重要性；返回 `[N]` 分数或 K 个索引，保留原 important/distinctive 预算。 |
| `mmtok` | `scorer`, `selector` | 原视觉、投影和文本参考；返回 `[text_count+N,N]` 覆盖矩阵或 K 个索引。参考的来源和 Projector 调用保持原路径。 |
| `zoo_prune` | `scorer`, `selector` | 原 Projector 输入与代理敏感度；返回 `[N]` 敏感度或 K 个索引。 |
| `saint` | `scorer`, `selector` | 当前轮交替 A 集的冗余向量；selector 返回当前 pass 的保留索引，原迭代次数、保护和总体预算仍有效。 |
| `pyramid_drop` | `source_scorer`, `scorer`, `selector` | source 接真实捕获 Q/K/mask；scorer 接应用边界值及来源评分；selector 接最终候选与 K。 |
| `d2pruner` | `source_scorer`, `scorer`, `selector` | 同上；执行顺序及各阶段原预算保持不变。 |
| `btp` | `source_scorer`, `scorer`, `selector` | 同上；原最终清空步骤不调用自定义评分/选择。 |
| `halfv` | `source_scorer`, `scorer`, `selector` | 同上；沿用原层位置和 Token 保留规则。 |
| `dart` | `scorer`, `diversity_scorer`, `selector` | scorer 接完整序列的 key-norm 优先级；diversity 接当前 pivot 与候选相似度；selector 保留原配置对应的有效预算。 |
| `feather` | `scorer`, `selector` | scorer 接原捕获点不带 RoPE 的 Q/K/logits；selector 的 K 分别是第一阶段 union 数或第二阶段最终数。 |
| `ivc_prune` | `coordinate_scorer`, `semantic_scorer`, `selector` | 坐标评分返回原 `[2,N]` cosine/sine 归约；语义评分区分 seed/foreground，选择沿用 strict/non-strict 数量语义。 |
| `finepruner` | `grouper`, `iac_selector`, `scorer`, `head_selector`, `selector` | IAC grouper 返回每 Token 整数桶标签，iac_selector 每桶一个代表；ARP scorer 返回 `[H,N]`，head_selector 选原预算的头，selector 选最终视觉项。 |
| `transprune` | `transition_scorer`, `scorer`, `selector` | transition 分别接 Attention、FFN 前后真实值；scorer 接对齐后的 TTV 历史及 IGA。mid 压缩后仍沿用原索引对齐后续 FFN。 |
| `sgl` | `scorer`, `selector` | scorer 接完整实际 guide attention 步骤；selector 在应用边界执行。guide 的加载、生成及来源对齐保持原路径。 |
| `visionzip` | `scorer`, `dominant_selector`, `contextual_selector`, `grouper`, `aggregator` | 替换显著项、上下文项及分组；值聚合仅处理上下文项，显著项和保护项直接保留。 |
| `prumerge` | `scorer`, `selector`, `grouper`, `aggregator` | 原中心及带权邻域的选择/聚合；邻域允许重叠，不能假定是无交集 partition。 |
| `aim` | `scorer`, `grouper`, `aggregator` | 逐轮 A/B 相似度、固定合并数的配对、实际值聚合；默认是当前节点等权均值，不是按最终原始成员重新求均值。 |
| `tome` | `encoder_scorer`, `encoder_grouper`, `encoder_aggregator`, `output_scorer`, `output_grouper`, `output_aggregator` | Encoder 与输出压缩两个原步骤分别配置；保留质量权重、保护项和 dense restore。没有无作用域的通用别名。 |
| `nuwa` | `stage1_scorer`, `regional_selector`, `stage1_selector`, `aggregation_weights`, `stage2_scorer`, `stage2_selector`, `stage2_source_scorer` | 第一阶段区域候选、固定预算和 `[K,N]` 聚合权重；权重允许重叠并冻结用于分支。第二阶段在实际 hidden 上评分选择；source 仅适用于原 Attention 来源模式。 |
| `restore` | `pristine_scorer`, `pristine_selector`, `anchor_scorer`, `anchor_selector`, `grouper`, `aggregator`, `calibration_bias` | 保留 pristine；残余分组须完整覆盖且含对应 anchor；校准返回原 `[B,1,Q,K]` bias。原代表位置与质量继续驱动布局。 |
| `pact` | `source_scorer`, `scorer`, `selector`, `grouper`, `aggregator` | 原 Attention 来源及应用评分/选择；grouper 返回 `{centers, labels}`，保持原 center_count，中心必须属于自身组。原 include_pruned 重分配分支允许非中心项为 −1，其余情况下 kept 项须完整分组。 |
| `sparsevlm` | `rater_scorer`, `rater_selector`, `source_scorer`, `scorer`, `selector`, `recycle_selector`, `cluster_scorer`, `cluster_selector`, `grouper`, `aggregator` | 文本 rater、来源评分、Token 选择、回收及聚类各在原步骤替换；grouper 返回回收集局部标签，保持 cluster_count 与中心自属。 |

PACT/SparseVLM 的 aggregator 仅接多成员组，单例保持原值，质量仍是实际成员数。主路径保存决定，重算与对应分支使用已保存的组，不重新评分或聚类。

评分与选择合同见 [图像规则检查函数](../../mllm_comp/compressors/image/_rules.py)。第一组的实际 request 与样例在 [选择规则测试](../../tests/unit/test_image_selection_rules.py)，来源和复合步骤在 [decoder 规则测试](../../tests/unit/test_image_decoder_rules.py)。上述六个合并/复合方法的输入差异必须保留，不能套用一个通用无权均值。

AIM/ToMe grouper 接 `values`（detach）、当前 source/destination 候选、`scores` 和 `merge_count`，返回固定数量的 `(source,destination)`。source 唯一，destination 可重复，必须属于当前轮合法分区且排除保护项。aggregator 接实际 `values`，顺序为 destination 后接已保存顺序的 sources，并提供 `indices`、`destination`、`token_sizes`、`default_values`。返回同设备、dtype、宽度的有限向量；单例与未合并项旁路。AIM 还提供 `round_index`，ToMe 提供 `batch_index`。可执行例子见 [AIM/ToMe 测试](../../tests/unit/test_image_aim_tome_custom_rules.py)。

Nüwa 的区域选择返回区域候选中的全局索引，最终 stage1 选择也须来自候选集；`aggregation_weights` 接原 similarity/proximity/default_weights 与选中项，返回 `[K,N]`。RESTORE 的 aggregator 接当前组实际 `values` 与全局 `source_indices`；重放保留 pristine，并对 anchor 组独立调用相同值函数。它可以是非线性函数，不要求等价于固定均值矩阵。见 [复合规则测试](../../tests/unit/test_image_compound_custom_rules.py)。

## 音频

以下 14 个 variant 均为方法名加 `.audio`，对应现有 `qwen25_omni` binding。支持位置沿用各自已有声明，共 14 个 binding。

| 方法 | 规则名 | 输入、输出与限制 |
| --- | --- | --- |
| `segmentwise_topk_audio` | `scorer`, `selector` | 接实际 values、原 scores、keep_count、item_index；selector 还接 segment_boundaries/budgets，须满足每段原预算。 |
| `uniform_sampling_audio` | `selector` | 接 values、default_indices、compression_factor、keep_count，返回 K 个索引；没有聚合步骤。 |
| `encoder_attention_topk_audio` | `scorer`, `selector` | 每 chunk 的 values、scores、keep_count、chunk_index；返回 N 分数或 K 索引，同策略复用于已支持的其他位置。 |
| `progressive_attention_pruning_audio` | `scorer`, `selector` | 同上，保留原逐层计划及长度更新。 |
| `adjacent_token_merging_audio` | `scorer`, `grouper`, `aggregator` | scorer 接 keys 与成对分数，返回 `floor(N/2)`；grouper 返回固定合并数的合法偶数 pair starts，保持不相交相邻配对。 |
| `uniform_average_pooling_audio` | `grouper`, `aggregator` | grouper 接原 groups、compression_factor 和 K，返回 K 个非空连续组，按时间覆盖前缀；允许原尾部丢弃语义。 |
| `unsupervised_segmentation_audio` | `scorer`, `grouper`, `aggregator` | scorer 返回 N−1 个相邻差异；grouper 返回完整连续 partition，组数可变。自定义 scorer/grouper 的 Projector 预规划保存实际组；checkpoint 按首次实际决策重放。 |
| `local_temporal_bipartite_merging_audio` | `grouper`, `aggregator` | 保留交替分区、局部窗口及每轮合并数；详细合同见 README。 |
| `similarity_attention_merge_prune_audio` | `scorer`, `grouper`, `aggregator`, `selector` | scorer 返回 merge/prune 两向量；完整连续分组至少 K 组，再选择 K 个组。原归一 merge 权重进入保存的计划。 |
| `fastadasp_weighted_token_merging` | `scorer`, `grouper`, `aggregator` | scorer 接实际 rotated-Key adjacent_similarity 与 merge_weights，返回同两向量；grouper 返回完整连续的 K 组。 |
| `audio_pyramid_attention_pruning` | `scorer`, `selector` | 真实 captured rotated Q/K、instruction/valid mask、audio_indices、absolute_positions；返回音频项分数和 K 个索引。 |
| `headrouter_dynamic_audio_pruning` | `scorer`, `selector` | scorer 接 values、scores、per_head_attention、profiles、centers、routing_sigma；selector 接 scores、K，保持原无 RoPE embedding 来源。 |
| `fastslm_hta_audio` | `transform` | 接 live values、K 和已注册 abstractor，返回 `[K,C]`；模块仍由原路径在优化器之前注册和加载，query-slot 布局固定。 |
| `structured_llm_layer_pruning` | `decision` | 接实际 model、semantic layers、num_layers、target_layers、remove_count，返回恰好指定数量的唯一合法层索引；原 prepare/applier 执行删层。 |

除 LTBM 的逐轮合同外，共享 audio aggregator 接 `values`（当前组、保留梯度）、`indices`、`group_index`、`weights`（已归一或 None），返回一个同 dtype/device/宽度的有限可微向量。配对/选择不求导，实际值函数在重放中保留梯度。方法特有输入及错误示例见 [音频规则测试](../../tests/unit/test_audio_custom_rules.py)，各执行位置、缓存和重算见 [原生执行测试](../../tests/integration/test_audio_custom_rules_cpu.py)。

## 视频

以下 9 个方法的 variant 均为方法名加 `.video`；均对应现有 `llava_onevision`、`qwen25_vl`、`qwen3_vl`，共 27 个 binding。视频和图像 VisionZip 的上下文规则名不同，应使用各自 variant 的名称。

| 方法 | 规则名 | 输入、输出与限制 |
| --- | --- | --- |
| `visionzip` | `scorer`, `dominant_selector`, `context_selector`, `grouper`, `aggregator` | 固定每帧显著/上下文预算；selector 返回当前帧局部候选索引，grouper 将非显著候选完整划分到各上下文 target，组须含对应 target。aggregator 接实际 values、group、target、assigned_indices，含显著单例。 |
| `dytok` | 上述五项，加 `frame_scorer`, `budget_allocator` | 复用视频 VisionZip 五项；frame_scorer 接实际 Token、原检测分数和 provider signals；budget_allocator 返回逐帧整数预算，保持原总预算及每帧容量。零预算帧旁路。 |
| `vqtoken` | `transform`, `grouper`, `aggregator` | transform 返回同 shape/dtype/device 的实际值；grouper 返回原固定 32 个非空完整组，索引属于当前视频；aggregator 返回每组一行。transform 和聚合在 DeepStack 实际值上重新调用。 |
| `fastvid` | `frame_grouper`, `saliency_scorer`, `saliency_selector`, `context_allocator`, `center_scorer`, `center_selector`, `grouper`, `aggregator` | frame_grouper 返回连续完整帧 partition，段数可变；context_allocator 保持总上下文预算和每帧容量。grouper 为每个目标帧划分段内候选，可有空组且不同目标帧可重叠。aggregator 接视频局部 target/assigned_indices，仅作用于上下文项；显著项直接保留。 |
| `metok` | `frame_grouper`, `frame_scorer`, `event_selector`, `keyframe_selector`, `pool_transform`, `prune_scorer`, `prune_selector` | frame_grouper 可改变连续事件数；event/keyframe selector 保持各自预算。pool_transform 接实际 `[H,W,D]`、output_shape、default_transform，返回 `[H'W',D]`。prune_scorer 使用原固定 LLM Q/K 来源；prune_selector 返回事件局部索引，接 stage_index/event_index，包含原最终 K=0 步骤。 |
| `prunevid` | `frame_grouper`, `static_selector`, `token_grouper`, `aggregator`, `vtp_scorer`, `vtp_selector` | frame_grouper 返回连续完整帧 partition，段数可变；static_selector 选帧内空间索引。token_grouper 返回当前候选局部索引组，保持原 `max(1,floor(N*cluster_ratio))` 数量，N≤14 时旁路。vtp_scorer 使用原真实 Q/K；vtp_selector 从所给静态/动态候选中返回局部视觉索引并保留各自预算。 |
| `dycoke` | `ttm_scorer`, `ttm_selector`, `decode_scorer`, `decode_selector`, `decode_decision` | TTM scorer/selector 接原相邻帧对，保留每对原数量及时间顺序。decode_decision 返回是否复用历史选择：True 复用、False 重选，复用须已有历史。decode_selector 返回绝对 visual cache 位置；非视觉 view 和物理完整 cache 由原路径维护。 |
| `longvu` | `frame_scorer`, `frame_selector`, `fusion_transform` | frame_selector 返回当前视频帧索引，保留 min/max 帧数限制；fusion_transform 接原注册模块、raw/DINO、选中帧、位置及 default_transform，返回原选中帧×144×输出宽度。原 Q3 分支仍用独立结构均值，不把自定义 SVA 应用于该分支。 |
| `llamavid` | `query_transform`, `context_scorer`, `context_aggregator`, `content_aggregator` | query_transform 与 context_scorer 返回原 query hidden / QK logits 形状并保留梯度。两 aggregator 接实际 values、weights、kind、branch，在所选自有 Projector 前后返回一行。Q3 先按原 raw→projected 规则映射参考权重，再在 branch 实际值上重新聚合；不要求非线性规则等价于固定权重矩阵。 |

视频 aggregator/pool/transform 返回同输入 dtype/device、规定 shape 的有限 Tensor；实际值函数必须保留梯度。硬分组、索引及用于离散选择的 scorer 不求导；LLaMA-VID 的 query/context logits 属于可微值路径。索引始终对应该次 request 的 values 或 candidates，不能将完整视频索引直接用于帧局部 request。完整函数 request 和独立输出公式见 [视频规则测试](../../tests/unit/test_video_custom_rules.py)，全部基座的正式配置、梯度、DeepStack、checkpoint 和 decode view 例子见 [视频原生执行测试](../../tests/integration/test_video_custom_rules_cpu.py)。Projector 前的参考特征仅用于计划，值函数作用于实际输入，不能用参考值替换实际值。


## 模块替换与证据

Qwen3-VL 的两个模块插件及九个策略均使用 `factory`，输入、模块注册和恢复合同见 [README](README.md#qwen3-vl-模块替换)，执行例子见 [模块测试](../../tests/integration/test_qwen3_module_custom_rules_cpu.py)。结构删层和 KV 视图使用上表的实际 decision/selector，不额外制造 scorer、grouper 或 aggregator。

这些规则覆盖的是已实现的执行能力。改变规则或位置的实验保留历史证据引用，但标记原配方证据不适用；它不会自动获得原配方的 Active、质量或训练结论。上述小模型正确性验证也不是重新收集真实 checkpoint 的支持证据。
