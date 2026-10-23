# Qwen2.5-Omni 八方法算法边界与自动检查参考

> 类型：Reference
> 适用范围：八种 AudioLLM 方法的算法语义与 O/E/T 自动检查
> 当前状态：Active

本文是**查判据的参考卡**，不是第二份操作手册。所有命令、顺序和证据目录只以 [八方法开发验证 Runbook](eight_methods_runbook.zh.md) 为准。

面向的读者是了解多模态压缩、但不熟悉本仓库内部实现的研究人员。正常验收不要求逐行读代码，也不要求手算完整 DPP、merge 或 attention 矩阵。

## 1. 一分钟速查

- [方法卡 1：Segmentwise Top-K](#p1)
- [方法卡 2：A-ToME](#p2)
- [方法卡 3：LTBM](#p3)
- [方法卡 4：SA-MAP](#p4)
- [方法卡 5：Encoder Attention TopK](#p5)
- [方法卡 6：Progressive Pruning](#p6)
- [方法卡 7：Audio-Pyramid](#p7)
- [方法卡 8：SPADE Operation-Only](#p8)
- [自动门禁 O/E/T](#automatic-gates)
- [训练、质量和效率的附加验收](#optional-gates)

开发验证只有三项：

| 门禁 | 研究人员只需要确认什么 | 通过标记 |
| --- | --- | --- |
| O：Oracle | 小型确定性输入是否得到预期索引、分组、长度和边界 | `PASS-O` |
| E：真实评估路径 | 真实 Qwen2.5-Omni checkpoint 能生成，Token/结构确实变化且不变量成立 | `PASS-E` |
| T：训练值路径 | loss 有限、音频值路径梯度非零；Operation 方法还要能导出重载 | `PASS-T` |

本次方法的 Oracle、真实 checkpoint 推理和 backward 均通过时，可以记录开发验证通过。
按改动独立选择检查；脚本不维护前置关卡或 PASS 状态文件。

开发验证通过 只证明“当前实现的评估/训练值路径可用”，不等于论文指标复现、完整训练收敛或性能优于 baseline。

## 2. 实现范围标签与阻断规则

| 标签 | 含义 |
| --- | --- |
| `PASS-ADAPTED` | 论文核心思想已迁移到 Qwen2.5-Omni，但存在已登记的 backbone、预算或生命周期适配 |
| `PASS-VARIANT` | 受相关工作启发的项目方法；不能写成对某篇论文的直接复现 |
| `PASS-PARTIAL` | 只实现论文中明确划定的子集 |
| `BLOCKED-O/E/T` | 对应自动检查失败；该方法停止在此处 |

以下情况必须阻断：

- 位置、对象或压缩方向错了；
- 实际预算与配置/验收卡不一致且没有解释；
- 跨音频、跨 chunk 或跨样本错误配对；
- 产生 NaN/Inf、长度元数据失配、时间顺序破坏或 decode/cache 失效；
- 将 `VARIANT`/`PARTIAL` 写成论文完整复现；
- 只有进程退出，没有对应门禁要求的断言和证据。

论文没有定义的取整、最小 Token 数、Qwen chunk、M-RoPE 和训练桥接规则，可以记为 adaptation；这类已知差异本身不导致失败。

这些标签仅帮助理解方法范围，不是开发验证中的人工签字步骤。

## 3. 规范来源地图

| 方法 | 规范来源 | 身份 |
| --- | --- | --- |
| Segmentwise Top-K | 已验证的项目基线契约；[Projector-Level 架构与 Debug 手册](debugging/projector_level.zh.md) | `ADAPTED` |
| A-ToME | 已验证的项目基线契约；[Encoder-Level 架构与 Debug 手册](debugging/encoder_level.zh.md) | `ADAPTED` |
| LTBM | *Locality Matters for Training-Free Audio Token Compression in ALMs*, arXiv:2605.25179v1 | `ADAPTED` |
| SA-MAP | *Similarity-Attention Boosted Token Compression via Merging-and-Pruning for Efficient Speech LLMs*, Preprints.org v1, 2026-06-09 | `ADAPTED` |
| Encoder TopK | Lee & Lee, *Token Pruning in Audio Transformers*, arXiv:2504.01690v2 | `ADAPTED` |
| Progressive | Chen et al., *Efficient Speech-to-Text Translation: Progressive Pruning for Accelerated Speech Pre-trained Model*, ICME 2024 | `ADAPTED` |
| Audio-Pyramid | 本项目 LLM-Level Token 契约；SpeechPrune 只作为相关依据；[LLM-Token 架构与 Debug 手册](debugging/llm_token_audio_pyramid.zh.md) | `VARIANT` |
| SPADE Operation-Only | *SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS*, §2.1 的结构删层子集；[LLM-Operation 架构与 Debug 手册](debugging/llm_operation_spade.zh.md) | `PARTIAL` |

PDF 页码均指 PDF 阅读器显示页。SA-MAP PDF 有一页封面，因此 PDF 页比稿件页大 1。

<a id="p1"></a>
## 方法卡 1. Segmentwise Top-K

**一句话目标：** 每个音频在 Audio Encoder 完成后、projector 之前，根据最后一层 encoder attention，在时间分段内保留高分 Token。

必须成立：

1. 精确位置是 `avg_pooler → ln_post` 后、`audio_tower.proj` 前；Encoder 的 32 层输入长度不被提前截断。
2. 每个音频独立处理；最后一层 encoder incoming attention 对齐到 pooler 后 Token，再做 segment allocation 和 Top-K。
3. 输出恢复时间顺序，placeholder、M-RoPE 和 projector 输入长度同步；Identity 路径必须与 baseline 等价。
4. 短音频、多音频和非整除预算不能跨音频借 Token，也不能保留零个 Token。

允许的迁移差异：平台使用 `floor(N × keep_ratio)`、最小保留数和 Qwen chunk/pooler 对齐规则；这些是项目契约，不宣称来自某篇论文。

代码入口：`mllm_comp/compressors/projector_level/segmentwise_topk.py`。自动证据：`test_segmentwise_topk.py`、Projector CPU/GPU smoke、compressed-training backward。

通过写法：开发验证通过；算法范围为项目既有 `ADAPTED` 基线契约。

<a id="p2"></a>
## 方法卡 2. A-ToME

**一句话目标：** 在指定 Audio Encoder 层的 MHSA residual 后、FFN 前，对每个当前 chunk 的不相交相邻 Token 对做相似度选择和均值合并。

必须成立：

1. 默认 active layers 是零基 `[2,5,8,11,14,17,20,23,26,29]`，压缩发生在 `post_mhsa_pre_ffn`。
2. 候选只能是 `(0,1),(2,3),...`；按 normalized key cosine 选最高分 pair，奇数尾 Token 原样保留。
3. 被选 value 取均值并保持梯度；score/pair 选择不进入梯度。
4. 每个 chunk 独立预算、逐层重建长度边界，输出时间有序，pooler/placeholder/M-RoPE 长度一致。

允许的迁移差异：默认层表、每 chunk 取整、Qwen 变长 FlashAttention controller 和 pooler 修复属于 Qwen 适配。

代码入口：`mllm_comp/compressors/encoder_level/adjacent_token_merging.py`。自动证据：`test_adjacent_token_merging.py`、Encoder CPU/GPU smoke、compressed-training backward。

通过写法：开发验证通过；算法范围为项目既有 `ADAPTED` 基线契约。

<a id="p3"></a>
## 方法卡 3. LTBM

**论文锚点：** PDF p.2 Fig.1、p.3 §3.1–3.2 Eq.(1)–(2)、p.4 Eq.(3)–(6)、p.5 Eq.(7)。论文方法位于 average pooler 后、projector 前，做局部二部 cosine matching，并保持时间顺序。

必须成立：

1. 位置是 pre-projector；按 source/destination 二部集合在局部窗口内寻找 cosine 相似 Token。
2. 合并达到精确平台 keep budget，未合并 Token 保留，结果恢复时间顺序。
3. 多个 source 可在同一轮进入同一 destination；该 destination 与本轮 source 等权求均值。
4. `w=8` 是论文默认窗口；`keep_ratio=0.5` 是论文实验点之一，不是质量保证。

必须登记的 `ADAPTED` 差异：

- 论文目标长度写 `round(ρL)`，平台使用 `floor(ρL)`；所以 `25×0.5` 为 `12`。
- 实现已按 S/D 子序列索引计算局部窗口，并在每轮先为所有 source 找最佳 destination，
  再按相似度排序后批量合并；进入下一轮时才重新划分当前序列。
- 同一 destination 及本轮分配给它的 source 按 Eq.(5) 普通均值合并；下一轮把该结果视为一个
  当前 Token，不恢复 original-mass 权重。
- 论文研究 training-free inference；T/Swift 只证明平台训练值路径兼容。

代码入口：`mllm_comp/compressors/projector_level/local_temporal_bipartite_merging.py`。自动证据：`test_projector_extension_methods.py -k ltbm`、extension GPU smoke、compressed-training backward。

通过写法：开发验证通过；算法范围为 `ADAPTED`，不能写成逐数值复现。

<a id="p4"></a>
## 方法卡 4. SA-MAP

**论文锚点：** PDF p.5 §3.1、p.6 Eq.(2)–(4)、p.7 Eq.(5)–(7) 与 Fig.2、p.8 Eq.(8)–(9)、p.14 Algorithm A1。顺序必须是 G-SAM merging 后再 ADPruner。

必须成立：

1. 位置是 projector 后、LLM 前；相邻相似组先 merge，再按剩余预算 prune。
2. 使用 encoder 倒数第二层 attention；merge value 是 attention-weighted mean。
3. pruning 同时利用相似性、多样性和 attention importance，并通过 greedy MAP/DPP 风格选择得到精确预算。
4. 当前 `threshold=0.80` 只可表述为论文 Qwen2-Audio speech-understanding、50% compression 的迁移初值。

必须登记的 `ADAPTED` 差异：

- grouping 已使用“下一个 Token 与组内每个 Token 的 cosine 再平均”；threshold grouping 独立于
  最终预算。若 grouping 后少于 `m`，实现会拒绝运行并要求提高 threshold，而不会截断 merge。
- merge importance 按 Eq.(4) 对 head 取 max；prune importance 独立使用 head-mean attention 的
  diagonal。两者都显式映射经过 Qwen 的二 Token pooler，这一桥接属于 Qwen 适配。
- `attention_power=1.0` 时 DPP kernel 为 Eq.(8) 的 `diag(A) L diag(A)`；可配置的 power 与
  `dpp_jitter=1e-6` 是平台数值扩展，实验必须记录实际值。
- DPP 使用独立掩码持续排除已选候选；归一化、核矩阵和递推均使用 FP64，并在选取期间关闭
  autocast，以处理重复特征下的数值误差。输出仍从原始合并特征取值，保留原 dtype 和梯度。
  此修复保留原核矩阵公式、分组阈值、预算及 jitter，不增加 TopK 补齐或对角正则化；
  FP64 的额外计算和内存成本需要在目标 GPU 上测量。
- 当前只覆盖 training-free path，不包含论文另加 Q/K linear layers 的两阶段 tuning。

代码入口：`mllm_comp/compressors/projector_level/similarity_attention_merge_prune.py`。自动证据：`test_projector_extension_methods.py -k sa_map`、`test_sa_map_dpp.py`（重复特征四档预算、独立行列式参考、BF16 autocast 与梯度）、extension GPU smoke、compressed-training backward。

通过写法：开发验证通过；算法范围为 `ADAPTED`，不能声称复现 `SA-MAP(tuning)`。

<a id="p5"></a>
## 方法卡 5. Encoder Attention TopK

**论文锚点：** PDF p.2 §3 Eq.(1)–(2)、§3.1；p.3 Fig.2。论文在第 4、7、10 个 ViT block 的 MHSA residual 后、MLP 前剪 Token，即代码零基 `[3,6,9]`。

必须成立：

1. 插入层 `[3,6,9]`，位置 `audio_encoder_layer_post_mhsa_pre_ffn`。
2. Qwen 使用论文 AudioMAE 式 received-attention：跨 head、跨 query 聚合每个 Token 收到的 attention。
3. 三个 active block 对当前长度重复使用同一 keep ratio，再送入后续 FFN/blocks。
4. 默认 `keep_ratio=0.90` 是论文扫描过的点；论文扫描 `0.4–0.9`，没有唯一默认值。

必须登记的 `ADAPTED` 差异：省略 `/N` 只差同 chunk 正常数、Top-K 排名等价；`floor`、最小 2 Token、per-chunk 选择、时间重排和 stride-2 pooler 修复均由论文留白后的 Qwen 适配。

低门槛长度核对：`50 → 45 → 40 → 36` 个 encoder Token，pooler 后为 `18` 个 audio Token。

代码入口：`mllm_comp/compressors/encoder_level/attention_topk.py`。自动证据：`test_encoder_extension_methods.py -k attention_topk`、extension GPU smoke、compressed-training backward。

通过写法：开发验证通过；算法范围为 `ADAPTED`。论文的 AudioMAE/AST 分类 MAC 与精度结果不是 Qwen 的硬门禁。

<a id="p6"></a>
## 方法卡 6. Progressive Pruning

**论文锚点：** PDF p.1 Fig.1(b)、p.2 Fig.2、p.3 §III-A Eq.(1)–(2)、p.3–4 §III-B Eq.(3)、p.4 Table I。论文在完整 block 后，按当前长度逐层删除低 received-attention Token；支持 fixed 和线性递增 rate。

必须成立：

1. 压缩位于 `audio_encoder_layer_output_post_block`，当前 block 完整执行后才影响下一层。
2. score 使用 incoming/received attention；fixed schedule 恒定，linear schedule 在首末 active stage 间做端点包含插值。
3. 正式 eval/train 配置使用 `[6,12,18]`，rate 为 `[0.05,0.125,0.20]`。
4. 低门槛长度核对：`50 → 47 → 41 → 32` 个 encoder Token，pooler 后 `25 → 16`。

必须登记的 `ADAPTED` 差异：

- 论文主实验是 12-layer HuBERT、initial pruning layer `6`，随后 selected layers 逐层剪；`[6,12,18]` 是本项目对 Qwen 深度和目标预算的三阶段映射，不是论文层号直译。
- Eq.(2) 下标与正文“received attention”语义有歧义，项目按正文使用列方向 incoming attention。
- Eq.(3) 未明确首尾 ordinal；项目采用 `position/(m-1)` 的端点包含插值。
- 论文完整训练包含无剪枝微调后再 retraining；T/Swift 只证明当前训练框架兼容。

代码入口：`mllm_comp/compressors/encoder_level/progressive_attention_pruning.py`。自动证据：`test_encoder_extension_methods.py -k progressive`、extension GPU smoke、compressed-training backward。

通过写法：开发验证通过；算法范围为 `ADAPTED`。

<a id="p7"></a>
## 方法卡 7. Audio-Pyramid

**身份边界：** 这是项目的 AudioLLM `VARIANT`。SpeechPrune PDF p.3 Fig.1、p.4 Eq.(4)–(13) 只支持“LLM 前 query/attention-aware 音频 Token pruning”这一相关动机；SpeechPrune 本身是 LLM 输入前的两阶段方法，不在多个 decoder layer 输出处逐级压缩。

必须成立：

1. 在配置的 decoder layers 后，只选择 audio observation Token；system/instruction/response/special Token 不被误删。
2. score 是 instruction query 对 audio key 的 received attention，按每阶段 keep ratio 形成逐级金字塔。
3. 只在 prefill 压缩；decode 绕过再次压缩，KV cache、position ids、M-RoPE 和 attention mask 保持一致。
4. eval generate、gradient-checkpoint backward 和 Prefill/Decode 计数均有自动断言。

项目默认层 `[8,17,26]` 和每阶段 `0.5` 是 Audio-Pyramid 自身实验配置，不来自 SpeechPrune。不要使用“SpeechPrune 复现”措辞。

代码入口：`mllm_comp/compressors/llm_level/audio_pyramid.py`。自动证据：`test_llm_level_compression.py`、LLM CPU/GPU eval-train smoke。

通过写法：开发验证通过；算法范围为项目 `VARIANT`。

<a id="p8"></a>
## 方法卡 8. SPADE Operation-Only

**论文锚点：** SPADE PDF p.2 §2.1 Eq.(1) 用 WLI 评估删层影响并选择低贡献层；p.3 §2.2 Eq.(2) 是 teacher 多级蒸馏。本项目只实现“已给定层索引后的结构删除”。

必须成立：

1. 在 prepare 阶段按显式或预计算 layer indices 删除完整 Thinker LLM layers，不能删光所有层。
2. 删除后 layer/module 索引连续，`num_hidden_layers`、state dict 和运行配置同步。
3. eval generate 与 backward 可用；导出 Transformers checkpoint、operation manifest、hash 后可以 reload。
4. 相同删除计划可复现，错误索引和不支持的 operation 字段会显式拒绝。

明确不在范围内：WLI 分析与自动选层、teacher recovery、CE/logit/latent/attention/embedding distillation、dynamic teacher target。Head/FFN/Talker pruning 是平台未来可扩展对象，也不是当前 SPADE 子集。

代码入口：`mllm_comp/compressors/llm_level/structured_layer_pruning.py` 与 `mllm_comp/export/operation_export.py`。自动证据：`test_operation_compression.py`、LLM CPU/GPU export-reload smoke。

通过写法：开发验证通过；算法范围为 `PARTIAL（Operation-Only）`。不能把 SPADE 完整系统的质量结果归因于当前实现。

<a id="automatic-gates"></a>
## 4. 自动检查 O/E/T

### PASS-O：小型 Oracle

每个方法必须同时有：

- 纯算法/配置单测：预算、索引、分组、值变换、非法参数；
- Qwen CPU integration：真实模块边界、长度修复、生命周期或导出契约。

只有 unit test 通过而 CPU integration 缺失，不能记 `PASS-O`。

### PASS-E：真实 checkpoint 评估路径

必须使用同一次同步后的代码、真实 Qwen2.5-Omni checkpoint 和单个可见 GPU。自动检查至少断言：

- generate 成功且输出有限；
- 方法实际激活，Token 数或结构按配置变化；
- audio unit/chunk 边界、时间顺序、placeholder/M-RoPE/cache 等适用不变量成立；
- Identity 方法适用时与 baseline 等价。

这不是 MMAU 质量门禁。

### PASS-T：训练值路径

Token 方法至少需要 finite loss 和非零、有限的音频值路径梯度。离散选择本身可以不求导，但被保留/合并 value 的链路不能断。

SPADE 还必须验证结构删层后的 backward、checkpoint export、hash manifest 和 reload。`PASS-T` 不代表训练收敛。

<a id="optional-gates"></a>
## 5. 可选检查：Plan、Swift、质量与效率

这些不属于 开发验证通过，避免研究人员为了验证算法实现先准备完整实验环境。

| 标记 | 证明什么 | 不证明什么 |
| --- | --- | --- |
| `PLAN-OK` | train YAML 能生成原生 Swift launch plan | 真正执行过训练 |
| `SWIFT-OK` | 真实 Swift 单步产生 checkpoint 与 Manifest | 收敛或最终质量 |
| `QUALITY-DONE` | 固定协议下的任务指标文件非空 | 优于 baseline 或达到论文指标 |
| `PROFILE-DONE` | 独立 profile run 有非空 timing/memory 记录 | 统计显著或跨机器可比 |

质量与效率必须分开运行。所有 baseline/compressed run 由 runner 强制使用相同 `max_new_tokens`（默认 16）和 `use_cache=true`；profile 不能代替 default-mode 质量结果。

性能结论必须在同一次代码同步、checkpoint、任务、数据版本、GPU、batch size 和 generation protocol 下比较 baseline。没有预先设定阈值时，只能写“实验已完成”，不能写“性能通过”。

## 6. 最终汇总规则

八种方法分别结论，不以“7/8 通过”掩盖失败方法：

| 方法 | 开发验证通过后的范围说明 |
| --- | --- |
| Segmentwise、A-ToME、LTBM、SA-MAP、Encoder TopK、Progressive | `PASS-ADAPTED` |
| Audio-Pyramid | `PASS-VARIANT` |
| SPADE Operation-Only | `PASS-PARTIAL` |

任何核心门禁缺失时写对应 `BLOCKED-*`，不要降级成模糊的“基本可用”。
