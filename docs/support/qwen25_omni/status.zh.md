# Qwen2.5-Omni 训练、压缩与分析验证状态

> 类型：Status
> 适用范围：Qwen2.5-Omni baseline、训练、AudioLLM v2 压缩、评测与 Thinker 分析能力
> 最后更新：2026-10-03（同步音频实验入口；既有真实模型证据未全量重跑）

本文汇总 Qwen2.5-Omni 基线与 active 方法的训练、压缩、评测和分析结果；候选方法的
逐项证据维护在[候选方法状态](audio_compression/candidate_methods.zh.md)。逐次日志、
checkpoint 和 benchmark 原始结果保存在运行目录，不进入仓库。

## 2026-10-02 音频实验接入

主线合入了音频运行时修复、研究脚本与新的训练/评测配方。既有八个 active、六个 target
方法的支持层级保持不变；本节列出的配置和脚本不作为新增真实模型或质量证据。

- 比例评测：[Encoder 三方法](../../../scripts/eval/lmms_eval/lmu/encoder_ratio_sweep_final/README.md)、
  [Projector 三方法](../../../scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/README.md)。
  目标保留率受 chunk、取整与 pooler 影响，比较时记录实际 Token 数。
- 训练入口：[3B/7B full SFT 与 LoRA 基线](../../../scripts/train/qwen2_5_omni/lmu/README.md)、
  [Encoder 比例训练](../../../scripts/train/qwen2_5_omni/lmu/encoder_ratio_train/README.md)、
  [Projector 比例训练](../../../scripts/train/qwen2_5_omni/lmu/projector_ratio_train/README.md)。
- 音频研究：[声学属性](../../../scripts/analysis/audio_acoustic/README.md)、
  [时间结构](../../../scripts/analysis/audio_temporal/README.md)、
  [类别信息](../../../scripts/analysis/audio_category/README.zh.md)、
  [冗余](../../../scripts/analysis/audio_redundancy/README.zh.md)、
  [sink](../../../scripts/analysis/audio_sink/README.md)、
  [静音冗余](../../../scripts/analysis/silent_redundancy_mmsu/README.md)、
  [共享 Probe 实验](../../../scripts/analysis/audio_four_probes/README.zh.md)。

这些研究脚本复用或扩展已有执行路径，不等同于在 active Analysis registry 中新增 Probe。
运行前按各入口核对数据路径、依赖、模型与资源；HPC/LMU 配方的存在不代表已取得 allocation
或完成整组实验。下文保留原验证条件，TODO 中的目标规模质量/效率与完整训练任务仍待完成。

## 1. 验证结论

固定最小验证权重为
`/home/WuHao/checkpoints/ckpts-omni/Qwen2.5-Omni-3B`。运行脚本允许通过
`MODEL_PATH` 或 `QWEN25_OMNI_MODEL_PATH` 覆盖该路径。

Qwen2.5-Omni-3B 的以下链路已经完成真实服务器复测：

- 图像全参数 SFT；
- 音频全参数 SFT；
- 训练 checkpoint 的 MME 图像评测；
- 训练 checkpoint 的 MMAU-test-mini 音频评测；
- lmms-eval 双卡数据并行与 rank-0 聚合；
- 平台训练/评测路径与参考 ms-swift/lmms-eval 路径的关键参数对齐。
- Segmentwise Top-K 与 A-ToMe 的动态单阶段音频压缩训练、checkpoint 与 Manifest；
- LTBM、SA-MAP、Encoder Attention TopK 与 Progressive Pruning 的真实 3B generate 和 loss backward。
- Audio-Pyramid 的真实 3B 多步 Decode、gradient-checkpoint backward 与 Prefill-only 选择；
- SPADE Operation-Only 的真实 3B 结构改写、generate/backward、结构 checkpoint、hash Manifest 与 reload 后生成。
- 14 个 AudioLLM 方法通过 Algorithm/Audio Variant/Qwen Binding/exact SupportUnit 的
  v2 路径解析；八个 active 与六个 target 状态保持不变，Qwen 生产路径不再回退旧 Descriptor；
- 七个共享 Analysis Probe 在 Thinker/LLM 音频 Token 上完成真实 3B baseline 最小验证，
  并与 Projector、Encoder、LLM Token、LLM Operation 四类代表压缩组合验证。

复测结果没有发现上述 active 基线与八个 active 压缩方法的剩余正确性阻塞。这个结论
不覆盖下文六个 `target` 候选方法。本文不固化某次具体分数；分数和逐样本输出属于
对应 run 的实验产物。

## 2. 训练基线

正式入口：

| 模态 | 配置 | 脚本 | 验证状态 |
| --- | --- | --- | --- |
| 图像 SFT | `configs/train/qwen2_5_omni/train_img_sft.yaml` | `scripts/train/qwen2_5_omni/train_img_sft.sh` | 已实跑 |
| 音频 SFT | `configs/train/qwen2_5_omni/train_aud_sft.yaml` | `scripts/train/qwen2_5_omni/train_aud_sft.sh` | 已实跑 |

两条基线共同使用：

- `tuner_type: full`；
- 仅训练 Thinker/LLM，冻结视觉编码器与 aligner；
- bfloat16、长度 4096、学习率 `1e-5`、1 epoch；
- 每卡 train batch 16、梯度累积 4、双卡全局 batch 128；
- DeepSpeed ZeRO-2；
- `ENABLE_AUDIO_OUTPUT=0`，不训练 Talker。

图像基线显式使用 `max_pixels: 451584`。音频基线使用 16 kHz 输入和 `use_audio_in_video: false`。这些模型/处理器原生参数位于 `backbone.config`，由 ms-swift adapter 映射为 Qwen 原生环境变量；训练器参数仍位于 `train.args`。

### 2.1 音频压缩训练与 backward

| Placement | Plugin | 正式配置 | 真实单步结果 | 证据层级 |
| --- | --- | --- | --- | --- |
| Projector-Level | `segmentwise_topk_audio` | `train_aud_sft_segmentwise_topk.yaml` | 251→62 audio tokens | Swift forward/backward、checkpoint、Manifest |
| Projector-Level | `local_temporal_bipartite_merging_audio` | `train_aud_sft_ltbm.yaml` | 25→12 audio tokens，输入梯度非零且有限 | 真实 3B backward + Swift bridge contract |
| Projector-Level | `similarity_attention_merge_prune_audio` | `train_aud_sft_sa_map.yaml` | 25→12 audio tokens，输入梯度非零且有限 | 真实 3B backward + Swift bridge contract |
| Encoder-Level | `adjacent_token_merging_audio` | `train_aud_sft_atome.yaml` | 251→93 audio tokens | Swift forward/backward、checkpoint、Manifest |
| Encoder-Level | `encoder_attention_topk_audio` | `train_aud_sft_encoder_topk.yaml` | 25→18 audio tokens，输入梯度非零且有限 | 真实 3B checkpointing backward + Swift bridge contract |
| Encoder-Level | `progressive_attention_pruning_audio` | `train_aud_sft_progressive_pruning.yaml` | 25→16 audio tokens，输入梯度非零且有限 | 真实 3B checkpointing backward + Swift bridge contract |
| LLM-Level Token | `audio_pyramid_attention_pruning` | `train_aud_sft_audio_pyramid.yaml` | 25→4 audio tokens，输入梯度非零且有限 | 真实 3B Prefill/Decode + checkpointing backward + Swift bridge contract |
| LLM-Level Operation | `structured_llm_layer_pruning` | `train_aud_sft_spade_operation_only.yaml` | 36→33 Thinker layers，减少 231,230,976 参数 | 真实 3B generate/backward + Transformers export/reload + Swift bridge contract |

八个 active 方法与六个 target 候选只在多模态 registry 维护 Algorithm、Audio Variant、
Qwen Binding 和 exact SupportUnit，并由 Projector Token、Encoder Token、LLM Token
和 LLM Operation 四个 Binding executor 服务。Swift bridge 按精确 v2 合同准备语义标签
和运行时上下文：Projector/Encoder 在 Thinker
forward 前压缩音频表示，LLM Token 在 decoder Prefill 层边界压缩，LLM Operation 在
optimizer 创建前完成结构改写。

证据层级需要分开理解：Segmentwise Top-K 与 A-ToMe 已通过完整 Swift 单步子进程、
checkpoint 和 Manifest 验收；新增四种方法已通过真实 Qwen2.5-Omni-3B
forward/backward、非零输入梯度以及 Swift bridge/launch-plan 回归；Audio-Pyramid 与
SPADE 也通过同级真实 backward，SPADE 另通过通用 Transformers 结构导出、hash Manifest
和 reload。后六种方法仍未执行完整 Swift checkpoint/Manifest run。这足以证明框架训练
路径与 autograd 可用，不等价于完整数据集收敛。

运行模式也已实测：`default` 的 timing/debug 均为空；`profile` 记录 forward 时间与
CUDA 峰值；`debug` 记录 32 个 Audio Encoder 层和 36 个 LLM 层，其中 A-ToMe 的
10 个启用层均有压缩前后 token 数。这里验证的是训练集成与单步可训练性，不代表
完整数据集收敛或最终任务质量。

四个执行族分别维护当前架构与最小 Debug 参考，八方法的接入边界另见扩展指南：

- [`Projector-Level 架构与 Debug 手册`](audio_compression/debugging/projector_level.zh.md)；
- [`Encoder-Level 架构与 Debug 手册`](audio_compression/debugging/encoder_level.zh.md)；
- [`LLM-Token Audio-Pyramid 架构与 Debug 手册`](audio_compression/debugging/llm_token_audio_pyramid.zh.md)；
- [`LLM-Operation SPADE 架构与 Debug 手册`](audio_compression/debugging/llm_operation_spade.zh.md)；
- [`Qwen2.5-Omni 音频压缩方法扩展指南`](audio_compression/extension.zh.md)。

### 2.2 六个 target 候选方法

六个候选已有实现、完整 v2 合同和 eval/train 入口，可按实际机制合同继续研究运行。
catalog 生命周期保持 `target`，不会进入 `ACTIVE_COMPRESSORS`；exact SupportUnit
保持 `declared`，Manifest 的 `compression_stages[].support_status` 如实记录该状态。

已有证据包括数值与原生小模型测试，以及 HTA 在选定 Operation 组合下的真实 3B
联合训练和完整 Swift 恢复。各方法的具体证据、剩余验证范围和权重入口统一见
[六个候选方法状态](audio_compression/candidate_methods.zh.md)。候选接口和已验证场景
不自动扩大为其他组合、目标规模收敛或质量支持。

## 3. 评测基线

| 任务 | 配置 | 双卡入口 | 验证状态 |
| --- | --- | --- | --- |
| MME | `configs/eval/lmms_eval/qwen25_omni_img.yaml` | `scripts/eval/lmms_eval/qwen25_omni_2gpu.sh` | 已复测 |
| MMAU-test-mini | `configs/eval/lmms_eval/qwen25_omni_aud.yaml` | `scripts/eval/lmms_eval/qwen25_omni_2gpu_aud.sh` | 已复测 |

共同运行条件：

- bfloat16 与 FlashAttention2；
- lmms-eval 默认 chat 路径，不传 `--force_simple`；
- 每个 worker 一份模型副本，由 Accelerate 做数据分片与聚合；
- `use_audio_in_video: false`；
- checkpoint 通过 `MODEL_PATH` 或 `QWEN25_OMNI_MODEL_PATH` 注入，脚本不绑定某次 run。

MMAU task 自己拥有 `max_new_tokens` 和 `do_sample`。平台音频配置设置 `until: []`，避免 lmms-eval 默认的 `"\n\n"` delimiter 在解码后截断回复，从而与当前原生 Qwen2.5-Omni chat wrapper 的完整回复评分行为一致。

### 3.1 新增压缩方法的真实 checkpoint smoke

四个新增方法使用同一 Qwen2.5-Omni-3B checkpoint 和单个真实音频样本完成
`generate`；均至少生成一个 token，并由 controller report 对账实际长度：LTBM
25→12、SA-MAP 25→12、Encoder TopK 25→18、Progressive Pruning 25→16。
这些结果验证真实模型集成，不替代目标规模 MMAU 的质量和效率比较。

### 3.2 LLM-Level 生命周期 smoke

Audio-Pyramid 在同一真实 Qwen2.5-Omni-3B checkpoint 上把 25 个 audio Token 依次压缩为
13、7、4；记录 1 次 Prefill、3 次 Decode，且 `selection_repeated_during_decode=false`。
该运行同时覆盖原生三轴 M-RoPE、标准 `DynamicCache` 和带 gradient checkpoint 的 SFT
loss backward，音频输入存在 12,800 个非零梯度元素。

SPADE Operation-Only 在 optimizer 创建前把 Thinker 从 36 层改写为 33 层，减少
231,230,976 个参数；改写后的模型完成 generate/backward，并以 `thinker_only` artifact
导出 Transformers checkpoint 与 hash Manifest。reload 后仍保持 33 层并再次生成 4 个
Token。该范围只验证显式/预计算层选择和完整层删除，不包含 WLI 分析、自动层选择、蒸馏、
head/FFN pruning 或 Talker。

### 3.3 AudioLLM v2 与 Analysis 最小复测

本轮使用现有 Qwen2.5-Omni-3B、合成 1 秒 16 kHz WAV 和完全离线环境完成最小复测。
没有下载/安装依赖，没有导出大型 checkpoint，也没有终止或干预服务器已有进程。每个
GPU 进程固定到单卡并使用独立输出目录。

| 范围 | 真实 3B 结果 |
| --- | --- |
| 未压缩 Analysis baseline | 七个 Probe 各产生一条非空 `AnalysisRecord`；0 skipped；Manifest invariant 通过 |
| Segmentwise Top-K + Representation | Projector 压缩后记录非空；0 skipped；25→6 audio Token，LLM 输入 81→62 |
| A-ToME + Representation | Encoder 压缩后记录非空；0 skipped；50→20 Encoder Token、25→10 audio Token |
| Audio-Pyramid + Representation | layer 8 出口压缩、layer 9 入口观察；记录非空；0 skipped；25→13→7→4 |
| SPADE + Representation | 36→33 Thinker layers 后记录非空；0 skipped；结构层号连续 |
| 四执行族 backward | Segmentwise、A-ToME、Audio-Pyramid 与 SPADE loss 有限，目标输入均有 12,800 个有限非零梯度元素 |

七个 Probe 复用 ImageLLM 的共享实现，Qwen 只在 semantic binding 与有效 Token layout
提供者中补充 audio 语义。范围限定 Thinker/LLM 中已经拼入 prompt 的音频 Token，不含
Audio Encoder 专用 Probe、Talker 或 Token2Wav。LLM Token 压缩按每层入口有效布局分组，
不会用压缩前 audio span 解释压缩后的 hidden state。

## 4. 已闭环的排查项

- 训练数据路径、样本规模和核心 Swift 参数已核对；
- 图像 `max_pixels` 已与参考运行对齐；
- 图像/音频 SFT 的实际 micro-batch 与梯度累积已回收到本地配置；
- lmms-eval 默认 chat 路由已确认，simple 仅由显式 `force_simple: true` 触发；
- `use_audio_in_video`、processor、dtype、attention implementation 和 generation kwargs 的所有权已明确；
- MMAU 输出截断差异已通过 `until: []` 修正并复测。

这些结论由回归测试和当前配置保持，不再保留逐次差分报告或一次性 OCR 诊断脚本。

## 5. 未覆盖范围

以下能力不能从图像/音频基线推断为已验证：

- video SFT；
- audio-video SFT；
- image/audio/video/audio-video LoRA 的目标规模训练；
- Talker 训练；
- 压缩训练的完整数据集收敛、最终任务质量与目标规模评测；
- LTBM、SA-MAP、Encoder TopK 与 Progressive Pruning 的完整 Swift checkpoint/Manifest run；
- Audio-Pyramid 与 SPADE Operation-Only 的目标规模 MMAU 质量/效率实验和完整 Swift checkpoint/Manifest run；
- 六个 target 候选尚未验证的工程场景与目标规模质量/效率；具体范围见[候选方法状态](audio_compression/candidate_methods.zh.md)，不包含其中已经完成的数值测试与 HTA 联合训练恢复场景；
- 多节点训练/评测或 tensor parallel。

对应配置可以通过 plan/contract 测试，但在完成真实运行前只标记为“契约通过”。

## 6. 回归要求

修改 Qwen2.5-Omni adapter、processor、runner、lmms-eval bridge、ms-swift bridge 或相关配置后，至少检查：

1. 八个原生训练 profile、八个 active 压缩训练配置，以及六个 target 方法的 eval/train 配置与生命周期 contract；
2. image/audio 配置的 lmms-eval launch plan；
3. 默认 chat 路由与显式 simple 路由；
4. 音频 `until: []`、`use_audio_in_video: false` 和 task generation ownership；
5. 图像 `MAX_PIXELS=451584`、音频 `SAMPLING_RATE=16000`；
6. 至少一个真实图像和音频样本；
7. 十四个方法的 Algorithm/Audio Variant/Qwen Binding/exact SupportUnit、exact length、identity、backward，以及六个 target 方法不进入 active registry但仍可显式解析的回归；
8. LLM Token 的 Prefill/Decode/cache/M-RoPE 与 LLM Operation 的 pre-optimizer apply/export/reload 回归；
9. 七个 Analysis Probe 的 Qwen Thinker audio Token 分组、teacher forcing、干预、压缩后 layer-entry layout、非空记录和零 skipped sample；
10. 涉及生成或 processor 数值行为时，复跑对应真实 checkpoint smoke 或 benchmark，而不是只检查最终 argv。
