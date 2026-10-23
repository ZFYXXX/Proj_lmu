# 项目里程碑与 TODO

最后更新：2026-10-03（同步主线能力及新增对比方法；未重跑历史真实模型验证矩阵）

本文是项目级里程碑和未完成 TODO 的事实来源。正式接口见[接口规范](platform/interface_spec.zh.md)，新增压缩方法见[开发指南](guides/2_dev_compression_method.zh.md)，Qwen2.5-Omni 的既有方法见[AudioLLM 压缩扩展](support/qwen25_omni/audio_compression/extension.zh.md)，具体能力状态由相应 `support/*/status.zh.md` 维护；[平台设计 v0.3](archive/platform_design_v0.3.zh.md)仅保留为历史设计基线。

## 维护规则

- 已有实现、自动化回归和真实运行证据后，任务才能从 TODO 移入“已完成里程碑”；
- `plan_only`、schema 校验和 CPU contract 只能证明配置或启动计划可构造，不能作为模型链路通过的证据；
- TODO 只保留尚未完成且有明确退出条件的事项，不记录逐次调试日志；
- 能力状态发生变化时，在同一次代码变更中同步更新本文和对应验证状态文档；
- 稳定设计和接口细节不在本文重复维护。

## 稳定后资产分类

| 资产 | 稳定后的处理 | 原则 |
| --- | --- | --- |
| `plan_only` / launcher `dry_run` | 保留唯一、明确的只读启动计划，清理重复别名 | 用户和 Agent 在昂贵运行前查看有效配置；计划通过不证明模型运行正确 |
| 原 schema-only `dry_run` | 已删除；definition validator 由真实命令和 runtime activation 共用 | 不维护与真实执行并行的第二套语义 |
| 原 `preflight_only` | 已删除；依赖检查统一使用 `scripts/diagnostics/validate_framework_integrations.py` | 避免多个相近 CLI 开关 |
| unit / contract / regression test | 保留有独立错误责任的最小集合 | 独立数学、实际组件与持久化行为优先；删除只重复配置、布尔镜像或已覆盖路径的场景 |
| 可重复 CPU smoke | 保留最小代表集 | 覆盖真实组件组合但不依赖昂贵 GPU |
| 真实 checkpoint GPU smoke | 保留最小端到端集合，改为 opt-in、定期或发布前运行 | 验证单元测试无法覆盖的真实模型集成 |
| 一次性排查脚本、临时 fixture 和历史运行报告 | 闭环后删除 | 不把 Vibe Coding 过程产物固化为长期负担 |

诊断和测试已按上述责任收敛：算法数学和梯度由 kernel/原生小模型验证，真实权重命令
聚焦资源加载、目标命中、位置/缓存、训练更新及重载。方法报告的完整合成字典矩阵、
生成样本后的重复布尔汇总和自然视频的算法分支穷尽要求已移除；基础布局、公共成功/失败、
卸载、阶段和实际参数门禁保留。本轮回归结果见下方“必要覆盖收敛”，现行命令与证据边界见
[VideoLLM 状态](support/video_compression/status.zh.md#证据与诊断代码的维护边界)。

## 状态定义

| 状态 | 含义 |
| --- | --- |
| 已验证 | 实现、回归测试和真实运行证据均存在 |
| 已激活 | 已实现、通过对应 contract/runtime 测试并进入 active registry；目标规模质量与效率证据另行说明 |
| 候选（target） | 已有精确 v2 合同和显式配置；满足实际机制合同即可研究运行，验证状态单独记录，不构成正式支持声明 |
| 契约通过 | 配置、启动计划或 CPU contract 已验证，但未完成目标规模实跑 |
| 当前 | 正在推进的主里程碑 |
| 未激活 | 只有设计或 schema，真实命令必须明确拒绝 |

## 代表验证权重

真实 checkpoint smoke 和最小能力验证使用下列模型；本地路径通过对应环境变量或配置指定：

| 基座 | 固定最小验证权重 |
| --- | --- |
| Qwen2.5-Omni | `Qwen2.5-Omni-3B` |
| Qwen2.5-VL | `Qwen2.5-VL-3B-Instruct` |
| Qwen3-VL | `Qwen3-VL-2B-Instruct` |
| LLaVA-OneVision | `llava-onevision-qwen2-0.5b-ov` |

Qwen2.5-Omni AudioLLM 与三个视觉基座共享多模态 v2 contract/session；各基座仍保留
独立 Binding executor 来维护物理不变量。具体实验仍可通过对应的 `MODEL_PATH` 环境变量覆盖模型路径。

## 当前概览

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 平台 core 与组合式 backbone | 已验证 | typed config、facade/runtime、state、validator、registry 和 manifest 基础已建立 |
| 自定义压缩规则 | 原生小模型通过（探索） | `operation.rules` 按方法开放实际适用步骤，复用位置、预算、布局、DeepStack、checkpoint 与 cache；默认配方及历史证据保留，探索规则不继承原质量/训练结论。新增方法不自动获得旧方法的规则接口。[规则目录](../examples/compression_rules/methods.md)、[配置和函数合同](../examples/compression_rules/README.md) |
| Qwen2.5-Omni 原生模型链路 | 已验证 | text/image/audio/video、Thinker、Talker/Token2Wav 和 trace 已通过真实 GPU smoke |
| Qwen2.5-Omni 图像/音频全参数 SFT | 已验证 | ms-swift 双卡训练、产物收集和最终模型评测已复测 |
| Qwen2.5-Omni MME/MMAU | 已验证 | lmms-eval chat 路径、双卡聚合和准确率复测通过 |
| Qwen2.5-Omni video/audio-video 与 LoRA | 契约通过 | 八份配置和薄脚本存在，尚未全部完成目标规模实跑 |
| Qwen3-VL 原生/lmms/Swift | 已验证（次主线） | native text/image/video 与一步框架 smoke 已完成；未作为当前训练质量基准 |
| Qwen3-VL 共享 Attention/FFN replacement | capability 已激活；精确 prepare 单元 declared | 两插件的共享模块替换仍限定 Qwen3-VL；6 种 Attention 与 3 种 FFN 策略已有数值、forward/backward 和生命周期测试，不将 capability 的 active 状态等同于精确 SupportUnit 或目标规模质量/效率证据 |
| Qwen 图像分支 Attention/FFN replacement | 已接入；精确 prepare 单元 declared | 两插件新增 `.image_llm` variant，接入 Qwen2.5-VL / Qwen3-VL；已有原生小模型 CPU 测试覆盖分模态、梯度、缓存、训练桥和保存恢复，完整 GPU Swift 训练/恢复及质量实验未认证 |
| ImageLLM 历史验证矩阵 | 已验证（推理 + 一步训练） | 原 25 个算法中，除 D2Pruner 外的 24 个算法在三基座上共有 78 个 active inference 单元和 72 个 `train_step_verified` 单元；该历史矩阵不包含新增 Bifocal 或 14 组对比方法，完整训练闭环未声明 |
| Qwen2.5-VL Bifocal | 已接入；forward/generate 单元 declared | Projector-Level 图像压缩已有 kernel、binding、eval/train 配置与 CPU 测试；未登记真实 checkpoint 推理/训练证据，不继承三基座历史矩阵 |
| OV 对比方法三基座适配 | 已接入；精确单元 declared | 14 组、17 个 ID、51 个 binding，包含 7 个 Token 与 10 个 Operation ID；原生小模型前向/梯度/生成及配置和训练桥检查通过。真实 checkpoint、校准权重、完整训练及质量/效率范围见 [接入说明](support/imagellm/comparison_methods.zh.md) |
| 双基座 MWOP（Mask 形式） | CPU 验证；prepare 单元 declared | 代码标识为 `fixed_mask`；LLaVA-OneVision / Qwen2.5-VL 头、路径、FFN 遮罩，Taylor/随机清单与保存恢复已接入；129 项 CPU 测试通过，真实大模型 GPU Swift CLI 训练与质量基准未完成，不声明物理加速 |
| VideoLLM 九算法 | 已验证（有限推理与最小训练/重载） | 三个视觉基座共 27 组真实 checkpoint 推理和 27 组最小训练/保存重载证据；不代表完整训练收敛、论文质量或性能基准 |
| Qwen2.5-Omni Projector-Level audio compression | 已验证（运行时） | Segmentwise Top-K、LTBM、SA-MAP 共用 plan、legalizer 与 pre/post-projector controller；三者均通过真实 checkpoint generate，完整 MMAU sweep 仍属于质量实验 |
| Qwen2.5-Omni Encoder-Level audio compression | 已验证（运行时） | A-ToMe、Encoder Attention TopK、Progressive Pruning 共用逐层变长 controller；三者均通过真实 checkpoint generate，新增两种 pruning 方法同时通过 gradient-checkpoint backward |
| Qwen2.5-Omni 音频压缩候选 | 候选（target） | FastSLM HTA、UniAvg、UniSamp、UnSeg、FastAdaSP、HeadRouter 已有精确合同、eval/train 配置及原生小模型数值与生命周期测试；目标规模运行和质量/效率证据按方法单独记录，不因研究运行通过自动晋级 |
| Qwen2.5-Omni 压缩训练 | 已验证（分层证据） | Segmentwise Top-K 与 A-ToMe 已完成真实 Swift 单步 LoRA、loss backward、checkpoint 和 Manifest；其余六种 active 方法已完成真实 3B backward 与 Swift bridge/launch-plan 回归，其中 SPADE 另完成结构 checkpoint/hash Manifest/reload；尚未完成各自的完整 Swift run、收敛和质量实验 |
| 真实压缩 pipeline | 当前 | Qwen2.5-Omni 的八种 active 方法与六种 target 候选均通过 Algorithm/Audio Variant/Qwen Binding/exact SupportUnit 进入同一 v2 session，四个 Qwen executor 保留既有语义；ImageLLM 同样复用 kernel/variant/binding；跨模态组合和完整质量实验仍待推进 |
| 多基座 analysis | 已激活 | 七个 Probe 已进入 active registry，并在三个 ImageLLM 最小 checkpoint 与 Qwen2.5-Omni 3B Thinker/LLM 音频 Token 上完成真实最小验证；Qwen baseline 产生 7 条非空记录，四类代表压缩后分析均无 skipped sample |
| 实验结果与 Manifest | 已接通 | analyze/eval/train 共用实验入口；成功保存有效配置、Manifest 和产物，失败保存结构化结果及原生退出码；各方法的真实运行与质量证据单独记录 |
| 独立研究方法的使用与恢复 | 代表流程已验证 | 可导入示例独立于 tests，先按实际方法接口验证执行复用，再用 `register_method` 注册；FirstK 音频已完成真实 Swift 更新、保存、恢复更新及 lmms-eval 单样本评测。各模态的方法接口与支持证据仍分别声明 |

2026-10-03 从当前注册表读取到 **68 个 Algorithm、72 个 Variant、176 个 Binding**：
图像 Token 97、新增图像对比 Operation 30、视频 27、音频 14、共享/分模态模块替换 6、MWOP Mask 2。
本轮对比方法接入新增 17 个 ID/51 个 binding；此前主线快照为 51/55/125。
这是实现目录的快照，包含 `declared` 和候选单元；不是已验证算法数，也不是自定义规则覆盖数。
后续以 `mllm-comp discover` 的精确 Binding 与 SupportUnit 为准。以上历史验证记录保留各自
日期、配方与模型范围，不因本次文档同步视为在当前提交全量复测。

## 2026-10-03 对比方法合入

实现提交 `d5c9dcb` 与文档提交 `10000c6` 已经由合并提交 `fc86f2d` 进入 `main`。
14 组方法增加 17 个 Algorithm、51 个三基座 Binding，精确 SupportUnit 保持 `declared`。
相关 CPU 回归 378 项通过，覆盖新增方法、原生 provider、训练桥、session/definition/registry、
MWOP 与分模态集成；这不是全仓库测试或真实 checkpoint benchmark。

真实 Swift/lmms-eval 全链路、针对目标权重的校准、质量/效率与新增参数的 adapter-only
保存恢复仍需验证。TwigVLM 仅有初始化分支，Short-LVLM 仍需外部 SCP 资产，Structural
仅接入 depth 分支；具体方法边界和配置统一见[对比方法接入](support/imagellm/comparison_methods.zh.md)。

## 2026-10-02 主线合入

主线先在 `5dbe05b` 合入音频、分模态与 Bifocal，再在 `4d03e19` 合入 MWOP（Mask 形式）。
本节记录实现接入，未完成的真实模型验证仍保留在 TODO。

- **音频实验**：接入音频压缩运行时修复、Encoder/Projector 比例评测与训练入口，以及
  声学、时间、类别、冗余、sink 等研究脚本。入口见 [Omni 状态](support/qwen25_omni/status.zh.md#2026-10-02-音频实验接入)。
  脚本和配置存在不证明对应 sweep 已完成，也不改变既有八个 active、六个 target 的证据边界。
- **分模态模块替换**：在 Qwen2.5-VL / Qwen3-VL 的指定 Attention/FFN 层中分别执行图像、
  文本分支，复用既有替换算子，接入 pipeline、Swift 训练桥、保存恢复和缓存生成。
  配置和限制见[分模态说明](../mllm_comp/modality_split/README.md)；精确 prepare 单元仍为 `declared`。
- **RefCOCO 定位评分**：Qwen3-VL 使用 0–1000 坐标，Qwen2.5-VL 使用实际处理后的图像像素
  坐标；评分协议 `qwen-grounding-v2` 与显式框解析已有单元测试。协议与比较条件见
  [评测指南](guides/1_run_evaluation.zh.md#refcoco-定位评分)，不将评分修复视为模型质量提升。
- **Bifocal**：新增 `bifocal.image.qwen25_vl` 的 forward/generate 执行及专用配置，
  原生位置、标签和图像占位符由基座适配。两项支持单元仍为 `declared`，见
  [ImageLLM 状态](support/imagellm/status.zh.md#bifocal-接入)。
- **MWOP（Mask 形式）**：以 `fixed_mask` 标识接入双基座头、注意力路径和 FFN 遮罩、Taylor/随机清单生成、完整模型与
  LoRA checkpoint 的 Mask 保存重载；有限组合为 ZOO-Prune 或层间 PyramidDrop。
  129 项 CPU 验证的范围与限制见[MWOP Mask](guides/fixed_mask.zh.md)。

## 已完成里程碑

### M0：平台骨架

- 组合式 `BackboneFacade`、`BackboneRuntime`、`ProcessedBatch` 和 `ForwardState`；
- capability catalog 与 active registry；
- typed YAML 与共享 validator；
- lmms-eval/ms-swift 固定版本 third-party 接入。

### M1：原生 backbone

- Qwen2.5-Omni 全模态输入、Thinker 文本、Talker 语音、原生 validator 和 trace；
- Qwen3-VL text/image/video、DeepStack/M-RoPE 结构与通用 text trace；
- Qwen2.5-VL text/image/video 与 text trace；
- LLaVA-OneVision text/image、AnyRes metadata 与 vendored Qwen source route；
- 四个 backbone 均以各自已验证的 active scope 进入 active registry。

### M1c：Qwen2.5-Omni 训练与评测基线

- 图像和音频全参数 SFT 使用服务器实测参数完成双卡训练；
- MME 与 MMAU-test-mini 使用正式项目配置完成双卡评测；
- 默认 chat/simple 路由、FlashAttention2/bfloat16、模态参数和输出解析完成差分排查；
- 音频配置显式 `until: []`，最终复测通过；
- 详细范围见[Qwen2.5-Omni 验证状态](support/qwen25_omni/status.zh.md)。

### M2a：Qwen2.5-Omni 六方法单阶段扩展

- 建立统一 compression registry 与 controller factory，评估和训练不再分别维护 plugin-name 分支；
- Projector-Level 新增 LTBM 和 SA-MAP，分别覆盖 pre-projector merging 与 post-projector merging-and-pruning；
- Encoder-Level 新增 Attention TopK 和 Progressive Pruning，分别覆盖 MHSA 后 pruning 与完整 block 后 progressive pruning；
- 四个新增方法均提交 eval/train 完整配置、算法/plan/controller/bridge 回归，并通过真实 Qwen2.5-Omni-3B generate 与 loss backward；
- 新增[音频压缩方法扩展指南](support/qwen25_omni/audio_compression/extension.zh.md)，固定两个 Level 的最小接入步骤、长度契约和验证矩阵。

### M2b：面向压缩研究人员的四执行族重构（历史，已被 M2f 取代）

早期 Descriptor 调度已在 M2f 删除，迁移背景见
[Omni v2 收敛记录](archive/omni_compression_v2_convergence.zh.md)。Audio-Pyramid 与 SPADE 的
既有运行证据见 [Qwen2.5-Omni 状态](support/qwen25_omni/status.zh.md)，当前扩展方式见
[压缩研究人员扩展指南](guides/2_dev_compression_method.zh.md)。

### M2c：ImageLLM 多算法推理与一步训练

- 25 个算法共用 kernel、variant 与三基座 binding，不复制 inference/training 实现；
- 24 个可运行算法在 LLaVA-OneVision、Qwen2.5-VL、Qwen3-VL 上保留精确 forward/generate 证据；
- 72/72 一步训练案例验证真实压缩、finite loss、非零梯度、optimizer step、布局修复与 gradient-checkpoint 重放；
- generate-only 方法使用 `use_cache=False` teacher forcing，离散选择不要求可学习，但 hidden states 不 detach；
- 当时 ms-swift 以精确 `train_step_verified` 作为启动门禁；后续框架重构已改为检查实际执行契约，历史证据仅描述验证范围；
- ToMe 已在三个基座的公开 `mllm-comp train` 入口完成一步训练，并写出压缩证据、checkpoint 与 Manifest；
- 该矩阵未包含 D2Pruner 的 calibrated prior 验证，六个 SupportUnit 保持 declared；当前已支持外部 prior 的加载与校验，具体边界见 [ImageLLM 状态](support/imagellm/status.zh.md#d2pruner-边界)；
- 完整 train lifecycle、video/audio、多图、质量和效率仍未声明。精确边界见 [ImageLLM 多算法状态](support/imagellm/status.zh.md)。

### M2d：Qwen3-VL LLM Attention/FFN replacement

- `qwen3_vl_attention_replacement` 与 `qwen3_vl_ffn_replacement` 已进入 compressor active registry；前者支持 linear attention、causal CNN、identity、LoLCATs linear-window、NHA 与 sparse-window，后者支持 low-rank、identity 与 two-layer MLP；
- 两个插件都在 `prepare` 阶段对 `llm.layer[n].attn` 或 `llm.layer[n].ffn` 执行保持 hidden-state I/O shape 的结构替换，供后续 generate 或 training forward 使用；runtime session 卸载时恢复原模块；
- 配置校验、planner/OperationApplier、独立数值参考、forward/backward、runtime loader/session 与 lmms-eval 参数链路均有回归测试，并提供真实 checkpoint 单图生成诊断入口；
- 此里程碑的 `.shared_llm` reload 契约是“加载原始 checkpoint 后重新应用 YAML”，不导出独立 Transformers replacement checkpoint；任何替换策略均不因保持 I/O shape 就获得 baseline 数值等价或质量收益声明。后续分模态替换、MWOP Mask 和 comparison 的范围见上方 2026-10-02/03 合入记录；不能将当时的子集限制当作当前全框架清单。

### M2e：多基座 analysis 激活

- `representation_dynamics`、`layer_importance`、`late_injection_early_exit`、`residual_stream_decomposition`、`prediction_dynamics`、`attention_flow` 与 `hierarchical_taylor` 已进入 analysis active registry；
- 七个 Probe 共享 lmms-eval 样本循环、Backbone semantic binding、session 生命周期、typed artifact envelope、`AnalysisRecord` 与 manifest，不各自维护数据加载路径；
- analysis 不再维护独立 schema-only dry-run；真实 `analyze` 在模型加载前通过 `analysis/definition.py` 解析 override 并校验有效定义；
- 顶层 compression dry-run 已删除；`analyze`、`eval`、`train` 和 runtime activation 共用 `pipeline/definition.py` 校验有效 compression definition；
- LLaVA-OneVision 0.5B、Qwen2.5-VL 3B 与 Qwen3-VL 2B 的固定最小 checkpoint 已完成真实 smoke 矩阵，Attention 路由质量守恒与 Taylor 分层分数/因果对照均保留有限证据；
- 本里程碑最初不包含 Qwen2.5-Omni；其 AudioLLM 扩展由 M2f 完成。

### M2f：Qwen2.5-Omni AudioLLM v2 收敛与分析扩展

- 将 14 个 Qwen 音频方法迁移到 `AlgorithmDescriptor → Audio Variant → Qwen Binding → exact SupportUnit`，保持 8 个 active 与 6 个 target 状态；
- Qwen 生产配置、catalog、eval/train bridge 与 runtime session 不再进入旧 Descriptor 调度；删除旧内置描述符、binding 和 controller factory 文件，不提供 fallback；
- Projector Token、Encoder Token、LLM Token 和 LLM Operation 四个 executor 继续维护 chunk、placeholder、M-RoPE、prefill/decode、KV cache、labels 与结构重排，不把 Qwen 物理逻辑扩散到共享 runtime；
- 七个共享 Probe 扩展到 Qwen Thinker/LLM audio Token，支持 audio Token group、`a2a/a2t/t2a/t2t` 边、late injection/early exit 和 Taylor teacher forcing；Audio Encoder、Talker、Token2Wav 不在本范围；
- 真实 Qwen2.5-Omni-3B 合成 1 秒 16 kHz 音频最小验证中，baseline 产生 7 条非空 AnalysisRecord；Segmentwise Top-K、A-ToME、Audio-Pyramid 和 SPADE 的压缩后分析均产生非空记录、无 skipped sample，Manifest invariant 通过；
- 离线最小 generate/backward 覆盖四个执行族；没有下载、安装、导出大型 checkpoint 或干预服务器已有进程。

## 已完成：实验生命周期与方法边界重构

本轮实现、相关验证和使用者/研究者/维护者的独立源码任务验收已完成。已完成的切片：

- CLI/YAML 和 Python 共用独立实验入口，保存生效配置、结构化结果和失败；默认输出隔离，显式扩展注册贯穿规划、执行与 Manifest。
- 纯 Operation 使用结构操作合同；PREPARE 先于请求机制安装，后续目标从改写后的真实模型解析。保存的物化阶段不会在公共恢复入口再次剪枝。
- FastV/PyramidDrop 的评分与方法状态共用实现，原生 provider 保留各自位置、cache、DeepStack 和重放修复；真实图像 checkpoint 的前向、反向和代表组合已验证。
- BTP/HalfV 的评分和历史状态也归到共享方法；独立图像算法可通过 `strategy_factory` 复用三个基座的选择机制，原生小模型验证了布局、梯度和代表 checkpoint 重放。BTP 跨两 GPU 的来源/应用检查已通过；真实 Qwen2.5-VL-3B 上两方法均完成一步训练、视觉参数和末层归一化参数更新、完整 forward 重放。该证据不等于完整训练恢复或质量评测。
- TransPrune 的 TTV 历史、固定 IGA 来源和待应用决策也由方法管理；三个基座共用临时模块观测，保留各自布局、逐层缓存与重算。原生小模型的位置、独立数值、梯度、生成及跨两 GPU 验证通过；真实 Qwen2.5-VL-3B 上完成一步训练，视觉和末层归一化参数均更新，完整 forward 重放一致。该证据不扩大为完整训练恢复或质量结论。
- 视频单目标 Projector 共用安装、清理、布局记录和 Qwen3 DeepStack 包装，VQToken 已迁入；独立 stride 示例在三原生小模型验证了输出、布局、梯度和关闭后的原生恢复。真实 Qwen3-VL-2B 的 VQToken 用既有单样本诊断完成五步末层 norm 训练、参数精确重载和压缩推理；该轻量参数重载不等于完整 optimizer 恢复或视频质量评测。
- 音频方法的数学与重放移到方法文件，Qwen 的探针、维度与物理适配保留在基座；规则能力由 Variant 声明，复用 LTBM 机制的仓库外方法已覆盖规划、训练桥、数值与梯度。
- 唯一可推导的配置字段自动补全；发现、两次实验比较和框架内恢复有共同 CLI/Python 入口。HTA 原位置重建、Operation＋Token 和 Swift 多阶段原生小模型训练恢复已验证。
- 统一依赖约束与安装说明已整理，量化的内部实现、外部组件和完整配方设计见[量化联合接入](platform/quantization_integration.zh.md)，尚无量化执行器。

真实 Swift CLI 的 Operation＋HTA 保存、完整恢复继续训练和推理已通过：Qwen2.5-Omni-3B
删去一层后，在同一模型上训练 HTA 与末层归一化参数；从第一步 checkpoint 恢复后的第二步
参数、Adam 状态、调度器和 RNG 与连续训练完全一致。公共恢复推理保持 35 层并加载训练后的
HTA，音频 Token 从 25 个压缩到 1 个。复现入口是
`scripts/diagnostics/validate_qwen25_omni_swift_resume.py`。
上述功能验证不代表全参数或 LoRA/DeepSpeed 恢复、训练收敛或基准质量收益，也不扩展全部方法组合的正式支持。

## 已完成：使用、研究与维护流程改进

- 复用现有机制的新方法可在一个文件中定义算法并调用一次 `register_method`；框架构造既有描述符，保留物理依赖与冲突，不复制原方法参数、位置声明或支持证据。音频、图像和视频的四个完整示例迁入 [examples/compression_extensions](../examples/compression_extensions/README.md)，不再要求使用者导入 tests。
- CLI/Python 按本次任务检查真正使用的模型、数据和配置字段；闲置任务占位符不再阻塞运行。启动计划显示有效配置，缺少任务/数据、未替换变量及输出目录冲突均给出定位；Manifest 不把未使用的数据占位符记录为数据身份。
- 研究指南按修改规则、注册方法、新增机制三种任务组织，从 700 行收敛到 175 行；模型专属接入细节移至参考文档。维护指南提供实际调用链、责任代码和按改动选择验证的说明；Session 删除无调用的参数、裸 executor 包装及单记录的重复循环。
- 修复真实训练 checkpoint 关闭 Thinker cache 后，恢复压缩评测重复处理音频的问题。动态压缩生成显式使用缓存默认值，保留原生参数优先级，用户明确关闭缓存时在请求前说明限制；训练 forward、baseline 和静态 Operation 保留原行为。

2026-09-09 的代表流程使用 Qwen2.5-Omni-3B、新 FirstK 方法与原生 Swift CLI：冻结其余参数，
训练末层 norm 一步并保存完整 checkpoint，再经 `restore_config` 恢复至第二步。两次参数均真实
更新，训练音频 Token 为 25→12。恢复后的原生 lmms-eval 完成 MMAU-test-mini 一个样本，
音频 Token 为 250→125，方法身份和运行阶段贯穿三个 Manifest。相关入口与生命周期回归
402 项通过，随后缓存修复相关回归 41 项通过（两批有重叠，不相加）；研究示例的 CLI 计划也已实跑。
该证据限于上述功能流程，不代表全参数收敛、基准质量收益或其他方法组合全部通过。

## 已完成：先验证执行复用，再注册方法

- VQToken、FastVID、VisionZip 共用视频 Projector 的安装、信号生命周期、布局记录与
  DeepStack 应用。内置与研究方法都提供压缩和重放；Stride 示例直接沿用所选执行器。
- DyCoKe 将视觉 Attention 评分和选择历史移入每个 cache 独立的策略；基座保留
  prefill、增量 decode、原生 RoPE 与物理缓存接入。新 StrideRecent 示例替换 Token
  选择和 KV 读取决定，沿用同一机制，完整物理 KV 仍保留。
- 音频 Encoder 规则跨 Projector/LLM 复用按实际接口检查，消除两种实现类的身份限制；
  下游参考直接重放方法自己的计划，不再从 A-ToMe 诊断字段重建决定。独立非相邻加权
  合并策略验证了连续阶段的参考、值和梯度，观察链结束后释放临时计划。
- Qwen3 Operation 的策略集合与模块构造回到方法中，安装器负责物理替换及状态归属。
  既有自定义 factory 与 Omni 的决定/应用边界保留。研究和维护指南按实际接口组织。

2026-09-09 在统一 Dev 环境分批验证了 447 项相关回归（重叠不重复计数），包括跨两张
GPU 的音频重放。真实 Qwen3-VL-2B 权重使用合成视频 patch 输入，完成 VQToken、FastVID、
VisionZip、DyCoKe 及两个研究示例的生成；两个示例均有有限 loss、视觉输入和末层 norm
梯度。StrideRecent 将 784 个视频 Token 选为 196 个，decode 每步只读取其中最近两个视觉
KV，保留全部非视觉 KV；末层 norm 更新后完整导出权重，再加载得到精确相同的参数和生成
Token。这是原生模型导出/重载的功能证据，不等于 Swift optimizer 恢复、训练收敛或质量/效率
收益，不提升研究示例的正式支持状态。接口细节见[执行参考](reference/compression_execution.zh.md)。

## 已完成：保留压缩边界，精简重复实现

- 纯图像选择可只返回 `ImageTokenSelection`；共享代码保留文本与结构 Token，并生成值、
  索引和布局。内置 FastV 与独立研究示例共用该实现，既有完整结果和合并/历史方法保留。
- 三个图像基座的 75 个物理合同集中维护，中央 registry 与适配器共用 variant、位置和
  metadata；原查询路径保留，LLaVA 回调时机、能力和修复要求不变。旧新结构字段逐项相等。
- Qwen3 Attention/FFN replacement 进入统一定义与 Session；九份配方使用 `algorithm`，
  旧 `plugin` 配方和序列化快照在读取时正规化。prepare、参数挂载及反向卸载沿用原机制，
  运行时基座依赖延迟到安装时导入。

2026-09-10 的远程相关回归为 1,212 通过、25 个 GPU 项跳过；随后双 GPU 与受影响入口
补测 298 通过，覆盖上述 25 项，两批有重叠。真实 Qwen3-VL-2B 使用固定 256×256 图像，
独立图像选择与 rank=16 FFN 替换通过公共 Session 组合：序列 80→48，有限 loss、非零梯度
与参数更新成立，checkpoint 重算只重放一次决定；生成三个 Token 时只在 prefill 选择，
增量 decode 不重选，卸载恢复原 FFN。此检查未验证完整 Swift 训练恢复或质量收益，
未提升研究方法支持状态。量化实现、联合音视频及其他组合仍按各自实际进展处理。

### 必要覆盖收敛

- 视频诊断共用环境初始化、CLI 参数和训练生命周期；方法保留信号、布局、标签、
  缓存及实际参数检查。原生小模型承担正常执行、梯度和重放，夹具只保留消费者需要的输入。
- 删除 42 份方法报告镜像测试、9 个命令转发 Shell，收敛机械矩阵和大型假模型。
  Omni 验证命令独立运行，不再依赖 PASS 状态编排或前置完整回归。
- 完整报告写入文件，终端只输出摘要；稳定布局训练仍逐步检查，只在首步存储完整布局。
  算法、执行机制和基座差异分别保留独立覆盖。相对 `ee2a11e`，诊断与测试代码净减 12,610 行。

2026-09-10 的收敛提交 `098be96` 完成统一 Dev 环境 CPU 回归：4,057 passed / 132 skipped；
跳过项需要 GPU、显式启用或两设备，不计为通过。后续公共报告/Shell 补测 51 passed，
输出与计数夹具补测 55 passed，均与前面的集合有重叠。依赖诊断、独立 Segmentwise oracle
和八方法启动计划也通过；启动计划不构成音频训练证据。

选定的 22 条真实权重路径全部通过：九条训练/新模型重载、十一条固定视频生成、
Qwen3 原生 text/image/video 与 Omni 原生多模态及语音输出。LongVU/LLaMA-VID 各三步，
验证可训练压缩模块严格重载；其余七条训练各五步。此前共享初始化与 ToMe 的三个原生
小模型 GPU 检查也获得有限 loss 和非零视觉梯度。这些结果没有重新认证全部 54 个历史
支持单元，也不替代完整 Trainer/optimizer 恢复、多卡专项或质量实验。
现行命令和历史证据入口见[VideoLLM 状态](support/video_compression/status.zh.md#证据与诊断代码的维护边界)。

### 公共行为复用

- LLaMA-VID、LongVU、METok 共用 Qwen 原始特征到 raster/坐标的恢复；DyToK、METok
  共用视频像素到 SigLIP 的转换，见 [`_qwen_video_inputs.py`](../mllm_comp/backbones/_qwen_video_inputs.py)。
  Qwen2.5 窗口逆序与 CLIP 归一化、Qwen3 原始顺序与 half normalization 继续分别处理，
  方法自己的 QFormer、DINO/SigLIP 信号与梯度上下文保持原有职责。
- Audio-Pyramid 与 FastAdaSP 共用 [`_rotary.py`](../mllm_comp/compressors/llm_level/_rotary.py)
  中的普通/三轴 RoPE 数学；评分范围、位置编码是否必需和梯度策略留在各算法。
- 六个 streaming Probe 共用产物发布函数，各自计算 aggregate/summary，保留文件名、
  schema、流式 raw 与专属 metadata；Taylor 独特的 scores/ranks 产物仍单独发布。
- Qwen3 诊断复用已有发布入口，六个 Qwen2.5 训练命令共用 grid/timing 记录，
  DyCoKe 复用生成报告并保留精确 decode 长度检查；各方法输入门槛继续独立。

2026-09-10 对上述改动执行相关 CPU/原生小模型验证，189 passed、0 skipped，涵盖
非恒等空间顺序、像素与 RoPE 数值/梯度、checkpoint backward、分析文件以及诊断失败和防覆盖。
本次未重跑完整测试、真实权重或 GPU 矩阵，不据此扩大支持声明。只增加普通函数，
未引入新执行框架、继承层次或配置开关。

### 执行、状态与注册收敛（2026-09-10 完成）

本轮以 `bd972a6` 为基线，代码与行为修复截至 `676f1f7`。已审查图像、视频、音频、
Operation、KV 及公共实验入口，完成实际重复的清理和代表流程验收。具体机制见
[执行参考](reference/compression_execution.zh.md)，入口与状态归属见
[架构说明](platform/architecture.zh.md)。

| 范围 | 完成的整理与保留的职责 |
| --- | --- |
| 图像 Encoder / Projector | 核对现有共享数学与 provider 边界；保留 ToMe dense restore、SAINT 逐层选择，以及 AnyRes、window、M-RoPE 和 DeepStack 的真实布局差异。 |
| 图像 LLM | 多方法复用 Method/Call、原生信号采集、布局和 checkpoint 重放；RESTORE 共用校准状态，SGL/Nüwa 复用选择机制。删除无消费者的逐方法包装；LLaVA FastV、PyramidDrop、PACT 的公开回调仍保留。 |
| 视频 | METok/PruneVid 六份 decoder 循环收敛为三份基座循环，方法持有各自评分与选择状态；Projector 继续复用已有执行。删除为诊断重复计算的 DeepStack 数值重放和自证标记。 |
| 音频 | LLM 层内 Attention→压缩→FFN 复用公共原生边界，前向与重算共用应用入口；删除临时零 MLP、重复归一化、无效转发和三段手写补齐。Encoder chunk、Projector 预计算与 LLM 稀疏位置分别保留。 |
| Operation / KV | 保持结构删除、模块替换、安装恢复和缓存读写的真实边界；删除 DyCoKe 恒等检查。已发布且有消费者的 Operation 导出合同、物理缓存归属校验继续保留。 |
| 参数与注册 | 12 个图像方法的重复参数白名单和默认值改为读取 Spec/Method；十组同结构的三基座注册改为直接声明表，所有图像 variant/binding 字段及顺序与改动前一致。 |
| 测试与诊断 | 保留独立数学、原生布局、梯度、重算、缓存与恢复验证；删除旧包装专属测试和重复布尔汇总。Omni GPU 配方共用正式配置与合成音频，未新增另一套测试体系。 |

训练/直接前向、生成 prefill 与增量 decode 继续区分；方法负责数学和本次决策，
provider 负责模型布局，checkpoint 保留必要的值重放，诊断记录不承担执行状态。
音频逐层语义标签仍供分析消费，不因它属于记录数据而删除。

按使用任务完成的代表验收如下；各指南沿实际调用链指向责任代码，不要求先阅读全部接口。

| 使用者 | 已验收的流程 |
| --- | --- |
| 压缩使用者 | YAML 配方、CLI/Python 实验入口、原生评测、结构化结果、分析和训练恢复。见[运行指南](guides/1_run_compression_experiment.zh.md)。 |
| 压缩研究者 | FirstK 独立方法复用已有音频执行；查询、计划、运行和压缩记录贯通。先确认执行接口，再注册；算法无需修改 runtime 或后端。见[开发指南](guides/2_dev_compression_method.zh.md)。 |
| 框架维护者 | 算法、provider、Session、原生后端与产物有明确责任入口；实际分析错误已沿媒体桥定位并修复。见[排错指南](guides/3_troubleshoot_runtime.zh.md)。 |
| Agent | 使用同一查询→计划→运行→比较入口及结构化失败结果，无需专用执行框架；非法比例在配置阶段定位且不创建运行目录。 |

公共流程使用真实 Qwen2.5-Omni-3B、BF16/eager 与本地 MMAU 一个样本：baseline 经
CLI、FirstK 经 Python 调用原生 lmms-eval 成功，音频 Token 250→62；查询、YAML
计划和结果比较的 CLI/API 输出一致，结果、Manifest 与逐请求记录齐全。远程同步目录
没有 Git 元数据，因此协议比较按既有合同返回 `matched: null` 和缺失的源码身份，
不能声称协议完全匹配；单样本结果也不是质量证据。

原生 Swift 以结构删层加 HTA 完成连续两步与恢复训练：只训练 HTA 与末层 norm，
两者均有实际参数更新；从第一步恢复到第二步的可训练参数、Adam、调度器、随机状态
与连续运行逐项精确一致。恢复推理保留 35 层，音频 25→1 并生成两个 Token。
验证复用[现有恢复诊断](../scripts/diagnostics/validate_qwen25_omni_swift_resume.py)，
范围限于该 full-tuner 冻结设置，不代表全参数收敛、LoRA 或 DeepSpeed 恢复。

压缩后表征分析修复了 MMAU 原生 `type/url` 媒体记录在 simple 桥中的接入问题，
共用媒体整理入口保留原任务对象。修复后 CLI 处理一个样本、零跳过，文本/音频两组
共十个有限统计值，Manifest invariant 通过；相关公共入口与后端合同 124 项通过。

图像执行的分批证据如下。计数按批记录，集合可能重叠，不加总为唯一测试数：

| 批次 | 原生小模型与适配器检查 | 双设备 | 真实 Qwen3-VL-2B 检查 |
| --- | --- | --- | --- |
| DART / D²Pruner | 两批 195、247 项通过，集合有重叠 | 12 项通过 | DART 序列 91→59；一次选择 |
| FEATHER / IVC-Prune | 两批 282、214 项通过 | 12 项通过 | 序列分别 91→43、91→59；分别两次、一次选择 |
| FinePruner | 239 项通过，包含相邻 Nüwa / SparseVLM 路径 | 3 项通过 | 序列 91→36；两次 IAC、一次 ARP |
| SparseVLM 与公共合并重放 | 主体 527 项通过；修复后补测 17 项通过，集合有重叠 | 6 项通过，包含 PACT | 序列 91→45；两次选择与回收合并 |
| PACT | 主体 253 项通过；补测 11 项及直接回调 3 项通过，集合有重叠 | 3 项通过 | 序列 91→28；一次聚类，后续层比例 Attention |
| RESTORE 校准状态 | 63 项通过；覆盖三基座生成、质量/掩码及相邻 Nüwa 路径 | 本批未新增 | 序列 91→43；decode 校准 28 层，各层物理缓存长度 44 |
| SGL target 选择 | 211 项通过，含相邻 Nüwa 路径；排序检查改在公共 kernel 观测 | 6 项通过 | 序列 91→59；guide 分数只消费一次，各层物理缓存长度 60 |
| Nüwa 第二阶段 | 主体 145 项通过；补测 99 项及自定义规则 7 项通过，集合有重叠 | 6 项通过 | 序列 91→43；两阶段各执行一次，各层物理缓存长度 44 |

图像真实权重检查得到有限非零视觉梯度，checkpoint 反向和两步缓存生成不重复选择。
SGL 使用同一 2B checkpoint 的两份独立模型验证 guide→target 传递，不作为大小模型
搭配的质量证据。旧包装清理的 369 项与 91 项检查不重叠；参数注册整理的 242 项覆盖
配置优先级、三基座入口和研究扩展，与此前集合重叠。

视频公共循环的 64 项相关检查和 1 项双 GPU 检查通过；真实 Qwen3-VL-2B 四帧样例
得到 METok 序列 167→39、PruneVid 167→55，有限非零视觉梯度、重算与生成通过，
28 层缓存分别为 40、56。METok 注入固定帧相似度，证据不覆盖 SigLIP。记录职责清理
的 280 项检查保留了独立数值和梯度预期。

音频边界的 411 项相关检查及 6 项双 GPU 检查通过；后续参数清理的 195 项包含上述
双 GPU 用例，集合重叠，覆盖两种 checkpoint、卸载、缓存生成及相邻图像/视频路径。
真实 Omni-3B 一秒音频中，
Audio-Pyramid 将 25 个音频 Token 压至 4 个，FastAdaSP 压至 13 个，均有有限非零
输入梯度，生成不重复选择。变长补齐与合并的 392 项检查另覆盖多音频区间、文本行、
位置与梯度，也包含上述双 GPU 用例；14 份旧 GPU 配方的正式配置解析不等于重跑
14 份真实模型实验。

本轮完成现有机制的重构与代表工程流程，不提升其他方法/组合的支持状态，不从代码
行数或诊断调用次数推断性能收益。目标规模质量与收敛实验、量化、动态 Operation、
通用运行时数量合同、联合音视频和长期 Agent 自改算法仍按各自 TODO 推进，未预建空实现。

## 既有 M2 真实压缩路径

VideoLLM 九算法迁移已完成 27/27 真实 checkpoint 推理与 27/27 最小训练/保存重载单元，
包括 DyCoKe 生成阶段的视觉 KV view pruning；54 个独立 video SupportUnit 已有精确证据。
G0–G7 已通过；最终安装环境聚合为 1842 passed / 1 skipped，独立审查及一次受影响复查
已完成，唯一 P2 非视频隔离问题已关闭。canonical 配置、本地辅助资源、测试派生 fixture 和
不复现论文质量/收敛/benchmark 的边界见 [VideoLLM 状态](support/video_compression/status.zh.md)。

完成标准：

1. 压缩动作真实减少目标模态 token 或修改目标 operation；
2. legalizer 修复模型要求的 grid/mask/cache/position 状态；
3. 压缩模型仍可生成，并由模型特定 invariant 验证；
4. baseline/compressed 使用同一评测协议产生可比较指标；
5. checkpoint、配置、版本、指标和关键 trace 进入可验证 Manifest。

## TODO

### 近期合入能力的后续验证

- [ ] 14 组新增对比方法：按精确方法/基座补齐真实 checkpoint 的 lmms-eval 与 Swift 训练闭环、目标权重校准和质量/效率对照；AdaLLaVA/Skip-Vision 另验证 adapter-only 参数保存恢复。CPU 原生测试不替代这些证据，Twig 与 Short-LVLM 的资产限制见接入说明。

- [ ] Bifocal：使用真实 Qwen2.5-VL checkpoint 完成压缩推理、一步 Swift 训练与保存重载，记录精确证据后再调整 SupportUnit；质量/效率实验单独报告。
- [ ] 分模态替换：补齐双基座 CUDA/BF16 与代表算子组合的真实验证，完成原生 Swift CLI 保存、恢复继续训练及恢复后评测；CPU bridge 测试不替代该闭环。
- [ ] MWOP（Mask 形式）：完成双基座真实 GPU 的代表 full/LoRA 训练、保存恢复和质量对照；稠密 Mask 的禁用比例不作为物理加速指标。

### P0：补齐音频质量与完整训练证据

- [ ] 使用同一 checkpoint 跑 MMAU baseline 与 50%/25%/10% Segmentwise Top-K，并完成质量/效率联合报告；
- [ ] 按论文默认参数完成 A-ToMe 评估，并验证完整训练收敛与质量；
- [ ] 完成 Segmentwise Top-K 与 A-ToMe 的评估/训练人工验收清单，保存可追溯 Manifest 和运行证据。
- [ ] 对 LTBM、SA-MAP、Encoder TopK 与 Progressive Pruning 完成目标规模 MMAU、完整 Swift checkpoint/Manifest 和质量/效率实验。
- [ ] 对 Audio-Pyramid 与 SPADE Operation-Only 完成目标规模 MMAU、完整 Swift checkpoint/Manifest 和质量/效率实验；SPADE 当前范围不含 WLI 分析、蒸馏、head/FFN pruning 或 Talker。

### P1：扩展压缩能力

- [ ] 按方法补齐音频候选的目标规模 generate、训练恢复和质量/效率证据；复用已有独立数值、原生小模型 backward/重放及 HTA 保存重载测试，缺少哪类证据就补哪类。工程支持与论文质量收益分别声明，不因单个组合通过自动晋升全部候选；

- [ ] 将 Qwen2.5-Omni Projector/Encoder Token 执行族的 `position_policy` 从当前固定的 `compact` 扩展为可配置策略，并至少实现、验证 `preserve_absolute`：Segmentwise Top-K 保留所选 Token 的原始 M-RoPE 位置；A-ToMe 明确定义合并 Token 的代表位置；同时覆盖后续文本位置、batch padding、eval/generate/train、Swift template bridge、legalizer、Controller 和 backbone-specific invariant。Audio-Pyramid 的 LLM Token 路径已经独立实现 `preserve_absolute`，不属于此 TODO；Projector/Encoder 路径完成前不得宣称支持该策略；
- [ ] 扩展多 stage 组合前，定义跨 stage 的 Token 预算、位置重建和逐层证据约束；

### 稳定后清理

- [ ] eval/train 真实入口稳定后，处理 `eval --dry_run` 与 `--plan_only` 的同入口别名；保留一个明确的启动计划名称，并同步该公开参数的用户说明；
- [ ] 清理已闭环的临时 trace、benchmark 差分脚本和历史运行报告，只保留可复现命令、固定 fixture 与必要 Manifest 证据；
- [ ] 需要扩大常规模态训练覆盖时，再分别实跑 video/audio-video 和 LoRA，不用配置存在替代运行证据。
