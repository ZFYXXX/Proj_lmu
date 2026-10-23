# LLM-Level 压缩

本目录提供六个插件实现，分属 LLM Token 与 LLM Operation 两个执行族；catalog 生命周期
为四个 active、两个 target。插件可包含多个 variant，表格不代表全框架 LLM-Level 方法总数：

| Plugin | 执行族 | 状态 | 执行阶段 | 当前范围 |
| --- | --- | --- | --- | --- |
| `audio_pyramid_attention_pruning` | `llm_token` | active | `generate` / `forward` | Thinker Prefill 中按层进行 audio attention pruning，Decode 不重复选择，并保留绝对 M-RoPE 位置 |
| `fastadasp_weighted_token_merging` | `llm_token` | target | `generate` / `forward` | 在 decoder attention residual 后、FFN 前按相邻 Key 相似度与全 prompt 累计注意力执行加权音频 Token 合并；支持 sparse/constant/decay 调度 |
| `headrouter_dynamic_audio_pruning` | `llm_token` | target | `generate` / `forward` | 使用无 RoPE QK probe、逐 head 选择性和 Gaussian profile routing，在单个 Prefill 边界裁剪 audio Token |
| `structured_llm_layer_pruning` | `llm_operation` | active | `prepare` | SPADE Operation-Only：按显式或预计算索引删除完整 Thinker LLM 层，并导出可重载的 Transformers checkpoint |
| `qwen3_vl_attention_replacement` | `llm_operation` | active | `prepare` | 共享 Qwen3-VL 与分模态 Qwen2.5-VL/Qwen3-VL 的 6 种保持 I/O shape 的 Attention replacement |
| `qwen3_vl_ffn_replacement` | `llm_operation` | active | `prepare` | 共享 Qwen3-VL 与分模态 Qwen2.5-VL/Qwen3-VL 的 low-rank、identity 或 two-layer MLP FFN replacement |

Audio-Pyramid、HeadRouter 与 FastAdaSP 复用动态 LLM Token controller；FastAdaSP 使用论文要求的 attention 后、FFN 前合并边界。FastAdaSP 和 HeadRouter 均有 eval/train 候选入口，SupportUnit 保持 `declared`，精确验证范围见[候选方法状态](../../../docs/support/qwen25_omni/audio_compression/candidate_methods.zh.md)。HeadRouter 默认内置官方公开的 Qwen2.5-Omni-3B layer-1 profile，其他 checkpoint 必须提供匹配 head 数的自定义 profile。

结构层剪枝在 runner 和 optimizer 创建前由 Qwen-specific `OperationApplier` 执行；评估和训练看到的是同一个已改写模型，导出 artifact 明确限定为 `thinker_only`。两个 replacement 插件也在 prepare 阶段安装。`.shared_llm` 在 Qwen3-VL 上由 runtime
session 管理可逆替换，从原始 checkpoint 加载后按配置重新应用，不导出独立 replacement
checkpoint；`.image_llm` 在 Qwen2.5-VL/Qwen3-VL 上联合编译图像/文本分支，并提供
专用保存恢复，见[分模态说明](../../modality_split/README.md)。两类精确 prepare 单元均为 `declared`。

Audio-Pyramid 与 FastAdaSP 的 Q/K 旋转共同使用 [`_rotary.py`](./_rotary.py)，支持普通 RoPE 与 Qwen 三轴 M-RoPE。辅助函数不决定评分范围或梯度：Audio-Pyramid 在 `no_grad` 下可选地应用位置编码；FastAdaSP 要求原生位置编码，继续使用全 prompt 注意力和相邻 Key 相似度。

Omni SPADE 仍只覆盖 `llm + layer + prune + structural_rewrite + remove_module`；其 WLI
分析、自动选择、蒸馏和 head/FFN pruning 未激活。全框架另外接入了
[MWOP Mask](../../../docs/guides/fixed_mask.zh.md)和[14 组对比方法](../../../docs/support/imagellm/comparison_methods.zh.md)
中的 Token 与 Operation 路径，它们在其他目录维护，不能用本目录清单概括全部能力。
任意通用组合、expert 剪枝和 Talker Operation 仍未提供；各方法的质量与效率须独立验证。

Qwen2.5-Omni 配置位于 `configs/eval/lmms_eval/qwen25_omni_aud_audio_pyramid.yaml`、`configs/eval/lmms_eval/qwen25_omni_aud_fastadasp.yaml`、`configs/eval/lmms_eval/qwen25_omni_aud_headrouter.yaml`、`configs/eval/lmms_eval/qwen25_omni_spade_operation_only.yaml` 及对应的 `configs/train/qwen2_5_omni/` 训练配置；Qwen3-VL 的九份 replacement 配置位于 `configs/eval/lmms_eval/qwen3_vl_*.yaml`，精确策略、诊断命令与 reload 边界见 [`Qwen3-VL backbone README`](../../backbones/qwen3_vl/README.md)。Audio-Pyramid 的逐层布局、prefill/decode 与最小排错见 [`LLM-Token 架构与 Debug 手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/llm_token_audio_pyramid.zh.md)，FastAdaSP 的论文/官方映射与调度边界见 [`FastAdaSP 方法卡`](../../../docs/support/qwen25_omni/audio_compression/fastadasp.zh.md)，HeadRouter 的论文/官方映射和 profile 边界见 [`HeadRouter 方法卡`](../../../docs/support/qwen25_omni/audio_compression/headrouter.zh.md)；SPADE 的 prepare、结构物化与 export/reload 见 [`LLM-Operation 架构与 Debug 手册`](../../../docs/support/qwen25_omni/audio_compression/debugging/llm_operation_spade.zh.md)。当前执行族总览见 [`Qwen2.5-Omni 音频压缩方法扩展指南`](../../../docs/support/qwen25_omni/audio_compression/extension.zh.md)，正式新增通用方法以 [`开发压缩方法指南`](../../../docs/guides/2_dev_compression_method.zh.md) 为准。
