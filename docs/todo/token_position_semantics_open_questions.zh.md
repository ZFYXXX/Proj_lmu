# Token 压缩的位置语义待确认事项

> 类型：通用架构 TODO / 开放设计问题
>
> 状态：Deferred，当前不实施
>
> 记录日期：2026-08-13；范围更新：2026-09-10
>
> 启动条件：已有代表方法验证；当具体方法或组合需要现有 Binding 无法表达的位置语义时再推进

## 1. 为什么记录这项 TODO

本问题最初来自 Qwen2.5-Omni 八种 active 音频方法中的两种 Token 位置策略：

- LLM 之前的六种 Encoder/Projector Token 方法使用 `compact`；
- LLM 内部的 Audio-Pyramid 使用 `preserve_absolute`；
- SPADE 只改变 LLM 层结构，不改变 Token 布局，因此不涉及 Token 位置策略。

这两种行为当前都已经有明确实现，但它们是否应该成为通用多模态压缩框架的最终语义，
目前尚未确认。特别是，“LLM 前的位置”和“LLM 内的位置”可以放在统一框架中讨论，
但不能在没有定义坐标类型的情况下都称为 `position_id`。

三条代表路径及 Omni v2 收敛已完成验证，见
[Qwen2.5-Omni 状态](../support/qwen25_omni/status.zh.md)。下面保留其位置差异与尚未解决的
通用问题，不把已完成的人工 Debug 列为前置待办，也不预先决定统一方案。

## 2. 必须区分的四种坐标

后续设计至少需要明确区分：

1. **来源坐标（source coordinate）**：Token 对应原始音频、图像或视频中的时间/空间位置；
2. **物理布局槽位（layout slot）**：Token 当前在压紧 Tensor 中的下标；
3. **模型语义位置（model position）**：交给 LLM RoPE、M-RoPE 或其他位置编码的位置；
4. **缓存位置（cache position）**：KV cache 中的写入和读取槽位。

它们可能相同，也可能不同。例如 Audio-Pyramid 压缩后可以出现：

```text
物理布局槽位：0, 1, 2
模型 M-RoPE：10, 12, 14
```

因此，“Tensor 已经 compact”不能自动推出“模型位置已经重新编号”；反过来，保留模型绝对
位置也不代表必须在 Tensor 或 KV cache 中保留空洞。

## 3. 当前实现的已确认行为

### 3.1 LLM 之前的六种方法

当前六种方法是：

- A-ToME；
- Encoder Attention Top-K；
- Progressive Pruning；
- Segmentwise Top-K；
- LTBM；
- SA-MAP。

它们均使用 `position_policy="compact"`。当前 Qwen Binding 会压紧 Audio placeholder、
mask、label 和相关序列字段，移除已有 `position_ids`，再让原生 `get_rope_index()` 按压缩后
的 Audio 长度重建 M-RoPE。

这些方法仍然保持输出的时间先后顺序，但不保留被删除 Token 造成的位置空洞：

```text
来源 Token： A  B  C  D  E
保留/合并后：A     C     E
进入 LLM 后：0     1     2   # 示意；实际是 Qwen 多轴 M-RoPE
```

对于 A-ToME、LTBM 和 SA-MAP，一个结果 Token 还可能对应多个来源 Token。此时“保留哪个
绝对位置”没有天然唯一答案，可能需要左端点、中心、加权中心或其他显式坐标聚合规则。
当前实现没有选择这些规则，而是使用 compact 布局。

### 3.2 LLM 内部的 Audio-Pyramid

Audio-Pyramid 在若干 LLM 层已经处理完 prompt 后才删除 Audio Token。此时 hidden state
已经使用了先前生成的 M-RoPE。当前实现使用 `position_policy="preserve_absolute"`：物理
Tensor 会被压紧，但保留 Token 继续 gather 进入该层时对应的 `position_ids` / position
embeddings。

```text
压缩前模型位置：10, 11, 12, 13, 14
保留 Token：     10,     12,     14
压缩后模型位置：10,     12,     14
压缩后物理槽位： 0,      1,      2
```

这避免了同一个 hidden state 在前半段 LLM 使用位置 12、后半段突然被解释成位置 11 的
层间位置语义跳变。`cache_position` 仍可根据物理 cache 布局重建，不应与保留的语义
M-RoPE 位置混为一谈。

### 3.3 多阶段组合的当前含义

如果先执行 Segmentwise-TopK，再执行 Audio-Pyramid：

1. Segmentwise-TopK 先建立一套 compact 的 LLM 输入位置；
2. Audio-Pyramid 随后保留的是这套 LLM 输入位置；
3. 它不会恢复到原始音频帧或原始 Projector Token 的来源坐标。

因此，当前的 `preserve_absolute` 实际含义是“保留进入该 LLM 压缩阶段时的模型位置”，
不是“始终保留原始模态坐标”。

## 4. 为什么当前不能直接统一成一个 `position_id`

LLM 前的压缩发生时，Audio feature 尚未最终绑定到 Thinker LLM 的位置语义。压缩后可以
先确定最终 placeholder 布局，再从 LLM 第 0 层开始使用一套一致的新 M-RoPE。

LLM 内的压缩则发生在已有 LLM 层已经消费位置编码之后。若在中间层直接重新编号，后续层
看到的位置基准会与前序层不同。这不是绝对禁止的算法，但它应被视为显式的“中途位置重置”
变体，并单独验证，而不是被当作无语义影响的 Tensor compact。

所以真正需要统一的是位置变化的描述方式，而不是强迫所有阶段采用同一种位置执行策略。

## 5. 后续需要确认的问题

在启动重构前，需要逐项回答以下问题：

1. 通用状态是否应同时保存 source coordinate、layout slot、model position 和 cache position？
2. `LayoutDelta` 只描述物理布局，还是同时携带来源坐标与模型位置变化？
3. 是否需要独立的 `PositionDelta` / position policy，避免把位置变化塞进 `LayoutDelta`？
4. `compact` 和 `preserve_absolute` 应属于算法变体、Backbone Binding，还是二者共同决定的 SupportUnit？
5. merge 方法的模型位置应使用左端点、中心、加权中心、连续值，还是统一重新编号？
6. LLM 前保留稀疏来源坐标时，后续文本 Token 的位置应该保持、前移还是重新生成？
7. LLM 中途重新编号是否有值得支持的算法场景？若支持，如何避免层间位置语义不连续？
8. 多阶段压缩组合时，`preserve_absolute` 应相对于原始模态、LLM 初始输入，还是当前 stage 输入？
9. Audio、Image、Video 的时间/空间坐标能否共用一个抽象，还是需要模态专属坐标结构？
10. Qwen M-RoPE、LLaVA 位置/AnyRes、Qwen3-VL DeepStack 等 Backbone 特性分别要求哪些独立 Binding 规则？
11. prefill、decode、KV eviction 和 cache compaction 中，model position 与 cache position 如何分别传播？

## 6. 暂定工作原则，而非最终方案

在上述问题确认前，后续代码不应：

- 把当前六种方法的 `compact` 宣称为所有 LLM 前压缩的通用规则；
- 把 Audio-Pyramid 的 `preserve_absolute` 宣称为所有 LLM 内压缩的通用规则；
- 用一个含义不明的 `position_id` 同时代表来源坐标、Tensor 下标、M-RoPE 和 cache slot；
- 因为 Tensor shape 已压紧，就默认语义位置也应重新编号；
- 因为保留绝对 M-RoPE，就在 KV cache 中保留无意义的物理空洞。

当前更稳妥的原则是：

> 统一坐标词汇、映射与审计方式；位置执行策略由算法阶段和 Backbone Binding 明确决定，
> 在证据不足时不强制统一成 `compact` 或 `preserve_absolute`。

具体方法若需要新的位置策略，应根据它的数学与原生模型行为验证，不能仅凭这些代表路径推导。

## 7. 新位置语义所需的验证材料

启动具体位置改动时，先复用已有记录；仅补充该改动缺少的观察，参考：

- A-ToME：来源 pair、逐层物理布局、pooler 后长度、placeholder 与最终 M-RoPE；
- Segmentwise-TopK：selected source indices、compact slots、placeholder 与最终 M-RoPE；
- Audio-Pyramid：每个 stage 的 entry/exit absolute positions、物理宽度、position embeddings、
  attention mask、prefill/decode 与 cache position；
- 一次 `Segmentwise-TopK → Audio-Pyramid` 组合推演，明确两次位置策略的参照系；
- 至少一个 merge 方法的来源组映射，确认未来是否确实需要代表位置或连续坐标。

这些材料用于回答设计问题，不要求现在新增一轮发布验收。

## 8. 相关文档

- [Token 数量合同与运行时决策重构](token_compression_cardinality_execution_model.zh.md)
- [Omni 向多模态 v2 收敛历史记录](../archive/omni_compression_v2_convergence.zh.md)
- [开发压缩方法指南](../guides/2_dev_compression_method.zh.md)
- [Audio-Pyramid 架构与 Debug 手册](../support/qwen25_omni/audio_compression/debugging/llm_token_audio_pyramid.zh.md)
- [Encoder-Level 架构与 Debug 手册](../support/qwen25_omni/audio_compression/debugging/encoder_level.zh.md)
- [Projector-Level 架构与 Debug 手册](../support/qwen25_omni/audio_compression/debugging/projector_level.zh.md)
