# 具身设计笔记

> 类型：Future design
> 当前状态：Reserved；非当前规范或已激活能力

这些内容刻意放在主设计之外，因为 embodied VLN/VLA 不属于 Phase 1a。当前只预留 `brain_only` compression 边界：被压缩组件是多模态 LLM brain，cerebellum/runtime 由具身合作者侧固定。本文件不表示 runtime 已实现或已激活。

## 范围

```text
compression_scope = brain_only
```

运行时约束：

- brain-only compression 不能访问 cerebellum module path；
- cerebellum 只由 embodied runtime 侧持有；
- 当前 active scope 不暴露 action-head compression；
- compressed-brain 实验必须固定 cerebellum、coordinator、simulator seed 和 episode list，才能做归因。

## `EmbodiedSystem` 接口

```python
@dataclass
class EmbodiedSystem:
    brain: Any
    cerebellum: Any
    coordinator: "EmbodiedCoordinator"
    runtime: Any
    compression_scope: Literal["brain_only"]
```

## `EmbodiedReproducibilityManifest` 可复现记录

闭环运行应输出：

- simulator name 和 version；
- simulator seed；
- environment asset hash；
- episode ids；
- start poses；
- model checkpoint id；
- model decoding config 和 seed；
- cerebellum version 和 seed；
- coordinator version；
- per-step trace path；
- trajectory artifact path。

## 闭环归因协议

每个 compressed-brain run 应配对：

```text
baseline brain
same cerebellum
same coordinator
same seed
same episode list
```

每步 trace 应记录：

- decision 来自 brain、cerebellum 还是 coordinator；
- brain output；
- cerebellum output；
- executed action；
- failure category，如有；
- 该 step 的 compression stats。

如果同一 cerebellum 下 baseline 跨 seed 不稳定，则 compressed-brain 结果标记为 low-confidence。
