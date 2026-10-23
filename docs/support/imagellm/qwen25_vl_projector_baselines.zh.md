# Qwen2.5-VL Projector 压缩对照候选

本页盘点当前仓库中可在 Qwen2.5-VL 图像路径比较的已有压缩器。它们是算法基线，
不是本研究的可训练统计 Projector；除非单独运行同一数据、同一 K 的实验，代码存在
或既有一步训练证据都不能代替本实验中的质量结论。

## 候选方法

| 方法 | Qwen2.5-VL 边界/机制 | 对本研究的比较价值 | 限制 |
| --- | --- | --- | --- |
| AIM | vision merger 后；特征相似 bipartite merge，可微地聚合 Token 值 | 代表相似度驱动的合并基线，可与固定规则/锚点分组同预算对照 | kernel 本身没有训练参数；不显式输入质量、位置或离散度 |
| PruMerge | vision merger 后；视觉接收注意力选中心，Key 相似度分配邻居 | 检查注意力质量和簇分配是否比纯特征锚点更有效 | 聚合规则含非归一化注意力和全局残差 Token，须报告实际输出数并统一到 K |
| VisionZip | vision merger 后；主导 Token + 上下文锚点分组 | 最贴近非均匀分组与质量/空间覆盖问题 | 配额分配和位置代表策略与本文固定 K 分组不完全相同 |
| ToMe | Qwen2.5-VL encoder/merger 边界；bipartite feature merge | 常用特征相似合并参照；仓库有 Qwen2.5-VL train-step 配方 | 执行边界在 encoder 内，不能当成 post-merger 固定分组的同位消融 |
| GAP / DivPrune | vision merger 后；按视觉显著度或特征多样性选择并丢弃 | 作为“只选择、不聚合”的控制组 | 不保留每个源 Token 的簇质量，需与合并组分开解释 |
| Bifocal | vision merger 后；coarse/fine 互补合并 | 可比较显式空间粗细结构路线 | 有独立复杂度和 ViT 统计；不是仅替换一个簇统计分支 |
| ZOO-Prune | vision merger 后；原生 merger 低秩敏感度代理评分后选择 | 对照 Projector 输出敏感度驱动的 Token 保留 | 是训练自由选择器，不是重建簇均值的学习型 Projector |

对应核心实现在 [`mllm_comp/compressors/image/`](../../../mllm_comp/compressors/image/)，
Qwen2.5-VL 物理合同在
[`contracts/qwen25_vl.py`](../../../mllm_comp/compressors/image/contracts/qwen25_vl.py)。
平台级能力边界和历史一步训练说明见
[`status.zh.md`](status.zh.md)；Qwen2.5-VL 的 ToMe 和 Bifocal 一步训练配方分别在
[`qwen25_vl_tome_train_step.yaml`](../../../configs/train/imagellm/qwen25_vl_tome_train_step.yaml)
和
[`qwen25_vl_bifocal_train_step.yaml`](../../../configs/train/imagellm/qwen25_vl_bifocal_train_step.yaml)。

## 本实验的对照顺序

1. 在固定 K=64 上先比较固定 grid、feature-anchor 与 AIM/PruMerge/VisionZip 的实际
   分组/聚合；GAP 或 DivPrune 作为纯选择控制。全部按 Qwen 原生 M-RoPE、同一文本
   前缀与同一 Layer-0 查询测量，不把不同边界方法伪装成同一分组。
2. Stage 2 的核心学习组仍按研究设计设定：Mean Only、Mass+Spatial Span、再加
   dispersion/residual 分支。现有免训练压缩器只用于外部算法参照，不替代等参数量控制。
3. 所有任务质量对照使用相同本地样本清单、提示、生成参数和输出预算；单图数或最终
   Token 数不一致的运行不进入主表。

当前阶段已通过 Layer-0 的 1,200 样本归因门槛；AIM/PruMerge/VisionZip 等尚未在这批
数据上完成同预算实测，因此这里只登记为候选，不报告优劣结论。

## Layer-0 一次压缩轨迹先导（16 张）

新增的冻结解码器轨迹 pilot 在 TextVQA、ChartQA、GQA、POPE 各取 4 张，将 merger
后的视觉序列分别按 grid 与 feature-anchor 压到 64，再比较相同文本位置上的教师/压缩
模型状态与 logits。Pilot 门槛通过：16/16 成功、0 跳过、压缩后文本 M-RoPE 三轴位置
精确保持、目标 Token 数正确、无非有限值。

按四个任务等权平均，feature-anchor 的教师文本 KL 为 0.245，grid 为 0.432；文本位置
预测 top-1 一致率分别为 88.8% 和 85.5%。这是小样本的 logits/state fidelity 诊断，
不是任务答案准确率，也不能证明任一方法已达到部署质量。逐层 drift 与样本明细见
`outputs/analysis/visual_merge_attribution/layer0_trajectory_pilot16_20261017_v3/`。

当前可说的有限结论是：Layer-0 的非均匀 feature-anchor 分组值得继续验证，尤其在
GQA、POPE 和 TextVQA pilot 上；不能据此宣称 Projector 学习方法已验证。下一步按既定
计划先完成 decoder 层 `{2,4,8,16}` 的 300 张/任务归因，再扩大真实压缩轨迹并与上述
仓库压缩器做同预算对照。
