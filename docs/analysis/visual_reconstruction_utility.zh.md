# 重构残差与任务效用：2026-10-20

目的：验证“全局摘要难以重构的信息是否值得优先保留”，同时区分空间频率、特征重构误差与答案贡献。当前为不训练的诊断代理，不复现或否定完整的 ReConV，不预设残差必然对应 OCR、语义或正交信息。

## 样本与冻结

从本地 ChartQA、TextVQA 各选 500 张独立图像，排除上一轮 `visual_evidence_interaction/20261020_v1/cohort.json` 的全部 1000 个图像内容哈希，并做跨任务去重。选择只读图像，不查看问题答案及模型表现。每任务 100 张开发、400 张确认；开发中的前 8 张为 smoke，不增加独立样本量。

配置保持 Qwen2.5-VL-3B 原 checkpoint/revision、BF16/eager、原生动态分辨率、原生 prompt 和评分。ChartQA greedy 最多 16 Token，TextVQA 最多 32。先通过 16 张真实模型 smoke，再跑 200 张开发并冻结，最后运行 800 张确认；确认全部完成才检验四项主要假设。

Smoke 中 dense 原生生成必须与诊断 cached 生成逐 Token 完全一致。额外的 cached/cache-free 比较若出现不同 greedy 输出，必须记录并做同一生成前缀下的逐步 logits 检查：BF16 KL≤.01、候选 top1 regret≤.25，且将相同权重精确转换 FP32 后 KL≤1e-6、最大 logits 差≤.002。之后精确恢复 BF16 权重和原始 buffer（特别是 FP32 M-RoPE inv_freq）。此规则在 smoke 调试中、开发和确认运行前固定；不以“结果接近”跳过位置或 cache 验收。

## 每图 43 个条件

- Dense、Feature-anchor 均值/原始代表 Token（两者使用相同原始位置）、同一均值改为重心位置、PruMerge+ K128，共 5 个参照。
- 共同 96 个网格摘要的单独输出（K96，仅解释预算增量）。
- 固定 K128：96 个共同网格均值摘要＋32 个原始 Token。11 种选择：局部重构残差高/低、问题文本 Layer-0 attention、rank-8 PCA 投影残差、特征多样性，以及各三 seed 的随机/空间匹配随机。所有选择共享摘要内容与原始代表位置；新增原始 Token 使用原始位置，整序列按源位置排列。空间匹配随机保持与高残差条件完全相同的每网格单元新增数量。
- 10 个等能量频率条件：在实际 H×W post-merger 特征网格上做正交 DCT，保持 DC，仅对相同数量的低频、高频、三 seed 随机 AC 系数衰减。按扰动向量平方范数匹配能量，两档剂量保持 1:2；受最弱频带能量约束时同步减小。记录实际剂量、BF16 量化后的能量误差（上限 3%），不把“扰动能量”误称为“移除的特征能量”。这些条件保持原始长度和位置，不作为实际压缩方法排名。
- 16 个局部条件：将每个矩形空间分区内部替换为该区均值，长度、位置、mask 不变，计算答案 NLL、生成得分、teacher KL。每图以 16 次干预计算评分与损害的 Spearman，再按图像汇总；常数相关性明确报告为未定义。

## 统计与解释

主要终点为完整生成的任务得分。确认检验为每任务的高残差减随机、高残差减空间匹配随机；三 seed 先在图内平均，再做 10,000 次图像级配对 bootstrap。四检验 Holm 校正，支持要求校正 p<.05 且 95% CI 下界>0。另报告实际意义门槛：平均收益至少 2 个百分点，未达到不能宣称获得稳定实用优势。

频率干预、局部相关性、PCA/attention/多样性比较、均值与代表 Token、位置消融均作为次要解释证据。报告 baseline 零/部分/满分子组以及压缩失败组，但不按结果剔除图像。预算、原始网格大小、实际剂量、时间、显存、原生任务得分及严格 ChartQA 匹配保存在逐条件记录。

若代理没有稳定任务收益，结论仅限于该代理。下一步可执行用户提出的最小可学习组合，配套同数据/步数/参数预算的简单池化适配与重构路由对照；不在当前确认运行中修改方法。

## 执行

```bash
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/analysis/prepare_reconstruction_utility.py \
  --output outputs/analysis/visual_reconstruction_utility/20261020_v1 \
  --previous outputs/analysis/visual_evidence_interaction/20261020_v1/cohort.json
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/analysis/run_reconstruction_utility.py \
  --root outputs/analysis/visual_reconstruction_utility/20261020_v1 --launch
```

运行器后台双 GPU 各单进程，按图像分片。只管理自己创建的子进程；任何数量、位置、生成、非有限值或冻结哈希错误停止本轮子进程并保留错误。JSONL 逐条件 fsync，可从同配置和同实现断点恢复。结果路径包括 cohort、preregistration、run_manifest、frozen、逐图 conditions/selection_plan、开发与确认报告。
