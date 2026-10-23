# Qwen2.5-VL 可学习组合：单 seed 小数据实验

本轮是用户提出的组合方向的最小可训练版本。依据 [原文和官方代码核查](learned_hybrid_sources.zh.md)，实现记录与完整论文方案的差异；保留既有无训练实验结果。

## 实验组与预算

| 组 | K128 构成 | 目的 |
|---|---|---|
| pool_adapter | 128 个网格均值＋零初始化残差适配器 | 学习对齐的简单控制 |
| summary_random | 64 个学习摘要＋64 个随机原始 Token | 参数量匹配控制 |
| summary_residual | 64 个学习摘要＋64 个重构残差原始 Token | 检验残差路由 |
| spectral_residual | 频域初始化的64个学习摘要＋64个残差原始 Token | 最小频域＋重构组合 |

学习摘要通过窄维多头跨注意力更新；位置条件解码器重构原始特征，InfoNCE 提供监督。残差采用归一化正交残差比例，选中项的门控前向恒为1，反向为有偏 STE。摘要和原始 Token 按源位置排列，使用原始代表 M-RoPE；文本位置完全不变。修改位于 native vision merger 后、LLM 输入前，支持真正压缩后的 cache decode。

后三组参数数量及初始化完全相同；pool 参数更少，报告实际数量。四组固定训练数据、顺序、LR、step 和 K，不预设复杂组必然领先。

## 训练与数据

- 原权重 Qwen2.5-VL-3B-Instruct、revision `1b5a7674d40a0ac9acef541a46fed12d710bd36b`，BF16 eager、batch1、动态分辨率 `[200704,1605632]`。冻结 vision/native merger/LLM，仅新模块参数进入 FP32 AdamW。
- TextVQA train 2048 张＋ScienceQA-IMG train 512 张，2560 张独立图像，每图一道固定问题。训练选择只读图像；排除全部 ChartQA test 和 TextVQA validation 图像（4675 个内容哈希），训练数据之间也去重。没有 ChartQA 训练监督，ChartQA 是迁移评估。
- 单 seed42，320 optimizer steps，梯度累积8，等于一遍2560图像。LR2e-4，warmup16，cosine，weight decay .01，梯度裁剪1。
- `CE + .5*teacher_KL + .1*InfoNCE`，pool 无重构项。教师和学生 logits 只计算答案预测位置，选择器不读取文本答案。
- 每16step和首末step保存 checkpoint；保存优化器、scheduler、RNG、micro index、配置和代码哈希。首step强制重新创建模块、读取权重和optimizer继续训练，并验证重载输出完全一致。仅用最终step320，避免按评测分数选checkpoint。

## 自动执行与验收

双GPU分两轮：pool＋spectral；random＋residual。先对四组各2step真实训练 smoke，验收 native dense生成一致、严格K128、非零有限梯度、主干无梯度、保存恢复一致。随后冻结代码/config/数据哈希并训练四组。任何子进程失败只停止本实验的兄弟进程，保留日志；不影响服务器其他任务。

训练完成后自动运行相同的16图生成smoke、200图开发、800图留出评测。生成使用原生ChartQA relaxed/strict和TextVQA完整答案软得分，保持16/32 greedy生成预算。逐图保存预测、Token数、压缩成员、位置和mask检查、NLL、teacher KL及时间。

留出集比较预设三项对照在两任务上的六个效应，10000次图像级paired bootstrap及Holm校正。单seed区间不代表训练seed波动；这些图像已经在先前实验中评测过，称为训练留出评测，不称全新独立确认。历史Feature-anchor/PruMerge+结果仅在同图dense预测和原始Token数全部一致时加入。

```bash
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/train/qwen25_vl/run_learned_hybrid.py \
  --root outputs/train/qwen25_vl/learned_hybrid_k128_20261021_v1 --launch
```

结果目录：`outputs/train/qwen25_vl/learned_hybrid_k128_20261021_v1/`，包括 training_samples、cohort、preregistration、frozen、run_manifest、models、logs、逐图条件和最终报告。
