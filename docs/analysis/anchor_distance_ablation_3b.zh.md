# 3B 锚点几何对照：只发现信号，不改变 7B 训练

实验根目录：`outputs/analysis/anchor_distance_ablation/3b_k128_n500_20261021_v2/`。
原 200 张队列尚未运行，已停止并标记为 superseded，保留原协议记录。
Qwen2.5-VL-3B、BF16、eager、原生动态分辨率；固定 K128，不使用任何 MLP。
所有组都输出原始未归一化特征的 FP32 簇均值，再转 BF16；使用同一簇重心 M-RoPE 规则，文本位置不变。

## 对照与可回答的问题

| 对照 | 锚点选择 / 分配 | 目的 |
|---|---|---|
| Dense | 不压缩 | 本轮重新计算的同模型上界 |
| Cosine farthest | 单位特征、最远优先、余弦分配 | 与原部署分组和值逐样本完全一致 |
| Unit L2 farthest | 单位特征、平方欧氏最远优先、欧氏分配 | 数学等价性检查；记录浮点近并列引起的差别 |
| Raw L2 farthest | 原始特征、平方欧氏最远优先、欧氏分配 | 模长是否包含有用信息 |
| Raw select / cosine assign | 原始欧氏选择，归一化余弦分配 | 区分锚点选择与分配的影响 |
| K-means++ one-pass | 单位特征、D² 采样，只分配一次 | 改变初始化选择目标的效果 |
| K-means 10 | 与上组相同初始化，最多 10 次普通欧氏 Lloyd 更新 | 进一步降低特征重构误差是否改善答案 |

所有条件的首锚点均固定为源 Token 0。这里是**固定首点的 K-means++ 对照**，不是随机首点的标准完整初始化。
两种 K-means 组均使用种子 11/23/47；中心更新后不重新单位化，不能混称 spherical K-means。
重复特征导致空簇时，确定性地从非单元素簇中拆出最大残差 Token；记录修复次数。

## 数据与统计

共 500 张独立图像：ChartQA / TextVQA 各 250 张，跨任务图像内容哈希互不重复。
每任务保留 100 张既有开发图像，再从原来的 400 张留出图像中按固定 seed 42 的哈希顺序选 150 张，不查看答案、预测或模型分数。
分别报告开发集 200 张、增补复核集 300 张，以及汇总 500 张的结果。
这些图像在历史实验中已经使用，因此是探索/复核，不是全新的独立确认。
16 张 smoke 只检查实现；它们已包含在开发图像中，不重复计入独立样本数。
每图 11 个条件；随机初始化种子先在图内平均，再对独立图像做 10,000 次配对 bootstrap。
ChartQA relaxed / strict、TextVQA 全部人工答案软得分；NLL 为解释指标。

记录锚点重合、完整分簇、原始和归一化 SSE、模长统计、位置/标签/缓存检查、诊断压缩耗时和 decoder prefill GPU 时间。
运行环境有其他任务，时间只是探索性测量；诊断压缩实现同时计算统计量，不是优化部署的 kernel，不声称端到端加速。
若重构误差下降但任务得分没有改善，不认定几何目标有效；若区间较宽且跨零，报告证据不足。
未归一化欧氏获得收益也不代表“模长=重要性”，后续需用模长打乱、尺度扰动验证。

## 执行

为避免争抢 7B 训练/评测 GPU，后台队列先等 7B 整体完成，再执行 3B smoke 和开发集。
通过独立 Session 入口复用现有原生任务加载/评分，不改动 7B 冻结中的 registry 或历史 checkpoint 代码。

```bash
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/analysis/qwen25_vl/run_anchor_distance_ablation.py \
  --root outputs/analysis/anchor_distance_ablation/3b_k128_n500_20261021_v2 \
  --source-run outputs/train/qwen25_vl/anchor_mlp_7b_k128_ddp_20261021_v1 --device cuda:1 --launch
```

产物：`protocol.json`、`cohort.json`、`run_manifest.json`、逐样本 `conditions.jsonl` / `summary.json`、失败记录，以及 smoke / development / confirmation（增补复核）/ all（500 张汇总）报告。
