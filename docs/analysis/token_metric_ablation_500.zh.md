# 同一 500 张图像：第二轮度量消融

运行根目录：`outputs/analysis/token_metric_ablation/3b_k128_n500_20261021_v1/`。
沿用上一轮 500 张的完整 cohort，字节哈希相同：ChartQA / TextVQA 各 250 张独立图像。
仍为 Qwen2.5-VL-3B、K128、不使用 MLP、不训练；不改 7B 权重和已完成评测。

## 冻结定义

| 方法 | 选择 / 分配空间 |
|---|---|
| Unit L1 | 每个原始 Token L2 归一化后，用 L1 距离 |
| Correlation | 每个 Token 减去自身通道均值，再单位化，用余弦；不是跨图像均值中心化 |
| Diagonal Mahalanobis | 单位特征除以冻结的正则化通道标准差，再用欧氏；变换后不再次单位化 |
| Query-response KV | 原生 Layer0 RMSNorm / QKV / M-RoPE；问题 Query 的 QK logits 与 V 的固定加权距离 |
| Query-response JS | 对每个视觉 Token 的 head × question-token 响应作 softmax，再比较 JS 距离 |

最后一组是**响应轴上的概率分布**，不是对视觉 Key 轴 softmax 后的原生注意力分布，不保证保留注意力质量。
所有方法首锚点固定源 Token 0，最远点遍历 / 最近锚点分配，并列选最小源索引，锚点固定归属自身以确保恰好 128 个非空簇。
合并仍取**原始未归一化特征**的 FP32 均值后转 BF16，位置仍用簇三轴重心，文本位置保持不变。

KV 距离平方固定为：

`mean_over_heads_and_question_tokens((q·(ki−kj)/sqrt(dh))²) + mean_over_native_value_dims((vi−vj)²)`。

不按答案调 λ，不读生成文本或答案 Token；只用 prompt 中当前问题的 Query。
明确实现 Qwen GQA 的 KV 共享，Key 使用原始 M-RoPE 后状态；最终压缩位置策略与所有对照相同。
JS 的 softmax 温度为 1，概率下限为 1e-12，再归一化；以自然对数定义 sqrt(JS)，距离计算按 8 个锚点分块控制显存。

## 马氏对角的独立校准

从既有训练清单按固定哈希选择 32 张 TextVQA train、32 张 ScienceQA train 图像，排除这 500 张图像内容。
物理上只读取 parquet 的 image 列，不读问题 / 答案。使用原生图像特征，不使用模型输出或标签。
先在每张图内对单位特征计算一阶 / 二阶矩，再对 64 张图等权平均。
方差使用 `0.9 × channel_variance + 0.1 × mean_channel_variance`，附加相对 1e-6 下限。
校准参数只保存一次，冻结后两卡共用；没有求高维完整协方差逆，也不把对角近似称为完整白化。

## 验收与报告

- 复用原来的 dense / cosine 任务结果；16 张 smoke 重新计算两组并验证生成 Token 一致、NLL 误差 ≤ 1e-5。
- smoke 对原生真实前向捕获的 Q/K/V 与位置编码响应做一致性检查。
- CPU 测试包括距离公式、GQA、JS、重复特征、全覆盖、K128、原始均值、文本位置、标签和答案泄漏隔离。
- 每图 5 个新增生成分支，加 2 个有来源记录的历史参照；种子不当作额外样本。
- 两卡按原图像分片；先 smoke，再 200 张开发 + 300 张复核，最终汇总 500 张。
- 原生生成分数为主，NLL / 几何误差为解释变量。每任务独立图像配对 bootstrap 10000 次，5 方法 × 2 任务的 10 项比较进行 Holm 校正。
- 这是复用图像的探索，不是新的独立确认。不得按本轮结果挑参数并继续称为确认实验。
- 分开记录选点、聚合及统计、响应提取和 decoder prefill 时间；旧 baseline 时间仅为历史值，不做跨运行速度排名。

```bash
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/analysis/qwen25_vl/run_token_metric_ablation.py \
  --root outputs/analysis/token_metric_ablation/3b_k128_n500_20261021_v1 \
  --source-run outputs/analysis/anchor_distance_ablation/3b_k128_n500_20261021_v2 \
  --training-run outputs/train/qwen25_vl/anchor_mlp_7b_k128_ddp_20261021_v1 --launch
```
