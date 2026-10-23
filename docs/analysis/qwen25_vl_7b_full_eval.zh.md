# 7B 固定 checkpoint 全量评测

运行目录：`outputs/eval/qwen25_vl/anchor7b_chartqa_ai2d_mlp_only_20261021_v3/`。
依次评测 ChartQA test 2500 题、AI2D test 3088 题；按用户要求不再运行原始 Feature-anchor 消融、TextVQA 和 PruMerge+。
旧 v1/v2 队列已停止并标记 superseded，保留全部旧产物和原始脚本快照；配置完全一致的 baseline/MLP 两组预测经过身份 / 生成 / 预算校验后复用，逐条件记录来源。
不再训练，不根据这些结果挑选 checkpoint；沿用 7B MLP 的 step 320。

两组统一使用 Qwen2.5-VL-7B、BF16、eager、原生动态分辨率、同一 prompt 和显式 greedy decoding / repetition_penalty=1：

- Dense，不压缩。
- Feature-anchor + 已训练 MLP，K128。

ChartQA / AI2D 最多生成 16 Token。
ChartQA 使用原生 relaxed score 并记录 strict；AI2D 使用原生多选 prompt / filter / exact-match。
全部题目保留，同时为所有方法报告相同的训练图像内容无重叠子集。
主分数按题目加权；配对 bootstrap 10000 次以独立图像组为重采样单元，避免将同图多题当作独立样本。

每个任务先运行 16 张独立图像 smoke，验证未压缩 native/cached 生成完全一致、K128、位置和 cache，失败即停。
GPU0 先运行 smoke / full shard0；GPU1 等现有 500 张距离消融完成，再运行 full shard1。
按图像哈希分片，所有同图问题在同一分片；不干预其他进程。

```bash
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/eval/lmms_eval/run_anchor7b_full.py \
  --root outputs/eval/qwen25_vl/anchor7b_chartqa_ai2d_mlp_only_20261021_v3 \
  --source-run outputs/train/qwen25_vl/anchor_mlp_7b_k128_ddp_20261021_v1 \
  --geometry-run outputs/analysis/anchor_distance_ablation/3b_k128_n500_20261021_v2 \
  --reuse-run outputs/eval/qwen25_vl/anchor7b_chartqa_ai2d_k128_20261021_v2 --launch
```

产物包含模型和 MLP 哈希、完整数据清单 / 图像指纹 / 训练重叠、逐题预测 / NLL / 原始和压缩 Token 数 / trace、失败状态及每任务 smoke/full 报告。
运行耗时包含诊断评分，且机器有其他任务，不作为优化部署速度结论。
