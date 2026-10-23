# 后续实验默认基座：Qwen2.5-VL-7B（2026-10-21）

用户已明确后续实验切换到 7B。历史 3B 配置、结果和实现哈希保持不变；新实验必须显式写出 7B 权重和独立输出目录，不能把历史表的 baseline 替换成 7B 数字。

## 本轮固定设置

- 原权重：`/home/WuHao/checkpoints/ckpts-vlm/Qwen2.5-VL-7B-Instruct`，只读。
- 本地 ModelScope 下载记录只给出 master，不能捏造 Hugging Face commit。`model_identity.json` 对全部五个 safetensors 分片、索引、模型/处理器/分词器配置计算 SHA256，用 `local-sha256:<digest>` 固定本轮 revision。
- 冻结视觉编码器、原生 merger/Projector、LLM 和 lm_head；只重训 `3584→256→3584` 残差 MLP，共 1,846,016 参数，FP32 参数与优化器、BF16 主干。
- 保持最初 Feature-anchor 的确定性余弦最远优先锚点、硬分组和 FP32 簇均值。固定 K128，使用簇重心三轴 M-RoPE；文本位置不变。没有引入动态预算或新的选择器。
- MLP 最后一层权重/偏置为零，step0 与不训练的均值压缩完全一致。不加载任何 3B MLP。
- 双卡 DDP，每卡一份完整冻结的 7B；仅同步 MLP 梯度。micro-batch=1/卡，梯度累积=4/卡，**全局有效 batch=8**，320 optimizer steps、2560 张图，保持与 3B 对照的训练量相同。
- 数据复用原固定清单：TextVQA train 2048＋ScienceQA-IMG train 512；图像内容去重并排除全部既定 ChartQA test/TextVQA validation 来源图像。不能使用测试题做训练。
- CE＋0.5×未压缩 **7B** 教师 logits KL；AdamW，lr=2e-4，weight decay=.01，16-step warmup、cosine、gradient clip=1。
- eager attention；use_fast=false；原生动态分辨率，min_pixels=200704、max_pixels=1605632。
- greedy、KV cache、**显式 repetition_penalty=1.0**。ChartQA 最多 16、TextVQA 最多 32 新 Token，不继承 checkpoint 默认的 1.05。

## 验收与队列

1. CPU 测试：3584 维零初始化、固定 K/位置/标签、保存恢复；双进程 Gloo 的梯度累积与串行全局 batch 相同。
2. 双卡真实 7B smoke：检查未压缩 native/manual 生成一致、零初始化特征/位置一致、压缩 KV decode 位置正确。先保存到 step2，再恢复至 step4；两卡 MLP 参数哈希必须相同。
3. 主训练：两卡共享 320-step 优化过程，不是两卡分别训练两种方法；训练清单每张图严格出现一次。检查冻结权重没有梯度。
4. 固定 step320 后自动进入同一 analysis 数据循环，评测 dense、Feature-anchor mean、训练 MLP 三组：16 张 smoke→200 张开发→800 张留出。比较均使用本轮 **7B** baseline；留出图像之前用过，明确不是新的独立发现确认。

输出：`outputs/train/qwen25_vl/anchor_mlp_7b_k128_ddp_20261021_v1/`。

启动：

```bash
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/train/qwen25_vl/run_anchor7b_ddp.py \
  --root outputs/train/qwen25_vl/anchor_mlp_7b_k128_ddp_20261021_v1 --launch
```

`run_manifest.json` 记录阶段、进程与日志；`model/manifest.json`、`model/training.jsonl` 和逐 rank 清单记录训练。checkpoint 保存优化器、scheduler 和每个 rank 的 RNG；恢复时将未提交日志尾部归档，避免将重放的样本误计两次。只管理本队列启动的进程，不改变其他服务器任务。

本轮是适配与小数据重新训练验证，不是重现 Bifocal 的完整评测，也不宣称端到端加速或方法新颖性。
