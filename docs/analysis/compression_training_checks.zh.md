# 可学习压缩失败排查（2026-10-21）

目的不是把组合结构继续堆大，而是区分：是否学会训练答案、简化结构是否伤害原有强基线，以及仅训练压缩器能否改善泛化。

## 原论文比较边界

Braco 原文 https://arxiv.org/html/2609.35232v1 的表 1 与 J.5 明确：95.2 是八项基准相对 Vanilla 的平均保留率，不是某个数据集的绝对准确率。25 Token 时 TextVQA 从 58.0 到 52.3，仍下降 5.7 pp。主实验使用 LLaVA-1.5-7B，558K 对齐＋665K 指令两阶段训练；指令阶段更新压缩模块、Projector 和 LLM。不能同我们冻结原生 Qwen2.5-VL、2560 张单遍训练直接比较。Qwen2.5 语言主干迁移也不等于原生 Qwen2.5-VL。

ReConV 原文 https://arxiv.org/html/2610.17412v1 的 §4.2 使用 CLIP＋Vicuna-13B；压缩在 Projector 前，两层压缩器，558K＋665K 数据，指令阶段包含 LLM LoRA。MT-VQA-v2 整体 79.99→70.93，优于其压缩对照并不等于几乎无损。我们的单层 post-merger、64+64 分配、归一化残差和全冻结主干均不同，不能称为官方复现。

这些差异只能解释“为什么不能直接比较”，不能证明我们只需扩大训练就能解决。原组合还落后于同模型的训练免费 Feature-anchor，因此先排查自身实现和优化目标。

## 固定检查

1. 从已有训练数据按图像哈希取 256 张（TextVQA train 192、ScienceQA train 64），对 dense、Feature-anchor、四种组合 step0/step320 分别生成和测答案 NLL。保留 TextVQA 全部人工答案，使用原生软评分。
2. 固定 64 张训练图像（48＋16），从 summary_residual step320 出发，只用答案 CE 反复训练 320 steps、累积 4，比较相同图像训练前后生成。该项只验证拟合能力，不证明新方法有效。
3. 固定原 Feature-anchor mean K128、原始簇成员、簇重心 M-RoPE，增加宽 256 的零初始化残差 MLP。step0 必须与原基线逐 Token 生成及 NLL 一致。使用相同 2560 张训练数据、320 steps、累积 8、CE＋0.5 teacher KL，视觉主干、原生 merger、LLM 全冻结。
4. 保存恢复和 GPU smoke 通过后，强基线分支自动执行 16 smoke、200 开发、800 留出图像评测，按图像 bootstrap。留出集之前使用过，明确不是全新的独立确认。

不修改旧 checkpoint 的实现哈希输入，不覆盖旧实验。首个 decode 为压缩 prompt 的最大三轴 M-RoPE＋1；文本位置不变，所有分支严格 K128。训练诊断只从 train 取样，沿用原来对全部 ChartQA test/TextVQA validation 图像的训练排除。

## 执行

运行根目录：`outputs/train/qwen25_vl/compression_training_check_20261021_v1/`。

入口：`scripts/train/qwen25_vl/run_compression_training_check.py --root <root> --source <learned_hybrid_run> --prepare --launch`。

两条后台分支：GPU0 训练集审计→64 张 CE-only 拟合；GPU1 强基线 smoke→2560 张训练→同协议留出评测。只管理自己启动的进程；任一分支失败时停止本轮后续任务并记录原因，不影响其他用户的任务。

产物：`protocol.json`、`run_manifest.json`、各分支 `manifest.json`/训练日志/逐图预测/答案 NLL/任务分数/恢复 checkpoint、`confirmation_report.json` 和最终 `final_report.json/md`。64 张检查不进入留出任务质量表，也不把训练得分当泛化结论。
