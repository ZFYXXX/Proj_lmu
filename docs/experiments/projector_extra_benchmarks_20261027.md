# Qwen2.5-7B / Qwen3.5-9B：POPE 与 MMStar 全量补充

本轮仅评测，不训练、不选 checkpoint、不重复已有 ChartQA/OCRBench/AI2D。

- MMStar：官方 `Lin-Chen/MMStar`，revision `bc98d668301da7b14f648724866e57302778ab27`，1500 题。
- POPE：本地 `Full/` 的 adversarial、popular、random 各 3000 题，共 9000。
- 每个模型各 10 组：dense、旧聚合 MLP、视觉选择器、图像全局上下文、问题全局上下文、已训练相关性＋保护、EADP、PruneSID、MMTok、Bifocal Stage1。共 40 个全量单元、210000 次生成。
- 所有压缩方法固定 post-merger `min(128,N)`，原生动态分辨率，保留各自已验收的 M-RoPE 与缓存路径。Bifocal 不冒称 Stage1+2。
- BF16、eager、seed42、greedy、repetition penalty1.05、最多生成16 token；沿用原生任务 prompt。
- 各模型/方法先跑两个任务各8条分层 smoke，通过后全量；无500张开发集。
- 检查图像内容哈希与2560张训练清单重叠，发现重叠停止准备，不静默删题。
- 固定现有训练 checkpoint、配置和源码哈希；3.5 图像上下文仅在原队列完成固定step320后解析并封存权重哈希。
- 运行从独立 worktree 快照导入，不提交Git、不推送、不中断旧任务。现有3.5训练与三任务评测优先，新增双GPU队列等其完成。

## 评分注意

MMStar 使用当前 lmms-eval 原生答案抽取与 `average`（18个细分能力等权），另外报告逐题 micro accuracy，不能混用。

POPE 保留当前框架严格 yes/no exact accuracy。独立计算 precision/recall/F1、模型回答Yes比例、无效回答比例及三策略分数。修复仅限新增任务的包装函数：原生工具的yes_ratio误统计标准答案比例；无效输出不得从正例recall分母消失。保存原始预测，严格匹配与论文其他规范化协议不可直接混为同一数字。

保存全部逐题预测、原始/实际token数、保留比例、生成耗时、显存和压缩trace。以图像内容为簇做10000次配对bootstrap，MMStar保持原生macro估计量。共享GPU耗时仅作描述，不当作严格加速比较。

MMStar部分题目来源于既有AI2D/ScienceQA等测试集，不是与旧评测完全独立的确认集。

## 入口

`scripts/eval/lmms_eval/run_projector_extra_benchmarks.py prepare|pipeline|report --root <独立结果目录>`

结果目录：`outputs/eval/projector_extra/pope_mmstar_k128_20261027_v1/`。

## 用户调整：先只跑 MMStar

原协议、执行快照与已完成的8组真实smoke均保留。独立调度器
`run_projector_extra_schedule.py` 从原快照导入模型/评测代码，只安排MMStar：
20组全量、30000次生成，双卡运行；POPE暂缓且不会自动接续。
仅替换仍在等待的本用户调度进程，不停止训练或他人任务。
单独保存调度脚本哈希、任务范围与`mmstar_report.md`；完成状态为
`selected_tasks_complete`，不宣称原40组矩阵全部完成。
