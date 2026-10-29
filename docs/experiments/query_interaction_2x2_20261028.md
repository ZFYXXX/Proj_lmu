# Qwen2.5-VL-7B：逐 Token 问题交互与任务集合监督

固定原基座、BF16/eager、原生动态分辨率、K128、原始视觉内容和 M-RoPE。
实验目录：`outputs/train/qwen25_vl/query_interaction_2x2_20261028_v1/`。
该次实现不加入信息回收、动态预算或 Qwen3.5。

## 四组

| 组 | 交互 | 训练目标 | 执行 |
|---|---|---|---|
| word_ce | 旧逐词匹配 | 原 CE + .5KL + 退火软硬约束 | 复用已完成 step320 |
| word_teacher | 同上 | 原目标 + 任务集合教师 KL，权重1 | 新训练 |
| context_ce | 问题上下文编码 + 每视觉 Token cross-attention + 两层评分 MLP | 原目标 | 新训练 |
| context_teacher | 同上 | 原目标 + 同一教师 KL，权重1 | 新训练 |

旧交互组初始化参数摘要与已有训练的初始摘要逐字节相同，训练样本、顺序、超参与步数一致。
新上下文组的两个训练臂也使用相同初始化。旧匹配分支491521个可训练参数，新上下文分支706177个。
视觉先验冻结。问题仅从最后一个 user prompt提取，不读答案；保留词序，编码器使用确定性正弦位置。
一层问题 self-attention/FFN、每视觉 Token 对问题的 cross-attention，以及两层评分 MLP都在压缩器中执行；不调用额外 LLM 层。

## 教师

固定原2560张训练图（TextVQA-train2048 + ScienceQA-train512），先复核与OCRBench/MMStar/AI2D的图像哈希无重叠。
每张图构造8个不同的同预算K128集合：冻结视觉选择器原集合，以及原生二维网格上不同区域保底的替换集合。
所有集合只保留原始Token，答案标签和文本位置同步压缩。
冻结Qwen计算各集合的标准训练答案平均NLL，以`softmax(-NLL/.25)`混合保留掩码形成总质量128的软标签。
这是有限候选池任务监督，借鉴任务损失搜索思路，不是EvoComp完整进化搜索、因果Token标签或测试集Oracle。
输出候选成员、损失、图像/问题/网格身份、逐文件哈希、优于原集合的比例和平均NLL改善。
教师只在训练集离线运行，不进入推理输入；两种结构使用同一份教师标签。

## 训练、验收与评测

沿用seed42、batch8、320步、单遍、AdamW2e-4、wd.01、warmup16、cosine、clip1。
保留原连续Top-K软训练，硬推理固定128。各新训练臂先验收真实7B identity、硬选择cache、梯度和4步连续/2+2恢复一致。
全局CPU验收包含词序敏感性、逐Token分数差异、梯度/冻结、标签质量守恒与成员检查、图像/问题对齐、标签与M-RoPE压缩、小型原生Qwen生成及checkpoint恢复。

真实7B基础验收安排在批量教师生成之前。完成后双卡分别生成两个教师分片，再让每卡执行一个训练/评测臂。
正式评测通过MLLM-Comp原生eval执行：OCRBench1000、AI2D3088（修正leading-option评分）、MMStar1500（原生类别宏平均）。
每个新臂先每任务8条smoke，通过后直接全量，没有500张开发集。
旧word_ce、dense、旧聚合MLP、图像上下文结果只读复用，不重复跑。
汇总结构、监督和二者交互的图像级配对bootstrap区间。固定step320，不按测试分数选checkpoint。

## 调度和版本

从独立worktree运行，记录commit、完整本地源码差异哈希、依赖、数据与checkpoint身份；不自动Git提交或推送。
和正在恢复的POPE/RefCOCO+/OK-VQA supervisor共用`outputs/runtime/qwen25_gpu_locks/gpu{0,1}.lock`。
获得两卡预留前不加载模型，manifest明确写queued，20秒更新心跳；保留他人进程。
资源保护失败保留日志并重试，已完成教师标签、checkpoint和评测单元校验后跳过；数据/数值/版本错误明确失败。
当前老supervisor在整个恢复序列期间持有预留，因此新实验排在这批任务之后；排队不等于已经开始训练。

启动：

```bash
MLLM_TOPK_HASH_LOCAL_SOURCE=1 MLLM_TOPK_NORMAL_GPU=1 OMP_NUM_THREADS=2 \
python scripts/train/qwen25_vl/run_query_interaction_2x2.py prepare --root RUN_DIR
MLLM_TOPK_HASH_LOCAL_SOURCE=1 MLLM_TOPK_NORMAL_GPU=1 OMP_NUM_THREADS=2 \
python scripts/train/qwen25_vl/run_query_interaction_2x2.py pipeline --root RUN_DIR
```

小网络读取问题或更换问题时索引变化，都不是有效性的充分证据；最终以硬压缩任务结果及配对区间判定。
# GPU allocation amendment (2026-10-28 evening)

User requested that the old MLP finish first, then prioritize the new method
on its freed GPU, keeping one GPU for the new datasets. The original v1 queue
had not started any GPU work and is preserved as `superseded_scheduling_only`.
Use a new hash-frozen source/run for this operational change; no Git commit.

`pipeline --gpus 0` owns only GPU0. Both immutable teacher data shards execute
sequentially on GPU0, followed by the three training/evaluation arms. All 2560
training examples, seed42, batch8, step320 and nine full evaluation cells stay
unchanged. GPU1 remains assigned to the existing RefCOCO+ and OK-VQA queue.

`split_qwen25_queues.py` preserves the in-flight GPU1 evaluator while replacing
the old dual-GPU CPU coordinator. It validates exact owner/PID start identities,
parks only the dispatcher, retires only its CPU supervisor, and independently
validates the carried native output before resuming the immutable legacy
pipeline with `--gpus 1`. It never signals model workers or other users' jobs.
Completed cells are reused; there is no replay of finished MLP evaluation.

