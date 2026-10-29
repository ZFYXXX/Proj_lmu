# Qwen3.5 图像全局上下文选择器补充实验

补齐此前Qwen2.5中有、Qwen3.5未独立训练/评测的image-context分支。它与纯视觉评分器和全局问题上下文选择器分别报告。

## 方法与训练

- 原生Qwen3.5-9B，post-merger、LLM输入前；冻结全部基座。
- 复用已完成step320的同基座视觉评分器初始化；附加图像特征均值→无仿射LayerNorm→线性投影→tanh，上下文加入逐Token隐藏特征后打分。
- context_mode明确为image；不读取问题或答案作为选择器输入。零初始化上下文分支必须严格保持视觉评分器初始得分/选集。
- 复用原2560张独立训练图像（TextVQA2048＋ScienceQA512），与三个评测集再次检查指纹不重叠，不改变清单。
- seed42、全局batch8、额外320步，与现有问题上下文分支相同；答案CE＋0.5 teacher KL＋退火mask BCE，AdamW2e-4、weight decay .01、warmup16、cosine、gradient clip1。
- 训练连续Top-K并保留完整序列，推理硬选min(128,N)个原始视觉Token并保留原位置。
- 固定最终step320，不根据测试成绩选择checkpoint。

## 版本与恢复

新增独立的qwen35_image_context_projector binding和qwen35_image_context_v1 checkpoint格式，复用已经验收的native compaction/cache生命周期。旧head的实现哈希和加载规则保持不变。

训练脚本增加独立image kind，包含优化器、调度器、数据顺序和随机状态恢复。真实训练smoke需通过两步训练及磁盘保存恢复后的参数精确重放；原基座不能产生梯度。

运行使用独立冻结worktree和run ID，记录源码、数据、基座和视觉prior权重哈希。按用户要求不为小改动自动提交Git，不覆盖旧输出，不修改共享数据/模型或他人进程。

## 队列

GPU0先运行训练保存恢复smoke，再执行正式320步。GPU1可并行执行真实9B生成验收：零上下文与原visual prior一致、K128、首个decode位置、dense identity生成分数一致，以及选择路径不调用问题tokenizer。

训练和生成验收通过后，双GPU按任务取队列，每项先8题smoke，再全量ChartQA2500题、OCRBench1000题、AI2D3088题。全部生成/计分通过MLLM-Comp统一lmms-eval入口，沿用已有原生动态分辨率、prompt、greedy、repetition penalty1.05和生成上限16/128/16。

不重复已有方法的全量。报告引用dense、纯视觉、全局问题的历史同协议结果；AI2D主展示统一为开头选项提取分，同时保存严格分。

计划结果目录：`outputs/eval/qwen35_vl/image_context_k128_20261027_v1/`。
执行入口：`scripts/eval/lmms_eval/run_qwen35_image_context.py`。
运行状态以manifest、training/image/manifest和逐任务complete标记为准。

## 启动记录

2026-10-27：44项相关CPU测试通过，四个已有Qwen3.5训练head的实现哈希保持有效。真实9B生成验收通过，包括零上下文保持visual prior、dense identity生成分数一致、不读取问题tokenizer和K128/cache/M-RoPE检查。

两步真实训练smoke及磁盘保存恢复后的参数精确重放通过。协调进程PID400757已启动；GPU0已进入正式320步训练，初次记录时完成2步。训练结束后自动在GPU0/1并行评测三个全量任务。该记录表示实验已启动，不代表训练或全量评测已完成。
