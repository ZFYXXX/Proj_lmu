# MMSU 全库 attention 干预实验（20–23）

**当前默认入口已经改为单次生成的 observe**，详见 [完整服务器命令](TERMINAL_COMMANDS_7B_LORA.md)。本文件描述保留的重型 suite，需要显式调用才会执行，不属于当前“只做必要分析”的流程。

本版本增加 suite 模式；原 observe 模式继续只观察、保留全部 token。所有代码都在本分析目录，不改主框架、任务 scorer 或 checkpoint。

**不需要 head 校准集。** 在读取预测前，用 checkpoint 的实际层数/head 数生成固定计划。在指定层内逐个扫描全部 head；每个条件均评估同一个完整 mmsufinal（5,000 条）。完整 B 的结果用于描述这套模型在 MMSU 上的效应，不再把同一 B 上挑出的最强 head 当作独立验证发现。

## 已实现的实验

| 项目 | 条件 | 主要结果 |
|---|---|---|
| 20 读取范围 | 各层/head 的毫秒距离分布；窗口和每 query 等连接数随机保留；softmax 前 mask、softmax 后置零分别运行 | head×距离、窗口半径×完整 B 准确率变化、配对 CI |
| 21 head 功能 | 每次只消融一个 head 的 out_proj 输入；同层随机单 head 对照；恢复原 head 输出 | layer×head 损失、head×MMSU 类别损失、随机对照差值、恢复检查 |
| 22 sink 机制 | 原始 attention 选高/低/随机等时长块；固定 shadow A 改 V；key 屏蔽后重归一化；恢复 V/head；特征级内容替换和时间移动 | incoming r×完整 B 损失、key/V 效应之差、位置/内容跟随 |
| 23 低权重证据 | 高/低/随机等预算块的保长均值替换、范数匹配替换、真实删除和恢复；TopK 同预算低分/随机回填 | 删除规则×完整 B、错对翻转、回填改善及配对 CI |

“属性”默认是原 MMSU 的 question_type、sub_category、task_name，并按官方 47 项任务映射加入 Perception/Reasoning × Semantic/Phonology/Paralinguistic 六组统计。它们衡量不同类型任务对 head 的依赖，不等同于独立声学属性真值，也不表示没有问题输入的 encoder 会按问题调整 head。已有固定读出器时可接入下文的 frozen probe；不会在 B 上拟合它。

全层版本中，21 与 22 默认扫描全部 32×20 组合。23 的实验对象仍是最终 encoder 输出位置，并非逐个中间层/head 删除。observe 的基准评估保持 100% token；suite 中明确命名的 delete/refill 条件会删减输出 token，不能把整个干预套件称为全部条件均无压缩。

## 20：范围和计算成本

距离为同一真实 chunk 内 |query_index-key_index| × 实际 encoder 步长（毫秒），包含 self。区间左闭右开：

0–20、20–40、40–80、80–160、160–320、320–640、640–1280、1280–2560、≥2560 ms。

先对每条音频的有效 queries 平均，再对音频等权。另存实际可见位置下的均匀参考。原 encoder 不允许跨 chunk 的连接仍保持不可见，不能用本实验声称它直接读取了这些远处位置。

窗口参数为**半径**，默认 80/320/1280 ms。每个条件只限制一层的所有 head。随机连接与该窗口在每个 query 上的保留数量相同，二者均保留 self，避免边缘位置预算不同或全屏蔽行。随机 mask 在 pre/post 条件间保持一致。

- pre：softmax 前将不允许的 logits 置为负无穷，重新归一化。
- post：对原 softmax 权重置零，不重新归一化。
- 这两个处理属于不同实验，不合并命名为同一种“窗口 attention”。

默认每 rank 的首条音频捕获一个预先固定层的实际 Q/K/V，独立计时 dense masked、FlashAttention-2 full 与 FlashAttention-2 window kernel，记录 CUDA Event 时间、额外峰值分配内存和输出误差。mask 创建、投影、生成不计入该 kernel 比较。CPU 明确返回 not_measured，不用 CPU 耗时假冒稀疏 GPU 结果。

接口依据 [FlashAttention 官方文档](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention)。window kernel 使用实际 window_size；普通布尔 mask 不宣称能节约计算。套件记录的 diagnostic_generation_seconds 包含观察/干预开销，不能作为生产推理速度。

## 21：无需校准的 head 扫描

计划内各层的每个 head 都有 zero 和 restore 条件。每次改变单个 head 在 out_proj 之前的输出，其余 head 和 token 残差流保留。

每个 head 的同层随机对照预先由 seed 选取最多三个其他 head。因为已经扫描全部 head，直接复用这些对照 head 的完整 B 结果，不重复生成等价条件。报告目标 head 准确率减去随机对照平均准确率及聚类 bootstrap CI。

属性分组结果来自全部原始样本的原任务类别，输出每个类别的样本数。head 之间的比较与类别热图是探索性结果，没有自动按显著性挑选 head，也没有声称校正后发现了独立属性功能。

## 22：sink 与功能贡献

每条音频先跑原始模型。候选块只依赖该次原始 incoming attention 和固定规则，之后所有条件复用这份参考。默认块长 80 ms、sink 预算为 eligible blocks 的 10%。块不跨真实 chunk，保留至少一个可用 key；高分、低分和随机条件的实际块数相同。

默认 sink 机制实验和第 21 项都扫描计划层的全部 head，不设 head 校准集。若预先决定只研究固定 head，可用 --sink-heads 0,1 明确缩小第 22 项范围；这不改变第 21 项的完整 head 扫描，也不改变每个条件的 5,000 条评估样本。

- v_zero：同一个 shadow A 下将选中 V 置零。
- v_norm：将选中 V 的方向替换为未选位置的均值方向，保留各自范数。
- key_mask：屏蔽相同 key，重算 softmax，改变竞争分配和 V 路径。
- restore_v：先改变 V，再恢复原 V，整个 B 继续走后续网络。
- restore_head：先算 key 干预，再恢复该处原 head 输出，整个 B 继续运行。

FA2 不返回 A。实现读取真实 Q/K/V，用与观察模式相同的 shadow QK/softmax 计算改动量，并将该 head 的 Δ(A@V) 加到原 kernel 输出上。其他 head 的输出保持原值。此为明确记录的 native-plus-shadow-delta 诊断，不声称和 FA2 内部每个有限精度操作位级等价。baseline 保存 shadow A@V 一致性检查；restore 条件应恢复原输出。应先查看重建误差和 restore_checks_passed，再解释机制。

输入诊断有两种：

- mel_roll：在每个原 chunk 内循环移动实际 log-Mel 的时间列；位移为偶数帧，记录 encoder token 对应关系。排除卷积边界附近位置和没有严格 stride-2 对应的奇数尾块。
- mel_mean：将该 chunk 的时间内容替换为其平均 log-Mel，保持帧数和 chunk 不变。

这两者是**特征域**诊断，不是重新录制/编辑波形后的自然音频，也不是保持所有任务答案的变换。每条原 MMSU 都参与处理，但 accuracy 设为空，原答案匹配另记 original_label_agreement。不把这个数混入普通任务损伤曲线；主要分析 attention 的位置相关与内容移动相关。

记录每块平移帧数和跟随统计的有效 token 数。最大 incoming 完全并列时，峰值位置记为缺失并记录并列数量，避免 argmax 总取开头导致假的位置跟随。候选块还记录 cutoff 同分数量，便于检查分数并列造成的选择偏差。

高 incoming 与功能损伤的共现、V/key 差异可以支持机制分析，仍不能用一个阈值判定所有 sink，也不能据此自动断言可以删除。

## 23：低权重证据

干预位置固定在 encoder 的原池化和 ln_post 之后、proj 之前。分数来自最后一层各 head 的原 incoming 平均，并按原池化坐标配对平均。因此本项测试 encoder 输出位置的任务贡献，不测试从中间层删除 token 所节省的 encoder FLOPs。

默认块长 80 ms；预算为 eligible blocks 的 5%、10%、20%。每条音频各规则选相同数量的完整块，未满一个块的尾部不混入不同长度预算；太短而无法干预的样本仍保留在完整 B，报告 active_samples。

- mean：选中块的表示替换为未选位置均值，保持序列长度。
- norm：替换方向并保留每个 token 原范数，保持序列长度。
- delete：实际删除选中表示，同时复用原 legalizer 修正 LLM 音频占位符、mask、位置和音频输出计数。原 Mel 输入保留。
- restore：均值替换后恢复选中原表示，运行完整 B。

同预算回填另外固定保留率 10%、25%、50%：先构造全局 TopK，用其中分数最低的一部分槽位（默认 25%）换入原被丢弃的低分位置或随机位置。三个条件保留 token 数完全相同。输出相对 TopK 的变化，以及低分回填减随机回填的配对差和回填曲线。

低分减高分、低分减随机的准确率差直接按同一音频配对计算 CI，不能用两个对 baseline 的独立区间代替。范围实验同样给出窗口减随机的配对差；sink 实验给出高分减随机以及 key-mask 减 V-zero 的配对差。

总分之外输出 correct→wrong、wrong→correct、both-correct、both-wrong。均值持平不掩盖逐题损伤。观察到低分干预更有害或回填更有效，可说明 attention 排序遗漏了任务有用位置；称为“语义丢失”仍需独立内容证据。

## 固定 probe（可选，不做 head 校准）

已有固定线性回归读出器时：

- NPZ 的 weight 为 [encoder_d_model, 属性数]，bias 为 [属性数]，attributes 为属性名字符串数组。
- 输入固定为 proj 之前表示的全时间均值，输出为 mean_hidden @ weight + bias。
- targets JSON 为原始 MMSU id → 属性名 → 数值真值，需覆盖全部 B。
- 通过建计划参数 --probe-weights 和 --probe-targets 接入。

记录预测与平方误差，比较每个 head/干预的属性误差变化。代码不训练 probe、不使用 probe 或测试答案选 head。未提供文件时，只做 MMSU 任务类别分析，不虚构属性 probe。输入变换条件的属性真值不默认保持，误差记缺失。

## 运行

在已有服务器评估环境和项目根目录执行；沿用原 mmsufinal 数据路径。依赖仍为 Transformers 4.57.3、现有 PyTorch/FA2 评估环境、NumPy、PyYAML，绘图用 Matplotlib。

入口专用配置为本目录的 [eval_mmsufinal.yaml](eval_mmsufinal.yaml)，明确 `tasks: [mmsufinal]`，不再加载通用四任务 YAML。实际 task YAML 和 scorer 继续使用原 `lmms_eval/tasks/mmsufinal/`。下面仅为显式选择重型 suite 时的入口；[完整 7B LoRA 终端命令](TERMINAL_COMMANDS_7B_LORA.md) 已按最新要求改为 observe。

重型 suite 的每个条件都使用完整 B。以下可选入口依次生成固定计划、评估、干预报告和原始表示报告，不会划分校准集。它不属于当前默认 observe 流程：

```bash
export CHECKPOINT_PATH=/absolute/path/to/7b-lora/checkpoint-4474-merged
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/analysis/audio_sink/run_mmsufinal_suite.sh
```

也可拆成四个命令，在模型加载前查看 plan 中的条件数量和总生成次数：

```bash
python -m scripts.analysis.audio_sink.suite_plan \
  --checkpoint "$CHECKPOINT_PATH" \
  --output runs/analysis/audio_sink/full_plan.json

python -m scripts.analysis.audio_sink.launch \
  --config scripts/analysis/audio_sink/eval_mmsufinal.yaml \
  --checkpoint "$CHECKPOINT_PATH" --mode suite \
  --suite-plan runs/analysis/audio_sink/full_plan.json \
  --output-dir runs/analysis/audio_sink/mmsu_attention_suite \
  --num-processes 4

python -m scripts.analysis.audio_sink.suite_report \
  --run-dir runs/analysis/audio_sink/mmsu_attention_suite

python -m scripts.analysis.audio_sink.report \
  --run-dir runs/analysis/audio_sink/mmsu_attention_suite
```

默认不传 --limit，必须每个条件都齐全覆盖 5,000 条才生成完整报告。每次用新输出目录，不支持静默跳过失败样本。每个 worker 加载一次模型，在每条音频上顺序运行条件；原 lmms-eval 结果保存原始模型预测，各条件使用相同 scorer 单独汇总。

若实际 checkpoint 是 32 层×20 heads，默认计划含 11,311 个条件加一个原始参考，共 **56,560,000 次生成**。计划生成时会打印实际数量，并保存 conditions_by_family。可用 --families range/head/sink/evidence 分别建计划，或预先指定 --layers 0,7,15,23,31；每个计划内的每个条件仍使用完整 B，不能根据 B 分数反复调整选择后声称独立验证。

一键入口对应的拆分变量为 ATTENTION_FAMILIES、ATTENTION_LAYERS、SINK_HEADS。例如只跑第 21 项的第 0 层，可设置 ATTENTION_FAMILIES=head、ATTENTION_LAYERS=0；该层全部 head 都在完整 5,000 条上评估。NUM_PROCESSES、MAIN_PROCESS_PORT、ANALYSIS_RUN_DIR 和 PYTHON_BIN 可分别指定进程数、端口、新输出目录及解释器。沿用已有任务数据路径；入口不自动同步服务器或提交 GPU 作业。

## 结果文件与验证

- suite_plan.json：冻结的全部条件、架构、seed、随机 head 对照。
- samples.rank*.jsonl：原始模型预测与观察统计。
- conditions.rank*.jsonl：逐条件逐样本预测、原任务分数、干预位置/预算、表示变化、诊断耗时。
- coverage.json：原始样本和逐条件覆盖检查。
- suite_summary.json：完整 B 分数差/CI、错对翻转、随机 head 对照、V/key 差、低分/高分/随机配对差、低分回填差、跟随诊断、重建误差、kernel 成本。
- condition_results.csv：每个条件一行的简表。
- suite_figures/：距离分布、head 损失/类别、窗口和低权重损伤曲线、sink 功能散点、位置/内容跟随图。head_loss_six_groups 含六组各自完整的 layer×head 损失热图，共享色标。完整数组在 JSON；suite 的范围/距离概览图展示预先固定的浅/中/深层，原始参考的全部层距离图由 report 分页输出。
- figures/、summary.json：原始参考的相似度、集中、全部层距离分页图、Mel 时间对齐、每层表示矩阵/PCA 和全部 head 的案例 attention 图；直接复用 suite 的原始参考记录。

默认 CI 按 audio_path 聚类；有可靠场景字段时可将字段加入参考记录，再给报告传 --group-field FIELD。类别热图给出描述性损失与样本数，不把每个格子当作通过多重检验的发现。

按用户要求，本次不在本地运行模型、pytest、绘图或实验，只做静态源码与语法检查。测试代码已保留，覆盖公式、HF encoder/Thinker、真实删除后的长度/位置契约，以及样本配对和恢复，但不声称这些测试已验证本次修改。**当前版本尚未运行目标 7B LoRA / FA2 / 多 GPU / 完整 MMSU**；不能把历史随机小模型或合成输入的结果当作研究证据。
