# Qwen2.5-Omni：mmsufinal 四项全量分析与 Taylor

所有执行代码、YAML、SH 和运行命令集中在本目录。完整命令见 [RUNNING_COMMANDS.zh.md](RUNNING_COMMANDS.zh.md)，环境变量集中在 [env.example.sh](env.example.sh)。另有独立的 `hierarchical_taylor` 入口，见下文。

两个 SH 都只使用 **`mmsufinal` 全部 5,000 条**。MMSU 是本文对该任务的简称，两份 YAML 都设置 `tasks: [mmsufinal]`。四项分析默认覆盖全部 Thinker LLM 层、全部 query heads，以及全部层分别删除 Attention/FFN 更新。无音频压缩，不加载 Talker。默认配置选择 7B；完整 3B、7B 或 merged checkpoint 均从其 `config.json` 读取真实层数/head 数。

## 文件

| 文件 | 用途 |
| --- | --- |
| [qwen25_omni_mmsufinal_four_probes.yaml](qwen25_omni_mmsufinal_four_probes.yaml) | MMSU-final 全部 5,000 条 |
| [env.example.sh](env.example.sh) | Python、checkpoint、GPU、数据路径和输出目录 |
| [run.sh](run.sh) | 在 mmsufinal 上跑四项分析，完成后自动生成报告 |
| [run_hierarchical_taylor.sh](run_hierarchical_taylor.sh) | 只在 mmsufinal 全集上单独跑 Taylor |
| [qwen25_omni_mmsufinal_hierarchical_taylor.yaml](qwen25_omni_mmsufinal_hierarchical_taylor.yaml) | MMSU 的全层 audio Taylor 配置 |
| [launch.py](launch.py) | 解析真实模型结构，保存有效配置，启动 GPU 分片，检查退出状态 |
| [worker.py](worker.py) | 原生 task 对话和 scorer；一次完整答案生成，随后执行所选分析；为 Taylor 准备标准答案 labels |
| [report.py](report.py) | 流式读取结果，核对覆盖，按样本合并分片，生成分组 JSON/CSV/PNG/PDF |
| [RUNNING_COMMANDS.zh.md](RUNNING_COMMANDS.zh.md) | 环境准备、计划、小规模 GPU 验证、全量运行、进度和重建报告命令 |

本目录复用项目现有的 `mllm_comp.analysis` 和 vendored lmms-eval，需放在同一版本项目的 `scripts/analysis/audio_four_probes/` 下运行。四项分析将 mmsufinal 按文档序号分给多个 GPU，每卡加载一个完整模型副本，无跨卡模型切分或 DDP。独立 Taylor 入口采用下述单 GPU 汇总方式。两个 SH 均可不传 benchmark，或显式传入 `mmsufinal`；不接受 `all` 或 MMAU 任务。

## 独立 hierarchical_taylor

使用 [run_hierarchical_taylor.sh](run_hierarchical_taylor.sh)，只在 **`mmsufinal` 全部 5,000 条**上运行 `hierarchical_taylor`。本文 Taylor 部分的 MMSU 均指 `mmsufinal`，配置为 `tasks: [mmsufinal]`。直接运行脚本即可，也可以显式传入 `mmsufinal`；不接受 `all` 或 MMAU 任务。遇到缺失标准答案的样本直接报错。

Taylor 观察的是组件激活对标准答案交叉熵 loss 的一阶敏感度。入口保留原生 chat prompt，用 `mmsufinal` 的 `doc_to_target` 标准答案字母构造 teacher-forced 输入，prompt labels 为 `-100`，loss 只在答案 Token 上取均值。它衡量的是标准答案的 loss 敏感度，不能直接当作自由生成的正确率。

默认覆盖全部 Thinker LLM 层、全部 query heads、全部 FFN 神经元，以及 path、head、FFN neuron、Attention/FFN sublayer、whole layer 等层级。音频模式下含 `a2a`、`t2a`、`t2t` 路径和 audio/text 分组。输出三类分数：`normalized_taylor`、`direct_removal_taylor`、`activation_magnitude`，并按分数种类和组件粒度分别排名；不同粒度的分数不混排。`ffn_bottom_fraction: 0.3` 用于低分 FFN 神经元集合的模态交叉检查，不会实际剪枝模型。

每次捕获 `layer_batch_size` 个层，但仍执行完整模型的 forward/backward。参数被冻结，没有 optimizer 更新。默认全层、每批 1 层：7B 每条 32 次 forward + 32 次 backward，3B 每条 36 次；此外保留一次完整原始答案生成及原生 scorer 判分。显存需要容纳完整模型和反向计算图，attention 仍随序列长度平方增长。

`mmsufinal` 在**一张 GPU、一个进程**中完成任务级累计，再由原生 Taylor probe 计算最终排名。`TAYLOR_GPU` 默认取 `CUDA_VISIBLE_DEVICES` 的第一个 ID；可显式覆盖。当前入口不合并多 GPU Taylor 分片。

默认 `save_raw: false`，保留完整任务级 FFN 神经元向量、任务分数、排名、逐样本 loss 摘要，以及 `samples.jsonl` 中的原始回答、判分和 target。若将 YAML 改为 `save_raw: true`，原生 probe 会先在内存累计所有逐样本神经元数据，全量运行内存成本很高。默认 `causal_validation.enabled: false`，即不额外执行真实删除验证；`direct_removal_taylor` 仍是一阶估计。需要删除验证时可在 Taylor YAML 显式开启，计算量会增加。

默认结果位置为 `$ANALYSIS_RUN_DIR/hierarchical_taylor/mmsufinal/shard-000/`，也可用 `TAYLOR_RUN_DIR` 覆盖 `hierarchical_taylor` 所在的根目录。其中：

```text
samples.jsonl
manifest.json
worker.log
analysis/hierarchical_taylor/
  hierarchical_taylor.task_scores.json
  hierarchical_taylor.task_ranks.json
```

Taylor 使用上述原生结果格式；四项分析的 `report.py` 不适用于这些文件。运行命令见 [RUNNING_COMMANDS.zh.md 第 7 节](RUNNING_COMMANDS.zh.md#7-单独运行-hierarchical_taylor)。

## 四项实验的实际范围

| Probe | 默认范围与输出 |
| --- | --- |
| representation_dynamics | 全层 system/text/audio 的 attn_res、attn_pure、ffn_res、ffn_pure、full_layer_res cosine |
| prediction_dynamics | 全层首输出 Token 的全部支持指标，包括 JS、双向 KL、top-1 一致率、最终候选 rank 和 top-k Token |
| attention_flow | 全层、全部 query heads、system/text/audio 的全部 query/key mass，以及文本/decision 读取音频的 entropy |
| residual_stream_decomposition | 全层分组 residual 幅度；每层分别删除 Attention/FFN update，输出 decision 与各 Token 组 logits 变化 |

每条样本只生成一次完整原始答案，用 mmsufinal 原任务 scorer 判分。生成和分析采用相同 `doc_to_messages` 对话。四个 Probe 另行执行所需 forward，并不共享一次 forward。删除实验比较 logits，不生成消融后的完整答案，不报告消融准确率。

逐条保存题目原始 id、准确率、原始回答、类别、难度、时长。按正确性、类别、子类别、难度、来源任务，以及 mmsufinal 所选样本时长中位数定义的短/长音频做分组。数据缺失字段记为 `unknown`；MMSU 没有提供的 speech/music/sound 分类不会被推断补上。报告中的无标签记录不计入准确率分母。

本次“全量”指原始音频的四项主实验；错配音频、加噪、时间打乱、纯文本对照不在当前四个基础 Probe 的执行条件中。范围写入报告，避免与配对干预实验混淆。

## 指标解释

- representation 的 res 指标衡量 residual 前后方向变化；pure 指标衡量更新方向与输入的对齐。热图的 `1-cosine` 不等于信息改写百分比。
- prediction 取原始提示之后的首输出位置。mmsufinal 要求只回答选项字母；保存最终首 Token 频数以检查是否符合该格式，不将完整答案生成轨迹等同于首 Token 分析。
- attention 是直接注意力，无跨层 rollout。高 mass 不等于因果重要；normalized entropy 是音频内部重新归一化后的集中程度。
- residual 的 `cosine_to_input` 是更新前后 residual 的 cosine。因果删除覆盖全部非 padding Token；audio 分组统计不是 audio-only 删除。
- 删除导致 JS/top-1 变化不等于准确率下降。写入大小与 JS 的散点只表示原始预测对该干预的敏感程度。

## 运行成本和配置

设观察层数 L、每次保留 attention 的层数 b、干预层数 C，每条样本四项分析需要约 `3 + ceil(L/b) + 2*C` 次完整 forward，另有一次完整原始答案生成。

7B 默认 32 层、b=1、全部层干预：99 次分析 forward/样本；mmsufinal 共 5,000 条，约 495,000 次分析 forward 加 5,000 次生成。3B 为 36 层、每条 111 次。多卡分摊样本但不减少总计算量。

Attention 矩阵为序列长度平方量级；b=1 不能消除单层长音频显存成本。Representation/residual 保留多层 hidden。脚本不会自动截断音频、减少样本或降低层数。部署后应先按运行文档执行小规模 GPU 验证。

通过 YAML 修改参数，或在 `run.sh` 后追加命令行覆盖：

- `--layers all` 或 `--layers 0,1`：观察层；显式传入时也限制因果层。
- `--causal-layers 0,10,20,31`：单独指定因果层。
- `--attention-layer-batch-size 2`：覆盖 YAML 中的层批大小。
- `--sample-limit 4`：mmsufinal 的样本上限；默认 YAML 为 null，全量。
- `--probes prediction_dynamics`：只跑指定子集；默认全部四项。
- `--dry-run`：仅打印计划，不加载模型、不创建输出。

数据绝对路径由原任务 YAML 或环境变量提供；入口只在内存中重定位，不改原任务文件。运行时保存有效配置和实际依赖版本。输出目录必须是新的；当前不支持中途续跑。

## 输出

四项分析的 `$ANALYSIS_RUN_DIR/mmsufinal/` 目录包含：

```text
analysis.resolved.yaml       实际模型、层、因果层、指标和样本上限
plan.json                   GPU 分片、数据覆盖项、forward 次数、分析语义
shard-000/                  每张 GPU 一个目录
  worker.log
  runtime.json              实际依赖版本、GPU、层数/head 数
  tasks.resolved.json        实际数据/对话/生成/scorer 配置
  dataset_inventory.json
  samples.jsonl             原始回答、判分、类别、时长、原始 id
  manifest.json
  analysis/<probe>/
    <probe>.aggregate.json
    <probe>.raw.json         逐样本的层/head/干预统计
report/
  README.zh.md
  summary.json              样本覆盖、准确率、时长分界、首 Token 频数
  groups.json               分组样本数、有标签数、原始准确率
  group_metrics.csv         所有组的完整均值和有限值计数
  prediction_depths.csv     逐样本在所选层中的持续 top-1 一致起点
  aggregates/<probe>/       按 task/组汇总，分片按样本加权
  figures/<task>/           主图及分类对比 PNG/PDF
```

图表包括五种 representation 热图、prediction 层曲线、六种 attention 层×head 热图，以及 residual 相对写入幅度、删除 JS 和写入大小—删除效应散点。分组 JSON/CSV 保留所有组；分组曲线同图最多十组，高基数组和来源任务保留在表格中。图展示诊断均值，不包含显著性检验或置信区间。

进程结束后核对样本、层、head 和两种删除覆盖；缺失数据会让入口失败，不能将部分结果当作完整实验。原始统计使用流式格式读取，避免一次加载整份多 GB 的 JSON。

## 本次验证

本地验证覆盖两份 SH 的 Bash 语法，以及执行套件的单元测试：32/36 层完整计划、观察层与因果层覆盖、
YAML/CLI 参数优先级、从其他工作目录启动时只选择 mmsufinal、原生 scorer、Taylor 标准答案输入、
多卡分片按样本合并、无标签样本、缺样本/层/head/干预检测，以及四类 PNG/PDF 导出。
图表布局检查使用明确的合成测试数据，不是 benchmark 实验结果。

当前完成的是本地代码与运行入口验证；没有加载真实 checkpoint 执行 GPU 小规模验证或启动全量任务。
服务器第一次使用时，按运行命令文档先执行 GPU 小规模验证，再运行全量。
