# 无压缩的音频 encoder 相似度与 attention 分析

**当前默认使用 observe：每条题目只生成一次答案。** 完整 mmsufinal 共 5,000 次生成，保留全部 token。在原模型 forward 内收集必要统计，结束后用已保存数据绘图。没有逐条件重复生成、head 消融、V/key 干预、真实删除或 kernel benchmark。

2026-09-22 更新：`bash scripts/analysis/audio_sink/run_mmsufinal_observe.sh` 从 checkpoint 读取实际架构，观察**全部 encoder 层、全部 heads**。Qwen2.5-Omni-7B 为 L0–L31 × H0–H19，共 **640 个 layer–head 组合**，不是只看 H1。全部 5,000 条均参与统计；只有 doc `0,1666,3333` 保存每层的详细表示和第 0 个 chunk 的全部 head attention 矩阵。不使用校准集或测试答案选层。

**[完整 7B 服务器命令](TERMINAL_COMMANDS_7B_LORA.md)** 使用上述 observe 入口，包含环境、checkpoint、全库评估与绘图。加载专用 [eval_mmsufinal.yaml](eval_mmsufinal.yaml)，明确 `tasks: [mmsufinal]`；数据和评分仍来自原 `lmms_eval/tasks/mmsufinal/mmsufinal.yaml`。输出为 `figures/`、`summary.json` 和 `six_group_analysis/`。

仍使用 128-query 分块重建、每 rank 首条音频的 shadow A@V 检查和 3 个固定案例，不保存全 B 的完整 attention 矩阵。相较旧五层版本，诊断的层数增加到 6.4 倍，统计文件及绘图数量随之增加；这不等于总推理耗时增加 6.4 倍。report 逐行读取 JSONL 并以数值数组保存 head 统计，避免整文件文本副本和大量 Python 浮点对象。进度条每完成一条题目更新；实际速度需在服务器测量。

旧 [20–23 干预套件](ATTENTION_EXPERIMENTS.zh.md) 和 `run_mmsufinal_suite.sh` 保留为显式选择的重型实验，不属于当前默认流程。服务器上已启动的 suite 不会因同步新文件自动停止；切换时应先在原终端停止旧作业。

使用 mmsufinal 和准确的完整 7B checkpoint，可为基础权重或已经合并 LoRA 的权重。脚本不会自动加载 adapter，旧结果目录名中的 `lora` 不代表实际加载了 LoRA。**所有 token 保留**；Global/Segmentwise TopK 只在分析记录里生成候选索引，不执行压缩。

| 用户列出的分析 | 全层版本的覆盖与运行方式 |
|---|---|
| 相邻 token 相似度 | observe：全部 32 层的 K/V/hidden 相似度；K/V 是拼接 head 后的 cosine，不是 640 条独立 head 相似度曲线 |
| attention mass、集中、距离、位置 | observe：全部 640 组合；全 B 统计及官方两大类、六子类汇总；虚拟 TopK 使用最后一层的 head 平均分数 |
| 21 head 功能 | suite：默认逐个扫描全部 640 heads，zero/restore、同层随机对照；额外输出六组 layer×head 损失热图 |
| 22 sink 机制 | suite：默认全部 640 组合的 V/key 干预与恢复，加 Mel 内容/时间变换；observe 只能发现候选，不能识别归一化机制 |
| 23 低权重证据 | suite：最终 encoder 输出上的等预算替换、删除、恢复与回填；不是逐个中间层/head 删除。delete/refill 条件会改变 token 数 |

旧五层结果仍只能支持 L0/7/15/23/31 的结论。全层结果需要重新跑一次全 B observe；修改绘图代码不能补出未采集的层。21–23 的代码存在不代表对应实验已经完成。

详细定义、原论文对应、图表与结论边界见 [实验设计](EXPERIMENT_DESIGN.zh.md)。只修改本分析目录，复用平台的原评估、生成和 scorer。

## 依赖与状态

在已有平台评估环境中运行，要求 Transformers **4.57.3**、PyTorch、NumPy、PyYAML；绘图使用 Matplotlib，官方类别元数据读取使用 PyArrow。不要把本地测试环境当作服务器的模型运行环境。

原 observe 版本此前通过 CPU 测试，包括真实 HF audio encoder 类的输出一致性。本次入口、统计和报告调整按要求仅做静态检查，没有在本地运行模型、测试或绘图，也没有目标 7B checkpoint / FA2 / 多 GPU 的实测速度。历史单元测试使用随机初始化的小模型与合成输入，图片仅用于测试。

## 历史可选检查：observer 输出一致性

下列额外 baseline/observe 对比不在当前一键入口中执行，当前入口只跑一遍完整 B。

在服务器项目根目录执行，保留原来已经验证的数据路径与环境。将 CHECKPOINT_PATH 设置为本次无压缩评估使用的准确 merged checkpoint；不要使用只有 adapter 的目录。每次使用新的输出目录。

```bash
export CHECKPOINT_PATH=/absolute/path/to/7b-lora/checkpoint-4474-merged
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD:$PWD/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"

python -m scripts.analysis.audio_sink.launch \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir runs/analysis/audio_sink/smoke_baseline \
  --mode baseline --num-processes 4 --limit 8

python -m scripts.analysis.audio_sink.launch \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir runs/analysis/audio_sink/smoke_observe \
  --mode observe --num-processes 4 --limit 8

python -m scripts.analysis.audio_sink.report \
  --run-dir runs/analysis/audio_sink/smoke_observe \
  --baseline-dir runs/analysis/audio_sink/smoke_baseline
```

检查 `summary.json`：`baseline_comparison.outputs_identical`、正确性转移、shadow A@V 误差。BF16/FA2 下重建权重可有数值误差；观察 hook 保留原模型输出。未定位差异前不将诊断峰值升级成研究结论。

`MMSUFINAL_DATASET_DIR` 不会替换任务 YAML 中的 JSONL 路径，沿用服务器原任务的已验证配置。脚本不改任务源文件。

## 完整 B：全部 5,000 条

一键入口不传 `--limit`，选层和案例均与答案无关。设置准确的 CHECKPOINT_PATH 和 MMSUFINAL_DATASET_DIR 后，在项目根目录运行：

```bash
bash scripts/analysis/audio_sink/run_mmsufinal_observe.sh
```

需要完整逐题 baseline 对比时，用 `--mode baseline` 对同一个完整 B 独立运行，再把它的目录传给 report 的 `--baseline-dir`。baseline/observe 的 checkpoint、精度、生成参数、任务配置与数据必须一致。

直接调用底层 `launch` 时，可用 `--layers all` 或不传该参数来观察全部层，也可给逗号分隔的零基层索引；`--example-chunk` 指定案例 chunk，`--query-block` 指定重建块大小。一键入口固定全部层、query-block=128、check-av-samples=1，不读取旧 suite 的 ATTENTION_FAMILIES/SINK_HEADS 环境变量。`--virtual-keep-ratio` 只是诊断标记数量，真实 token 保留率始终为 100%。若没有观测最后一层，不计算与当前 Segmentwise 方法对应的虚拟候选。

## 输出与解读

- `eval.resolved.yaml`、`observation.json`、`launch.json`、`runtime.rank*.json`：准确配置和运行来源。
- `samples.rank*.jsonl`：原始 id、预测、原任务分数、音频级逐层/逐 head 摘要。
- `example.rank*.doc*.npz`：固定案例的时间、Mel、incoming r、相邻 cosine 和单 chunk 表示/attention。
- `coverage.json`、`manifest.json`：全库覆盖与完成状态；原 lmms-eval 结果保留在同一 run 下。
- `summary.json`：汇总、配对检查、有效音频数、bootstrap CI、绝对开头/后续块起点比较、重建误差和 cutoff 同分情况。
- `figures/overview.{png,pdf}`：全库相似度、head 集中程度、局部位置与候选覆盖。
- `figures/sink_diagnostics.*`：每层/head 的 top10% 质量、incoming entropy、跨 query 读取比例和起点偏好。
- `figures/attention_distance_l*_l*.*`：所有观察层的 head×毫秒距离质量及均匀参考差，每页最多 4 层，32 层共 8 页。完整数组与平均距离保存在 `summary.json` 的 attention_distance；不包含遮挡或性能损失实验。
- `figures/doc*_timeline.*`：时间对齐图。
- `figures/doc*_layer*_representation.*`：每个观察层的 Gram 矩阵、PCA、相似度与 attention 共现。
- `figures/doc*_layer*_all_heads.*`：每个案例、每层的全部 20 heads 的 query×key 图，共用 0–1 概率色标；仅为选定 chunk，不是全音频矩阵。
- `figures/doc*_attention.*`：指定 chunk 中列质量最高的 head，明确是极值案例。
- `six_group_analysis/`：按官方任务映射关联 Perception/Reasoning × Semantic/Phonology/Paralinguistic，输出八列表格 `tables.md`、逐样本/逐层/逐 head CSV、`summary.json` 和 8 类图。`06_all_heads_mass`、`07_all_heads_peak_r`、`08_all_heads_start` 各含六个组的完整 32×20 热图；选定七个 head 的旧图仅作补充。

六组报告会在一键评估完成后自动生成，也可只对已采集的结果执行：

```bash
python -m scripts.analysis.audio_sink.group_report \
  --run-dir "$ANALYSIS_RUN_DIR" --dataset-dir "$MMSUFINAL_DATASET_DIR"
```

它只读取结果和 `data/*.parquet` 中的 id/类别字段，不加载模型。支持已有的五层结果和新的全 32 层结果；需要完整 5,000 条，并核对 47 个任务的映射、类别一致性和音频路径唯一性。此脚本使用的逐条 bootstrap 以本 MMSU 的不同音频路径为单位；若路径不唯一会停止，不能直接套用于多问题共享音频的数据。

PCA 和单案例不证明 sink；Segmentwise 约束带来的覆盖改善也不证明 sink。客观证据来自完整 B 的 incoming 分布、跨 query 读取、位置复现，以及相邻/远处表示的对照。高 cosine 支持合并候选，不能证明合并无损。语义贡献、频率依赖和归一化机制均不由本轮直接判定。

默认 bootstrap 按 audio_path 聚类。若多个路径来自同一个原始场景，先把可靠的场景 group 字段补入记录，再使用 `--group-field FIELD`；不要把 tokens 或 chunks 当独立样本。

## CPU 验证

```bash
python -m pytest scripts/analysis/audio_sink -q
```

本实现的 K 相似度是拼接 head 后的 cosine；H 为 MHSA 残差后、FFN 前。均匀、对角、单列读取均有独立测试，避免把低 query entropy 或 argmax 同分偏向误当成 sink。峰值位置和 top10% 质量还保留实际 chunk 长度对应的均匀参考，避免短尾块和取整造成伪偏好。

本地验证环境为 Python 3.12 / PyTorch 2.8.0 / Transformers 4.57.3，未替换项目的服务器环境。
