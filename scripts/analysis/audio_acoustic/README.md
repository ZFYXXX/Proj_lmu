# MMSU-final 独立声学分析

本目录保留 **10 个分析入口**：原始观测、频带必要性、跨频带交互、能量与任务必要性、音高属性、包络属性、停顿代理、Frequency Energy Analysis、表示时间频谱、Mel 分辨率。

**所有条件保持模型原生的 encoder／projector token 数，不执行额外 token 合并、删除或下采样。** `merge` 实验及其广播／真压缩实现、`--keep-ratios`、`--merge-methods` 已移除。`pitch`、`envelope`、`rhythm` 只使用原始答案分析输入属性分布和各属性组的基线准确率，不再执行压缩条件。模型自带的卷积、平均池化与 projector 保持原生行为。

入口只在自己的 worker 进程中安装临时观测、波形／Mel 干预和保长度平滑 hook，不修改 backbone、压缩方法注册表、训练流程和原任务文件。复用原生 `LMMSEvalAdapter`、`mmsufinal` 提示词／生成参数／评分器；仅从相邻 `audio_temporal.metrics` 复用 JSON 序列化辅助函数，不再导入其压缩 controller。

环境要求：**完整 Qwen2.5-Omni 3B／7B 或同结构完整 checkpoint、Transformers 4.57.3、16 kHz 单音频**。不需要 probe 权重、外部训练集或额外声学网络。每卡一个模型副本，`runtime_batch_size=1`，按题并行。默认观察 C、5 个固定深度的 E_l、P_in／P_out；`--layers all` 可观察全部音频 encoder 层。

详细实验：[EXPERIMENTS.zh.md](EXPERIMENTS.zh.md)。服务器命令：[TERMINAL_COMMANDS.md](TERMINAL_COMMANDS.md)。

## 双卡完整运行

```bash
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export RUN_GROUP="$PROJECT_ROOT/runs/analysis/acoustic_all_$(date +%Y%m%d_%H%M%S)"
bash scripts/analysis/audio_acoustic/run_all_mmsufinal_2gpu.sh --plan-only
bash scripts/analysis/audio_acoustic/run_all_mmsufinal_2gpu.sh
```

先设置 `CHECKPOINT_PATH` 和项目 Python 环境。默认 **56 个唯一干预＋每题 1 个原始基线＝285,000 次生成**，每条件完整覆盖 5,000 题。独立运行全部 10 个入口需 425,000 次，共享节省 140,000 次；相较移除合并前的 405,000 次共享运行，减少 120,000 次。音高／包络／停顿报告共用原始基线，不增加生成次数。

`RUN_GROUP` 和 `$RUN_GROUP.log` 必须是新路径。输出逐题记录、冻结计划、覆盖检查及 `report/<experiment>/` 的 10 份报告；不存在自动断点续跑。总脚本只接受 `--plan-only`／`--help`。

## 小规模验证与单项运行

```bash
# 两题覆盖全部 56 个干预，共 114 次生成。
OUTPUT_DIR="$PROJECT_ROOT/runs/analysis/acoustic_all_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/analysis/audio_acoustic/run_experiment_2gpu.sh all --limit 2

# 单项原始音高属性分析：无干预，每题生成一次。
OUTPUT_DIR="$PROJECT_ROOT/runs/analysis/acoustic_pitch_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/analysis/audio_acoustic/run_experiment_2gpu.sh pitch

# 高频必要性：省略 --limit 时每个条件覆盖完整 5,000 题。
OUTPUT_DIR="$PROJECT_ROOT/runs/analysis/acoustic_frequency_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/analysis/audio_acoustic/run_experiment_2gpu.sh frequency_energy
```

`--limit` 只用于 smoke，且不能小于 worker 数。需缩小低通网格时可传 `--cutoffs-hz 4000`；不再接受合并方法或 token 保留比例参数。单卡／多卡共享脚本为 `run_experiment.sh`，默认 `NUM_PROCESSES=4`；双卡脚本固定 2 个 worker。

## 输出与统计

| 输出 | 内容 |
| --- | --- |
| `analysis.json`、`eval.resolved.yaml`、`launch.json` | 实验条件、checkpoint、层、数据覆盖、启动参数、预计生成次数 |
| `runtime.rank*.json` | 实际依赖版本、编码器来源、dtype／backend |
| `samples.rank*.jsonl` | 每题一次原始答案、原评分器得分、线性谱能量、表示统计、输入声学属性 |
| `conditions.rank*.jsonl` | 每题×干预的答案、得分、移除能量、层漂移和原生 token 数 |
| `coverage.json` | 全部条件是否完整、唯一、配对覆盖同一批题目 |
| `example.rank*.doc*.npz` | 默认 doc 0／1／2 的 Mel、hidden、名义时间坐标和声学代理轮廓 |
| `report/<experiment>/summary.json` | 基线、条件／层／类别汇总及恒等对照结果 |
| `conditions.csv`、`category_damage.csv` | 条件指标和题型／细分任务损伤；纯观测入口只有表头 |
| `*.png`、`*.pdf` | 输入能量、频带损伤、交互、层漂移、时间频谱或属性基线图，按入口生成 |

JSON／CSV 准确率使用 0–1，图中准确率使用百分比；`damage_pp` 使用百分点。干预损伤为原始准确率减去条件准确率。置信区间按 `audio_path` 聚类 bootstrap，默认 1,000 次、95% 区间。干预默认容忍损伤 `TOLERANCE_PP=1.0`，需事先设定；smoke 或恒等对照失败时不作达标判定。逐条件区间未进行多重选择校正。

`question_type` 是 MMSU 的 perception／reasoning，不是 semantic／acoustic 标签。原题答案保持固定；干预后的得分用于诊断原任务信息保留情况，不是重新标注被改变后的音频。

## 原始声学属性报告

`pitch`、`envelope`、`rhythm` 各有 **0 个干预条件**，独立运行各生成 5,000 个原始答案；在 `all` 中共享同一份基线。

| 入口 | 原始输入属性 | 解释边界 |
| --- | --- | --- |
| `pitch` | 中位 F0、F0 四分位距（cents）、voicing 比例 | ACF 单音代理，不测 hidden 读出能力 |
| `envelope` | 平均 log-RMS、log-RMS 标准差、P90−P10 范围 | 数字幅度与起伏，不是校准声压级或重音标签 |
| `rhythm` | 全部低活动帧比例、≥200 ms 低活动段平均时长、这些段每秒数量 | 低活动结构，不是真实语速／音节率 |

ACF／RMS 使用 40 ms 窗、10 ms hop；F0 搜索 50–800 Hz，周期性阈值 0.6，至少 5 个有效帧才报告 clip 级音高统计。活动阈值为最大 RMS 以下 40 dB、下限 −80 dB，并排除近零幅度。

每种属性仅根据唯一音频的输入计算 1/3 和 2/3 分位点，不使用 QA 分数；同值不拆分，恒定值和缺失值单列。各组报告原始准确率、音频聚类 95% CI、题目数与音频数，另按任务／子类别／时长／原 token 数细分。这些结果描述输入属性与原模型准确率的关联，不测压缩敏感性，也不进行因果隔离。

| 属性输出 | 内容 |
| --- | --- |
| `attribute_assignments.csv` | 每题原始属性、分组、时长、原 token 数及任务信息 |
| `attribute_baseline.csv` | 属性组原始准确率、95% CI、样本数；空组指标留空 |
| `attribute_controls.csv` | 属性组内部按任务／子类别／时长／原 token 数细分的原始准确率 |
| `attribute_<name>.png/pdf` | 原始属性直方图及分位点、各组原始准确率和 CI |
| `envelope_examples.png/pdf` | 所选示例的原始 log-RMS 轮廓 |

不再生成合并曲线、压缩损伤散点、`attribute_damage.csv` 或 probe 输出。已有完整结果时可只重建报告，无需重新推理：

```bash
"$PYTHON_BIN" -m scripts.analysis.audio_acoustic.report --run-dir "$RUN_GROUP" --experiment pitch
```

## 执行边界与验证

- 沿用现有 GPU、解码、NumPy／PyTorch／Matplotlib 环境。`MMSUFINAL_DATASET_DIR` 指向含 `data/*.parquet` 的根目录；`MMSU_JSONL` 单独覆盖问题 JSONL；原配置可用时无需设置。
- 波形滤波保持采样率和时长，Mel 粗化广播回原维度，P_in 时间低通保持序列长度。Mel 的均值操作仍是保尺寸的频率分辨率干预。
- 条件表示与原始表示直接在原生 token 网格上比较；长度改变会被拒绝。时间坐标是采样／池化中心的名义坐标，hidden 维度不是 Hz。
- `diagnostic_seconds` 包含分析和 hook 开销，不用于宣称推理加速。本套件不再验证 token 合并效果。

```bash
python -m pytest -q scripts/analysis/audio_acoustic/test_audio_acoustic.py
bash -n scripts/analysis/audio_acoustic/run_experiment.sh
bash -n scripts/analysis/audio_acoustic/run_experiment_2gpu.sh
bash -n scripts/analysis/audio_acoustic/run_all_mmsufinal_2gpu.sh
```

测试使用小型随机 Transformers 编码器／Thinker，无需下载 checkpoint；覆盖全部条件的长度保持、真实冻结 runner、信号处理、属性基线报告、共享覆盖与双卡启动参数。完整 7B／MMSU GPU 验证须以服务器实际日志为准。
