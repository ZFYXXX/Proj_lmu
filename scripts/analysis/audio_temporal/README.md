# MMSU-final 音频时间性质分析

本目录是一个独立分析入口，只在运行时安装只读 hook 或请求级干预，不修改
`mllm_comp`、模型权重、MMSU-final 提示、问题、答案和原任务评分器。原“实验 2：
排除公共方向、背景和说话人解释”已删除，CLI 中不存在 `experiment=2` 或对应条件。

## 可独立运行的实验

| CLI 名称 | 对应分析 | 是否重跑干预条件 |
| --- | --- | --- |
| `capture` | 公共表征捕获与时间映射：`C/E_l/E_L/P_in/P_out` | 否 |
| `provenance` | 实验 13：原生 LTBM 合并后 token 的实际时间来源、跨度、空洞与碎片数 | 否；1 次压缩生成 |
| `similarity` | 实验 1：多尺度毫秒 lag-cosine | 否 |
| `regions` | 实验 3：silent/stable/transient/boundary 区域分析 | 是 |
| `predictability` | 实验 4：连续时间块预测 | 是 |
| `substitution` | 实验 5：高相似 token 等长替代 | 是 |
| `locality` | 实验 7：匹配 cosine 和 token 数后的 local/global 对照 | 是 |

详细定义、变量控制和判据见 [EXPERIMENTS.zh.md](EXPERIMENTS.zh.md)。

## 运行

在项目根目录执行一个实验：

```bash
export CHECKPOINT_PATH=/absolute/path/to/full-or-merged-checkpoint
bash scripts/analysis/audio_temporal/run_experiment.sh similarity \
  /absolute/path/to/runs/audio-temporal-similarity
```

完整 MMSU-final 默认 5,000 道题。先用小样本验证运行链：

```bash
bash scripts/analysis/audio_temporal/run_experiment.sh provenance \
  /absolute/path/to/runs/audio-temporal-ltbm-provenance-smoke \
  --limit 8 --num-processes 1
```

按依赖顺序逐个执行完整实验（每个输出目录都必须尚不存在）：

```bash
bash scripts/analysis/audio_temporal/run_experiment.sh capture       runs/temporal-00-capture
bash scripts/analysis/audio_temporal/run_experiment.sh provenance    runs/temporal-13-provenance
bash scripts/analysis/audio_temporal/run_experiment.sh similarity    runs/temporal-01-similarity
bash scripts/analysis/audio_temporal/run_experiment.sh regions       runs/temporal-03-regions
bash scripts/analysis/audio_temporal/run_experiment.sh predictability runs/temporal-04-predictability
bash scripts/analysis/audio_temporal/run_experiment.sh substitution  runs/temporal-05-substitution
bash scripts/analysis/audio_temporal/run_experiment.sh locality      runs/temporal-07-locality
```

若目标环境的解释器不是 `python`，可预先设置例如 `PYTHON_BIN=python3.10`。

也可直接调用：

```bash
python -m scripts.analysis.audio_temporal.launch \
  --experiment predictability \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir /absolute/path/to/run \
  --block-ms 40,80,160,320 \
  --replace-ratios 0.1

python -m scripts.analysis.audio_temporal.report \
  --run-dir /absolute/path/to/run
```

`run_experiment.sh` 会在数据收集完成后自动生成报告。输出目录必须尚不存在，避免覆盖已有结果。

### 双卡运行精简后的完整核心实验

`run_core_suite_2gpu.sh` 会依次运行原生 LTBM 的 `provenance`、基线
`similarity`、`regions`、精简后的 `predictability` 和 `substitution`。
`similarity` 同时保存公共表征，因此不再单独重复 `capture`；该套件不运行实验 7 的
`locality`。总计算量为 21 个完整 MMSU 等价遍历，即 105,000 次
生成；两张 GPU 只负责并行分片，不改变总生成次数。

```bash
bash scripts/analysis/audio_temporal/run_core_suite_2gpu.sh
```

默认使用 GPU `0,1` 和本项目远端环境的 MMSU-final、Python、checkpoint 路径；仍可在
启动前通过同名环境变量覆盖。为避免 SSH 断开终止任务，建议在 `tmux` 中执行。

## 输出

- `analysis.json`：运行前冻结的层、lag、预算和全部条件。
- `samples.rank*.jsonl`：通常为无压缩基线；`provenance` 为真实 LTBM 压缩生成，包含逐轮配对和每个输出 token 的来源。
- `conditions.rank*.jsonl`：逐题配对干预结果、耗时和 provenance。
- `example.rank*.doc*.npz`：预先指定样本的表征、时间轴、区域与覆盖数据。
- `coverage.json`：每个条件是否覆盖全部目标题目。
- `report/*.csv`：lag、区域、stage 长度、原生 provenance 和条件汇总；实验 13 另有 LTBM provenance 表。
- `report/*.png` 与 `report/*.pdf`：论文图和检查图。
- `report/summary.json`：机器可读的最终统计与解释限制。

## 重要边界

1. 区域标签是由输入 log-Mel 活动度与 spectral flux 生成的代理标签；MMSU-final
   没有逐帧 VAD/事件真值，报告不会把代理标签写成真值。
2. 所有减长干预位于原生 `avg_pooler + ln_post` 后、`proj` 前的 `P_in`。
   因而能测下游 projector/LLM token 减少和端到端生成耗时，不能宣称减少了 audio
   encoder 自身 FLOPs。
3. 实验 7 的自定义 merge 不跨音频文件或 encoder 硬分块；实验 13 使用原生 LTBM，
   它可能跨硬分块，报告会统计实际跨越数量，不能套用实验 7 的约束。
4. 毫秒覆盖是显式的 token 来源映射，不等同于深层网络的理论或有效感受野。
5. lag 相似性和 provenance 本身只说明表征/来源结构；要判断任务可替代性还需看
   等长替代。实验 13 的压缩分数没有同次运行的配对无压缩基线，不能单独解释为压缩损伤。
6. 为减少耗时比较被固定执行位置混淆，所有预先冻结的条件按全局题号做循环轮换；
   `conditions.rank*.jsonl` 的 `evaluation_order` 保存该条件在本题中的实际执行位置。
