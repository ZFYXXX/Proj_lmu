# 10.32.7.14 服务器执行命令

以下在 `/home/LuBowen/code/MLLM-Comp-Platform` 执行，要求服务器使用当前版本的 `audio_acoustic`。沿用已有 MMSU 路径，不需要 probe、外部训练音频或额外声学模型。全部条件保持原生 token 数。

## 环境与完整双卡评估

```bash
source /home/LuBowen/miniconda3/etc/profile.d/conda.sh
conda activate /home/LuBowen/miniconda3/envs/mllm-comp-platform

export PROJECT_ROOT=/home/LuBowen/code/MLLM-Comp-Platform
export PYTHON_BIN=/home/LuBowen/miniconda3/envs/mllm-comp-platform/bin/python
export CHECKPOINT_PATH=/home/WuHao/checkpoints/ckpts-omni/Qwen2.5-Omni-7B
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
export RUN_GROUP="$PROJECT_ROOT/runs/analysis/acoustic_all_$(date +%Y%m%d_%H%M%S)"
cd "$PROJECT_ROOT"

# 应显示 56 个唯一条件、285000 次生成、10 个实验视图。
bash scripts/analysis/audio_acoustic/run_all_mmsufinal_2gpu.sh --plan-only

# 正式运行，结束后生成全部报告。
bash scripts/analysis/audio_acoustic/run_all_mmsufinal_2gpu.sh
```

每张 GPU 加载一个完整模型，固定两个 worker，按题并行。完整套件只启动一次评估，每种条件仍覆盖全部 5,000 题。相较删除合并实验前的共享运行，生成次数由 405,000 降为 285,000；当前 10 个入口各自运行需 425,000 次，共享执行节省 140,000 次。

`RUN_GROUP` 和 `$RUN_GROUP.log` 必须是新路径；失败会保留输出，不自动断点续跑。原 MMSU 配置可用时无需设置数据路径；覆盖参数为问题 JSONL 的 `MMSU_JSONL` 和含 `data/*.parquet` 根目录的 `MMSUFINAL_DATASET_DIR`。

## 双卡小规模验证

默认完整条件网格的两题验证，共 114 次生成：

```bash
OUTPUT_DIR="$PROJECT_ROOT/runs/analysis/acoustic_all_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/analysis/audio_acoustic/run_experiment_2gpu.sh all --limit 2
```

检查 `coverage.json` 和各报告的 `identity_controls_passed`。缩小低通网格可以加 `--cutoffs-hz 4000`。已移除 `merge`、`--keep-ratios` 和 `--merge-methods`。

## 单项运行与报告重建

```bash
# 只做原始音高属性统计与分组准确率，无干预。
OUTPUT_DIR="$PROJECT_ROOT/runs/analysis/acoustic_pitch_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/analysis/audio_acoustic/run_experiment_2gpu.sh pitch

# 高频低通必要性实验。
OUTPUT_DIR="$PROJECT_ROOT/runs/analysis/acoustic_frequency_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/analysis/audio_acoustic/run_experiment_2gpu.sh frequency_energy

# 已有完整结果时，只重建报告，不重新推理。
"$PYTHON_BIN" -m scripts.analysis.audio_acoustic.report \
  --run-dir "$RUN_GROUP" --experiment pitch --bootstrap 1000

# 重建全部 10 份报告。
"$PYTHON_BIN" -m scripts.analysis.audio_acoustic.report \
  --run-dir "$RUN_GROUP" --bootstrap 1000 --tolerance-pp 1.0
```

共享记录在 `$RUN_GROUP/`，报告在 `$RUN_GROUP/report/<experiment>/`，总索引为 `report/summary.json`，日志为 `$RUN_GROUP.log`。音高／包络／停顿三项输出原始属性分布与原始分组准确率，不输出压缩损伤、合并曲线或属性解码误差。

属性报告包含 `attribute_assignments.csv`、`attribute_baseline.csv`、`attribute_controls.csv` 与 `attribute_<name>.png/pdf`；包络报告另有 `envelope_examples`。其他干预继续输出配对损伤、类别和层漂移指标。

## 默认成本

| 入口 | 干预数（含恒等） | 单独运行生成次数 | 共享方式 |
| --- | ---: | ---: | --- |
| observe | 0 | 5,000 | 原始基线 |
| band_importance | 17 | 90,000 | 与 energy_importance 共享删带 |
| complementarity | 11 | 60,000 | 共享原始基线、波形恒等 |
| energy_importance | 25 | 130,000 | 共享删带条件 |
| pitch／envelope／rhythm（各自） | 0 | 5,000 | 只复用原始基线，无额外生成 |
| frequency_energy | 11 | 60,000 | 共享原始基线、波形恒等 |
| representation_spectrum | 4 | 25,000 | 共享原始基线 |
| spectral_resolution | 7 | 40,000 | 共享原始基线 |
| **all** | **56 个唯一干预** | **285,000** | **10 份报告** |
