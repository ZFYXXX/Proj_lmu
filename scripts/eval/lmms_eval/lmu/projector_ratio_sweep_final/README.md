# Qwen2.5-Omni Projector 六方法评估

复用平台现有算法，在五个 benchmark 上运行当前实现支持的保留率。每份 YAML 只启用一个 projector stage；本文的 10%、25%、75%、50% 指最终音频 token **保留率**，不是删除率，也不是 Encoder 方法的逐层参数。

默认任务顺序：`audiomarathon2 → mmaunew3 → mmsufinal → mmarfinal → mmau_pro`。

## 文件与支持范围

| 方法 | 入口 SH | 注册状态（forward/generate） | 实际执行档位 | YAML 数 |
| --- | --- | --- | --- | --- |
| Segmentwise Top-K | `run_segmentwise_topk.sh` | active | 10% → 25% → 75% → 50% | 20 |
| UniAvg | `run_uniavg.sh` | cpu_contract_verified / target | 10% → 25% → 50% | 15 |
| UniSamp | `run_unisamp.sh` | cpu_contract_verified / target | 10% → 25% → 50% | 15 |
| UnSeg | `run_unseg.sh` | cpu_contract_verified / target | natural，每个 benchmark 一次 | 5 |
| LTBM | `run_ltbm.sh` | active | 10% → 25% → 75% → 50% | 20 |
| SA-MAP | `run_sa_map.sh` | active | 10% → 25% → 50% | 15 |

共 **8 个 SH、90 份 YAML**，入口共用本目录的 `common.bash`。`run_sa_map_keep50.sh` 仅运行 SA-MAP 的 50% 保留率，复用 `configs/sa_map/keep50/` 中的五份 YAML；`run_unisamp_keep50.sh` 仅运行 UniSamp 的 50% 保留率，复用 `configs/unisamp/keep50/` 中的五份 YAML。MMAU-Pro 沿用上一层的 `run_mmau_pro_stage.sh`。移动到服务器时保持目录位于 `scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final/`。

`cpu_contract_verified / target` 表示已有算法、variant、Qwen binding 和开发运行入口，且已通过独立数值与真实随机初始化 Transformers encoder 的 CPU 接入检查；尚未完成真实 checkpoint 验证并晋升 active。完整 `algorithm/variant/binding` 可运行，Manifest 应记录实际 CPU 验证状态。UnSeg preflight 的三维输入整理问题已修复；CPU 回归及七组真实 checkpoint 验证入口见[候选方法说明](../../../../../docs/support/qwen25_omni/audio_compression/candidate_methods.zh.md)。

六个算法身份：

| 方法 | algorithm / plugin | 插入位置 |
| --- | --- | --- |
| Segmentwise Top-K | `segmentwise_topk_audio` | pooler、ln_post 后，projector 前 |
| UniAvg | `uniform_average_pooling_audio` | projector 前 |
| UniSamp | `uniform_sampling_audio` | projector 前 |
| UnSeg | `unsupervised_segmentation_audio` | projector 前 |
| LTBM | `local_temporal_bipartite_merging_audio` | projector 前 |
| SA-MAP | `similarity_attention_merge_prune_audio` | projector 后，LLM 前 |

这些方法均不在 Audio Encoder 内部各层反复压缩。相对未压缩输入，主要减少送入 LLM 的音频 token；pre-projector 方法也减少 projector 处理量。UnSeg 需要额外的 audio-encoder preflight 来确定 placeholder 数，不能仅根据 token 减少就宣称端到端更快。

## 参数与不能生成的档位

| 目标保留率 | Segmentwise / LTBM `schedule.keep_ratio` | SA-MAP `schedule.keep_ratio` | UniAvg / UniSamp `schedule.compression_factor` |
| --- | --- | --- | --- |
| 10% | 0.10 | 0.10 | 10 |
| 25% | 0.25 | 0.25 | 4 |
| 75% | 0.75 | 本套配置已移除 | 不支持 |
| 50% | 0.50 | 0.50 | 2 |

- **Segmentwise Top-K**：`num_segments: 10`、`allocation: balanced_exact`，按最后一层 encoder attention 在时间分段内选 Top-K，恢复时间顺序。
- **LTBM**：`window_size: 8`，按局部二部分组 cosine 匹配，多轮均值合并到目标预算。
- **SA-MAP**：先相似度分组并按 attention 加权合并，再用 attention/DPP 选择到目标预算。保留已有 `similarity_threshold: 0.80`、`attention_power: 1.0`。7B LoRA 的 `keep75/audiomarathon2` 已出现合并后候选不足，因此本目录移除 SA-MAP 的 keep75 调度和五份 YAML，默认运行 10%、25%、50%。**剩余三档仍需真实数据验证；若分组后的 token 数小于目标 K，现有实现仍会报错。** 本次不修改算法、分组阈值或 DPP 参数。
- **UniAvg / UniSamp**：[Towards Audio Token Compression in Large Audio Language Models，v1 §IV](https://arxiv.org/html/2511.20973v1) 中的整数步长方法。UniAvg 的池化 kernel 和 stride 都为整数 K；UniSamp 选择 `0, K, 2K, ...`。现有代码明确要求 `keep_ratio = 1/K`，`0.75` 会报错。因此两个 SH 明确提示不支持 keep75，只运行 10%、25%、50%；未生成会报错的 keep75 YAML。
- **UnSeg**：同一论文中的自然分段方法，`budget_policy: natural_peaks`、`boundary_detector: strict_local_peak`，相邻 token 的 cosine dissimilarity 严格局部峰值决定分段，每段取均值。输出数为峰值数 + 1；代码明确拒绝 `keep_ratio`。配置放在 `configs/unseg/natural/`，不重复运行四次并伪装成四档。

因此当前实现无法构成 6 × 4 × 5 = 120 个有效固定比例实验。若以后必须统一四档，需要另行定义非整数重采样及固定预算分段变体，并与原论文实现区分。

对一个含 N 个 projector 输入/输出音频 token 的音频单元：

- Segmentwise / LTBM / SA-MAP：`K = min(N, max(1, floor(N × keep_ratio)))`；SA-MAP 还需满足上述分组条件。
- UniAvg：`max(1, floor(N / compression_factor))`；通常丢弃不足一个完整池化窗口的尾部，N 小于窗口时对整段取均值。
- UniSamp：`ceil(N / compression_factor)`，包含起始 token。
- UnSeg：由内容决定。论文的 8.23 tokens/s 是实验观测值，不能作为每条音频的固定预算。

短音频和多音频逐项取整会使实际总体保留率偏离标签。比如 N=25、目标 10% 时，前三种固定预算方法和 UniAvg 输出 2 个，UniSamp 输出 3 个。

## 服务器环境与数据

沿用之前评估所用的 Python、GPU、模型和 MMAU-Pro evaluator 环境变量。数据路径需要对应服务器现有文件：

- `AUDIOMARATHON_TASK_DIR` 指向 AudioMarathon2 task 包，`AUDIOMARATHON_ROOT` 指向其音频根目录。
- `MMSUFINAL_DATASET_DIR`、`MMARFINAL_DATASET_DIR` 分别指向包含 `data/*.parquet` 和 `_audio_cache/` 的数据目录。
- 服务器 `lmms_eval/tasks/mmsufinal/mmsufinal.yaml` 与 `mmarfinal/mmarfinal.yaml` 中的 `dataset_kwargs.data_files.test` 必须分别指向服务器上的 `MMSU.jsonl` 和 `MMAR.jsonl`。上述环境变量不会替换 JSONL 路径；复制本地 task YAML 后需要更新其中的 `/Users/...` 路径。
- `mmau_pro` 需要 `MMAU_PRO_QWEN_JUDGE_MODEL` 和 `MMAU_PRO_NVEMBED_MODEL`，小样本运行也需要；plan-only 不加载 evaluator。

你之前确认的 cuDNN 冲突来自 `LD_LIBRARY_PATH` 开头的 `/software/all/devel/cuda/12.4/lib64`。在同一环境的评估命令子 shell 中、启动 SH 前可移除这个已定位目录，保留后续 MPI/Intel 路径：

```bash
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi
"$PYTHON_BIN" -c 'import torch; print(torch.__version__, torch.backends.cudnn.version())'
```

此修正针对已确认的节点环境，不写入通用 SH。

## 启动

以下片段接在已有的完整环境设置之后执行；在服务器上使用已上传的新目录。

```bash
export SWEEP_DIR="$PROJECT_ROOT/scripts/eval/lmms_eval/lmu/projector_ratio_sweep_final"
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"
export MMSUFINAL_DATASET_DIR="$DATASET_ROOT/mmsu"
export MMARFINAL_DATASET_DIR="$DATASET_ROOT/mmar"

unset ADAPTER_PATH MODEL_PATH MERGED_MODEL_PATH OUTPUT_DIR
export CHECKPOINT_PATH="$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged"
export RUN_TAG="projector-7b-lora-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_projector_sweep_final

# 计划检查：不加载模型、不合并 LoRA、不运行 evaluator、不创建输出目录。
for method in segmentwise_topk uniavg unisamp unseg ltbm sa_map; do
  bash "$SWEEP_DIR/run_${method}.sh" --plan_only || break
done
```

小样本评估，建议先在已经跑通数据路径的 `audiomarathon2` 上逐方法验证：

```bash
export RUN_TAG="smoke-projector-7b-lora-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_projector_smoke

for method in segmentwise_topk uniavg unisamp unseg ltbm sa_map; do
  BENCHMARKS=audiomarathon2 bash "$SWEEP_DIR/run_${method}.sh" \
    --num_processes 4 --use_cache --sample_limit 1 --debug || break
done
```

`--sample_limit 1` 对任务组中的每个子任务分别生效，不代表整组只读取一个样本。`--debug` 用于观察真实压缩，不能用 plan-only 代替。

完整五任务评估，各方法在相同 GPU 上顺序运行：

```bash
export RUN_TAG="eval-projector-7b-lora-$(date +%Y%m%d-%H%M%S)"
export EVAL_NAME=audio_projector_sweep_final
export BENCHMARKS="audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro"

bash "$SWEEP_DIR/run_segmentwise_topk.sh" --num_processes 4 --use_cache
bash "$SWEEP_DIR/run_uniavg.sh" --num_processes 4 --use_cache
bash "$SWEEP_DIR/run_unisamp.sh" --num_processes 4 --use_cache
bash "$SWEEP_DIR/run_unseg.sh" --num_processes 4 --use_cache
bash "$SWEEP_DIR/run_ltbm.sh" --num_processes 4 --use_cache
bash "$SWEEP_DIR/run_sa_map.sh" --num_processes 4 --use_cache
```

只运行 SA-MAP 50% 档位时，沿用上述模型、数据与 evaluator 环境，执行下面对应的一条命令：

```bash
# 五个 benchmark 的 50% 小样本评估。
RUN_TAG="smoke-sa-map-keep50-$(date +%Y%m%d-%H%M%S)" \
  bash "$SWEEP_DIR/run_sa_map_keep50.sh" --num_processes 4 --use_cache --sample_limit 1 --debug

# 五个 benchmark 的 50% 完整评估。
RUN_TAG="eval-sa-map-keep50-$(date +%Y%m%d-%H%M%S)" \
  bash "$SWEEP_DIR/run_sa_map_keep50.sh" --num_processes 4 --use_cache
```

独立入口仍支持 `BENCHMARKS` 筛选任务，输出仍位于 `sa_map/keep50/<task>/`，不重复创建配置。

只补跑 UniSamp 50% 时，设置 `CHECKPOINT_PATH` 为待评估模型的合并权重，并沿用上述数据与 evaluator 环境：

```bash
RUN_TAG="eval-unisamp-keep50-$(date +%Y%m%d-%H%M%S)" \
  bash "$SWEEP_DIR/run_unisamp_keep50.sh" --num_processes 4 --use_cache
```

该入口默认依次运行五个 benchmark，仅使用 `compression_factor: 2`、`keep_ratio: 0.5`。输出位于 `unisamp/keep50/<task>/`；支持 `BENCHMARKS`、`--plan_only`、`--sample_limit` 和 `--debug`。`7B LoRA (2e-5)` 中的学习率由训练 checkpoint 决定，不是本次评估的参数。

每个 SH 遇到错误立即停止。普通任务与 MMAU-Pro 保留原 benchmark 的 generation 协议和评分入口；压缩只作用于被评估的 Qwen 模型。

## 输出与判定

默认输出：`runs/lmu/eval/<RUN_TAG>/<EVAL_NAME>/<method>/<keep10|keep25|keep75|keep50|natural>/<task>/`。每个有效设置有独立的 Manifest、结果和 debug 产物。

小样本完成后应核对：

1. Manifest 的 algorithm、variant、binding、projector placement、支持状态与本表一致；stage 已启用。
2. `debug_diagnostic*.jsonl` 中 controller 报告了实际应用的压缩，输入与输出音频 token 数对得上。UnSeg 应记录自然峰值、preflight 和实际输出数，不应出现伪造的固定比率。
3. 占位符、生成结果和任务评分完成；没有仅启动模型却未执行到压缩的情况。

`--plan_only` 和本地 CPU 核验不证明服务器数据、四 GPU 推理、MMAU-Pro judge 或压缩后的质量已验证。本次检查结果见 `VALIDATION.md`。

## 其他设置

- 进程数优先级：`EVAL_NUM_PROCESSES` → CLI `--num_processes` → visible GPU 数；单 GPU 测试时同步设置 `EVAL_NUM_PROCESSES=1`。
- MMAU-Pro 可通过 `MMAU_PRO_CUDA_VISIBLE_DEVICES`、`MMAU_PRO_NUM_GPUS` 单独设置 GPU；它们默认继承普通任务的设置。
- 普通任务端口起点依次为 29674、29684、29694、29704、29714、29724，MMAU-Pro 为起点 + 100。
- 可以用 `AUDIOMARATHON_MAIN_PROCESS_PORT`、`MMAUNEW3_MAIN_PROCESS_PORT`、`MMSUFINAL_MAIN_PROCESS_PORT`、`MMARFINAL_MAIN_PROCESS_PORT`、`MMAU_PRO_MAIN_PROCESS_PORT` 覆盖对应端口。
- `--config`、`--tasks`、`--checkpoint`、`--target`、`--compression_mode`、`--output_dir`、`--manifest` 由 sweep 管理；通过 `CHECKPOINT_PATH` / `MODEL_PATH` / `ADAPTER_PATH`、`BENCHMARKS`、`OUTPUT_DIR` 修改对应输入。
- `--dry_run` 等同于 `--plan_only`；也可传入 `--debug`、`--profile` 和平台支持的生成参数。
