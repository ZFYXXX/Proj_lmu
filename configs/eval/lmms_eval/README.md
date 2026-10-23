# lmms-eval 可运行配置

本目录只保存项目级完整评测配置：

普通配方省略固定输出路径，每次调用默认生成 `runs/<experiment.name>/<run_id>/`。
显式 `--output_dir` 使用精确目录；未指定 `--manifest` 时采用该目录下的
`manifest.json`。论文 sweep、LMU 和 ImageLLM probe 配方保留 run tag / 环境变量
协议，14-bench wrapper 也继续传入时间戳目录。它们不会再自动追加 run_id，已有输出
必须换新位置；完整规则见[运行压缩实验](../../../docs/guides/1_run_compression_experiment.zh.md)。

| 配置 | 模型 | 任务 | 状态 |
| --- | --- | --- | --- |
| `qwen25_omni_img.yaml` | Qwen2.5-Omni | MME | 已完成双卡复测 |
| `qwen25_omni_aud.yaml` | Qwen2.5-Omni | MMAU-test-mini | 已完成双卡复测 |
| `qwen25_omni_aud_segmentwise_topk.yaml` | Qwen2.5-Omni | MMAU-test-mini + Projector-Level audio pruning | 真实 checkpoint smoke 完成，待目标规模 MMAU 验证 |
| `qwen25_omni_aud_segmentwise_topk_paper.yaml` | Qwen2.5-Omni | MMAU-test-mini + 论文式 50/25/10% sweep 模板 | 由 sweep 脚本注入保留率与独立 run tag |
| `qwen25_omni_aud_atome.yaml` | Qwen2.5-Omni | MMAU-test-mini + Encoder-Level A-ToMe | 真实 checkpoint smoke 完成，待目标规模 MMAU 验证 |
| `qwen25_omni_aud_ltbm.yaml` | Qwen2.5-Omni | MMAU-test-mini + Projector-Level LTBM | 真实 3B generate smoke 完成，待目标规模 MMAU 验证 |
| `qwen25_omni_aud_sa_map.yaml` | Qwen2.5-Omni | MMAU-test-mini + Projector-Level SA-MAP | 真实 3B generate smoke 完成，待目标规模 MMAU 验证 |
| `qwen25_omni_aud_encoder_topk.yaml` | Qwen2.5-Omni | MMAU-test-mini + Encoder-Level Attention TopK | 真实 3B generate smoke 完成，待目标规模 MMAU 验证 |
| `qwen25_omni_aud_progressive_pruning.yaml` | Qwen2.5-Omni | MMAU-test-mini + Encoder-Level Progressive Pruning | 真实 3B generate smoke 完成，待目标规模 MMAU 验证 |
| `qwen25_omni_aud_audio_pyramid.yaml` | Qwen2.5-Omni | MMAU-test-mini + LLM-Level Audio-Pyramid | 真实 3B 多步 Decode 与 backward 完成，待目标规模 MMAU 验证 |
| `qwen25_omni_aud_fastslm_hta.yaml` | Qwen2.5-Omni | MMAU-test-mini + Projector-Level FastSLM HTA | target；数值/小模型及选定组合的真实 3B 联合训练恢复通过；需 HTA 权重，其他场景待验证 |
| `qwen25_omni_aud_uniform_average_pooling.yaml` | Qwen2.5-Omni | MMAU-test-mini + Projector-Level UniAvg | target；数值与原生小模型测试通过，真实 3B 生命周期待验证 |
| `qwen25_omni_aud_uniform_sampling.yaml` | Qwen2.5-Omni | MMAU-test-mini + Projector-Level UniSamp | target；数值与原生小模型测试通过，真实 3B 生命周期待验证 |
| `qwen25_omni_aud_unsupervised_segmentation.yaml` | Qwen2.5-Omni | MMAU-test-mini + Projector-Level UnSeg | target；数值、自然峰值预算与小模型重放测试通过，真实 3B 生命周期待验证 |
| `qwen25_omni_aud_fastadasp.yaml` | Qwen2.5-Omni | MMAU-test-mini + LLM-Level FastAdaSP | target；真实 3B 压缩/输入梯度/生成代表场景通过，完整生命周期与 MMAU/效率待验证 |
| `qwen25_omni_aud_headrouter.yaml` | Qwen2.5-Omni-3B | MMAU-test-mini + LLM-Level HeadRouter | target；官方 profile 与算法已接入，待真实 3B 生命周期和目标规模 MMAU 验证 |
| `qwen25_omni_spade_operation_only.yaml` | Qwen2.5-Omni | MMAU-test-mini + LLM-Level SPADE Operation-Only | 真实 3B 结构改写、生成与导出重载完成，待目标规模 MMAU 验证 |
| `qwen25_vl_imagellm_tome_forward_probe.yaml` | Qwen2.5-VL | MME + ToMe forward probe | 真实 3B、2 样本验证；永久 image feature 均为 `279→271` |
| `qwen25_vl_imagellm_visionzip_forward_probe.yaml` | Qwen2.5-VL | MME + VisionZip forward probe | 真实 3B、2 样本验证；永久 image feature 均为 `279→32` |
| `qwen25_vl_imagellm_pyramid_drop_forward_probe.yaml` | Qwen2.5-VL | MME + PyramidDrop forward probe | 真实 3B、2 样本验证；LLM 序列分别到 `75`、`74` |
| `qwen25_vl_14bench.yaml` | Qwen2.5-VL | 14 个 image benchmark | 单卡 baseline 入口 |
| `qwen3_vl.yaml` | Qwen3-VL | 配置指定任务 | integration smoke |
| `qwen3_5_mme_smoke.yaml` | Qwen3.5 / Qwen3.8 | 本地 MME 单样本 baseline | Transformers 5.19 image-only integration smoke |
| `qwen3_5_tome_mme_smoke.yaml` | Qwen3.5 / Qwen3.8 + ToMe | 本地 MME 单样本 compressed | Post-merger ToMe binding; average 50% feature retention |
| `qwen3_5_mme_full_divprune.yaml` | Qwen3.5 / Qwen3.8 + DivPrune | 本地 MME | Projector-boundary greedy max-min selection; 50% feature retention for a matched ToMe budget |
| `qwen3_5_eval_base.yaml` | Qwen3.5 / Qwen3.8 | GQA / MMBench | Shared SDPA baseline; select task and smoke limit at launch |
| `qwen3_5_eval_tome.yaml` | Qwen3.5 / Qwen3.8 + ToMe | GQA / MMBench | Shared SDPA ToMe comparison at 50% retention |
| `qwen3_5_eval_divprune.yaml` | Qwen3.5 / Qwen3.8 + DivPrune | GQA / MMBench | Shared SDPA DivPrune comparison at 50% retention |
| `tasks/qwen3_5_local/` | Local GQA testdev and MMBench EN dev | GQA / MMBench | Uses Yin's local parquet files; MMBench uses the static dev metric, with no Hub dataset/API dependency |
| `scripts/eval/lmms_eval/qwen3_5_method_comparison_lane.sh` | Qwen3.8 comparison runner | MME / GQA / MMBench | Runs MME DivPrune on both GPUs, then queues local GQA/MMBench Base, ToMe, and DivPrune comparisons across two GPU lanes |
| `scripts/eval/lmms_eval/qwen3_5_bf16_comparison_lane.sh` | BF16 Qwen3.8 comparison runner | MME / GQA / MMBench | Full BF16 Base/ToMe/DivPrune queues; one process per GPU, local parquet tasks, 16-token generation cap |
| `scripts/eval/lmms_eval/qwen3_5_bf16_comparison_lane.sh` | BF16 Qwen3.8 comparison runner | MME / GQA / MMBench | Full BF16 Base/ToMe/DivPrune queues; one process per GPU, local parquet tasks, 16-token generation cap |
| `qwen3_vl_14bench.yaml` | Qwen3-VL | 14 个 image benchmark | 单卡 baseline 入口 |
| `qwen3_vl_linear_attention.yaml` | Qwen3-VL | MME + Linear Attention replacement | capability active；精确 prepare 单元 declared；质量/效率待实验 |
| `qwen3_vl_cnn_attention.yaml` | Qwen3-VL | MME + CNN Attention replacement | capability active；精确 prepare 单元 declared；质量/效率待实验 |
| `qwen3_vl_identity_attention.yaml` | Qwen3-VL | MME + Identity Attention replacement | prepare 单元 declared；模块输入恒等对照，不等于 baseline 数值 |
| `qwen3_vl_lolcats_linear_window.yaml` | Qwen3-VL | MME + LoLCATS Linear Window replacement | capability active；精确 prepare 单元 declared；质量/效率待实验 |
| `qwen3_vl_nha.yaml` | Qwen3-VL | MME + NHA replacement | capability active；精确 prepare 单元 declared；质量/效率待实验 |
| `qwen3_vl_sparse_window_attention.yaml` | Qwen3-VL | MME + Sparse Window Attention replacement | capability active；精确 prepare 单元 declared；质量/效率待实验 |
| `qwen3_vl_ffn_low_rank.yaml` | Qwen3-VL | MME + Low-Rank FFN replacement | capability active；精确 prepare 单元 declared；质量/效率待实验 |
| `qwen3_vl_ffn_identity.yaml` | Qwen3-VL | MME + Identity FFN replacement | prepare 单元 declared；模块输入恒等对照，不等于 baseline 数值 |
| `qwen3_vl_ffn_two_layer_mlp.yaml` | Qwen3-VL | MME + Two-Layer MLP FFN replacement | capability active；精确 prepare 单元 declared；质量/效率待实验 |
| `llava_onevision_14bench.yaml` | LLaVA-OneVision（Qwen route） | 14 个 image benchmark | 单卡 integration 入口；checkpoint-specific 证据见状态页 |

上述表格列出常用入口，完整文件清单以本目录为准。新增 [Bifocal](../../../docs/support/imagellm/status.zh.md#bifocal-接入)、
[MWOP Mask](../../../docs/guides/fixed_mask.zh.md)、[分模态替换](../../../mllm_comp/modality_split/README.md)
及 [`configs/comparison/`](../../comparison/) 的三基座 DOP-V 各有独立配置与验证范围。

六个音频 target 的精确证据和剩余验证见[候选方法状态](../../../docs/support/qwen25_omni/audio_compression/candidate_methods.zh.md)。
标为 `target` 的六份配置只能通过精确 plugin 名显式运行，不会进入正式 active registry。
它们用于继续验证，不构成质量、效率或训练支持声明。FastSLM HTA 入口必须同时设置：

```bash
MODEL_PATH=/path/to/Qwen2.5-Omni-3B \
FASTSLM_HTA_CHECKPOINT_PATH=/path/to/Stage3_FastSLM.pt \
  bash scripts/eval/lmms_eval/qwen25_omni_1gpu_aud_fastslm_hta.sh --plan_only
```

若 full-SFT checkpoint 同时包含 Qwen 和 HTA，可把两个变量指向同一目录。脚本不会使用
某次带时间戳的本地 run 作为默认路径。六个候选的激活门禁见
`docs/support/qwen25_omni/audio_compression/candidate_methods.zh.md`。

ImageLLM forward probe 只用于验证 `lmms-eval` 请求是否真实进入
`ExecutionPhase.FORWARD` 压缩回调。配置中的 `forward_only_probe: true` 会对每个请求执行一次
`use_cache=False` 的完整前向，并将末位 logits 的 argmax token 解码为
`generate_until` 协议传输值。输出的 `compression_runtime.json` 必须包含非零
`invocation_count` 和实际 `LayoutDelta`，否则平台会让评测失败。

这种探针返回的 MME 分数不代表生成质量，也不能用于 baseline/compressed 质量或效率比较；
结果清单会写入 `benchmark_quality_claimed: false`。MME 必须按同一图片的 Yes/No
问题成对聚合，因此这三份 smoke 配置固定 `sample_limit: 2`。

单卡运行示例：

```bash
QWEN25_VL_MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
MLLM_COMP_IMAGELLM_PROBE_ROOT=/tmp/imagellm-forward-probe \
CUDA_VISIBLE_DEVICES=1 \
python -m mllm_comp.cli.main eval \
  --config configs/eval/lmms_eval/qwen25_vl_imagellm_tome_forward_probe.yaml \
  --device cuda:0
```

Qwen2.5-Omni 正式入口：

```bash
MODEL_PATH=/path/to/checkpoint \
bash scripts/eval/lmms_eval/qwen25_omni_2gpu.sh --plan_only

MODEL_PATH=/path/to/checkpoint \
bash scripts/eval/lmms_eval/qwen25_omni_2gpu_aud.sh --plan_only

MODEL_PATH=/path/to/checkpoint \
bash scripts/eval/lmms_eval/qwen25_omni_2gpu_aud_segmentwise_topk.sh --plan_only

MODEL_PATH=/path/to/checkpoint \
bash scripts/eval/lmms_eval/qwen25_omni_2gpu_aud_segmentwise_topk_paper_sweep.sh --plan_only

MODEL_PATH=/path/to/checkpoint \
bash scripts/eval/lmms_eval/qwen25_omni_1gpu_aud_atome.sh --plan_only
```

LLaVA-OneVision 14-bench 单卡入口：

```bash
MODEL_PATH=/path/to/llava-onevision-qwen2-7b-ov \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/lmms_eval/llava_onevision_14bench_1gpu.sh
```

设置 `EVAL_LIMIT=1` 可让 14 个任务各跑一个最小样本，用于验证完整加载、数据、
生成、聚合与结果落盘链路。

服务器可用内存不足以让 lmms-eval 同时持有全部 14 个 image dataset 时，使用顺序
smoke 入口；它逐任务启动平台评测、限制 1 个样本，并为每个任务保留独立 results
JSON 与 Manifest：

```bash
MODEL_PATH=/path/to/llava-onevision-qwen2-7b-ov \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/lmms_eval/llava_onevision_14bench_smoke_1gpu.sh
```

两个 baseline 入口默认暴露 GPU `0,1`，使用两个 Accelerate worker 和端口 `29601`。可通过 `CUDA_VISIBLE_DEVICES`、`MAIN_PROCESS_PORT` 和 `PYTHON_BIN` 覆盖机器相关设置。脚本不绑定某次训练 run。

压缩入口默认端口为 `29611`，输出目录与 baseline 分离。所有音频配置显式使用 `runtime_mode: default`：正常推理不注册效率或逐层调试 hook，也不写平台 diagnostic JSONL。运行时可用互斥参数覆盖：

- `--profile`：采集 generation、prefill、decode/token、近似 TTFT、CUDA 峰值显存与 attention scoring overhead；baseline/compressed 分别写 `baseline_diagnostic.rank*.jsonl` 和 `compression_diagnostic.rank*.jsonl`。
- `--debug`：不计时，只统计 processor、legalizer、Audio Encoder 各层、pooler、post-norm、projector 与 LLM 各层的 `text/audio/image/video/pad/unknown` Token 数，写 `debug_diagnostic.rank*.jsonl`。
- 不传参数或 `--runtime_mode default`：普通评测路径，无上述诊断文件。

效率评测必须给 baseline 和 compressed 命令都加 `--profile`。逐层调试建议单卡、单样本运行，例如：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval/lmms_eval/qwen25_omni_2gpu_aud_segmentwise_topk.sh \
  --debug --num_processes 1 --sample_limit 1
```

Segmentwise Top-K 与 A-ToMe 的断点式验收清单分别见
`docs/support/qwen25_omni/audio_compression/debugging/projector_level.zh.md` 和
`docs/support/qwen25_omni/audio_compression/debugging/encoder_level.zh.md`；八方法共用的 Descriptor、
四执行族、训练/评估契约和最小验证矩阵见
`docs/support/qwen25_omni/audio_compression/extension.zh.md`。Audio-Pyramid 与 SPADE
Operation-Only 的配置分别覆盖 LLM Token 的 Prefill/Decode 和结构重写路径；两者已通过
真实 Qwen2.5-Omni-3B 生命周期 smoke，但尚未运行目标规模 MMAU 质量/效率实验。
FastSLM HTA、UniAvg、UniSamp、UnSeg、FastAdaSP 与 HeadRouter 属于单独的 target
候选集，不计入上述八方法验证结论。

两份效率 summary 生成后，使用 `scripts/analysis/compare_audio_compression_runs.py` 并同时传入 baseline/compressed Manifest，将 lmms-eval 共同质量指标与效率差、speedup 和六个源文件哈希合并成一份审计报告。比较器会强制检查 checkpoint、任务、generation/并行协议以及 Manifest trace 路径一致性。

论文式 sweep 默认端口为 `29621`，按 `r50 -> r25 -> r10` 顺序执行，固定 `S=10` 和 `allocation=paper_floor`，并写入三个独立 run 目录。直接加载该模板前必须设置 `SEGMENTWISE_KEEP_RATIO` 与 `SEGMENTWISE_RUN_TAG`；推荐始终通过 sweep 脚本运行。

评测任务、数据集和 task generation kwargs 由 lmms-eval task YAML 拥有。项目配置负责模型运行时、任务选择、并行方式和平台需要显式覆盖的差异。音频配置中的 `until: []` 用于与当前原生 Qwen2.5-Omni chat wrapper 的完整回复评分行为对齐；`max_new_tokens` 和 `do_sample` 继续由 MMAU task 提供。

配置中的 `backbone` 是本次运行的完整模型声明，不依赖额外的 `configs/backbones/` 片段。
