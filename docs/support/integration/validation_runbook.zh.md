# 原生 CLI 集成人工验证手册

> 类型：Runbook
> 文档归属：Support / Integration
> 适用范围：lmms-eval 与 ms-swift 的独立人工复核
> 当前状态：Active；日常开发按改动选择验证，不要求完整执行本手册

## 1. 验证目标

人工验证需要同时确认两层事实：

1. lmms-eval/ms-swift 的原生日志、进度、结果表、cache/resume、标准文件仍在；
2. lmms-eval 的模型请求实际经过平台 Processor、StateBuilder 和 Runner，而不是
   上游内置 Qwen wrapper。

## 2. 环境与 GPU

目标环境固定 `transformers==4.57.3`。先记录：

```bash
python -V
python - <<'PY'
import importlib.metadata as m
for name in ['mllm-comp-platform', 'lmms-eval', 'ms-swift', 'transformers', 'torch']:
    print(name, m.version(name))
PY
```

每次 GPU smoke 前运行 `nvidia-smi`。其他用户进程不是自动阻塞条件：只要剩余
显存足以安全运行本次 1–4 样本 smoke，就继续；不得终止、暂停或干扰其他用户。
剩余显存不足或存在明显 OOM 风险时才停止并记录。

## 3. 安装与依赖诊断

使用当前开发 workspace 对应的环境，人工与 agent 均在该工作区验证。
本项目不设固定 Review 工作区；记录验证的提交或未提交改动，验证通过后提交，再按授权合并到 `main`。
并行验证其他版本时使用独立 Dev 工作树和环境。editable install 必须指向本次验证的源码：

```bash
python -m pip install -e /path/to/current/dev/workspace
python scripts/diagnostics/validate_framework_integrations.py
```

检查：

- `status=passed`，lmms-eval 和 ms-swift 的 `available` 均为 true；
- `lmms_eval.models` 列出四个基座的 simple/chat 入口，`ms_swift.loaders` 列出对应原生 loader；
- 报告 `model_load_attempted=false`；
- 版本不是 `not-installed`；
- 诊断期间没有模型权重或数据集加载日志。

## 4. `plan_only`

```bash
python -m mllm_comp.cli.main eval --config /path/to/qwen25_eval.yaml --plan_only
python -m mllm_comp.cli.main train --config /path/to/qwen25_train.yaml --plan_only
```

检查 JSON 中包含完整 argv、环境变量、cwd、版本和预期输出路径。评测 argv 必须有：

- `--model qwen2_5_omni`（Qwen3-VL 为 `mllm_comp_qwen3_vl`）；
- `mllm_comp_spec=...`；
- `--output_path`、`--tasks`、`--batch_size` 和 `--seed`。

`--limit` 只在配置或 CLI 明确设置样本上限时出现；完整 benchmark 不应为了满足
检查清单而添加 limit。`--device` 同样是可选项：只设置 `CUDA_VISIBLE_DEVICES` 并让
Accelerate 选择进程内 `cuda:0` 时，argv 可以不包含该参数。

多卡 plan 必须满足：

- `--num_processes 2` 生成一个平台父进程和一个 Accelerate/Torchrun launcher，不能
  生成两个 `mllm-comp eval` 父进程；
- Accelerate argv 含 `--multi_gpu --num_processes 2 -m lmms_eval`，Torchrun argv 含
  `--nproc_per_node 2 -m lmms_eval`；
- plugin spec 的 `expected_world_size` 为 2，且 `device` 为 `null`；
- 多卡时显式传 `--device`，或只暴露一张 GPU，却请求两个进程，都必须在模型加载前
  明确报错。

默认配置不得包含 `--force_simple`，此时 lmms-eval 应选择
`MLLMCompChatLMM`。只有配置 `eval.args.force_simple: true` 后，argv 才应包含一次
`--force_simple` 并选择 `MLLMCompLMM`。

Swift argv 必须以 `swift sft` 开头，并保留 dataset/template/resume/deepspeed 等
原生参数。Qwen2.5-Omni 环境含 `ENABLE_AUDIO_OUTPUT=0`；Qwen3-VL 不含该变量。

## 5. Qwen2.5-Omni MME 单样本 smoke

低成本 smoke 配置位于 `examples/eval/`，项目级完整 MME/MMAU 配置位于
`configs/eval/lmms_eval/`。先让示例脚本生成一份计划，再把样本数覆盖成 1 条：

```bash
export CUDA_VISIBLE_DEVICES=0
export QWEN25_OMNI_MODEL_PATH=/path/to/Qwen2.5-Omni-3B
export MLLM_COMP_RUN_DIR="$PWD/runs/qwen25_omni_mme_single"

bash examples/eval/eval_qwen25_omni_3b_mme.sh --plan_only --sample_limit 1
bash examples/eval/eval_qwen25_omni_3b_mme.sh --sample_limit 1
```

不要在基线对齐 smoke 中传 `--generation_max_new_tokens`、temperature 或 top_p；MME
task YAML 自己的 generation kwargs 应保持生效。HPC 上实际 dataset path 继续由
lmms-eval 的 MME task 配置决定，平台不会覆盖该路径。

通过标准：

- 终端出现 lmms-eval INFO 日志、请求构建与 `Model Responding` 进度；
- 结束时出现 lmms-eval 原生 `make_table()`；
- 输出目录存在原生 `*_results.json` 和（启用时）`*_samples_mme.jsonl`；
- 平台 Manifest 在原生进程结束后生成；
- Manifest 的 backend args 记录实际 argv、版本、result/sample path；
- 默认日志/断点证明模型只在
  `plugin.MLLMCompChatLMM -> load_backbone_runtime()` 加载；
- 请求进入 ProcessorAdapter、StateBuilder 和 Runner。

## 6. Qwen3-VL 单样本 smoke

```bash
export CUDA_VISIBLE_DEVICES=0
export QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
export MLLM_COMP_RUN_DIR="$PWD/runs/qwen3_vl_mme_single"

bash examples/eval/eval_qwen3_vl_mme.sh --plan_only --sample_limit 1
bash examples/eval/eval_qwen3_vl_mme.sh --sample_limit 1
```

检查 `--model mllm_comp_qwen3_vl`，默认不得出现 `--force_simple`，并确认
`MLLMCompChatLMM` 被实例化。MME 只能证明 image 请求；video 必须使用含视频样本的
lmms-eval task 另行验证。audio 或 audio-in-video 必须给出明确错误。若需要回归 simple
协议，在临时 runnable config 中显式设置 `eval.args.force_simple: true`，并确认选择
`MLLMCompLMM`；simple 结果不能替代默认 chat 路径的验收。

Qwen2.5-VL 与 LLaVA-OneVision 通过同一插件类进入平台，但使用独立 model id。至少先
检查两份 14-benchmark 配置的启动计划：

```bash
QWEN25_VL_MODEL_PATH=/path/to/Qwen2.5-VL-Instruct \
bash scripts/eval/lmms_eval/qwen25_vl_14bench_1gpu.sh --plan_only

LLAVA_ONEVISION_MODEL_PATH=/path/to/llava-onevision \
bash scripts/eval/lmms_eval/llava_onevision_14bench_1gpu.sh --plan_only
```

前者 argv 必须使用 `--model mllm_comp_qwen25_vl`，后者必须使用
`--model mllm_comp_llava_onevision`。需要验证真实请求时，使用对应
`*_14bench_smoke_1gpu.sh` 入口，并保持每个任务的原生结果与平台 Manifest 分离落盘。

ImageLLM 配置生成器使用同一个 registry，覆盖三基座的 24 方法 eval 和 23 方法 Swift train 配置；
这是历史 active 配方范围，不包括新增 Bifocal、14 组对比方法、MWOP（Mask 形式）或分模态替换。具体命令、一步训练证据要求和 D2Pruner 的外部 prior 要求见
[ImageLLM 验证手册](../imagellm/validation_runbook.zh.md)。

## 7. lmms-eval 两卡 smoke

在至少两张卡可见的机器上，先做 Qwen2.5-Omni 两卡 smoke：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export QWEN25_OMNI_MODEL_PATH=/path/to/Qwen2.5-Omni-3B
export MLLM_COMP_RUN_DIR="$PWD/runs/qwen25_omni_mme_2gpu"

bash examples/eval/eval_qwen25_omni_3b_mme.sh \
  --plan_only --num_processes 2 --sample_limit 4 --main_process_port 29601
bash examples/eval/eval_qwen25_omni_3b_mme.sh \
  --num_processes 2 --sample_limit 4 --main_process_port 29601
```

通过标准：两个 worker 的 world size 都是 2、分别使用两个 local-rank 设备；
lmms-eval 只在 rank 0 保存和打印最终聚合结果；平台只产生一个 plugin spec 和一个
Manifest；结果中的 effective sample 数是任务总数而不是每 rank 重复计数。不得因为
其他用户有程序运行就自动阻塞；仍以可见卡上的剩余显存是否足够为准，不干扰其进程。
若端口冲突，改用未占用的 `--main_process_port`，不要停止其他用户的 launcher。

Torchrun 回归可把两条命令增加 `--distributed_launcher torchrun`。它和 Accelerate
应进入同一个平台插件与 lmms-eval 聚合路径。真实多卡结果确认后，再进行完整 MME；
单卡 smoke 或纯命令契约测试不能替代真实两卡验收。

## 8. ms-swift 一步 smoke

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python -m mllm_comp.cli.main train \
  --config /path/to/qwen25_omni_sft.yaml \
  --max_steps 1
```

检查原生 Swift 参数摘要、日志与进度直接显示；`logging.jsonl`、
`trainer_state.json` 和 checkpoint 由 Swift 生成；平台结束后收集最新路径并写入
Manifest。resume、DeepSpeed 或分布式选项应出现在最终 `swift sft` argv 中。

Qwen2.5-Omni 动态音频压缩训练不能只按上述通用产物验收。还必须检查 Swift template
pre-forward、legalizer、方法执行点、loss backward、compression evidence 和 Manifest
gate。四个执行族重构后的职责划分和故障定位见 [`Projector-Level`](../qwen25_omni/audio_compression/debugging/projector_level.zh.md)、
[`Encoder-Level`](../qwen25_omni/audio_compression/debugging/encoder_level.zh.md)、
[`LLM-Token / Audio-Pyramid`](../qwen25_omni/audio_compression/debugging/llm_token_audio_pyramid.zh.md)和
[`LLM-Operation / SPADE`](../qwen25_omni/audio_compression/debugging/llm_operation_spade.zh.md)架构与 Debug 手册。自动 gate 的统一命令见
[`Omni 八方法 Runbook`](../qwen25_omni/audio_compression/eight_methods_runbook.zh.md)。

## 9. Qwen2.5-Omni 差分验证

不要只比较最终 MME 总分。固定同一环境、checkpoint、task YAML 和 20–100 个
doc_id，分别运行上游内置基线与平台插件，保存 sample 日志和额外 trace。逐样本比较：

- system prompt 与 task pre_prompt/post_prompt；
- apply_chat_template 后文本；
- `use_fast`、`process_mm_info` 输入输出与 processor kwargs；
- input_ids 长度、image_grid_thw、pixel/audio/video 输入；
- dtype、attention implementation、use_audio_in_video；
- max_new_tokens、do_sample、temperature/top_p/top_k；
- 生成 token IDs、decode、原始 Yes/No 与最终得分。

报告分类：完全一致、仅格式不同、Yes/No 内容翻转、processor/token 不一致、
generation 配置不一致。greedy 模式下平台不得继续传
`thinker_temperature/thinker_top_p/thinker_top_k`。

已经闭环的差异包括：默认 chat/simple 路由、processor 与 attention 运行条件、
generation kwargs 所有权，以及 MMAU 默认 `"\n\n"` delimiter 的解码后截断。生产配置
默认不传 `--force_simple`；音频 MMAU 配置显式使用 `until: []`。图像/音频训练与
MME/MMAU 最终复测状态见 `docs/support/qwen25_omni/status.zh.md`。

fast/slow processor、dtype、attention implementation、generation kwargs 和输出解析仍应
在数值行为变更时记录，但它们是回归变量，不再维护独立的逐次差分报告。

## 10. 最终证据清单

- 完整 unit/contract 测试结果；
- 依赖诊断与 `plan_only` JSON；
- lmms-eval 原生日志、进度、表格、result/sample 和平台 Manifest；
- Swift 原生日志、checkpoint/logging.jsonl 和平台 Manifest；
- 需要数值回归时生成的 Qwen2.5-Omni 固定样本差分 artifact；
- Qwen3-VL 同机制 smoke；
- 至少一次真实两卡 lmms-eval smoke，以及单一 Manifest/正确样本聚合证据；
- GPU 剩余容量与测试时进程记录；
- 所有未支持能力和失败原因，不得使用静默降级。
