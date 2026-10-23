# lmms-eval integration

状态：active（analysis/eval）。

`adapter.py` 负责启动计划、原生结果收集和 Analysis 样本迭代；`diagnostics.py` 负责把已有
机器记录整理为人类可读结果。Analysis 只有在所选 Probe 声明
`requires_teacher_forcing=True` 时才额外构造参考答案输入（当前为 HierarchicalTaylor）；
其他 Probe 仍保留目标文本和来源元数据，共用同一次样本循环。

评测采用 Native CLI First：`mllm-comp eval` 只生成插件规格和原生命令，并继承当前
终端的 stdout/stderr。单进程直接执行 `python -m lmms_eval`；`num_processes > 1`
时由同一个平台父进程执行 `accelerate ... -m lmms_eval` 或
`torch.distributed.run ... -m lmms_eval`。`lmms-eval` 继续拥有任务分片、跨 rank
聚合、请求循环、日志、tqdm、cache、标准结果文件和最终结果表。

分布式评测是单机数据并行：每个可见 GPU 加载一份模型副本。平台只写一次 plugin
spec，并在整个原生 launcher 结束后只收集一次结果、写一次 Manifest。插件校验
`Accelerator.num_processes` 与计划一致，并让每个 worker 使用其 local-rank 设备；
因此多卡模式禁止显式固定 `--device`。诊断 trace 会自动拆成 `.rank<N>` 文件，避免
并发写同一路径。

`plugin.py` 通过公开的 `lmms_eval.models` entry point 为 `qwen2_5_omni`、
`mllm_comp_qwen25_vl`、`mllm_comp_qwen3_vl` 与
`mllm_comp_llava_onevision` 注册模型实现。Qwen 默认不传
`--force_simple`，由 lmms-eval
按自己的规则选择 chat；只有 `eval.args.force_simple: true` 才显式选择 simple。
两条路径都通过平台的 `load_backbone_runtime()`、ProcessorAdapter、StateBuilder
和 Runner 执行。

chat 路径消费任务的 `doc_to_messages`，并使用 lmms-eval 的 image/video kwargs
转换为 Hugging Face 消息；simple 路径保持 `context + doc_to_visual` 协议。因此
平台不会把两种上游请求协议混为一条自定义路径。

当前支持单轮 `generate_until` 文本结果。Qwen2.5-Omni 支持 image/audio/video；
Qwen3-VL 支持 image/video 并拒绝 audio；LLaVA-OneVision 当前固定走 simple
`doc_to_visual` 的 image-only 路线。`loglikelihood`、multi-round、Talker
metric output 和未导出的 in-memory pipeline-stage eval 会显式报错。

详见 `docs/platform/framework_integration.zh.md` 与
`docs/support/integration/validation_runbook.zh.md`。
