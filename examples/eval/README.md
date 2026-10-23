# 可运行评测示例

本目录保存可以直接交给 `mllm-comp eval` 的示例配置。模型和输出路径由配套 shell
脚本设置为环境变量；如需在其他
机器运行，优先覆盖环境变量，不要把机器路径写回 YAML。

这里保留 `MLLM_COMP_RUN_DIR` 的配套脚本协议，它是精确输出目录，重复实验需换新值。
新建普通配方可以省略 `output`，让平台生成 `runs/<experiment.name>/<run_id>/`；
显式目录下的默认 Manifest 为 `manifest.json`。结构化结果和失败处理见
[运行压缩实验](../../docs/guides/1_run_compression_experiment.zh.md)。

平台公共字段见[配置字段参考](../../docs/reference/config.zh.md)；lmms-eval task
拥有的数据、prompt 和 metric 不在平台配置中重复定义。

- `qwen25_omni_3b_mme.yaml`：Qwen2.5-Omni-3B 的 MME 基线配置；配套入口为
  `examples/eval/eval_qwen25_omni_3b_mme.sh`。
- `qwen3_vl_mme.yaml`：Qwen3-VL 的完整 MME 基线配置；配套入口为
  `examples/eval/eval_qwen3_vl_mme.sh`。

`dataset_path`、split 和 generation kwargs 都归 lmms-eval task YAML 所有。这两个
平台示例只通过 `eval.tasks: [mme]` 选择任务，不重复声明 task 已经拥有的数据集路径、
split 或生成参数。`data.dataset_id` 不会改变 lmms-eval 实际读取的数据，因此 eval
示例不填写该字段，平台 Manifest 在无法可靠解析 task YAML 数据来源时记录 `null`。

HPC 中人工修改过的 lmms-eval task 数据集路径属于有效的运行环境配置。平台同步、
验证和排查不得覆盖它，也不得仅因它不是上游默认路径而判定为错误；实际数据来源应以
该环境中的 task YAML、最终 lmms-eval argv 和原生结果为准。

项目级完整 MME/MMAU 配置位于 `configs/eval/lmms_eval/`。一次性 OCR 子集、逐样本差分和具体 run 路径不作为长期示例保存；需要回归时应从正式 task/config 生成临时诊断配置。

两个示例默认 `num_processes: 1`，因此仍直接执行 `python -m lmms_eval`。单机多卡时
不需要复制或改写 task 配置，只需让目标 GPU 可见并覆盖进程数：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash examples/eval/eval_qwen25_omni_3b_mme.sh --num_processes 2
```

默认 launcher 是 `accelerate`；可用 `--distributed_launcher torchrun` 切换。共享机器上
若默认 rendezvous 端口冲突，可增加 `--main_process_port 29601`。示例脚本不会自行设置
`CUDA_VISIBLE_DEVICES`，因此会尊重 HPC scheduler 或调用者已有的 GPU 分配。多卡时
不要传 `--device cuda:0`：每个 lmms-eval worker 必须按自己的 local rank 选择设备。
