# lmms-eval 与 ms-swift 原生 CLI 接入设计

> 类型：Architecture
> 适用范围：外部评测与训练框架的职责、参数和数据边界
> 当前状态：Active

## 1. 架构决策

本项目采用 **Native CLI First, Platform Orchestration Only**。

平台只负责实验配置、少量跨后端参数、Backbone/压缩能力、原生命令构造、
运行环境与版本记录，以及在原生进程结束后生成 Manifest。lmms-eval 和
ms-swift 继续负责各自的参数解析、任务/数据、主循环、日志、进度、缓存、
分布式执行和标准产物。

禁止在平台中复制上游评测循环、Trainer、日志系统、进度条、结果表或缓存系统。

## 2. lmms-eval 数据流

```text
mllm-comp eval
  -> 解析平台配置并生成 plugin spec
  -> 单进程启动 python -m lmms_eval
     或由同一父进程启动 accelerate/torchrun -> N 个 lmms-eval worker
     （stdout/stderr 直接继承终端）
  -> lmms-eval 公开 entry point 发现 chat/simple 双实现
  -> 默认 MLLMCompChatLMM；显式 --force_simple 时为 MLLMCompLMM
  -> 插件调用 load_backbone_runtime()
  -> lmms task 构建 request
  -> LMMSEvalRequestBridge
  -> ProcessorAdapter.prepare_inputs()
  -> StateBuilder.build()
  -> Runner.generate()
  -> lmms-eval 聚合指标、保存标准结果/sample、打印 make_table()
  -> 平台读取原生结果并生成 Manifest
```

### 2.1 正式模型插件

`pyproject.toml` 使用 `lmms_eval.models` entry point。插件类位于
`mllm_comp/integrations/lmms_eval/plugin.py`，不修改 third_party。

插件覆盖四个平台 model id，并为每个 id 同时提供 chat/simple 类：

| 平台 Backbone | lmms-eval `--model` | 默认 chat | 显式 simple |
|---|---|---|---|
| `qwen25_omni` | `qwen2_5_omni` | `MLLMCompChatLMM` | `MLLMCompLMM` |
| `qwen25_vl` | `mllm_comp_qwen25_vl` | `MLLMCompChatLMM` | `MLLMCompLMM` |
| `qwen3_vl` | `mllm_comp_qwen3_vl` | `MLLMCompChatLMM` | `MLLMCompLMM` |
| `llava_onevision` | `mllm_comp_llava_onevision` | `MLLMCompChatLMM` | `MLLMCompLMM` |

launcher 默认不传 `--force_simple`，遵循 lmms-eval 的原生路由规则。chat 类消费
`doc_to_messages` 并保留 lmms-eval 的 image/video kwargs；只有配置
`eval.args.force_simple: true` 时才附加 `--force_simple`，使用
`context + doc_to_visual`。TaskManager 看到的是上表注册后的平台 model id；task YAML
若维护 model-specific `lmms_eval_specific_kwargs`，必须显式包含该 id 或提供
`default`，不能假设上游内置 model id 的分支会自动命中。后续新增 Backbone 只需增加
model-id 映射并核对 task-specific kwargs，不需要复制评测循环。

### 2.2 参数边界

平台拥有：model/checkpoint、tasks、batch size、limit、device、output_path、
compression mode 和 baseline/compressed target。它们不能在 `eval.args` 中覆盖。

随机种子默认取 `experiment.seed`；`eval.args.seed` 可按 lmms-eval 原生格式覆盖，
例如用 `[0, 1234, 1234, 1234]` 分别指定其四项 seed。

其他原生参数保留原名称放入 `eval.args`，例如：

```yaml
eval:
  args:
    log_samples: true
    verbosity: INFO
    include_path: /path/to/custom/tasks
    use_cache: /path/to/cache
    cache_requests: "true"
```

布尔 `true` 转成 flag，`false` 不传；标量直接传；标量列表转成逗号分隔值。
`cache_requests` 的原生取值为 `"true"`、`"refresh"` 或 `"delete"`；上例中的 `"true"` 必须保留引号。
极少数无法用 YAML 映射表达的参数可通过 `native_argv` 原样追加。

lmms-eval 0.7.2 的 `simple_evaluate()` 有 `bootstrap_iters`，但原生 CLI parser
没有对应 flag。平台会显式拒绝该字段，避免静默丢失；若需要它，应先由上游
公开 CLI 参数或升级到已提供该参数的版本。

### 2.3 单机多卡边界

`eval.num_processes` 默认为 1。大于 1 时，平台把同一个 lmms-eval 子命令包装为：

```bash
python -m accelerate.commands.launch --multi_gpu --num_processes 2 -m lmms_eval ...
# 或
python -m torch.distributed.run --nproc_per_node 2 -m lmms_eval ...
```

`eval.distributed_launcher` 可选 `accelerate`（默认）或 `torchrun`；
`eval.main_process_port` 可在共享机器上指定本地 rendezvous 端口。对应 CLI 覆盖为
`--num_processes`、`--distributed_launcher` 和 `--main_process_port`。

平台父进程始终只有一个：它写一次 plugin spec，启动一次原生 launcher，并在所有
worker 结束后收集一次结果和写一次 Manifest。数据分片、padding、gather、指标聚合和
rank-0 保存继续由 lmms-eval 负责。插件读取 Accelerate 的 process index/world size，
按 local rank 选择设备，并校验实际 world size 与 launch plan 一致。

这是每卡一份模型副本的数据并行，不是把单个模型切到多卡的 tensor parallel。
`runtime_batch_size` 是每个 worker 的 batch size。多卡时禁止 `--device cuda:0`，否则
所有 worker 会争用同一张卡；应通过 scheduler 或 `CUDA_VISIBLE_DEVICES` 指定可见卡。
如果显式的 `CUDA_VISIBLE_DEVICES` 少于 `num_processes`，平台会在启动前报错。示例脚本
不会覆盖调用者已有的 GPU 可见性。分布式诊断 trace 使用每 rank 独立文件。

### 2.4 输出与 Manifest

平台始终设置原生 `--output_path`。lmms-eval 自己保存：

- `<output>/<model>/<timestamp>_results.json`
- `<output>/<model>/<timestamp>_samples_<task>.jsonl`（启用 `log_samples` 时）

平台不接管保存过程，只在原生进程成功结束后查找本次新建/更新的结果文件。
Manifest 的 `eval_backend.args` 记录完整 launcher argv、内部 lmms-eval argv、进程数、
launcher、rendezvous 端口、实际 `CUDA_VISIBLE_DEVICES`（若调用者有设置）、cwd、实际版本、output_path、result path、
sample paths、plugin spec 和 checkpoint。

## 3. ms-swift 数据流

```text
mllm-comp train
  -> 将平台公共字段与 train.args 转成完整 swift sft argv
  -> 启动 swift sft（stdout/stderr 直接继承终端）
  -> Swift 负责 ModelLoader、template、dataset、Trainer、DDP/DeepSpeed、resume
  -> Swift 保存 checkpoint、logging.jsonl、trainer_state.json
  -> 平台在成功结束后收集最新原生产物并生成 Manifest
```

平台不再导入 `swift.pipelines.train.sft.sft_main`。

平台拥有 `model`、`model_type`、`template`、`dataset`、`output_dir`、
`max_steps` 和 `resume_from_checkpoint`。其他 Swift 参数在 `train.args` 中按原名
透传；列表保留为一个 flag 后跟多个值，`native_argv` 可追加完全原样的参数。

Qwen2.5-Omni 使用 `qwen2_5_omni` model/template 和
`ENABLE_AUDIO_OUTPUT=0`，训练边界为 Thinker。Qwen2.5-VL、Qwen3-VL、
LLaVA-OneVision 分别使用 `qwen2_5_vl`、`qwen3_vl`、`llava_onevision_qwen`
model/template，训练边界均为 language model，且不接受 audio-output 控制。
原版 LLaVA-OneVision checkpoint 还必须把仓库内
`third_party/backbones/LLaVA-NeXT` 作为 `train.args.local_repo_path`；配置生成器会自动写入该路由，
并关闭 vision-tower checkpointing，保留 language-model gradient checkpointing，避免 ToMe
pinned callback 被视觉编码器重放。

Qwen2.5-Omni 的模型/处理器原生模态控制按原名写入 `backbone.config`：
`max_pixels`、`use_audio_in_video`、`sampling_rate`、`video_min_pixels`、
`video_max_pixels`、`fps`、`fps_min_frames`、`fps_max_frames`、`frame_factor`。
适配器只处理这些明确列出的字段，并只映射配置中显式出现的值；未出现的字段使用
上游默认值。所有这些字段都映射为对应的 Qwen 原生环境变量。
训练 batch、优化器、冻结策略和 LoRA 等参数仍属于 `train.args`。

Qwen2.5-Omni 压缩训练使用同一个 `RuntimeCompressionDispatcher` session。
loader 返回模型后先执行 `PREPARE` Operation，再依据改写后的结构安装 `FORWARD`
Token 阶段；阶段按依赖排序。音频 encoder、projector、LLM 的已有组合规则继续生效，
前向复用 Omni 的多阶段计划与 request，Swift template 仍负责原生特征写入。
每个阶段分别记录安装、训练 forward 和对应 loss backward 的证据。

优化器、调度器、RNG 与 checkpoint 保存恢复仍归 Swift。`train.resume` 完整恢复时，
先从原始模型来源重建相同 pipeline，再由 Swift 恢复权重与训练状态；
不要把这个 checkpoint 同时设为 `train.checkpoint_in`，导致结构再次剪枝。
`resume_only_model=true` 会提前替换模型来源，因此不用于压缩训练的完整恢复。
Swift 的模型保存完成及训练退出后，平台关闭 session、移除 hooks；结构改写本身不可回滚。
`checkpoint_in` 则表示一次训练的初始权重来源，只能使用该 Swift loader 能加载的格式，
不能将框架单独导出的 Thinker 权重假定为 Swift 的完整复合模型 checkpoint。

Omni 只含静态删层 Operation 时保持原生 packing/padding-free 路径；包含动态 Token
阶段时继续禁止两者。此例不能推广到全部 Operation：新增图像 comparison、MWOP 和
分模态替换各自有布局限制，应按实际 binding 校验。

ImageLLM 动态 token compression 则直接加载原始预训练 checkpoint，在 loader 完成且
optimizer 创建前安装精确 binding；Swift 的 training forward、backward 和保存流程保持原生。
平台不会把不可重载的 in-memory 结构改写伪装成 Swift checkpoint。

新增图像 comparison 的 Operation 使用 `prepare`，Token 使用训练 `forward` / 评测
`generate`。训练桥区分 Token 减少和 Operation 实际执行；新增可学习参数的原生完整
checkpoint sidecar 有 CPU 测试，不能据此声明 Swift/PEFT adapter-only 恢复已验证。
配置与精确限制见[接入说明](../support/imagellm/comparison_methods.zh.md)。

## 4. 原生命令示例

平台入口：

```bash
bash examples/eval/eval_qwen25_omni_3b_mme.sh --plan_only
bash examples/eval/eval_qwen25_omni_3b_mme.sh
bash examples/train/train_qwen25_omni_sft.sh --plan_only
bash examples/train/train_qwen25_omni_sft.sh
```

Qwen3-VL 对应入口为 `examples/eval/eval_qwen3_vl_mme.sh` 和
`examples/train/train_qwen3_vl_sft.sh`。可运行 YAML 位于 `examples/eval/`、
`examples/train/`；项目级完整配置位于 `configs/eval/` 和 `configs/train/`。
ImageLLM 的任意已验证算法/基座配置由 `scripts.train.render_imagellm_config` 生成。

等价的 lmms-eval 核心形态（spec 由平台生成）：

```bash
python -m lmms_eval \
  --model qwen2_5_omni \
  --model_args pretrained=/path/to/model,mllm_comp_spec=/path/to/plugin.json \
  --tasks mme \
  --batch_size 1 \
  --output_path /path/to/output
```

两卡数据并行入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash examples/eval/eval_qwen25_omni_3b_mme.sh \
  --num_processes 2 \
  --main_process_port 29601
```

Swift 核心形态：

```bash
ENABLE_AUDIO_OUTPUT=0 swift sft \
  --model /path/to/model \
  --model_type qwen2_5_omni \
  --template qwen2_5_omni \
  --dataset /path/to/data.jsonl \
  --output_dir /path/to/output
```

## 5. 当前显式能力边界

- lmms-eval 插件仅支持 `generate_until` 文本指标；不支持 loglikelihood、
  multi-round 或 Talker 音频结果。单轮 simple/chat 请求均已支持。
- 分布式评测当前支持单机数据并行；尚未声明多节点或 tensor-parallel 能力。
- Qwen3-VL 不接受 audio/audio-in-video。
- 图像与音频压缩通过精确 Binding 接入原生请求。动态 Token 执行发生在 forward/generate，
  结构 Operation 在 runner/optimizer 之前的 prepare 阶段执行；实际方法与证据使用
  `mllm-comp discover` 查询。
- 单步训练、cached generate、结构导出与恢复分别需要对应证据，不能互相推导。
  D2Pruner 需要用户提供与模型及评分层匹配的 calibrated prior；加载与校验已实现，
  校准流程未提供。没有历史 SupportUnit 本身不阻止满足实际契约的研究方法运行。
- ms-swift Talker 训练尚未实现。
- lmms-eval 0.7.2 原生 CLI 未暴露 `bootstrap_iters`。

这些限制都不能被静默降级为未压缩模型或不同的上游执行路径。

## 6. 验证记录

- 当前 Qwen2.5-Omni 训练/评测结论：`docs/support/qwen25_omni/status.zh.md`
- 外部框架可重复验证步骤：`docs/support/integration/validation_runbook.zh.md`
- 已闭环的路由、processor、generation 和输出解析问题由回归测试保持，不再维护逐次差分报告。
