# 配置参考

`examples/` 中的 YAML 是最小可运行示例，不会列出所有可选参数。下面分别给出
Analysis、Evaluation、Training 和 compression stage 的带注释配置。复制时只保留
当前任务需要的 section。

平台会先展开 `${NAME}` 环境变量，再解析 YAML。未展开的变量、未知字段以及缺少
`experiment.name`、`backbone.name`、模型来源或数据来源的有效配置会在启动前
失败。

代码块中的标记按该命令真实运行时的要求填写：

- `必填`：缺少后无法启动该命令。
- `条件必填`：满足注释中的条件时必须填写。
- `可选`：可以省略；有平台默认值时会直接写在注释里。

`backbone.config`、`analysis.probes[*]`、`eval.generation`、`eval.args`、
`train.args` 和 compression stage 的子配置由对应 Backbone、Probe、后端或 plugin
解释，因此其中的字段是否必填还要以所选实现为准。

## Analysis

`analysis` 配置负责选择 task、样本范围和 Probe。Probe 自己拥有的参数写在对应
`probes` 项下。

```yaml
experiment:                              # 必填：所有运行的基本信息
  name: my_analysis                      # 必填：实验名称
  # run_id: analysis-001                 # 可选：每次实验默认生成新 ID
  seed: 42                               # 可选，默认 42：随机种子

backbone:                                # 必填：模型及其加载方式
  name: qwen25_vl                        # 必填：已注册的 Backbone 名称
  model_id: Qwen/Qwen2.5-VL-3B-Instruct # 条件必填：与 model_path 至少填写一个
  # model_path: /path/to/model           # 可选：本地权重；存在时优先于 model_id
  model_revision: revision-or-commit     # 可选：仓库 revision 或 commit
  trust_remote_code: false               # 可选，默认不设置：是否允许远程模型代码
  config:                                # 可选，默认 {}：传给该 Backbone 的加载参数
    torch_dtype: bfloat16                # 可选：模型计算精度
    attn_implementation: eager           # 可选：Attention 实现

pipeline:                                # 可选：省略时不配置压缩 stage
  stages: []                             # 可选，默认 []：baseline 保持为空

analysis:                                # 必填：Analysis 的任务和 Probe
  backend: lmms_eval                     # 可选，默认 lmms_eval：样本输入后端
  compression_mode: none                 # 可选，默认 none：none 或 compressed
  phase: pre_compression                 # 可选，默认 pre_compression：结果所属阶段
  tasks: [mme]                           # 必填：非空的 lmms-eval task 列表
  runtime_batch_size: 1                  # 可选，默认 1：每次 forward 的样本数
  sample_limit: 1                        # 可选，默认不限制：正整数
  probes:                                # 必填：至少包含一个 enabled Probe
    - name: representation_dynamics      # 条件必填：name 或 method 至少填写一个
      enabled: true                      # 可选，默认 true：是否执行该 Probe
      layers: [0]                        # 可选，默认 all：层编号列表或 all
      metrics: [cosine]                  # 可选，默认 [cosine]：输出指标
      token_groups: [system, text, vision] # 可选：默认 system、text、vision
      save_raw: false                    # 可选，默认 false：是否保存逐样本结果

output:                                  # 可选：省略时自动分配独立运行目录
  run_dir: ${RUN_DIR}                    # 可选：精确运行目录；应为空目录或新目录
  manifest_path: ${RUN_DIR}/manifest.json # 可选：默认 run_dir/manifest.json
```

`probes[*]` 没有统一的子字段全集。当前 Probe 的配置与结果说明见
[运行分析](../guides/1_run_analysis.zh.md)。

## Evaluation

Evaluation 只在平台配置中选择 lmms-eval task 和运行协议。数据集路径、split、
prompt、样本转换与 metric 仍由 lmms-eval task YAML 管理。

```yaml
experiment:                              # 必填：所有运行的基本信息
  name: my_eval                          # 必填：实验名称
  # run_id: eval-001                     # 可选：每次实验默认生成新 ID
  seed: 42                               # 可选，默认 42：传给 lmms-eval

backbone:                                # 必填：模型及其加载方式
  name: qwen25_omni                      # 必填：已注册的 Backbone 名称
  model_path: ${MODEL_PATH}              # 条件必填：与 model_id 至少填写一个
  # model_id: Qwen/Qwen2.5-Omni-3B       # 可选：模型仓库 ID；model_path 优先
  # model_revision: revision-or-commit   # 可选：仓库 revision 或 commit
  trust_remote_code: false               # 可选，默认不设置：是否允许远程模型代码
  config:                                # 可选，默认 {}：传给该 Backbone 的加载参数
    torch_dtype: bfloat16                # 可选：模型计算精度
    attn_implementation: flash_attention_2 # 可选：Attention 实现
    enable_talker: false                 # 可选：Qwen2.5-Omni 是否加载 Talker

pipeline:                                # 可选：baseline 可以省略
  stages: []                             # 条件必填：compressed 时至少一个 stage

eval:                                    # 必填：Evaluation 的任务和运行协议
  backend: lmms_eval                     # 可选，默认 lmms_eval：评测后端
  compression_mode: none                 # 可选，默认 none：none 或 compressed
  target: baseline                       # 可选：默认根据 compression_mode 推导
  tasks: [mme]                           # 必填：非空的 lmms-eval task 列表
  runtime_batch_size: 1                  # 可选，默认 1：每个 worker 的 batch size
  sample_limit: 10                       # 可选，默认不限制：最多评测多少个样本
  # checkpoint_path: /path/to/model      # 可选：优先于 backbone.model_path/model_id
  num_processes: 1                       # 可选，默认 1：本机 worker 数
  distributed_launcher: accelerate      # 可选，默认 accelerate：或 torchrun
  # main_process_port: 29601             # 可选：多进程 rendezvous 端口
  runtime_mode: default                  # 可选，默认 default：或 profile、debug
  generation:                            # 可选，默认 {}：本次生成协议
    max_new_tokens: 128                  # 可选：最多生成 token 数
    use_cache: true                      # 可选：是否使用生成 KV cache
    temperature: 0.0                    # 可选：采样温度；0 表示贪心生成
    top_p: 1.0                           # 可选：nucleus sampling 阈值
  args:                                  # 可选，默认 {}：额外 lmms-eval 参数
    log_samples: true                    # 可选：是否保存逐样本输出
    verbosity: INFO                      # 可选：lmms-eval 日志级别

output:                                  # 可选：省略时自动分配独立运行目录
  run_dir: ${RUN_DIR}                    # 可选：精确评测输出目录
  manifest_path: ${RUN_DIR}/manifest.json # 可选：默认 run_dir/manifest.json
```

运行方式和多卡约束见[运行评测](../guides/1_run_evaluation.zh.md)。

## Training

`train` 保存平台拥有的训练入口字段；具体 SFT、LoRA、batch、optimizer 和保存策略
放在 `train.args` 中，由当前 training backend 解释。

`experiment.seed` 默认传给 ms-swift；显式 `train.args.seed` 覆盖该默认值，
`train.args.native_argv` 中的 `--seed` 最后优先。实际启动参数保存在运行结果的
backend `argv` 中。该约定不覆盖方法自身的局部随机种子或恢复 checkpoint 的随机状态。

```yaml
experiment:                              # 必填：所有运行的基本信息
  name: my_train                         # 必填：实验名称
  # run_id: train-001                    # 可选：每次实验默认生成新 ID
  seed: 42                               # 可选，默认 42：随机种子

backbone:                                # 必填：模型及其加载方式
  name: qwen25_omni                      # 必填：已注册的 Backbone 名称
  model_path: ${MODEL_PATH}              # 条件必填：与 model_id 至少填写一个
  # model_id: Qwen/Qwen2.5-Omni-3B       # 可选：模型仓库 ID；model_path 优先
  # model_revision: revision-or-commit   # 可选：仓库 revision 或 commit
  trust_remote_code: false               # 可选，默认不设置：是否允许远程模型代码
  config: {}                             # 可选，默认 {}：传给该 Backbone 的加载参数

pipeline:                                # 可选：普通训练可以省略
  stages: []                             # 条件必填：compressed 时至少一个 stage

train:                                   # 必填：Training 后端及其参数
  backend: ms_swift                      # 可选，默认 ms_swift：训练后端
  compression_mode: none                 # 可选，默认 none：none 或 compressed
  runtime_mode: default                  # 可选，默认 default：或 profile、debug
  # checkpoint_in: /path/to/model        # 可选：CLI --checkpoint 写入此字段，优先级最高
  # resume: /path/to/checkpoint          # 可选：恢复 Trainer 状态，不是输入模型
  max_steps: 1                           # 可选：最大训练步数；最小验证通常设为 1
  args:                                  # 可选，默认 {}：以下字段交给 ms-swift
    datasets: ['${DATASET_PATH}']        # 条件必填：与 data.dataset_id 至少填写一个
    trainable_branch: thinker            # 可选：限制可训练的模型分支
    enable_audio_output: false           # 可选：Qwen2.5-Omni 是否训练音频输出
    torch_dtype: bfloat16                # 可选：训练计算精度
    tuner_type: lora                     # 可选：微调方式
    per_device_train_batch_size: 1       # 可选：每张卡的训练 batch size
    gradient_accumulation_steps: 1       # 可选：梯度累积步数
    learning_rate: 0.0001                # 可选：学习率
    lora_rank: 4                         # 可选：LoRA 的 rank
    lora_alpha: 8                        # 可选：LoRA 的缩放系数
    target_modules: [all-linear]         # 可选：LoRA 要适配的模块

data:                                    # 条件必填：未在 train.args 指定数据时使用
  dataset_id: '${DATASET_PATH}'          # 条件必填：数据集 ID 或本地数据路径
  # dataset_commit: revision-or-commit   # 可选：数据集 revision 或 commit
  sample_split: train                    # 可选：写入 Manifest 的数据 split

output:                                  # 可选：省略时自动分配独立运行目录
  run_dir: ${RUN_DIR}                    # 可选：精确实验目录；原生 backend 可在其中创建子目录
  manifest_path: ${RUN_DIR}/manifest.json # 可选：默认 run_dir/manifest.json
```

`train.args.datasets` 是 ms-swift 实际读取的数据；`data` 用于记录可追溯的数据身份。
完整训练配置见 `configs/train/`，运行方式见
[运行训练](../guides/1_run_training.zh.md)。

## Compression stage

Compression 不是独立的第四种配置。把下面的 `pipeline` 放入 Analysis、Evaluation
或 Training 配置，并将对应命令的 `compression_mode` 改为 `compressed`。

未注释的字段组成一个可用 stage 骨架；注释掉的字段按算法需要选择。不要同时使用
`binding` 和 `bindings`，`algorithm` 与 `variant` 必须成对出现。

```yaml
pipeline:                                      # 条件必填：compressed 运行需要此 section
  extensions: []                                # 可选：package.module:register 入口列表
  stages:                                      # 条件必填：compressed 时至少一个 enabled stage
    - name: audio_segmentwise_topk             # 必填：stage 名称；同一 pipeline 内唯一
      enabled: true                            # 可选，默认 true：是否启用该 stage
      placement: projector_level               # 必填：input_level、encoder_level、projector_level 或 llm_level
      execution_phase: forward                 # 必填：prepare、forward 或 generate
      plugin: segmentwise_topk_audio           # 必填：已注册的 compressor/plugin

      method: token_compression                # 必填：token_compression 或 operation_compression；quantization 暂不可运行
      object_family: tensor_slot               # 必填：必须与 method 对应；此处为 tensor_slot
      object_subtype: audio_encoder_output_pre_projector # 可选：更具体的压缩对象
      modalities: [audio]                      # 条件必填：token_compression 必须非空
      token_roles: [observation]               # 可选，默认 []：目标 token 的语义角色
      depends_on: []                           # 可选，默认 []：依赖的其他 stage 名称

      # v2 contract 按需使用；algorithm 和 variant 必须同时填写：
      # algorithm: my_algorithm                # 条件必填：使用 v2 contract 时填写
      # variant: my_algorithm.audio             # 条件必填：必须属于上面的 algorithm
      # binding: qwen25_omni.projector          # 条件必填：与 bindings 二选一
      # target_modalities: [audio]              # 条件必填：必须与 variant 声明一致
      # conditioning_modalities: []             # 可选，默认 []：必须与 variant 声明一致
      # preserved_modalities: [text]            # 可选，默认 []：必须与 variant 声明一致
      # coupling_policy: independent            # 可选，默认 independent：必须匹配 variant
      # media_alignment: temporal               # 可选，默认 none：必须匹配 variant

      # compound stage 用 bindings 替代上面的 binding：
      # bindings:                               # 条件必填：多 binding stage 使用
      #   - binding: qwen25_omni.audio_encoder  # 必填：已注册的物理 binding 名称
      #     target_modalities: [audio]           # 可选：填写时必须匹配 binding
      #     depends_on: []                       # 可选，默认 []：binding 内部依赖
      #     order: 0                             # 可选，默认 0：多个 binding 时不可重复
      #     config: {}                           # 可选，默认 {}：该 binding 的参数

      operation:                                # 可选，默认 {}：由 plugin 解释
        type: keep                              # 可选：本 plugin 只接受 keep
        budget_type: keep_ratio                 # 可选：描述预算类型；本 plugin 不读取
        target_site: audio_encoder_output_pre_projector # 可选：本 plugin 的目标位置
      criterion:                                # 可选，默认 {}：由 plugin 解释
        score_source: encoder_last_layer_attention # 可选：token 评分来源
        ranking: descending                     # 可选：描述排序方向；本 plugin 不读取
      schedule:                                 # 可选，默认 {}：由 plugin 解释
        granularity: per_audio                   # 可选：描述预算粒度；本 plugin 不读取
        keep_ratio: 0.25                        # 可选，本 plugin 默认 0.25：保留比例
      config: {}                                # 可选，默认 {}：其他 plugin 参数

      # 以下字段可选、默认 {}，具体格式尚未形成跨 plugin 的公共约定：
      # activation_scope: {}                    # 可选：激活范围
      # quantization: {}                        # 可选：量化参数
      # calibration: {}                         # 可选：校准参数
```

`operation`、`criterion`、`schedule`、`config` 等 mapping 由具体算法解释，不存在
跨算法的完整子字段全集。新增方法或 binding 时阅读
[开发压缩方法](../guides/2_dev_compression_method.zh.md)；公共 stage 约束见
[接口规范的 Config 规则](../platform/interface_spec.zh.md#8-config-规则)。

位置探索复用同一 stage，不增加方法专用配置层。适用字段如下；具体 site、评分来源和
步数约束由方法与基座共同校验，完整示例见[运行压缩实验](compression_positions.zh.md)。

| 字段 | 含义 |
| --- | --- |
| `operation.target_site` | 实际应用位置，例如投影前后、完整层输出或 Attention 残差后且 FFN 前；不能仅凭 site 名跨越不兼容阶段 |
| `operation.layers` | 零基的实际应用层，须为递增、非负且无重复的整数列表；评分层和预算仍使用该方法原有字段 |
| `operation.layer_edge` | 完整层 site 默认为 `after`；显式使用时须同时给出 `layers`。`before k` 归一为前一层输出，`before 0` 为首层输入；层内 site 不接受此字段 |
| `operation.targets.<target_id>` | 复合方法按已有 target 名设置对应步骤的位置字段，保留其他步骤及依赖顺序；单目标方法可使用根级简写 |

不指定位置覆盖时保持原配方；不同表达解析到同一物理边界时只执行一次。Manifest 记录
实际位置及评分来源，原历史证据保留引用；移动后的探索配置不能继承标准配方的精确验证结论。
模块替换、结构删层和 KV 操作继续使用各自已有模块目标及生命周期配置，不使用 Token 边界替代。

各现有方法通过 `operation.rules.<规则名>` 配置适用的评分、选择、分组、聚合、变换或
操作决策。每项为 `{callable: 'package.module:function', params: {...}}`，省略时保持
原规则；Qwen3 的模块替换使用 `factory`。旧 `criterion.scorer` 和
`operation.selector/grouper/aggregator` 归一到同一入口，同名重复配置报错。
规则与位置分别配置，不要求每个方法有相同接口。改变规则的实验保留历史引用，但不继承
原配方的精确证据。全部方法的名字、对应基座和函数合同见
[规则目录](../../examples/compression_rules/methods.md)，配置样例见
[自定义压缩规则](../../examples/compression_rules/README.md)。

新增三基座 comparison 的参数放在各 stage 的 `config`，不自动接受历史方法的
`operation.rules` 或任意 target site。Token 训练/评测分别使用 `forward`/`generate`，
Operation 两者都使用 `prepare`；后者表示安装时机，不限制运行时决策次数。可运行
DOP-V 配方及各方法校准字段见[对比方法说明](../support/imagellm/comparison_methods.zh.md)。

## 实验输出与研究扩展

CLI 和 `run_experiment(config, task=...)` 共同分配输出身份。`output` 可整体省略，
默认目录是 `runs/<experiment.name>/<run_id>/`；显式 `run_dir` 是精确目录，不再
追加 run ID。已有非空目录或 Manifest 会报错。`manifest_path` 省略时使用
`run_dir/manifest.json`。`plan_only` 检查输出位置但不创建文件。

相对 `run_dir` 和显式 `manifest_path` 按调用时工作目录解析一次，外部 Manifest
不相对于 `run_dir` 拼接。执行使用配置副本：其中 `output.run_dir`、
`output.manifest_path`，以及结果 `paths` 的 `run_dir`、`effective_config`、`result`、
`manifest` 都记录绝对路径；调用者传入的配置不变。显式路径的实际位置保持不变，
也不会因路径规范化而追加 run ID。

实际执行前保存可重新加载的 `effective_config.json` 及 `result.json` 运行状态；
配置快照保留完整配方；本次命令为 `compression_mode: none` 时，Manifest 的压缩阶段为空，
不会在 baseline 完成后再解析闲置压缩配方。压缩运行的阶段和位置取自执行前已解析的定义。
结束后更新结构化结果，只有成功执行写入 Manifest。CLI/Python 的结果字段和示例见
[运行压缩实验](../guides/1_run_compression_experiment.zh.md)。

从已保存训练产物恢复时，调用 `restore_config(run_dir, task=..., checkpoint=None)`，
或以 `--from_run` 替代 CLI 的 `--config`。返回配置保留任务配方，重置运行身份与输出，
并区分完整权重、已物化 Operation、HTA 和原生训练状态；具体支持范围见
[从运行恢复](../guides/1_run_compression_experiment.zh.md#从运行恢复)。

`pipeline.extensions` 是显式注册入口列表，例如 `[my_research.methods:register]`。
每个函数接受本次实验的独立 registry；父进程和原生 worker 都必须能导入该模块。
扩展不修改全局 catalog，也不需要额外运行模式。研究方法即使没有正式 SupportUnit
也可在满足实际合同后执行；接入代码和限制见
[研究人员扩展指南](../guides/2_dev_compression_method.zh.md#4-注册自己的方法)。

查询同一扩展时，使用 `mllm-comp discover --extension my_research.methods:register`
或 `discover_capabilities(extensions=["my_research.methods:register"])`，不需要启动
模型。`discover`、`compare` 及其 Python API 都不引入新的配置 section；完整用法见
[能力查询与结果比较](../guides/1_cli_and_config.zh.md#6-能力查询与结果比较)。

## 环境变量和命令行参数

- `${NAME}` 在 YAML 解析前展开；真实运行不允许留下未定义变量。
- `--run_id`、`--output_dir` 和 `--manifest` 分别修改 `experiment.run_id`、
  `output.run_dir` 和 `output.manifest_path`。
- `analyze --checkpoint` 修改 `backbone.model_path`。
- `eval --checkpoint` 修改 `eval.checkpoint_path`。
- `train --checkpoint` 修改 `train.checkpoint_in`；`--resume` 只恢复 Trainer 状态。
- Analysis 和 Evaluation 的 `--tasks`、`--sample_limit` 只影响本次运行；
  Training 使用 `--max_steps` 临时限制步数。

命令行覆盖先合并进基础配置，再对本次有效配置执行 runnable-config 检查。可以通过
CLI 补齐模型、任务或输出路径，最终配置仍须满足运行要求。实际命令和参数优先级见
[CLI 与配置](../guides/1_cli_and_config.zh.md)。

## 当前不要使用的字段

`backbone.source_path`、`train.script_path` 和 `train.args_path` 虽然仍被 loader
接受，但当前 active runtime/backend 不读取。新配置应使用 `model_path`、
`model_id` 和 `train.args`。
