# 顶层命令设计

> 类型：Specification  
> 适用范围：`mllm-comp` 顶层路由、共享参数和命令生命周期  
> 当前状态：Active

本文集中定义用户直接调用的顶层命令。算法配置、外部框架参数和具体分析方法不在
这里重复展开，分别参见[接口规范](interface_spec.zh.md)、
[外部框架接入](framework_integration.zh.md)和
[分析运行指南](../guides/1_run_analysis.zh.md)。

## 1. 命令集合

当前提供三个实验命令和两个实验辅助命令：

| 命令 | 职责 | 正常成功产物 |
|---|---|---|
| `mllm-comp analyze` | 在真实样本循环中运行一个或多个 analysis Probe | analysis artifact、`AnalysisRecord`、Manifest |
| `mllm-comp eval` | 通过 active eval backend 运行 baseline 或 compressed 评测 | 原生评测结果、指标、Manifest |
| `mllm-comp train` | 通过 active train backend 运行 baseline 或 compressed 训练 | 原生 checkpoint、训练日志、Manifest |
| `mllm-comp discover` | 查询实际声明、执行机制与验证证据 | 能力 JSON，不加载模型 |
| `mllm-comp compare` | 比较两个已记录实验 | 指标差异、缺失项和协议差异 JSON |

顶层命令采用封闭集合。平台不提供独立 `dry_run`、`validate_manifest` 或 `infer`
命令。只有形成真实、可维护的端到端生命周期后，才能增加新的顶层命令；不得先暴露
占位入口。

查看当前集合或某个命令的实时参数：

```bash
mllm-comp --help
mllm-comp analyze --help
mllm-comp eval --help
mllm-comp train --help
mllm-comp discover --help
mllm-comp compare --help
```

## 2. 路由与共享生命周期

顶层 parser 只识别第一个位置参数并把其余参数交给对应命令。每个命令独立拥有自己的
parser，不能在根 parser 中复制子命令参数。

```text
argv
  -> 顶层路由 analyze | eval | train
  -> 子命令 parser
  -> 加载 YAML
  -> 应用显式 CLI override
  -> run_experiment：解析有效配置、校验 definition、分配独立输出
  -> 只读计划，或记录有效配置/运行状态后真实执行
  -> 记录结构化结果；真实成功后写 Manifest
```

`discover` 直接投影同一个注册表；`compare` 读取已有实验产物。两者不创建实验或加载
模型，分别复用 Python 的 `discover_capabilities` 和 `compare_runs`。

### 2.1 配置与 override 优先级

优先级从高到低为：

1. 本次命令显式传入的 CLI override；
2. YAML 中对应 `analysis`、`eval` 或 `train` section；
3. 命令定义的 backend 默认值。

CLI override 在 runnable-config 检查前应用，因此可以补齐或替换本次模型、任务和
路径。检查针对合并后的有效配置，未解析环境变量仍然报错。CLI 与 Python API
共用 `run_experiment`，输入配置不会被一次运行修改。

应用 override 后，三个命令都会调用共享 compression definition validator。使用
`compression_mode=compressed` 时，至少要有一个 enabled stage；使用 `none` 时，
配置中的 stage 不会被激活。

### 2.2 共享参数

| 参数 | analyze | eval | train | 语义 |
|---|:---:|:---:|:---:|---|
| `--config PATH` | ✓ | ✓ | ✓ | 实验 YAML；与 `--from_run` 二选一 |
| `--from_run PATH` | ✓ | ✓ | ✓ | 从已有实验的有效配置和真实 checkpoint 构造独立实验 |
| `--log_level LEVEL` | ✓ | ✓ | ✓ | 平台日志级别接口；不等同于上游框架 verbosity |
| `--compression_mode none\|compressed` | ✓ | ✓ | ✓ | 覆盖本次运行的压缩模式 |
| `--output_dir PATH` | ✓ | ✓ | ✓ | 覆盖 `output.run_dir` |
| `--manifest PATH` | ✓ | ✓ | ✓ | 覆盖 `output.manifest_path` |
| `--run_id ID` | ✓ | ✓ | ✓ | 覆盖 `experiment.run_id` |
| `--plan_only` | ✓ | ✓ | ✓ | 只读解析与计划，不写入运行文件 |
| `--tasks a,b` | ✓ | ✓ | — | 覆盖任务列表 |
| `--sample_limit N` | ✓ | ✓ | — | 限制样本数，主要用于 smoke |
| `--device DEVICE` | ✓ | ✓ | — | 本次 analysis/eval 设备覆盖 |
| `--runtime_batch_size N` | ✓ | ✓ | — | 每个 analysis/eval worker 的 batch size |

训练设备、分布式策略和框架私有参数继续由 train backend 及 `train.args` 管理，不在
顶层 `train` 命令复制。

使用 `--from_run` 时，`--checkpoint` 选择要恢复的产物；省略则读取源训练结果中记录的
checkpoint。恢复按产物类型重建原方法位置、避免重复物化剪枝；完整训练状态仍交由
Swift 恢复。支持的产物及限制见[运行压缩实验](../guides/1_run_compression_experiment.zh.md)。

## 3. 执行模式

三个命令均提供真实执行与只读计划模式：

| 模式 | 参数 | 加载模型/数据 | 启动上游任务 | 写 Manifest |
|---|---|:---:|:---:|:---:|
| 真实执行 | 不传控制参数 | 是 | 是 | 成功后写入 |
| 启动计划 | `--plan_only` | 否 | 否 | 否 |

两种模式都会先加载完整配置、应用 CLI override 并校验 effective compression
definition。`eval/train --plan_only` 在 `output.plan` 和 `output.versions` 中输出
argv、环境、工作目录、依赖版本和预期产物；`analyze --plan_only` 输出
`output.definition`。计划不创建输出目录，不能证明模型 forward 或训练已经成功。只检查已安装框架与插件时，使用
`python scripts/diagnostics/validate_framework_integrations.py`，不为它维护独立命令模式。

`eval --dry_run` 当前只是 `eval --plan_only` 的遗留别名，不是顶层命令。新脚本和文档
必须使用 `--plan_only`。

## 4. `analyze`

### 4.1 职责边界

`analyze` 复用 lmms-eval 样本循环和 Backbone semantic binding，在模型内部注册 Probe，
收集结果后清理 hook。它不维护独立数据循环，也不提供只校验 schema 的旁路。

```text
effective config
  -> AnalysisDefinition
  -> load backbone runtime
  -> lmms-eval sample loop
  -> Probe observe/intervene
  -> artifacts + AnalysisRecord
  -> Manifest
```

### 4.2 参数分组

| 分组 | 参数 | 说明 |
|---|---|---|
| 模型与阶段 | `--checkpoint`、`--phase` | checkpoint 覆盖；记录 `pre_compression` 或 `post_compression` 阶段 |
| Probe 选择 | `--probe NAME` | 可重复；覆盖 YAML 中启用的 Probe 集合 |
| 通用 Probe override | `--layers`、`--metrics`、`--save_raw` | 临时覆盖层、指标和 raw artifact 行为 |
| 方法特定 override | `--token_groups`、`--collect_points`、`--operations`、`--target_token_types`、`--attn_edges` | 只应用到声明支持该字段的 Probe |

逗号分隔参数会在 definition 阶段规范化。某个 override 没有任何选中 Probe 支持时必须
报错，不能静默忽略。完整方法配置和逐方法脚本见
[分析运行指南](../guides/1_run_analysis.zh.md)。

最小示例：

```bash
mllm-comp analyze \
  --config examples/analysis/qwen25_vl_analysis_matrix.yaml \
  --probe representation_dynamics \
  --sample_limit 1
```

## 5. `eval`

### 5.1 职责边界

`eval` 默认选择 `lmms_eval` backend。平台负责 effective config、plugin spec、原生命令
构造、进程启动、结果定位和 Manifest；任务循环、数据分片、指标聚合和标准结果文件由
lmms-eval 负责。

### 5.2 参数分组

| 分组 | 参数 | 说明 |
|---|---|---|
| backend 与模型 | `--backend`、`--checkpoint` | backend 默认取 `eval.backend`，再回退到 `lmms_eval` |
| 结果语义 | `--target baseline\|compressed` | 未指定时根据 effective compression mode 推导 |
| generation | `--generation_max_new_tokens`、`--use_cache` / `--no-use_cache`、`--temperature`、`--top_p` | 覆盖 `eval.generation` 中对应字段 |
| 分布式 | `--num_processes`、`--distributed_launcher`、`--main_process_port` | 单机多进程数据并行；launcher 为 `accelerate` 或 `torchrun` |
| 运行观测 | `--runtime_mode`、`--profile`、`--debug` | 三者互斥；后两个是 runtime mode 快捷方式 |
| 控制模式 | `--plan_only` | 输出启动计划，不执行评测 |

`runtime_batch_size` 是每个 worker 的 batch size。`num_processes > 1` 时，不应把所有
worker 显式固定到同一个 `--device cuda:0`。完整 native CLI 与参数归属见
[外部框架接入](framework_integration.zh.md)。

启动计划示例：

```bash
export QWEN25_OMNI_MODEL_PATH=/path/to/Qwen2.5-Omni-3B
mllm-comp eval \
  --config configs/eval/lmms_eval/qwen25_omni_img.yaml \
  --plan_only
```

## 6. `train`

### 6.1 职责边界

`train` 默认选择 `ms_swift` backend，也允许 catalog 中声明 train interface 的其他 active
backend。平台负责 effective config、压缩定义、原生命令构造、进程状态、产物定位和
Manifest；Trainer、优化器、数据管线、分布式训练和 checkpoint 保存由上游 backend
负责。

### 6.2 参数分组

| 分组 | 参数 | 说明 |
|---|---|---|
| backend 与输入 | `--backend`、`--checkpoint`、`--resume` | 选择 backend、训练输入 checkpoint 或恢复状态 |
| smoke 与产物 | `--max_steps`、`--artifact_format transformers` | 临时步数上限和可重载压缩产物格式 |
| 运行观测 | `--runtime_mode`、`--profile`、`--debug` | 三者互斥 |
| 控制模式 | `--plan_only` | 输出启动计划，不执行训练 |

训练框架的其余参数写入 `train.args`，不要为每个上游 flag 增加平台 CLI 参数。

ImageLLM 压缩训练会在启动 backend 前检查方法、基座、phase 和实际机制限制。
历史 `train_step_verified` 证据描述已验证范围，缺少证据不阻止合法研究运行。
`python -m scripts.train.render_imagellm_config` 可生成三基座的 24 方法 eval 配置和
23 方法 Swift train 配置；SGL 缺少 forward Binding，D2Pruner 未纳入已验证配方，
两者不由该生成器提供训练配置。

启动计划示例：

```bash
export QWEN25_OMNI_MODEL_PATH=/path/to/Qwen2.5-Omni-3B
export MLLM_COMP_TRAIN_DATASET=/path/to/train_audio.jsonl
export MLLM_COMP_TRAIN_RUN_DIR=/path/to/output
mllm-comp train \
  --config configs/train/qwen2_5_omni/train_aud_sft.yaml \
  --plan_only
```

ImageLLM 的静态入口示例为
`configs/train/imagellm/qwen25_vl_tome_train_step.yaml`；完整生成参数和真实一步验收见
[ImageLLM 验证手册](../support/imagellm/validation_runbook.zh.md)。

## 7. 输出、错误与退出状态

- 三个命令最终输出统一 JSON envelope，包含 `status`、`run_id`、`paths`、
  `returncode`、`native_returncode`、`output` 和 `error`。原生子进程日志仍直连
  stdout/stderr；程序和 Agent 应读取 `result.json`，不把整个 stdout 当成一个 JSON。
- 参数错误、runnable-config 错误和 definition 错误在开始昂贵工作前失败。
- `0` 表示命令请求的模式成功完成；依赖不可用或平台可识别的配置/集成错误使用
  非零状态；native eval/train 进程异常时保留对应的 shell return code 语义。
- 只有真实执行成功后才写 Manifest。plan 和失败运行都不能产生成功
  Manifest。

调用方不得通过匹配日志文本判断成功，必须检查进程退出状态和预期产物。

省略输出路径时，每次运行生成 `runs/<experiment.name>/<run_id>/`。
显式 `output.run_dir` 是精确目录，框架不会在其中再追加 run ID；非空目录及已有
Manifest 拒绝使用。Manifest 省略时写入 `run_dir/manifest.json`。
执行前写 `effective_config.json` 与 `result.json` 的运行状态，成功更新结果并写
Manifest；已准备输出目录的失败实验保留结构化错误，不生成成功 Manifest。
Python 返回相同内容的 `ExperimentResult`，完整示例见
[运行压缩实验](../guides/1_run_compression_experiment.zh.md)。

## 8. 能力发现与比较

```bash
mllm-comp discover --backbone qwen25_omni --modality audio
mllm-comp compare runs/baseline/run-1 runs/compressed/run-2 --output comparison.json
```

发现结果分别展示实际执行器和精确证据，缺少证据不等于没有实现。`--extension`
可重复指定显式注册入口。比较复用原生指标与已有音频效率协议检查；缺失指标不会按
零处理，协议不一致会单独列出。二者成功返回 0；可识别的输入或产物错误返回 2 并输出
结构化错误。`compare --output` 仅把同一结果额外写入指定文件。

## 9. 演进规则

新增或修改顶层命令时，必须同时满足：

1. 命令对应真实生命周期，而不是 schema 展示器或占位入口；
2. 实验命令复用有效配置与 definition 校验，辅助命令读取同一注册表和实验产物；
3. CLI 只保留跨 backend 的稳定参数，框架私有参数留在对应 `*.args`；
4. 明确 normal 与 plan 语义是否适用，不增加含义重叠的开关；
5. 同步本规范、`--help`、CLI 参数测试和至少一个真实或 contract 入口测试；
6. 删除命令时同时删除 dispatcher、模块、测试、示例和活跃文档引用。

具体 command lifecycle 的数据对象和激活规则继续以
[接口规范](interface_spec.zh.md)为准。
