# CLI 与配置

> 类型：Guide
> 
> 文档层级：1 / 平台使用者
>
> 目标结果：查询已有能力，启动可追溯的 Analysis、Evaluation 或 Training，并比较运行记录
>
> 适用范围：`mllm-comp analyze`、`eval`、`train`、`discover` 与 `compare`
>
> 当前状态：Active

第一次使用时不要从空 YAML 开始，也不需要先记住全部参数。最短流程是：

```text
选择命令 → 复用现有配置或脚本 → 设置本机路径 → 按需用 CLI 覆盖 → 检查产物
```

以下命令均从仓库根目录执行，并假定已经按
[项目 README](../../README.md#环境安装)完成安装。

## 1. 顶层命令

平台提供三个实验命令和两个查询命令：

- `mllm-comp analyze`: 观察或干预模型内部状态，参考 `examples/analysis/`, 详细说明见 [教程](1_run_analysis.zh.md).
- `mllm-comp eval`: 评估，参考 `examples/eval/`, 详细说明见 [教程](1_run_evaluation.zh.md).
- `mllm-comp train`: 训练，参考 `examples/train/`, 详细说明见 [教程](1_run_training.zh.md).
- `mllm-comp discover`: 查询能力声明、binding 和已有验证证据。
- `mllm-comp compare`: 读取两个运行目录，比较指标及记录的协议。


Note:
- `examples/` 保存教学和最小运行入口
- `configs/` 保存项目当前使用的完整评测和训练
- 压缩配置附着在 Analysis、Evaluation 或 Training 运行上。如何选择 baseline/compressed 协议见 [压缩实验](1_run_compression_experiment.zh.md)。
- 顶层命令支持的参数详细信息可以通过运行 `--help` 查看。

## 2. 命令形式

三个实验命令遵循同一个形式；`discover` 和 `compare` 不需要 `--config`，用法见[能力查询与结果比较](#6-能力查询与结果比较)：

```bash
mllm-comp <analyze|eval|train> --config /path/to/config.yaml [CLI overrides]
```

从已有运行的实际 checkpoint 恢复时，用 `--from_run /path/to/run` 替代 `--config`。
两者互斥；恢复后仍支持原命令的任务、设备和输出覆盖，并创建独立新实验。
完整用法与产物边界见[从运行恢复](1_run_compression_experiment.zh.md#从运行恢复)。

例如，下面使用现有 Qwen2.5-VL 配置运行一个样本、一个层的
`representation_dynamics`：

```bash
mllm-comp analyze \
  --config examples/analysis/methods/qwen25_vl_representation_dynamics.yaml \
  --checkpoint /absolute/path/to/Qwen2.5-VL-3B-Instruct \
  --device cuda:0 \
  --run_id representation-dynamics-demo-01 \
  --output_dir outputs/analysis/representation_dynamics/demo-01 \
  --manifest outputs/analysis/representation_dynamics/demo-01/manifest.json
```

把 `demo-01` 替换为本次运行的唯一名称，不要复用已有非空输出目录。项目提供的
shell wrapper 会设置常见环境变量和输出路径，适合直接运行；上面的显式 CLI 形式更
适合确认每个值最终来自哪里。

## 3. 参数配置

同一个值不要同时维护在多个位置。按它的生命周期决定归属：

| 位置 | 适合保存 | 示例 |
| --- | --- | --- |
| YAML | 需要复用和复现的实验协议 | Backbone、task、Probe、generation、train args、compression stages |
| CLI | 只影响本次运行的覆盖 | checkpoint、sample limit、GPU、进程数、max steps、输出路径 |
| 环境变量 | 与机器或调度环境相关的值 | 本地模型路径、数据路径、run root、`CUDA_VISIBLE_DEVICES` |
| shell wrapper | 一组反复使用的启动约定 | 配置路径、环境变量默认值和公共 CLI 参数 |

配置解析后的覆盖关系是：`项目默认值 < YAML < 明确的 CLI override`



环境变量通常先替换 YAML 中的 `${...}`，或由 wrapper 转换成 CLI 参数。机器绝对
路径不要写回长期配置。

阅读一份 YAML 时，按下面的顺序确认即可：

1. `experiment`：本次运行的名称、run ID 和 seed；
2. `backbone`：模型系列、checkpoint 或 model ID；
3. `analysis`、`eval` 或 `train`：当前命令的任务协议；
4. `pipeline.stages`：仅 compressed 运行需要；
5. `data`：训练数据或平台能够可靠记录的数据身份；
6. `output`：新的运行目录和 Manifest 路径。

Note:
- 公共字段的含义、默认值和 CLI 对应关系见[配置字段参考](../reference/config.zh.md)。
- 具体写法仍以 `examples/` 和 `configs/` 中的现有配置为起点。
- 新增或修改 compression stage 时，不要从本指南推导字段，转到[开发压缩方法](2_dev_compression_method.zh.md)。

## 4. 常用 CLI 覆盖

`analyze`、`eval`、`train` 都要求 `--config` 或 `--from_run`，并支持以下覆盖：

| 参数 | 用途 |
| --- | --- |
| `--run_id` | 修改本次运行身份 |
| `--output_dir` | 修改运行产物目录 |
| `--manifest` | 修改 Manifest 路径 |
| `--compression_mode` | 在 `none` 与 `compressed` 间选择 |
| `--log_level` | 修改平台日志级别 |

`--output_dir` 指定精确目录，不会追加 run ID；省略配置中的 `output.run_dir` 时，
默认生成 `runs/<experiment.name>/<run_id>/`。相对输出路径按启动时工作目录解析，
结果 `paths` 和有效配置中的输出路径记录为绝对路径。`--manifest` 也按启动目录
解析一次，可以指向运行目录之外；省略时为本次 `run_dir/manifest.json`。
已有非空运行目录或已有 Manifest 会被拒绝。

命令专属的常用覆盖如下：

| 目的 | Analysis | Evaluation | Training |
| --- | --- | --- | --- |
| 更换输入模型 | `--checkpoint` | `--checkpoint` | `--checkpoint` |
| 恢复训练状态 | — | — | `--resume` |
| 降低最小运行成本 | `--sample_limit` | `--sample_limit` | `--max_steps` |
| 选择任务或方法 | `--tasks`、`--probe` | `--tasks` | 写入 YAML |
| 选择设备或多卡 | `--device` | `--device`、`--num_processes` | 由 launcher 和可见设备控制 |


## 5. 启动前后检查

三个实验命令都可以先查看计划：

```bash
mllm-comp eval --config /path/to/eval.yaml --plan_only
mllm-comp train --config /path/to/train.yaml --plan_only
mllm-comp analyze --config /path/to/analysis.yaml --plan_only
```

`--plan_only` 检查配置和输出位置，不创建文件或加载模型。Evaluation 和 Training
返回原生 argv、环境、工作目录、依赖版本和预期产物；Analysis 返回解析后的分析定义。
计划不能作为真实链路通过的证据；Analysis 可随后使用一个样本、一个 Probe 和一个层
完成最小真实运行。

真实运行结束后至少确认：

1. 命令成功退出，原生框架没有隐藏失败；
2. 本次使用了新的输出目录；
3. `manifest.json` 记录了预期的配置、checkpoint 和运行身份；
4. Analysis artifact、lmms-eval 结果或 trainer/checkpoint 等原生产物真实存在。

需要交接结果时，同时提供实际命令、effective config、source commit、Manifest 和
原生产物路径。配置已经有效但运行失败时，转到
[运行时排错](3_troubleshoot_runtime.zh.md)。

## 6. 能力查询与结果比较

### 查询能力

```bash
mllm-comp discover --backbone qwen25_omni --modality audio --phase generate
mllm-comp discover --kind analysis
```

`discover` 输出 JSON，不加载或实例化模型，也不构造压缩执行器。可用过滤器如下：

| 参数 | 含义 |
| --- | --- |
| `--kind` | `compressor`（默认）、`backbone`、`analysis`、`integration` |
| `--name` | 精确能力名称；查询 compressor 时为 algorithm 名称 |
| `--method` | `token_compression`、`operation_compression`、`quantization` |
| `--variant` | 精确 variant 名称 |
| `--binding` | 精确 binding 名称 |
| `--backbone` | 目标 backbone 名称 |
| `--modality` | `text`、`image`、`video`、`audio`、`scene_3d` |
| `--phase` | `prepare`、`forward`、`generate` |
| `--extension` | 显式 `package.module:register` 入口，可重复传入 |

除 `--name` 外，后七项只用于 `--kind compressor`。例如接入自己的研究模块后，
可通过 `--extension my_research.methods:register` 查询其声明；扩展注册只作用于
本次查询，不修改全局 catalog。`--method quantization` 当前没有已实现 binding 可返回。

压缩查询逐 binding/phase 返回目标模态、参数默认值、`executor_available` 和
`support`。`capability_status`、可用执行器、精确支持证据分别描述不同事实：
没有证据的研究 binding 仍可被查询到；一次推理通过不能证明训练或 checkpoint
重载已通过。具体解释和 Python 示例见[选择能力和实验问题](1_run_compression_experiment.zh.md#1-选择能力和实验问题)。

Qwen3-VL Attention/FFN replacement 已通过 `prepare` Binding 注册，可查询其实际
执行器与支持证据：

```bash
mllm-comp discover --backbone qwen3_vl --method operation_compression --phase prepare
```

这些算法出现在 `items` 中。`catalog_only` 仅列出尚未进入压缩 Algorithm 注册表的
catalog 条目，不补造 variant、phase 或支持证据。

### 比较运行目录

```bash
mllm-comp compare /absolute/path/to/baseline_run /absolute/path/to/compressed_run
mllm-comp compare /absolute/path/to/baseline_run /absolute/path/to/compressed_run \
  --output reports/comparison.json
```

两个位置参数都是运行目录。命令读取 `result.json`、Manifest 和有效配置快照；
也能读取只有默认 `manifest.json` 的历史目录。结果输出为 JSON，`--output` 额外
写入指定 JSON 文件；该参数不会重新运行实验。

`metrics` 按 task、split、metric 对齐数值，`protocol` 报告已知差异和缺失证据。
请结合运行状态及 `protocol.matched` 阅读差值，含义见
[比较已完成的运行](1_run_compression_experiment.zh.md#比较已完成的运行)。预期的输入或文件错误
会输出 `status: failed`、错误类型和说明，退出码为 `2`；比较命令成功退出只表示完成了
记录比较，不表示两次实验都成功或协议匹配。
