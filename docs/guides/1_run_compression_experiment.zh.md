# 运行压缩实验

> 类型：Guide
>
> 文档层级：1 / 平台使用者
>
> 目标结果：使用已有 SupportUnit 完成协议一致的 baseline/compressed 对比，并保存质量与运行证据
>
> 适用范围：当前 ImageLLM、VideoLLM、Qwen2.5-Omni 与 Qwen3-VL operation replacement 的标准配方和位置探索配置；实际能力以精确 SupportUnit 和 runtime 校验为准
>
> 当前状态：Active

本指南面向“使用已有算法做实验”。当前可运行的方法见
[压缩方法支持范围](../support/compression_methods.zh.md)。修改算法或新增执行机制时阅读[开发压缩方法](2_dev_compression_method.zh.md)。算法文件或 YAML 存在不等于
能力可运行，实际选择必须落到精确的算法、variant、backbone、模态和 phase。

## 先完成一条主流程

1. 选择对应模型的现有配方，填写模型路径及当前任务的数据。
2. CLI 加 `--plan_only`，或 Python 传 `plan_only=True`，检查 `output.effective_config` 和实际后端计划。
3. 正式执行后读取 `result.json`：先看状态，再看指标、checkpoint 和运行目录。
4. 继续训练或评测训练产物时用 `--from_run` / `restore_config`，保留原方法和模型结构。
5. 要修改算法时转到[扩展示例](../../examples/compression_extensions/README.md)，仍使用这条运行与恢复路径。

训练、评测和分析分别检查自己实际使用的输入。训练的模型可来自 `train.checkpoint_in`，
评测可来自 `eval.checkpoint_path`；两者未指定时使用 `backbone.model_path/model_id`。
评测数据来自 `eval.tasks`，训练数据来自 `train.args.datasets` 或 `data.dataset_id`；
不需要为无关任务补齐占位字段。

参数按消费者放置：模型加载选项（例如 `attn_implementation`）写在 `backbone.config`；
单卡设备使用 CLI `--device` 或 Python `run_experiment(..., device="cuda:0")`。
`eval.args` 直接传给原生 lmms-eval CLI，`train.args` 传给原生 Swift；不要把模型加载
选项混入评测命令参数。计划里的 `output.plan.argv` 可以查看最终命令。


## 1. 选择能力和实验问题

先回答：

1. 压缩目标是 image、audio、video 还是 operation；
2. 使用哪个 backbone 和 checkpoint；
3. 目标位置是 input、encoder、projector 还是 LLM；
4. 需要 forward、generate、prepare 还是训练证据；
5. 研究问题是正确性、质量、效率、训练还是组合效果。

当前精确状态来源：

- [ImageLLM 多算法](../support/imagellm/status.zh.md)；
- [新增 14 组对比方法及三基座配置](../support/imagellm/comparison_methods.zh.md)；
- [VideoLLM 九算法](../support/video_compression/status.zh.md)；
- [MWOP Mask](fixed_mask.zh.md)与[分模态替换](../../mllm_comp/modality_split/README.md)；
- [Qwen2.5-Omni](../support/qwen25_omni/status.zh.md)；
- [Qwen3-VL](../support/qwen3_vl/README.md)；
- [项目进度](../progress.md)。

一个 active inference 单元不能自动证明 training；一个 backbone 的证据不能外推到
另一个 backbone；image variant 也不能外推到 video。

可以先按研究场景查询真实 binding 和证据，不必启动实验或加载模型：

```bash
mllm-comp discover --backbone qwen25_omni --modality audio --phase generate
```

Python 使用相同查询接口，返回可序列化的字典：

```python
from mllm_comp.experiment_tools import discover_capabilities

capabilities = discover_capabilities(
    backbone="qwen25_omni", modality="audio", phase="generate",
)
for item in capabilities["items"]:
    print(item["algorithm"], item["variant"], item["binding"], item["support"])
```

还可按 `name`、`method`、`variant`、`binding` 过滤；查询其他类别时使用
`kind="backbone"`、`"analysis"` 或 `"integration"`，可用 `name` 指定精确名称。
自己的研究扩展通过 `extensions=["my_research.methods:register"]` 显式注册到本次查询。
查询不实例化模型或压缩执行器，也不把扩展发布到全局 catalog。`executor_available`
只表示声明了可调用的执行器工厂；训练、反向传播、checkpoint 重载等事实应分别检查
`support` 中的证据，不能从 `capability_status` 或空的证据列表推导出来。
完整 CLI 过滤器见[能力查询](1_cli_and_config.zh.md#查询能力)。

## 2. 最小闭环

### CLI：先查看计划，再执行

`mllm-comp eval`、`train`、`analyze` 共同调用 `run_experiment`。三者都接受
`--plan_only`：检查配置与输出位置，返回计划，不创建目录、加载模型或执行任务。

例如先查看一个最小分析计划；确认模型和任务数据可用后，去掉 `--plan_only` 执行：

```bash
mllm-comp analyze \
  --config examples/analysis/methods/qwen25_vl_representation_dynamics.yaml \
  --checkpoint /absolute/path/to/Qwen2.5-VL-3B-Instruct \
  --device cuda:0 --plan_only
```

以 Qwen2.5-Omni 音频 Segmentwise Top-K 为例，先在相同 checkpoint 和样本范围上运行
baseline，再运行 compressed：

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B \
bash scripts/eval/lmms_eval/qwen25_omni_2gpu_aud.sh --sample_limit 1

MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B \
bash scripts/eval/lmms_eval/qwen25_omni_2gpu_aud_segmentwise_topk.sh --sample_limit 1
```

共享机器上按实际 GPU 分配调整可见设备和进程数。最小运行应确认：

- 两条命令都通过同一 lmms-eval task 和 chat/generation 协议；
- compressed runtime 命中目标 stage；
- token 或结构按配置发生真实变化；
- legalizer/invariant 通过；
- 两次运行分别写入独立 Manifest 和输出目录。

训练继续使用已有 ms-swift 配方。例如查看压缩训练计划：

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B \
DATASET_PATH=/absolute/path/to/train_audio.jsonl \
OUTPUT_DIR=/absolute/path/to/new_training_run \
bash scripts/train/qwen2_5_omni/train_aud_sft_segmentwise_topk.sh --plan_only
```

去掉 `--plan_only` 后执行训练。脚本提供的 `OUTPUT_DIR` 按该配方解析为显式目录，
例如 Segmentwise 配方会使用其下的 `segmentwise_topk/`；真实 checkpoint 和训练日志
位置以结果中的后端输出为准。`train.resume` 指定恢复输入，恢复训练仍写入新的实验目录。

### Python：每次调用是独立实验

Python 与 CLI 使用相同配置、能力校验、后端和产物逻辑，不需要自己安装压缩 hook。
下面直接复用分析配方；`eval` 和 `train` 分别传对应配置与 `task`：

```python
import json

from mllm_comp.analysis.definition import AnalysisOverrides
from mllm_comp.config import config_to_dict, load_config
from mllm_comp.experiments import run_experiment

config = load_config(
    "examples/analysis/methods/qwen25_vl_representation_dynamics.yaml"
)
config.backbone.model_path = "/absolute/path/to/Qwen2.5-VL-3B-Instruct"
overrides = AnalysisOverrides(sample_limit=1)

plan = run_experiment(
    config, task="analyze", device="cuda:0",
    analysis_overrides=overrides, plan_only=True,
)
print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))

# 正式执行；再次调用会开始另一次独立实验。
result = run_experiment(
    config, task="analyze", device="cuda:0", analysis_overrides=overrides,
)
report = result.to_dict()
if report["status"] == "failed":
    print(report["error"])
else:
    print(report["paths"]["run_dir"])

# run_experiment 使用配置副本，不会把生成的 run_id、目录或覆盖值写回输入。
original_recipe = config_to_dict(config)
```

`run_experiment(config, task="eval")` 和 `run_experiment(config, task="train")`
分别运行 lmms-eval 与 ms-swift 配方。需要重用配方时重用配置即可，不重用已结束的
模型 runtime。若本任务实际使用的字段包含 `${...}`，在 `load_config` 前设置对应环境变量。
未使用任务的字符串占位符可以保留；例如评测不要求填写训练数据。配置的整体类型仍须合法。

### 输出目录与覆盖

普通配方可以省略整个 `output`。默认目录是
`runs/<experiment.name>/<run_id>/`，未指定 `run_id` 时每次生成新的运行身份。
配置或 CLI 显式指定 `run_dir` / `--output_dir` 时，它就是精确目录，不会追加
`run_id`；只更改 `run_id` 不会移动一个显式目录。

`manifest_path` / `--manifest` 省略时采用本次 `run_dir/manifest.json`。例如只指定
`--output_dir /absolute/path/to/new_run` 就能同时确定默认 Manifest 位置。
显式 Manifest 可以另放，但已有 Manifest 或非空运行目录都会被拒绝，不覆盖旧实验。
已有空目录可以使用。失败后使用新目录；自动目录配方直接再次运行即可。

相对 `run_dir` 和显式 `manifest_path` 都按调用时的工作目录解析一次；外部 Manifest
不会相对于 `run_dir` 再拼接。`ExperimentPaths` 的 `run_dir`、`effective_config`、
`result`、`manifest` 全部记录为绝对路径，有效配置中的 `output.run_dir` 和
`output.manifest_path` 同步记录实际绝对位置。调用者传入的配置对象保持原值，因此
可以继续复用；规范化路径不会改变显式目录的位置，也不会额外追加 run ID。

训练脚本的环境变量目录、论文 sweep 的 run tag、分析 wrapper 的时间戳目录，以及
诊断脚本的专用工件目录仍是显式协议。继续使用这些入口时按其变量或 CLI 参数指定新目录，
不要假定公共默认规则会替它们追加目录。`plan_only` 也检查目录冲突，但不写文件；
计划里自动生成的运行身份不会被下一次正式调用自动复用。

每个正式实验的最小产物如下：

| 文件 | 用途 |
| --- | --- |
| `effective_config.json` | 运行前写入，包含已解析配置、实际任务默认值、运行身份、输出路径和分析覆盖，可重新加载 |
| `result.json` | 运行前写入 `running`，结束后更新为 `succeeded` 或 `failed`，统一提供路径、退出码、任务输出和错误 |
| `manifest.json` 或显式 Manifest | 成功后写入的实验、压缩和验证记录；失败不生成成功 Manifest |

重新使用 `effective_config.json` 时，它已经包含显式运行身份与目录；先为新实验设置
新的输出位置，或清除这些字段以重新使用自动目录。原生后端结果、analysis artifact、
checkpoint 和日志仍保留各自格式，通过结果中的路径找到。

### 从运行恢复

`restore_config` 读取运行目录内的 `effective_config.json` 和 `result.json`，返回供
`run_experiment` 使用的配置。默认 checkpoint 取训练结果实际记录的输出；不会把
评测的输入模型当成训练产物，也不会从目录名猜测最新 checkpoint。

```python
from mllm_comp.experiments import restore_config, run_experiment

config = restore_config("/absolute/path/to/training_run", task="eval")
config.eval.tasks = ["mmau_test_mini"]
config.eval.sample_limit = 1
result = run_experiment(config, task="eval")
```

CLI 使用同一入口；`--from_run` 替代 `--config`，其他任务参数仍可覆盖：

```bash
mllm-comp eval --from_run /absolute/path/to/training_run \
  --tasks mmau_test_mini --sample_limit 1 --plan_only
mllm-comp train --from_run /absolute/path/to/training_run --max_steps 4
```

恢复清空原 `run_id`、运行目录和 Manifest 路径，也清除原诊断输出文件名；不修改
原运行。默认生成新目录，也可显式指定新的 `--output_dir` 和 `--manifest`。
如果中断运行已有实际 checkpoint、结果尚未记录它，可用
`--checkpoint /absolute/path/to/checkpoint` 显式选择；该参数在恢复模式下选择恢复产物，
由入口决定它属于模型权重、HTA 模块还是 Swift resume。训练的 `--from_run` 与
`--resume` 不同时使用。

| 产物 | 恢复行为 |
| --- | --- |
| 完整 Hugging Face 权重目录 | 要求 `config.json` 和真实权重文件；分片索引引用的文件必须存在。Eval/Analysis 加载该模型 |
| 已物化的删层 checkpoint | 根据 checkpoint 中的 `mllm_comp_operations` 或历史 Operation 导出记录，停用已经应用的同名 stage；没有证据时明确报错 |
| 完整 checkpoint 内的 HTA | 保留原位置、模块名称和架构配置，再从该 checkpoint 加载对应 HTA 权重 |
| 单个 HTA state-dict 文件 | 通过显式 `checkpoint` 选择；原配方必须只有一个启用的 HTA stage，基座保持原值。具体张量与形状由现有 HTA loader 校验 |
| Swift 训练 checkpoint | 保留原初始模型和压缩结构，将实际 checkpoint 传给 `train.resume`；原生 Swift 负责权重、optimizer、scheduler、RNG 等状态恢复 |
| PEFT adapter | 可交原生 Swift 继续训练；当前 Eval/Analysis 没有通用 PEFT 加载或合并入口，需要提供完整合并权重 |
| selected-parameter / final-norm 诊断产物 | 明确拒绝；它仅用于既有诊断验证，不是完整模型或原生训练状态 |

训练恢复要求原生 `trainer_state.json`，不把诊断的 `training_state.json` 当作替代。
恢复入口显式使用 `resume_only_model=false`，不继承旧配方的仅权重热启动模式。
Thinker-only checkpoint 不包含 Talker，恢复 Eval/Analysis 时自动关闭 Talker。

Omni 的动态 Token 压缩生成默认启用 KV Cache，恢复评测不沿用训练时关闭缓存的默认值。
这条生成路径暂不支持显式 `use_cache=False`；直接训练 `forward` 不受此限制。

`restore_config` 只检查配置与产物结构，不加载模型、不执行训练，也不重新验证历史
支持声明；实际加载和执行仍由现有模型与后端完成。原配方的相对输入路径保持原值，
跨工作目录使用时应确认模型、数据和外部模块路径仍有效。

### 结构化结果与失败

返回值是 `ExperimentResult`，`to_dict()` 和 CLI 最终输出采用同一结构：
`schema_version`、`task`、`status`、`run_id`、`paths`、`returncode`、
`native_returncode`、`output`、`error`。计划状态为 `planned`，计划位于
`output.plan` / `output.versions`；分析计划位于 `output.definition`。
三种计划均提供 `output.effective_config`，可直接核对 CLI 覆盖、默认值和展开后的
压缩身份；未使用任务的占位符仍按原值保留。
`paths` 包含 `run_dir`、`effective_config`、`result`、`manifest`。

可预期的失败返回 `failed`，`error` 包含 `type`、`stage`、`message`；应检查
`returncode`，不要仅判断是否存在输出文件。后端退出码保存在 `native_returncode`，
例如原生信号退出 `-15` 对应公共退出码 `143`。意外 Python 异常在能够写入结果时记录失败，
随后重新抛出。未取得可用输出位置的失败不会覆盖冲突目录中的文件，也不保证产生新的结果文件。

CLI 保留上游日志，因此 stdout 可能同时出现日志和最终 JSON。脚本或 Agent 跟踪正式
实验时应读取 `result.json`；计划没有结果文件，直接使用返回的计划内容。

### 遇到失败，先看可修改的字段

| 结果或错误 | 下一步 |
| --- | --- |
| `error.stage` 为配置/校验，包含 `${...}` 和字段路径 | 填写列出的字段，或在加载 YAML 前设置对应环境变量 |
| 缺少 `eval.tasks` / `analysis.tasks` | 指定当前任务的名字；训练数据不能代替评测 task |
| 输出目录冲突 | 选择新的 `--output_dir`，或省略显式目录让框架生成独立运行目录 |
| 方法、binding 或阶段不匹配 | 在计划有效配置中检查实际展开身份，并查询该机制的声明 |
| 后端执行失败 | 读取 `native_returncode` 与该运行的原生日志，再查[分层排错](3_troubleshoot_runtime.zh.md) |

脚本或 Agent 可以按相同步骤读取计划、运行状态和错误，再修改配置开启新实验。
规划不执行模型，单次成功也不自动晋级方法的正式支持状态。

## 3. 完整实验流程

### 固定协议

在扩大样本前固定：

```text
checkpoint + revision
task + split + prompt/chat protocol
generation kwargs + seed
precision + cache + batch
sample set
compression definition
quality/profile 输出位置
```

Baseline 与 compressed 之间只改变 compression definition 和输出身份。不要同时改变
checkpoint、prompt、batch 或 generation 参数后把差异归因于压缩。

### 配置 stage

通用 v2 stage 展开后使用精确身份：

```text
algorithm -> variant -> binding
```

并保证 placement、phase、target modalities、operation、budget/schedule 与 binding 一致。
Qwen2.5-Omni AudioLLM 与 ImageLLM 都使用这条 v2 路径。请求配置可以省略由注册表
唯一补全的字段；存在多个合法 variant、binding 或 phase 时须显式选择。完整生效配置会
记录解析后的身份，`plugin` 与 `algorithm` 保持一致。SupportUnit 单独记录精确范围的
验证证据，缺少历史证据不阻止满足实际执行契约的研究运行。

下面是已有 Segmentwise Top-K 配方的 stage 片段；模型、数据和任务字段继续使用
[完整示例](../../examples/compression/qwen25_omni_audio_segmentwise_topk.yaml)：

```yaml
pipeline:
  stages:
    - name: audio_segmentwise_topk
      algorithm: segmentwise_topk_audio
      variant: segmentwise_topk_audio.audio
      binding: segmentwise_topk_audio.audio.qwen25_omni
      placement: projector_level
      execution_phase: forward
      target_modalities: [audio]
      operation:
        type: keep
        target_site: audio_encoder_output_pre_projector
      schedule:
        keep_ratio: 0.25
```

预算与评分参数按方法解释，不应将一个方法的层号或比例规则套用到另一个方法。
需要探索位置时，先查[压缩位置与评分来源](../reference/compression_positions.zh.md)：
[Audio](../reference/compression_positions.zh.md#audio)、
[Image](../reference/compression_positions.zh.md#image)、
[Video](../reference/compression_positions.zh.md#video) 分别列出允许的边界、评分可用性和布局约束。
先在有效配置与运行记录中确认实际位置，再比较质量与成本；改位置不自动获得原位置的证据。

### 从正确性扩大到质量和效率

推荐顺序：

1. deterministic/contract 证据已经存在；
2. 单样本真实 checkpoint 运行；
3. baseline/compressed 小样本协议对齐；
4. 完整质量评测；
5. 独立 profile 运行；
6. 汇总 Manifest、指标和 trace；
7. 需要训练时再进入 compression-aware training。

`--profile` 用于效率采集，`--debug` 用于逐阶段或逐层统计。两者可能改变同步、内存和
时延，不应默认同时开启，也不应把 debug 运行当作正式效率结果。

### 比较已完成的运行

使用两次结果中 `paths.run_dir` 给出的实际目录：

```bash
mllm-comp compare /absolute/path/to/baseline_run /absolute/path/to/compressed_run \
  --output reports/comparison.json
```

Python 返回同一份字典；需要保存时可自行写入 JSON：

```python
from mllm_comp.experiment_tools import compare_runs

comparison = compare_runs(
    "/absolute/path/to/baseline_run", "/absolute/path/to/compressed_run",
)
print(comparison["protocol"])
print(comparison["metrics"])
```

先检查 `left`、`right` 中的运行状态、退出码和缺失产物，再查看 `protocol`：

| `protocol.matched` | 含义 |
| --- | --- |
| `true` | 已检查的协议字段一致，所需记录没有缺失 |
| `false` | 存在已知差异；`differences` 给出字段和两侧值，`missing` 仍可能非空 |
| `null` | 未发现已知差异，但缺少判断所需的记录，不能确认匹配 |

例如，同步到服务器的源码目录或安装包没有本项目的 Git 元数据时，Manifest 不记录
源码身份；此时 `missing` 会指出 `source_tree_hash` / `source_scope`，比较仍返回已记录
的指标，但不会确认协议完全匹配。这种情况与评测执行失败不同，应分别检查运行状态和
比较证据。源码记录范围见[Manifest 约定](../platform/interface_spec.zh.md)。

`metrics` 按 `(task, split, metric)` 对齐原始数值，`absolute_delta` 是右侧减左侧，
`relative_delta` 是该差值除以左侧值。缺失指标保留为 `null`，不会按零处理；左侧值为
零时不计算相对差值。训练指标来自原生训练结果；分析保留两侧 `analysis_records` 的
摘要、样本数和产物路径，不把不同 Probe 的摘要强行转成统一质量指标。
`config_differences` 另列有效配置差异，省略输出身份等运行位置字段。

这些是记录中的数值差异；正差值不一定代表更好的指标，协议匹配也不能单独证明
压缩带来了因果提升。仍需根据指标方向、样本范围和实验控制解释结果。

`audio_efficiency` 当前只比较 Qwen2.5-Omni 的 baseline/compressed 音频 profile
记录。两侧均为 `runtime_mode=profile`，且通过已有音频比较器对协议、trace、样本数量
和运行环境的检查时，才返回效率比值；证据不足时返回 `status: unavailable` 及原因。
该结果不外推为图像或视频的速度结论。比较只读取现有产物，不重新执行模型。

### 多 stage

多 stage 必须明确顺序和依赖，并连续维护布局：

- 后续 stage 使用前一 stage 的有效 token/layout；
- token budget、position、mask、placeholder 和 cache 不能从原始输入重新猜测；
- 多目标流需要明确 coupling policy 和 alignment；
- 两个 independent 单模态 stage 不等于联合音视频方法。

当前 contract fixture 可以验证定义组合，但不自动证明目标组合在真实模型上 active。

逐方法的多步骤位置、评分依赖和 DeepStack/cache 处理见
[压缩位置与评分来源](../reference/compression_positions.zh.md)。更改应用层之前，应先确认
该步骤的原评分来源在新边界已经可用，并保留其阶段顺序与预算约束。

## 4. 验收与声明边界

一次压缩实验至少分开回答：

| 问题 | 所需证据 |
| --- | --- |
| 算法是否按定义执行 | kernel/Strategy 断言和 budget/layout 记录 |
| 是否接入真实模型 | checkpoint runtime、目标位置与 invariant |
| 是否保持质量 | 完整 baseline/compressed 同协议指标 |
| 是否提高效率 | 独立 profile 与资源统计 |
| 是否支持训练 | backward、gradient、optimizer、checkpoint/reload |

单样本生成成功只能证明真实链路 smoke；一步训练只能证明对应训练单元；论文默认参数、
目标规模质量和收敛仍需独立实验。保存结果时应同时保留 baseline/compressed 的
Manifest、原生框架结果和实际运行命令，不能只保存最终比较表。
