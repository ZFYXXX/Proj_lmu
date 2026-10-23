# 开发压缩方法

第一次扩展可直接修改下面的 FirstK 示例：算法和注册在一个 Python 文件中，
配方继续使用现有训练、评测命令。完成第 2 节即可运行，后面的执行细节按需查阅。

## 1. 选择需要改动的范围

| 你想做什么 | 写什么 | 从哪里开始 |
| --- | --- | --- |
| 在现有方法中更换评分、选择、分组或聚合 | 对应 rule 函数/对象与 `operation.rules` | [规则目录](../../examples/compression_rules/methods.md)、[配置示例](../../examples/compression_rules/README.md) |
| 新方法复用现有信号、位置和执行过程 | 算法实现、必要声明与一次 `register_method` | [完整示例](../../examples/compression_extensions/README.md)及下文 |
| 新增当前没有的信号、位置、状态或模型接入 | 方法实现与对应 Binding/provider | 本文第 5 节，再查模型专用参考 |

只修改评分时，不需要注册新算法；希望新方法具有独立身份和参数时，使用第二条路径。
框架继续负责安装、物理布局、标签、位置、缓存和卸载。方法负责自己的数学、所需输入
以及需要重放的决定。不同执行族的真实接口见第 3 节。

查看已有实现时，[14 组新增对比方法](../support/imagellm/comparison_methods.zh.md)提供三基座
Token/Operation 接入实例，但不自动支持下文所有 rule 或位置探索接口。选择复用机制前
应核对实际 Binding 与对应执行器。

## 2. 先跑一个可修改的完整方法

打开 [first_k_audio.py](../../examples/compression_extensions/first_k_audio.py)，
它按时间顺序保留前 K 个音频 Token。修改自己的方法时只需关注：

1. `FirstKStrategy.compress`：输入特征及预算，返回保留的值、原索引和方法记录。
2. `ALGORITHM`：自己的方法名称。
3. `register`：明确复用的执行机制并接入策略；原有预算和配置规则随计划复用。

核心选择逻辑是：

```python
indices = torch.arange(keep_count, device=hidden_states.device)
return hidden_states.index_select(0, indices), indices, {}
```

替换这段逻辑后，仍须返回预算要求的数量和对应原索引。合并或不同预算使用第 3 节
对应接口。注册复用原计划和配置规则，不需要改名插件类或重复描述操作类型。

在[训练配方](../../examples/compression_extensions/first_k_audio_train.yaml)中同步
`algorithm` 名称，调整 `config.keep_ratio`。评分和分段默认值已由方法提供。
从仓库根目录填写模型、数据路径，先检查计划：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export QWEN25_OMNI_MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B
export MLLM_COMP_TRAIN_DATASET=/absolute/path/to/audio_train.jsonl
mllm-comp train --config examples/compression_extensions/first_k_audio_train.yaml --plan_only
```

查看 `output.plan`，去掉 `--plan_only` 执行，再从 `result.json` 读取结果和产物路径。
Python 调用及训练恢复见[示例说明](../../examples/compression_extensions/README.md)。

若报 `Invalid config for algorithm 'researcher_first_k_audio': keep_ratio ...`，
先修改配方中的 `config.keep_ratio`（范围为 `0 < keep_ratio <= 1`）。复用来源不支持
计划/配置复用时，注册错误会给出来源方法；此时按第 4 节显式声明自己的接口。

## 3. 先确认执行机制能否复用

先看执行器实际向方法传入什么、消费什么结果，再决定如何注册。能够填写相同的
Binding 不代表能替换算法；复用应由另一算法沿同一执行器完成值处理与生命周期来验证。

| 示例 | 算法作者实现 | 执行机制负责 |
| --- | --- | --- |
| [图像 LLM 选择](../../examples/compression_extensions/image_selection.py) | Method 创建请求状态，`select` 返回 `ImageTokenSelection` | 根据位置生成值和布局，处理原生层边界、mask/位置/cache、DeepStack 或 AnyRes 修复与重放 |
| [音频精确选择](../../examples/compression_extensions/first_k_audio.py) | `Strategy.compress`，复用现有预算 | Projector 接点、数量计划、placeholder 与 M-RoPE |
| [音频合并](../../examples/compression_extensions/ltbm_audio.py) | 需要的分组、聚合规则和方法参数 | 原分组/合并重放及音频布局 |
| [视频 Projector](../../examples/compression_extensions/stride_video.py) | 可调用方法与 `replay(features, result)` | 视频布局、基座桥、信号捕获的生命周期、Qwen3 旁路接入；VQToken/FastVID/VisionZip 共用该执行器 |
| [视频 Token 与 decode KV 视图](../../examples/compression_extensions/stride_recent_video.py) | Token 方法、`replay`、为每个 cache 创建的 `new_decode_policy` | prefill 布局、单 Token decode、原生 RoPE、完整缓存写入和稀疏读取视图 |
| Qwen3 Attention/FFN 替换 | 方法的 `plan` 与 `build_replacement`，或已有 `factory` rule | 定位和替换模块、参数归属、重复安装检查与回滚 |

这些接口差异来自它们实际需要的请求状态、精确预算和旁路重放。复用同一种机制时
不必重新学习或重写其他执行族。示例的算法身份保留既有 `researcher_*` 名称，但代码
位于 `examples`，无需导入测试模块。

纯图像选择方法只需要返回当前序列中要保留的图像位置，例如：

```python
from mllm_comp.compressors.image import ImageTokenSelection

return ImageTokenSelection(kept_image_indices=selected_positions)
```

位置使用当前完整序列的索引；执行器保持原顺序，保留全部文本和受保护的结构 Token，
并生成压缩值、删除索引和 `LayoutDelta`。算法可以附加评分与诊断信息，不需要依赖
`FastVResult` 或重复实现 gather。内置 FastV 也使用这段共享处理。合并、历史选择等
方法仍返回其实际需要的结果，不能仅用一组索引表达数值聚合。

需要模型信号时，先看 `compressors/image/decoder_selection.py` 中已有 Method/Call。
`capture_input` / `capture_output` 请求固定层的原生 Q/K，provider 负责实际投影；
`output_hidden_layer` 请求完成层后的归一化特征。DART 组合这两种信号，D²Pruner
使用固定层的最后提示 Token attention，两者沿同一个选择、布局修复与重放路径执行。
需要完整层输出及其投影时，用 `output_feature_layers` 指定来源层，在
`record_output_features` 接收完成后的特征、该层模块和原生输入 RoPE。FEATHER 复用
无 RoPE 打分函数，IVC-Prune 复用坐标与 Value 投影函数；Call 保留原规则所需信号，
不持有模块。Qwen3 在 DeepStack 注入后交付完成层特征。这个回调只在首次前向执行，
checkpoint 重算和增量 decode 沿用已有决定。
FinePruner 展示多阶段复用：IAC 在应用特征上选择；ARP 请求固定输入 Attention。
Call 将已有逐头分数映射到 IAC 保留的图像列，并在最终选择后释放，不改动公共 provider。
SparseVLM 通过 `initialize` 在首次层执行前读取提示嵌入，选定文本评分者；之后的
Attention 捕获、选择与合并仍走同一调用链。合并结果用 `LayoutDelta` 表达成员和输出
顺序，公共布局修复与 checkpoint 重放消费这份决定，方法无需逐基座实现合并重算。
PACT 用 `output_projection_layers` 指定完成层及其投影权重所在层，在
`record_output_projection` 接收原生 RoPE 前 Q/K 和旋转后的 Key；评分与聚类仍由方法负责。
它在 `prepare_attention_mask` 中把聚类质量加到后续层的 Attention 掩码；适配器根据
方法的 `requires_eager_attention` 安装并恢复实际所需的 Attention 实现。
请求对象在选择后释放临时信号，不把一次调用的 Tensor 保存到方法配置中。
需要 checkpoint 绑定资源的方法可实现 `prepare(runtime)`；它在 hook 安装前加载并
校验资源，位置规划不会加载资源。D²Pruner 的校准先验使用此入口，算法仍不持有模型。

Qwen3 Attention/FFN 替换同样通过 Algorithm、Variant、Binding 和 Session 安装。
新配方用 `algorithm` 选择方法，保留 `config.strategy` 和 `schedule.targets`；
替换参数在 optimizer 创建前挂到模型上。更换模块构造时，先使用已有
[`operation.rules.factory`](../../examples/compression_rules/README.md#qwen3-vl-模块替换)。

视频 Projector 的方法由无参数 `strategy_factory()` 创建，压缩与重放都接收 `config`
和 `rules`。更换算法不需要自己安装 hook 或重新实现 DeepStack；需要现有编码器信号时，
选择带相应信号捕获的机制。KV 视图策略的 `plan_view` 返回本步保留位置；该机制保留完整
物理 KV，不因此宣称减少缓存存储。新缓存表示需要自己的物理接入。

### 结构位置、请求阶段和状态

LLM/Decoder 表示模型结构位置。prefill 是生成时处理提示词，decode 是增量生成，
teacher-forced 训练是直接前向；不能把所有 `forward` 都叫 prefill。`generate` 包含
prefill 与 decode，方法在哪一段改变 Token 或 KV 必须由机制明确说明。

可学习参数归模型；一次请求的信号和决定归请求；生成历史归对应 cache。训练重算
使用已保存的决定，不重复推进动态选择。具体实现见
[状态归属](../platform/architecture.zh.md#3-stage-与运行时)和
[方法执行参考](../reference/compression_execution.zh.md)。

## 4. 注册自己的方法

第 2 节的简写适用于保留既有计划、预算和配置规则的策略替换。需要不同计划、参数
规则或执行接口时，向同一个 `register_method` 传入 `AlgorithmDescriptor`，显式提供
数学内核、操作语义、位置类型及配置校验；参见[图像示例](../../examples/compression_extensions/image_selection.py)。
`variant` 默认按方法名和目标模态命名，也可显式指定。

自己的模块通过 `pipeline.extensions: [my_research.methods:register]` 加载，只注册到本次
实验的 registry，不修改全局 catalog；所有原生 worker 都需要能导入这个模块。

### 复用什么，自己声明什么

- `algorithm`：字符串表示复用来源的计划和配置；`AlgorithmDescriptor` 表示自己声明这些语义。
- `reuse_bindings`：明确借用哪些兼容的模型接入；保留其物理 targets、结构句柄、阶段和安装依赖。
- `strategy_factory`：替换已有执行器公开的策略；或者提供 `executor_factory`，明确接上该执行族的方法接口，二者选一。
- `config_defaults`、`config_validator`、`supported_rules`、`conditioning_modalities`、`position_resolver`：仅声明新方法实际使用的内容，默认不复制原方法的这些算法设置。

helper 生成普通 Algorithm、Variant 和 Binding，不另建执行协议。你无需分别克隆三层
对象、删除旧规则或清理原方法的论文身份。物理冲突仍然保留，不能把复用解释为任意
方法可以叠加。原方法的 SupportUnit 和验证结果不会复制到新方法。

完整的底层声明仍可通过 `registry.register_algorithm/variant/binding` 表达。只有
新机制需要它时再读[架构四层模型](../platform/architecture.zh.md#2-四层能力模型)与
[能力合同源码](../../mllm_comp/compressors/contracts.py)。合法研究方法不需要先登记 SupportUnit 才能运行。

## 5. 什么时候修改框架

先写清楚现有机制缺少的实际行为，再改对应层：

| 新需求 | 修改入口 |
| --- | --- |
| 已有算法增加基座 | 新基座 Binding 及其物理适配，复用算法 |
| 新模态或联合音视频 | Variant 描述真实目标/条件/保留模态及对齐，Binding 连接对应流 |
| 新信号、层位置或旁路行为 | 对应 provider/controller，明确采集和消费时机 |
| 新的缓存表示或 decode 状态 | 实际 cache/Attention 接口；Token 的 LayoutDelta 不替代 KV 存储 |
| 新的结构替换或量化模块 | prepare/加载/保存恢复的实际接点，必须在 optimizer 前安装所需参数 |

量化当前仍是[接入设计](../platform/quantization_integration.zh.md)，不能仅注册名字
便当作已实现。分别开启两个单模态方法也不等于联合音视频压缩。

维护者的具体调用链和修改位置见[运行时排查](3_troubleshoot_runtime.zh.md#先按修改任务找到责任代码)。
新增代码后仍走原 session、后端和产物入口，不为方法创建另一个 Trainer 或 runner。

## 6. 用足够的验证完成研究迭代

先为自己的数学写一个小型确定性例子：给定输入，明确期望索引、分组、数值或预算。
再选择该执行族的一条已有原生接入回归，确认方法实际被调用且布局正确。

| 改了什么 | 验证到哪里 |
| --- | --- |
| 仅替换算法/rule，物理机制不变 | 自己的数学断言，加对应执行族的代表接入 |
| 新增位置、状态或旁路处理 | 覆盖改变的布局、请求/重算或缓存行为 |
| 声明训练可用 | 有限 loss、目标梯度、参数更新；声明完整训练恢复时还要保存并继续更新 |
| 宣称质量或效率收益 | 同协议 baseline/compressed 评测或 profile |

已有研究入口回归见
[test_researcher_extension_contract.py](../../tests/unit/test_researcher_extension_contract.py)，
三模态分别有相应 native integration。不要为新增名字复制全部方法×基座矩阵，也不要求
每个研究方法新建单独的 registry、adapter 和 GPU 测试文件。

实验可运行与正式支持声明分开。需要公开某个基座/模态/阶段的能力时，再按
[验证与激活指南](3_verify_capabilities.zh.md)登记该范围的真实证据。没有历史证据不是
阻止研究的理由，接口不匹配和实际不变量失败仍应直接修复。
