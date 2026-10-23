# MLLM-Comp-Platform 接口规范

> 类型：Specification
> 当前状态：Active
> 规范语言：中文
历史设计来源：`../archive/platform_design_v0.3.zh.md`

本文只定义当前代码契约、命令生命周期、验证规则和激活条件。顶层 CLI 的集中说明见[顶层命令设计](top_level_commands.zh.md)；当前项目进度见 `../progress.md`，研究人员接入步骤见 `../guides/2_dev_compression_method.zh.md`，Qwen2.5-Omni 既有方法见 `../support/qwen25_omni/audio_compression/extension.zh.md`；v0.3 设计文档只解释历史动机；概念解释见 `core_concepts.zh.md`；完整配置见 `configs/`。

## 1. 范围与状态标签

### 1.1 范围

本文描述配置、方法契约、运行生命周期与基座接入边界，不维护方法数量或 Active 名称表。
当前能力使用 `mllm-comp discover` 查询；论文分类与精确支持的区别见
[压缩方法支持范围](../support/compression_methods.zh.md)。历史 Phase 1a 的范围留在归档设计，
不用于判断当前实现。

### 1.2 状态标签

Catalog 的 `CapabilityStatus` 只有以下三个值：

| 标签 | 含义 |
|---|---|
| `active` | 已实现、通过相应验证，并进入 active registry |
| `target` | 待激活的能力；实际可执行范围由能力种类和执行合同决定 |
| `future` | 保留的后续方向 |

压缩支持另由精确 `SupportUnit` 的 `SupportStatus` 和证据记录表达，`declared` 属于
这套支持状态，不能用于 Catalog；完整含义见[能力验证指南](../guides/3_verify_capabilities.zh.md)。

Backbone、Analysis 和后端运行注册表只接收 `active` 条目。压缩方法的全局 `target`
不等同于不可执行：其运行准入取决于选中算法的 schema、Binding、phase、模型结构和
执行冲突；历史证据随实验记录保留。`active` 也不表示任意模型、位置或训练方式均已验证。

- Algorithm/Variant/Binding 描述算法、模态与执行机制；SupportUnit 描述验证范围。
- 动态 Token 方法使用 `forward` 或 `generate` 生命周期；生成请求内须区分 prefill 与增量 decode，实际决策范围由方法声明。当前静态结构 Operation 在 `prepare` 改写模型。结构位置与生成阶段见[术语约定](core_concepts.zh.md#结构位置与生成阶段)。
- definition、plan-only 和 CPU 合同检查不能替代真实 checkpoint、恢复或质量证据。

## 2. 模块边界

| 模块 | 可以包含 | 不可以包含 |
|---|---|---|
| `core` | 跨基座 state/action/plan/manifest/invariant | 具体模型加载、算法实现 |
| `backbones` | facade、profile、processor/model/state/legalizer/runner | 通用压缩算法 |
| `compressors` | 平台原生压缩算法和选择器 | 外部框架调用 |
| `pipeline` | plan 执行、legalization、operation apply、trace | 算法评分逻辑 |
| `analysis` | probe、hook、analysis artifact | 独立重复的数据循环 |
| `integrations` | lmms-eval、训练框架 wrapper | 可复用压缩算法 |
| `export` | checkpoint/operation artifact | 训练和评测流程 |
| `runtime` | processor/model 加载、设备设置、运行资源 | 压缩算法评分与后端任务循环 |

算法不得放入 `third_party`、临时 wrapper 或 shell script。

## 3. 命令生命周期

`configs/` 只保存完整的可运行 eval/train 配置，不提供模板目录或 YAML 组合机制。可运行 analysis 配置位于 `examples/analysis/`，compression contract 示例位于 `examples/compression/`。真实 analyze/train/eval 入口必须先调用 `validate_runnable_config()`，拒绝缺失关键字段或仍含未解析环境变量的配置。

### 3.1 共享 definition 校验

平台不暴露独立 compression dry-run。`analyze`、`eval` 和 `train` 在应用 CLI override
后、进入 backend 前调用 `resolve_compression_definition()`；runtime activation 对直接
调用再次执行同一校验。校验覆盖：

1. backbone 和命令 backend capability；
2. runtime batch 上限；
3. enabled stage、唯一名称与跨方法 stage 不变量；
4. method descriptor 和 compressor schema；
5. v2 Algorithm/Variant/Binding contract、phase、依赖与冲突；关联精确 SupportUnit 证据。

definition 校验不调用模型 forward、planner、legalizer 或 operation applier，不能单独
证明真实 pipeline 可运行。需要只查看框架启动计划时，使用真实 `analyze`/`eval`/`train` 命令的
`--plan_only`；该选项仍会先经过同一个 definition validator。

### 3.2 共同实验入口

CLI 与 Python 都调用 `mllm_comp.experiments.run_experiment()`：

```text
配置副本 + 显式覆盖
  -> 配置与 definition 校验
  -> 分配本次运行身份和路径
  -> plan_only 返回计划，或写 effective_config/result 并执行任务
  -> 收集原生产物、结果和压缩运行记录
  -> 写成功 Manifest，或记录结构化失败
```

Analysis 通过 lmms-eval 数据接口取得样本，在拥有的 BackboneRuntime 上安装压缩和 Probe，
执行 `forward/generate` 后保存产物，并在成功或失败时释放 Probe 与运行资源。
Eval 构造原生 lmms-eval 启动计划，由模型插件把压缩接入实际请求；Train 构造 Swift 计划，
由 ModelLoader 在 optimizer 创建前安装结构操作和动态执行器。

checkpoint、训练日志与有效参数由原生后端产生，平台记录返回路径。一次执行成功、单步
backward、保存恢复与训练收敛是不同证据。具体 Python 返回值、目录规则和失败处理见
[运行压缩实验](../guides/1_run_compression_experiment.zh.md)。

## 4. Core 数据模型

### 4.1 `TokenSpan`

```python
@dataclass(frozen=True)
class TokenSpan:
    sample_index: int
    start: int
    end: int
    modality: Literal["text", "image", "video", "audio", "scene_3d"]
    role: Literal["instruction", "observation", "answer"]
    source: str | None = None
    media_unit: str | None = None
    time_span: tuple[float, float] | None = None
    encoder_origin: str | None = None
    alignment_group: str | None = None
```

`start` inclusive，`end` exclusive。`modality_token_spans` 是唯一 canonical span 字段。Qwen2.5-Omni 原生接入产生 image、video 和 audio span；Qwen3-VL 产生 image/video span，视频时间戳对齐时同一模态可以有多个不连续 span；scene_3d 仍为预留枚举。

### 4.2 `ForwardState`

稳定字段组：

| 组 | 字段 |
|---|---|
| token carrier | `input_ids`, `inputs_embeds` |
| derived token fields | `attention_mask`, `position_ids`, `labels` |
| image | `pixel_values`, `image_grid_thw` |
| video | `video_pixel_values`, `video_grid_thw`, `video_second_per_grid` |
| audio | `audio_values`, `audio_attention_mask`, `audio_feature_lengths` |
| provider-native | `native_model_inputs` |
| normalized metadata | `token_type_mask`, `modality_token_spans`, `modality_metadata` |
| runtime | `hidden_states`, `sample_ids`, `stage_history` |

约束：

- StateBuilder 创建 state；
- compressor 只读 state；
- legalizer 返回新的 state，不原地修改输入；
- `stage_history` 只能由当前压缩执行路径的 owner 写入：通用 token pipeline 写自己的执行结果，Qwen2.5-Omni Binding controller 写其 v2 Stage 结果；integration 不得绕过执行器直接写入；
- backbone-specific position/grid 规则不塞进通用 core。

### 4.3 Runtime、输出与 trace

`BackboneRuntime` 由单次命令生命周期持有，包含 model、processor、device 和 dtype；facade 不保存重量级运行时对象。

```python
@dataclass
class TextGenerationOutput:
    token_ids: Any
    text: list[str]

@dataclass
class SpeechGenerationOutput:
    waveform: Any
    sample_rate: int
    speaker: str | None = None
    codec_token_ids: Any | None = None

@dataclass
class GenerationResult:
    text: TextGenerationOutput | None = None
    speech: SpeechGenerationOutput | None = None
    traces: TraceArtifacts | None = None
    sample_ids: list[str | int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

通用 text logits/hidden states/attentions/KV cache 只在 `TraceRequest` 明确请求时通过 hook 保留。Qwen2.5-Omni 额外提供 `thinker_reply_part`、Talker 中间量与 Token2Wav 输入 trace；复合 `generate()` 默认只返回完整 text sequence，或 `(text sequence, waveform)`，Talker codec tokens 不得伪装成默认模型返回值。Qwen3-VL 只返回 text，speech 必须为空。

当前验证的 Transformers 4.57.3 没有公开的 Qwen2.5-Omni 复合文本/语音流式返回契约。平台明确记录 `streaming_supported=false`，不模拟流式能力。Omni 语音输出 batch size 为 1；Qwen3-VL 完整结果 streaming 路径也尚未激活。

### 4.4 `TokenAction`

```python
@dataclass
class TokenAction:
    op: Literal["keep", "drop", "merge", "pool", "reorder", "reweight"]
    target_indices: Tensor
    merge_groups: Tensor | None = None
    weights: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

当前保留式 token compressor 推荐产生 `op="keep"`，由 legalizer 统一应用。

### 4.5 `LegalizationPolicy`

稳定 policy：

- position：`compact`、`preserve_absolute`、`m_rope_rebuild`、`backbone_default`；
- mask：`rebuild_from_tokens`、`preserve`、`backbone_default`；
- label：`rebuild_ignore_index`、`preserve`、`backbone_default`；
- coordinate：`preserve`、`drop_removed`、`backbone_default`。

Qwen2.5-Omni 的具体 M-RoPE layout 由 Qwen legalizer 和 backbone-specific invariant 负责。

### 4.6 `OperationAction`

保留最小语义：

- operation：`skip`、`mask`、`prune`、`replace`；
- target：layer、attention、FFN、head、channel、expert、module、projector；
- apply mode：runtime gate、static mask、structural rewrite；
- semantic `module_path`；
- route：`v2v`、`v2t`、`t2v`、`t2t`。

下面列出早期两类 OperationAction/applier 参考子集，不代表全部 Operation binding。
第一类是 Qwen2.5-Omni Thinker 的完整 LLM 层剪枝：

```text
op=prune
target_kind=layer
target_scope=llm
apply_mode=structural_rewrite
shape_effect=remove_module
requires_checkpoint_export=true
```

该子集由 prepare-time Qwen `OperationApplier` 执行，并已经通过结构 checkpoint、哈希 Manifest、reload 和重载后生成。

第二类是 Qwen3-VL LLM Attention/FFN 的保持形状替换：

```text
op=replace
target_kind=attention | ffn
target_scope=llm
apply_mode=structural_rewrite
shape_effect=preserve_io_shape
requires_checkpoint_export=false
```

Attention 插件激活 `attn_linear_attention`、`attn_cnn`、`attn_identity`、`attn_lolcats_linear_window`、`attn_nha` 和 `attn_sparse_window`；FFN 插件激活 `ffn_low_rank`、`ffn_identity` 和 `ffn_two_layer_mlp`。两者在原始 checkpoint 加载后按 YAML 安装到 `llm.layer[n].attn` 或 `llm.layer[n].ffn`，session 卸载时恢复原模块；reload 必须重新应用配置，不导出独立 replacement checkpoint。

以上替换指 `.shared_llm` 路径；另有 Qwen2.5-VL/Qwen3-VL `.image_llm` 的
[分模态编译与保存恢复](../../mllm_comp/modality_split/README.md)、双基座
[MWOP Mask](../guides/fixed_mask.zh.md)和[三基座 comparison 方法](../support/imagellm/comparison_methods.zh.md)。
它们通过各自 Binding 执行，精确单元保持 `declared`；已有具体 mask/gate/skip 实现
不代表通用 OperationAction executor 覆盖任意目标、任意组合或 Talker。
`OperationAction`、`OperationApplyResult` 与 `OperationExportRecord` 从稳定 `mllm_comp.core` 顶层重导出；更完整的 route/composition helper 仍从其显式模块导入。

### 4.7 `CompressionPlan`

plan 是 plugin 意图与 pipeline 执行之间的唯一桥：

```python
@dataclass
class CompressionPlan:
    stage_name: str
    placement: Literal["input_level", "encoder_level", "projector_level", "llm_level"]
    execution_phase: Literal["prepare", "forward", "generate"]
    modalities: set[str]
    token_roles: set[str]
    method: Literal["token_compression", "operation_compression", "quantization"]
    object_family: Literal["tensor_slot", "module_unit", "weight", "weight_activation"]
    object_subtype: str | None
    token_actions: list[TokenAction]
    legalization_policy: LegalizationPolicy | None
    operation_actions: list[OperationAction]
```

`quantization` 是预留的 method marker，尚无可执行量化方法。格式、校准和 export 规则见
[量化接入设计](quantization_integration.zh.md)，不属于当前稳定接口。

### 4.8 Metadata、Trace 与 Manifest

`StageMetadata` 记录 changed indices/spans、stats 和 artifacts。token trace 至少记录：

- sample id；
- stage 和 placement；
- original/kept token count；
- affected spans；
- changed indices。

成功 Manifest 是本次实验的结果快照。schema v2 为已有调用保留单数
`compressor_name/config` 投影，同时使用 `compression_stages` 保存逐阶段 identity、
SupportUnit evidence 和 layout 摘要，并通过 `per_stage_stats` 保存逐阶段统计。结构上已经
能够记录多 stage；组合是否正确仍需对应交互证据，不能从记录字段存在推导。
共享实验入口还保存 `effective_config.json` 与 `result.json`；失败不会伪造成功 Manifest。

`run_experiment` 把本次执行前解析的 `CompressionDefinition` 传给 Manifest，复用其
有效 Binding 和算法配置；baseline 不记录未执行阶段，完整配方仍保存在配置快照中。
单独调用 `build_run_manifest(config)` 时保留已有的配方投影用途；记录一次实际命令时应传入
其 `compression_definition`，避免把其他任务的启用配方当作本次执行。

## 5. Invariant

### 5.1 通用 invariant

`ForwardStateInvariant` 检查：

- batch 和 sequence carrier 基本一致性；
- label 对齐；
- canonical span 的 sample、顺序、边界和重叠；
- 可用 tensor 的 NaN。

通用 invariant 不应假设所有 backbone 的 `position_ids` 都是普通 `[B, L]`。Qwen M-RoPE 等 layout 必须由 backbone-specific invariant 检查；在该检查实现前，相关路径不能激活。

### 5.2 Operation invariant

通用 invariant 检查 action id、semantic module path、依赖无环、width-wise coupled modules 和 export signature。Qwen applier 还检查真实层计数、config 更新、参数变化与导出记录；这些结构检查仍不能替代重载和模型生成/backward 验证。

### 5.3 错误

| 错误 | 用途 |
|---|---|
| `ConfigError` | 配置或 override 非法 |
| `RegistryError` | capability 未激活或 factory 非法 |
| `InvariantError` | state/plan invariant 失败 |
| `ManifestValidationError` | manifest 不完整或非法 |
| `BackendIntegrationError` | 外部框架失败 |
| `NotActivatedError` | reserved/target 能力尚未实现 |

未实现能力必须失败，不允许静默跳过。

## 6. Backbone 接口

### 6.1 Capability Profile

profile 至少声明：

- name、family、code origin、source ref、loader strategy；
- image/video/audio 支持；
- output modality；
- position system；
- processor output keys 和 required state fields；
- span strategy 和 projector boundary；
- analysis/eval runtime batch 限制。

实际可用基座及能力通过 `mllm-comp discover` 查询，验证范围见
[文档索引中的具体能力与验证](../README.md#具体能力与验证)。早期接入里程碑编号不表示运行能力。

### 6.2 Component Protocols

```python
@dataclass
class BackboneRuntime:
    model: Any
    processor: Any
    model_path_or_id: str | None = None
    revision: str | None = None
    device: Any | None = None
    dtype: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedBatch:
    model_inputs: dict[str, Any]
    sample_ids: list[str | int]
    prompt_lengths: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

class ProcessorAdapter(Protocol):
    def capabilities(self) -> BackboneCapabilities: ...
    def load_processor(self, model_path_or_id: str, *, revision=None, **kwargs) -> Any: ...
    def prepare_inputs(self, processor, raw_sample, *, phase: str, **kwargs) -> ProcessedBatch: ...
    def detokenize(self, processor, token_ids, **kwargs) -> list[str]: ...

class ModelAdapter(Protocol):
    def capabilities(self) -> BackboneCapabilities: ...
    def load_model(self, model_path_or_id: str, *, train: bool, revision=None, **kwargs) -> Any: ...
    def structural_handles(self, model: Any) -> Mapping[str, Any]: ...
    def external_code_info(self) -> dict[str, Any]: ...

class StateBuilder(Protocol):
    def build(self, processed_batch: ProcessedBatch, *, raw_batch=None, phase="eval") -> ForwardState: ...

class NativeValidator(Protocol):
    def validate_state(self, runtime, state, processed_batch) -> list[InvariantViolation]: ...
    def validate_generation(self, runtime, state, processed_batch, result) -> list[InvariantViolation]: ...

class Legalizer(Protocol):
    def apply(self, state: ForwardState, *, token_actions, policy, stage_name) -> ForwardState: ...

class Runner(Protocol):
    def forward_train(self, runtime: BackboneRuntime, state: ForwardState) -> Any: ...
    def generate(self, runtime, state, *, processed_batch, trace_request=None, **kwargs) -> GenerationResult: ...

class OperationApplier(Protocol):
    def validate(self, *, model, structural_handles, actions) -> list[InvariantViolation]: ...
    def apply(self, *, model, structural_handles, actions, stage_name, dry_run=False) -> OperationApplyResult: ...
```

`forward_train` 是已有的直接前向接口，Analysis 也使用它；方法名本身不表示进行了训练或生成
prefill。是否提供 labels、是否 backward 和是否携带 KV Cache 由具体调用决定。
Qwen2.5-VL/Qwen3-VL 的视频 teacher forcing 在 labels 存在且未设置 `use_cache` 时关闭缓存，
仍拒绝绕过 generate 生命周期的显式 cached direct forward。

### 6.3 Composed Facade

```python
@dataclass(frozen=True)
class BackboneFacade:
    processor_adapter: ProcessorAdapter
    model_adapter: ModelAdapter
    state_builder: StateBuilder
    legalizer: Legalizer
    runner: Runner
    operation_applier: OperationApplier | None = None
    native_validator: NativeValidator | None = None
```

构造时 processor/model capability profile 必须相同。registry 类型：

```python
BackboneFactory = Callable[[], BackboneFacade]
ACTIVE_BACKBONES: dict[str, BackboneFactory]
```

配置只通过 `backbone.name` 选择内置能力，例如：

```yaml
backbone:
  name: qwen25_omni
  model_id: ${QWEN25_OMNI_MODEL_ID}
  model_revision: ${QWEN25_OMNI_REVISION}
```

factory、profile、状态和激活阶段由 `mllm_comp.registry.catalog` 的同一条 capability 记录提供，不得复制到 YAML。模型特有且随运行变化的 loader 参数放入 `backbone.config`。

Factory 的支持状态由 catalog 投影；真实模型、输入输出与外部框架证据见对应基座状态页。

### 6.4 Structural Handles

后续 operation compression 使用稳定语义路径：

```text
llm.layers
llm.layer[i]
llm.layer[i].attn
llm.layer[i].ffn
vision_encoder
audio_encoder
projector
```

plugin 不得通过猜测源码属性名定位模块。

## 7. Compressor、Analysis 与 Pipeline

### 7.1 Compressor

```python
class CompressorPlugin(Protocol):
    name: str
    version: str | None
    def validate_config(self, config: dict[str, Any]) -> None: ...
    def plan(self, *, state, context, config) -> CompressionPlan: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
```

这是参考 Plan 接口：该类 plugin 不修改 state/model，只产生 `CompressionPlan`。
真实运行方法不必先实现这个接口。内置 ImageLLM 与 Qwen2.5-Omni
AudioLLM 方法由公开的 AlgorithmDescriptor、ModalityVariantDescriptor、
BackbonePlacementBinding 声明，exact SupportUnit 单独记录证据；共享 session、lmms-eval 与
ms-swift 不按论文方法名分支。同一执行族、同一 target site 的新方法应实现算法对象并
注册 v2 四层合同，不应修改共享执行链。

`CompressionMethodDescriptor` 只保留给显式注入的非 Qwen 仓库外 compressor；内置
descriptor registry 当前为空。它不是 Qwen 生产路径、fallback 或新内置方法的扩展 API。

Binding 的 `strategy_factory` 与 `executor_family` 是显式执行字段；
`exclusive_runtime_slot`、`prepare_after_slots` 和 `canonical_executor_family` 描述执行槽、
依赖与位置适配。metadata 只保留附加描述。研究模块通过 `pipeline.extensions` 中的
`package.module:register` 向本次独立 registry 注册，不修改全局 catalog。

当前方法、位置和证据使用 `discover` 查询；新增方法步骤见
[研究者指南](../guides/2_dev_compression_method.zh.md)。

### 7.2 Analysis

`AnalysisSample` required keys：

- `sample_id`
- `task_name`
- `processor_outputs`

可选 raw doc、media 和 metadata。analysis plugin 消费由 lmms-eval wrapper 构造的 sample，不维护第二套数据循环。

analysis runtime 只从 active registry 解析内置 Probe，并让所有 Probe 共享
lmms-eval 数据循环、Backbone semantic binding、typed artifact envelope、
`AnalysisRecord` 和 manifest。内置能力、支持基座和验证范围见
[多基座分析运行指南](../guides/1_run_analysis.zh.md)；catalog 是能力状态的唯一
事实来源。

### 7.3 Pipeline 与 runtime activation

通用 token pipeline 必须：

- 按 stage 顺序处理；
- 从 registry 解析 active plugin；
- 明确拒绝 operation action，避免在 forward state 路径误做结构改写；
- mutation 前后运行 invariant；
- 生成 `StageMetadata` 并更新自己所执行 stage 的 history；trace、训练 evidence 和 Manifest 由调用该执行结果的 runtime/integration 生命周期收集。

`CompressionPipeline.apply` 已实现上述 token plan、legalization、通用 invariant、metadata
和 stage history 的 reference 执行，保留现有模块导入与调用合同。真实 eval/train/analyze
通过 `activate_runtime_compression()` 与 `RuntimeCompressionDispatcher` 投影
Algorithm/Variant/Binding/SupportUnit，由 `CompressionSession` 管理依赖、安装、回滚
和释放；Qwen executor family 在 Binding provider 内执行。

Qwen LLM Operation 不进入通用 token pipeline，而是在同一 v2 session 的 Binding
activation 中于 post-load、pre-runner、pre-optimizer 生命周期执行。

视频 executor 通过 `backbones/_video_bridge.py` 统一选择 native bridge、所有权和
projector callback site；`backbones/_provider_patching.py` 提供保持签名的可恢复方法替换。
各 executor 仍拥有算法回调及其安装、回滚和卸载顺序。Qwen2.5-Omni 两条音频压缩路径
共享原生 RoPE 长度参数转换，请求状态及是否启用转换仍由各自 controller 管理。

## 8. Config 规则

顶层可包含以下 section，不再使用与目录语义重复的模板布尔标记；当前任务无关的
section 可以省略，由 loader 使用 typed 默认值补齐：

```text
experiment
backbone
pipeline
analysis
train
eval
data
output
```

共享配置不得写本机绝对路径。环境变量展开后的路径进入 effective config 和 manifest。

stage 公共字段：

| 字段 | 状态 |
|---|---|
| `name`, `enabled`, `placement`, `execution_phase`, `plugin` | stable |
| `method`, `object_family`, `object_subtype`, `modalities`, `token_roles` | stable |
| `operation`, `criterion`, `schedule`, `config` | token/operation 公共入口 |
| `algorithm`, `variant`, `binding` | v2 方法与执行身份 |
| `depends_on` | 每个 Stage 独立通过机制、phase、依赖和冲突校验 |
| `activation_scope` | 按方法 schema 指定作用层范围 |
| `quantization`, `calibration` | 已解析的预留字段；不表示量化执行机制已实现 |

loader 必须把每个 `pipeline.stages` mapping 立即规范化为 `CompressionStageConfig`。真实命令、definition validator 和 runtime activation 只消费这一种 typed 表示；未知的顶层、section 或 stage 字段必须在加载时失败。

配置示例：

- baseline analysis：`examples/analysis/qwen25_vl_analysis_matrix.yaml`
- compression definition：`examples/compression/qwen25_omni_audio_segmentwise_topk.yaml`

完整可运行配置位于 `configs/eval/lmms_eval/`、`configs/train/qwen2_5_omni/` 和
`configs/train/imagellm/`；配置生成脚本从 registry 选择已声明方法。接口声明不能推导正式支持，
单 Stage 证据不能推导任意组合。动态 Token 训练使用 `execution_phase: forward`，
结构剪枝使用 `execution_phase: prepare`；共享 session 按实际依赖安装。
`train.backend=ms_swift` 本身不代表未知 plugin 或 Talker 训练可用。

## 9. Registry 与激活

- `catalog` 投影内置 capability provider、profile 和生命周期状态；`ACTIVE_*` 名称由其生成。
- 多模态算法、变体和基座执行由 compression registry 声明；验证结论由精确 SupportUnit 记录。
- 显式研究扩展克隆内置 registry 后注册，definition、runtime、原生 worker 和 Manifest 使用相同身份。
- 尚无证据的 Binding 可用于满足实际契约的研究运行；未实现的执行器与模型机制不能靠状态标签补齐。
- backbone registry value 是 facade factory；返回类型或 profile 不一致时失败，错误包含 capability 名称。

查看实际能力而非维护第二份列表：

```bash
mllm-comp discover --kind backbone
mllm-comp discover --kind analysis
mllm-comp discover --kind integration
mllm-comp discover --backbone qwen25_omni --modality audio
```

`support` 中的单步训练证据不能代替 cached generate、恢复继续训练或最终任务质量。
输入模态能力也不能推导该模态已有压缩机制。
`discover` 的 `items` 表示精确 v2 Binding；只有 catalog 声明的旧版能力单列为
`catalog_only`，不伪造相同的 phase、模态或支持证据。

## 10. 最小示例

### 10.1 Projector Token 压缩

```text
audio ForwardState
  -> Segmentwise plan fixes each audio N_i -> K_i budget
  -> Qwen legalizer compacts placeholder/mask/labels/M-RoPE state
  -> Projector controller selects real encoder outputs
  -> generate or training forward/backward
  -> report and evidence record realized counts
```

完整参数以 `configs/` 中的 Segmentwise YAML 为准。

### 10.2 LLM Operation

```python
OperationAction(
    action_id="prune_llm_layer_10",
    op="prune",
    target_kind="layer",
    target_scope="llm",
    module_path="llm.layer[10]",
    apply_mode="structural_rewrite",
    shape_effect="remove_module",
    original_shape=(36,),
    pruned_shape=(35,),
    target_indices=[10],
    requires_checkpoint_export=True,
)
```

该 action 由 Qwen-specific applier 在 prepare 阶段执行；完整导出仍必须经过 checkpoint
hash Manifest 与 reload 验证。

### 10.3 Quantization（reserved）

```yaml
method: quantization
object_family: weight_activation
```

这里只保留方法族入口。format、scheme、calibration 和 export 字段在首个真实量化实现排期时确定。

## 运行诊断与源码身份

视频 executor 的 `runtime_metadata()` 仅返回最近一次视频请求的详细记录
（`records_scope=latest_video_request`），调用计数继续累计。诊断脚本应在请求后
读取当前列表，使用累计调用计数验证是否执行；不能再按上一请求的列表长度切片。
这些诊断列表不承担梯度、重放或 cache 修复的计算状态。

lmms-eval 每个请求、三个图像基座 runner 的每次 forward/generate，以及 ms-swift
图像训练的外层 forward 开始时清理 activation 的上一请求诊断记录。分析 Probe
调用同一 runner，因此多样本分析同样有界；image 多层及 generate 内多次 decode 的
记录在同一请求内保留，生命周期、累计计数和计算状态不清理。压缩评测产生每 worker
一份 schema 2 的 compression runtime JSON 摘要及 `.requests.jsonl` 详情。
详情逐请求追加，摘要包含累计请求数、当前 activation 状态、记录数量、最近请求
和 trace 路径；完整 token 映射在 JSONL 中。每次完成写入后关闭详情文件并原子刷新
摘要，因此后续模型请求失败时，已完成请求的记录仍可读取。新运行重新初始化这两份产物。
顶层 Manifest 的 `compression_runtime_N` 与 `compression_requests_N` 分别引用它们。
效率汇总继续读取 `lmms_eval_diagnostic_N`，不把压缩详情或 pretty 文本作为效率输入。

新 Manifest 的 `source_scope=runtime_sources_v1`：`source_tree_hash` 只识别正在执行的
平台 checkout 中的 `mllm_comp`、脚本、补丁、包配置，以及本次实际 backend 使用的 vendored
运行包（LLaVA 的 `llava`、lmms-eval 的 `lmms_eval`、ms-swift 的 `swift`）。
只读取源码/资源格式（Python、JSON、YAML、TOML、文本、shell、Jinja 与补丁），
不扫描测试、文档、任务日志、归档、checkpoint 或输出内容；未初始化子模块记录其固定 Git 引用。
有效实验配置仍单独写入，源码身份从执行包位置解析，不从配置文件位置推断。
`source_commit` 保留平台提交引用，`source_dirty` 表示上述源码范围存在未提交变化。
analyze 根据执行结果中的 backend 识别运行包，也覆盖默认 lmms-eval 的 analysis-only
配置。源码删除在暂存前后使用相同摘要，两种状态均标记为 dirty；提交删除仅更新提交引用和 dirty。
安装目录没有本项目 Git 元数据时，三项源码身份和 scope 均为 null，不借用其他仓库的身份。

音频对比在新 scope 下比较源码摘要及原有 checkpoint、任务、版本和生成协议，
不因仅文档提交或源码暂存状态差异拒绝同代码实验。旧 Manifest 按原来的严格规则互比；
新旧来源范围不同，不能混作同一源码协议。
