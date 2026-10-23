# 开发和接入 Backbone

> 类型：Guide
>
> 文档层级：2 / 压缩研究人员
>
> 目标结果：把一个新多模态模型接入统一 runtime，并在声明范围内完成原生、外部框架和压缩验证
>
> 适用范围：新增 backbone 或为已有 backbone 增加新的原生模态能力
>
> 当前状态：Active

Backbone 接入不是编写一个 Transformers wrapper。平台要求 processor、model、state、
validator、legalizer、runner 和 structural handles 形成可组合的 `BackboneFacade`，再由
lmms-eval、ms-swift、analysis 和 compression binding 使用同一模型语义。稳定协议见
[接口规范](../platform/interface_spec.zh.md#6-backbone-接口)。

## 1. 扩展目标与准备

先固定精确范围：

- 模型系列与固定 revision；
- text/image/audio/video 输入；
- native forward、generate 与 training forward；
- lmms-eval、ms-swift 或 analysis 是否在本轮接入；
- 需要暴露的 encoder、projector、decoder、attention、FFN 等语义 handle；
- 当前不声明的模态、phase 和外部框架。

参考实现按相似结构选择：

| 特征 | 参考目录 |
| --- | --- |
| AnyRes、外部 LLaVA source | `mllm_comp/backbones/llava_onevision/` |
| Qwen2.5 视觉布局和 M-RoPE | `mllm_comp/backbones/qwen25_vl/` |
| DeepStack、operation replacement | `mllm_comp/backbones/qwen3_vl/` |
| image/audio/video、Thinker/Talker | `mllm_comp/backbones/qwen25_omni/` |

不要复制另一个 backbone 的类再用大量条件分支修正结构差异。共享算法放在 compressor
或公共组件，模型物理差异留在当前 backbone。

## 2. 最小接入

建议先形成下面的文件闭环：

```text
mllm_comp/backbones/<name>/
├── __init__.py
├── backbone.py
├── implementation.py
├── validation.py
└── README.md

tests/unit/test_<name>_backbone.py
tests/integration/test_<name>_gpu_smoke.py
```

最小版本只声明一个真实支持的模态和一条 native 路径，完成：

1. 在 `profiles.py` 和 catalog 中声明能力；
2. 加载 processor/model 并构造 `BackboneRuntime`；
3. processor 生成 `ProcessedBatch`；
4. state builder 生成带精确 token spans 的 `ForwardState`；
5. validator 拒绝不满足模型约束的输入；
6. runner 完成一次真实 checkpoint forward 或 generate；
7. session 关闭后不残留 hook 或模型补丁。

最小 native 闭环通过前，不要同时接入 compression、analysis 和两个外部框架。

## 3. 完整实现流程

### 组合 Facade

`BackboneFacade` 组合而不是继承所有行为。各组件职责为：

- processor：原始多模态输入到模型 batch；
- model：加载权重和提供模型对象；
- state builder：从 batch 和模型语义构造 `ForwardState`；
- validator：检查输入、状态、生成和模型特定 invariant；
- legalizer：压缩后修复 mask、position、grid、placeholder、cache 等状态；
- runner：统一 native forward/generate/train 边界；
- structural handles：向 operation、analysis 和 binding 暴露稳定语义位置。

不要让 integration 或 compressor 重新解析 provider 私有 tensor 来补 state builder 的
缺口；也不要让 legalizer 执行论文算法。

### 注册和加载

在 Catalog 中登记 Backbone factory、profile 和实际支持范围。尚待实现或验证的
Backbone 使用 `target`，后续方向使用 `future`；完成声明范围的验证后设为 `active`。
`registry/backbones.py` 只把 `active` provider 纳入运行注册表，其他状态不能通过
`get_backbone()` 加载。已激活基座仍须校验具体输入、模态和模型合同，不能回退到近似模型。

这是 Backbone 的注册门槛。压缩方法的 `target` 若已有可用 Binding，可在满足实际执行
合同后用于研究运行；其支持证据由独立的 SupportUnit 记录。

配置示例应包含固定 model ID/revision，并允许通过 `--checkpoint` 或 wrapper 环境变量
覆盖本地路径。服务器绝对路径不写入长期 YAML。

### 外部框架

lmms-eval 接入复用平台 plugin 和统一 model bridge；新 backbone 需要映射到正式
`ModelManifest`，验证 chat/simple 请求实际经过平台 Processor、StateBuilder 和 Runner。

ms-swift 接入需要明确 model type、template、target modules、dataset 语义和训练边界。
“能生成 argv”只证明计划可构造，至少要执行一个真实 batch 才能声明 integration 通过。

### Analysis 与 compression

Analysis 通过 `analysis/bindings.py` 解析 decoder layer、attention、FFN、norm、lm head、
input embeddings 和特殊旁路等语义。新增 backbone 不应在每个 Probe 中增加模型名分支。

Compression 复用已有 Algorithm/Variant，只为新模型增加
`BackbonePlacementBinding`、executor/provider、legalizer 和精确 SupportUnit。binding
不得复制共享 kernel。多个同步分支使用 compound binding 和同一个布局决定。

## 4. 测试、激活与边界

推荐证据顺序：

1. config/profile/registry contract；
2. processor 与 state 的确定性单测；
3. validator/legalizer invariant；
4. provider/executor CPU contract；
5. native 真实 checkpoint 最小运行；
6. lmms-eval 单样本；
7. 声明训练时的 ms-swift 一步训练；
8. analysis semantic binding；
9. 每个 compression SupportUnit 的独立证据。

只有完成本轮声明范围的真实 checkpoint 证据后，才把对应 catalog/SupportUnit 激活。
一个模态或 phase 通过不产生隐式支持矩阵。验证方法见
[能力验证](3_verify_capabilities.zh.md)，运行失败见
[运行时排错](3_troubleshoot_runtime.zh.md)。
