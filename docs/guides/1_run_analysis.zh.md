# 多基座机制分析：运行与结果解读

> 类型：Guide
>
> 文档层级：1 / 平台使用者
>
> 目标结果：选择并运行已激活的 Probe，读取可追溯的分析产物
>
> 适用范围：LLaVA-OneVision、Qwen2.5-VL、Qwen3-VL 与 Qwen2.5-Omni 当前 active analysis 单元
>
> 当前状态：Active

本文面向准备运行分析实验的人。它主要回答：

1. 研究问题应该选择哪个 Probe；
2. 怎样运行单方法示例或完整矩阵；
3. 结果文件应该怎样阅读；
4. 需要理解实现或继续开发时应该从哪里进入。

第一次使用建议依次阅读第 2～6 节；需要理解实现时可以直接阅读第 7 节。

## 1. `analyze` 的范围

`mllm-comp analyze` 在 lmms-eval 提供的真实样本上观察或干预模型内部状态，并把
结果写成 analysis artifact 和 `AnalysisRecord`。它不负责重新实现 dataset、计算
完整 benchmark 准确率、自动生成剪枝配置、训练模型或覆盖 checkpoint。

多个 Probe 共用同一个模型、lmms-eval 样本迭代器和 `AnalysisSample`。因此完整
矩阵只遍历一次数据集，但干预、分层采集和 Taylor 仍可能执行额外 forward 或
backward。

当前 active 范围为：

| Backbone 参数 | 模型系列 | 最小验证 checkpoint |
| --- | --- | --- |
| `llava_onevision` | LLaVA-OneVision | `lmms-lab/llava-onevision-qwen2-0.5b-ov` |
| `qwen25_vl` | Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `qwen25_omni` | Qwen2.5-Omni（Thinker/LLM 音频 Token） | `Qwen/Qwen2.5-Omni-3B` |
| `qwen3_vl` | Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct` |

实际 model revision 以
[`examples/analysis/methods/`](../../examples/analysis/README.md) 和四份
`*_analysis_matrix.yaml` 为准，避免在指南中维护第二份版本表。Qwen2.5-Omni 的
七个 Probe 只观察 Thinker/LLM 中已经拼入 prompt 的音频 Token，不进入 Audio
Encoder、Talker 或 Token2Wav。HardConcrete Gate、LoRA recovery、压缩训练和速度
优化也不属于 analysis runtime。

## 2. 快速开始

以下命令均从仓库根目录执行。

### 2.1 运行一个轻量 Probe

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-VL-3B-Instruct \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/analysis/run_analysis_method.sh representation_dynamics
```

`MODEL_PATH` 可以省略；省略后使用 YAML 固定的 Hugging Face 模型和 revision，也可
使用 `QWEN25_VL_MODEL_PATH`。单方法示例固定 `sample_limit: 1` 和 `layers: [0]`，
用于先验证配置、模型绑定和产物结构。

默认输出位于：

```text
outputs/analysis/methods/<backbone>/representation_dynamics/<timestamp>/
├── manifest.json
└── analysis/
    └── representation_dynamics/
        └── representation_dynamics.aggregate.json
```

这个最小运行不只以进程退出码为准。还应确认 checkpoint 和一个真实样本确实被加载、
Probe 生成非空 `AnalysisRecord`、Manifest 的 invariant 通过，并且没有 skipped sample。
配置校验或外部框架诊断不能替代这些真实运行证据。

### 2.2 运行完整矩阵

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-VL-3B-Instruct \
bash scripts/analysis/run_analysis_matrix.sh qwen25_vl
```

矩阵脚本允许用 `--probe NAME` 只选择其中的方法；Qwen 音频矩阵使用：

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B \
bash scripts/analysis/run_analysis_matrix.sh qwen25_omni
```

矩阵脚本支持 `llava_onevision`、`qwen25_vl`、`qwen25_omni` 和 `qwen3_vl`。
脚本只负责选择 YAML、设置设备和创建新输出目录；实际执行仍由统一的
`mllm-comp analyze` CLI 完成。

## 3. 如何选择方法

先按研究问题选择一个 Probe，不要一开始就运行全部方法。下表中的成本是相同层数
和样本数下的相对成本。

| 研究问题 | 推荐 Probe | 类型 | 主要额外执行 | 相对成本 |
| --- | --- | --- | --- | --- |
| 各层对目标模态/文本表征改变了多少？ | `representation_dynamics` | 观察 | residual hook forward | 低 |
| 模型从哪一层开始形成最终答案？ | `prediction_dynamics` | 观察 | logit-lens forward | 低 |
| 哪些 Head 直接把注意力路由到目标模态 Token？ | `attention_flow` | 观察 | 分层 Attention forward | 中高 |
| Attention/FFN 向 residual stream 写入了什么？ | `residual_stream_decomposition` | 观察 + 干预 | 观察加 update deletion | 中高 |
| 删除 Layer、Attention、FFN 或模态边会怎样？ | `layer_importance` | 干预 | baseline 加每个 intervention | 高 |
| 目标模态信息何时进入、何时可以停止交互？ | `late_injection_early_exit` | 干预 | baseline 加每个阈值 | 高 |
| Path、Head、Neuron、Sublayer、Layer 谁对损失重要？ | `hierarchical_taylor` | 梯度 + 干预 | forward/backward 加因果对照 | 最高 |

推荐从 `representation_dynamics` 和 `prediction_dynamics` 开始，再用
`attention_flow` 定位候选层或 Head，之后运行干预方法，最后用
`hierarchical_taylor` 做细粒度排名和因果交叉验证。观察指标只能说明“发生了
什么”；Attention mass 或 residual norm 较大并不自动等于因果重要性。

## 4. 配置和 CLI

不要从空文件猜字段。Qwen2.5-VL 与 Qwen2.5-Omni 各有七个单方法配置，统一索引在
[`examples/analysis/methods/`](../../examples/analysis/README.md)；四个 Backbone
的完整矩阵也由该索引说明。运行 Qwen 音频单方法时设置
`ANALYSIS_BACKBONE=qwen25_omni`。

### 4.1 通用 `analysis` 字段

| 字段 | 含义 | 建议起点 |
| --- | --- | --- |
| `tasks` | lmms-eval task 名称 | 先用一个小 task |
| `sample_limit` | 传给 analysis backend 的样本上限 | `1`，确认后再扩大 |
| `runtime_batch_size` | lmms-eval runtime batch | 当前使用 `1` |
| `compression_mode` | 分析原模型或已应用压缩的模型 | baseline 用 `none` |
| `phase` | 写入 `AnalysisRecord` 的阶段 | baseline 用 `pre_compression` |
| `probes` | 同一数据循环中打开的 Probe 列表 | 第一次只启用一个 |

每个 Probe 的专属字段在第 5 节及对应 YAML 中给出。

### 4.2 常用 CLI override

启动脚本会把后续参数原样传给 `mllm-comp analyze`：

```bash
bash scripts/analysis/run_analysis_matrix.sh qwen3_vl \
  --probe prediction_dynamics \
  --layers 0,8,16 \
  --tasks mme \
  --sample_limit 4 \
  --save_raw
```

| 参数 | 作用 |
| --- | --- |
| `--probe NAME` | 从矩阵中选择一个 Probe；可重复传入 |
| `--layers all` 或 `--layers 0,8,16` | 覆盖所选 Probe 的层 |
| `--tasks a,b` | 覆盖 lmms-eval task |
| `--sample_limit N` | 覆盖样本数 |
| `--metrics a,b` | 覆盖所选 Probe 的指标 |
| `--save_raw` | 额外保存逐样本结果 |
| `--checkpoint PATH` | 覆盖模型路径；脚本也支持 `MODEL_PATH` |

## 5. 七个 Probe

### 5.1 `representation_dynamics`

回答每层 Attention 和 FFN 对不同 Token 组的 residual 表征改变了多少。它采集
`before_attn`、`attn_only`、`after_attn`、`ffn_only` 和 `after_ffn`，并计算：

- `attn_res`：Attention 前后 residual；
- `attn_pure`：输入 residual 与纯 Attention update；
- `ffn_res`：FFN 前后 residual；
- `ffn_pure`：Attention 后 residual 与纯 FFN update；
- `full_layer_res`：整层输入与输出。

主要读取 `payload.layers` 和 `payload.task_aggregates`。cosine 越接近 1 表示方向越
相似；较低值表示方向改变更大，但不能单独解释为性能贡献。

- 配置：[`qwen25_vl_representation_dynamics.yaml`](../../examples/analysis/methods/qwen25_vl_representation_dynamics.yaml)
- 脚本：`bash scripts/analysis/run_analysis_method.sh representation_dynamics`
- 产物：`representation_dynamics.aggregate.json`
- 代码：[`representation_dynamics.py`](../../mllm_comp/analysis/probes/representation_dynamics.py)、[`representation.py`](../../mllm_comp/analysis/common/representation.py)

### 5.2 `prediction_dynamics`

回答中间层何时形成与最终层相近的 first-answer-token 分布。所选层 hidden state 会
经过模型原生 final norm 和 LM head，再与最终输出比较；它不是完整自由生成轨迹。

- `js`、`kl_final_to_layer` 越低，层分布越接近最终分布；
- `top1_matches_final` 越接近 1，层 top-1 越稳定；
- `rank_of_final_top1` 越低，最终答案候选越早进入前列。

- 配置：[`qwen25_vl_prediction_dynamics.yaml`](../../examples/analysis/methods/qwen25_vl_prediction_dynamics.yaml)
- 脚本：`bash scripts/analysis/run_analysis_method.sh prediction_dynamics`
- 产物：`prediction_dynamics.aggregate.json`
- 代码：[`prediction_dynamics.py`](../../mllm_comp/analysis/probes/prediction_dynamics.py)、[`prediction.py`](../../mllm_comp/analysis/common/prediction.py)

### 5.3 `attention_flow`

回答每层每个 Head 把多少直接 Attention mass 从一类 query 路由到一类 key。视觉
配置使用 `system`、`text` 和 `vision`，Qwen 音频配置使用 `system`、`text` 和
`audio`；常用指标包括同模态、文本到模态、决策到模态的 mass 和条件熵。这里不做
cross-layer rollout，也不把 Attention 权重直接解释为因果重要性。

主要读取 `payload.layers.<layer>.heads`、层均值和 `mass_conservation`。该方法必须
使用 `attn_implementation: eager`；`layer_batch_size` 越大，forward 次数越少，
但显存峰值越高。

- 配置：[`qwen25_vl_attention_flow.yaml`](../../examples/analysis/methods/qwen25_vl_attention_flow.yaml)
- 脚本：`bash scripts/analysis/run_analysis_method.sh attention_flow`
- 产物：`attention_flow.aggregate.json`
- 代码：[`attention_flow.py`](../../mllm_comp/analysis/probes/attention_flow.py)、[`common/attention_flow.py`](../../mllm_comp/analysis/common/attention_flow.py)

### 5.4 `residual_stream_decomposition`

同时观察 Attention/FFN residual update，并验证删除 update 后答案分布如何变化。
`payload.residual_stats` 报告 norm、ratio 和 cosine；`payload.causal` 报告把所选层
Attention 或 FFN update 置零后的 KL、JS 和 top-1 变化。较大的 `update_norm` 只
表示写入较大，不能替代真实删除结果。

主要字段是 `layers`、`causal_layers`、`causal_operations`、`token_groups`、
`metrics`、`ranking_metric` 和 `collect_residual_stats`。

- 配置：[`qwen25_vl_residual_stream_decomposition.yaml`](../../examples/analysis/methods/qwen25_vl_residual_stream_decomposition.yaml)
- 脚本：`bash scripts/analysis/run_analysis_method.sh residual_stream_decomposition`
- 产物：`residual_stream_decomposition.aggregate.json`
- 代码：[`residual_stream_decomposition.py`](../../mllm_comp/analysis/probes/residual_stream_decomposition.py)、[`residual.py`](../../mllm_comp/analysis/common/residual.py)

### 5.5 `layer_importance`

比较 Layer、Attention、FFN 或指定跨模态 Attention 边被删除后的输出分布变化。
`operations` 支持 `layer`、`layer_strict`、`attn` 和 `ffn`；`attn_edges` 使用
target-to-source 记法，例如 `t2v` 表示文本 query 到视觉 key；Qwen 音频使用
`a2a`、`a2t`、`t2a`、`t2t`。Attention 干预只
删除 causal mask 中原本可见的边，不会暴露 future token；没有安全可删除边时会
标记 unavailable。

主要读取 `payload.layers` 和 `payload.task_rank_aggregation`。KL/JS 越高通常表示
干预影响越大，`changed_prob_on_base_top1` 越低表示影响越大。

- 配置：[`qwen25_vl_layer_importance.yaml`](../../examples/analysis/methods/qwen25_vl_layer_importance.yaml)
- 脚本：`bash scripts/analysis/run_analysis_method.sh layer_importance`
- 产物：`layer_importance.aggregate.json`
- 代码：[`layer_importance.py`](../../mllm_comp/analysis/probes/layer_importance.py)、[`interventions.py`](../../mllm_comp/analysis/common/interventions.py)

### 5.6 `late_injection_early_exit`

回答目标模态—文本交互需要从哪一层开始，以及持续到哪一层之后仍然有用。视觉
Backbone 的 `target_modality` 为 `image`，Qwen 音频为 `audio`：

- `early_exit(k)` 保留第 `0..k` 层跨模态交互，阻断 `k` 之后的双向跨模态边；
- `late_injection(k)` 阻断 `k` 之前的跨模态边，在第 `k` 层重新注入 baseline
  目标模态状态，再恢复后续交互。

每个阈值均与无干预 baseline 比较。主要读取 `payload.early_exit`、
`payload.late_injection` 和 `payload.task_rank_aggregation`；影响较小的阈值不等于
已经得到可泛化的安全剪枝结论。

- 配置：[`qwen25_vl_late_injection_early_exit.yaml`](../../examples/analysis/methods/qwen25_vl_late_injection_early_exit.yaml)
- 脚本：`bash scripts/analysis/run_analysis_method.sh late_injection_early_exit`
- 产物：`late_injection_early_exit.aggregate.json`
- 代码：[`late_injection_early_exit.py`](../../mllm_comp/analysis/probes/late_injection_early_exit.py)、[`interventions.py`](../../mllm_comp/analysis/common/interventions.py)

### 5.7 `hierarchical_taylor`

在同一 teacher-forced loss 下，对 Path、Head、FFN Neuron、Attention/FFN
Sublayer 和 Whole Layer 建立统一的重要性视图。对组件贡献 `z` 和损失梯度 `g`：

```text
normalized_score     = mean(abs(g * z))
direct_removal_score = abs(sum(g * z))
```

`normalized_score` 用于跨样本和任务排名；`direct_removal_score` 近似整体删除组件的
损失变化，二者不能互换。`causal_validation` 会在同粒度内选择组件，并与数量相同、
互斥且由 seed 固定的随机对照比较 teacher-forced native loss delta。

该方法需要参考答案、teacher-forced backward、eager Attention 和语义 projection
handle。主要产物是 `hierarchical_taylor.task_scores.json` 与
`hierarchical_taylor.task_ranks.json`；仅在 `save_raw: true` 时写 raw 层级结构。

- 配置：[`qwen25_vl_hierarchical_taylor.yaml`](../../examples/analysis/methods/qwen25_vl_hierarchical_taylor.yaml)
- 脚本：`bash scripts/analysis/run_analysis_method.sh hierarchical_taylor`
- 代码：[`hierarchical_taylor.py`](../../mllm_comp/analysis/probes/hierarchical_taylor.py)、[`taylor.py`](../../mllm_comp/analysis/common/taylor.py)、[`taylor_validation.py`](../../mllm_comp/analysis/common/taylor_validation.py)

## 6. 怎样阅读输出

除 Taylor 外，每个 Probe 默认写一个 `*.aggregate.json`；`save_raw: true` 时再写
`*.raw.json`。共同 envelope 包含 `schema_version`、`artifact_kind`、`probe`、
`phase`、`backbone`、`metadata` 和 `payload`。

建议按以下顺序检查：

1. 打开 `manifest.json`，确认 `analysis_records` 数量与所选 Probe 数量一致；
2. 检查 `summary.successful_samples`、`summary.skipped_samples` 和
   `artifact_paths`；
3. 在 aggregate 中确认 `metadata`、`payload.binding_evidence`、配置层和 Token 组；
4. 再读取 `payload.layers`、`payload.causal` 或 Taylor score/rank；
5. 跨 task 比较时使用 `task_rank_aggregation`，不要混排不同 task 的原始量纲；
6. 出现跳过样本时先处理原因，不要只解读剩余有限值。

`save_raw: false` 不等于没有证据。aggregate 仍保留样本数、task 聚合、配置、
binding、跳过原因和质量检查；只有逐样本诊断时才应开启 raw，Taylor 的 FFN 向量
尤其可能生成很大的文件。

## 7. 实现导读

### 7.1 运行路径

```text
YAML / CLI -> definition 校验 -> BackboneRuntime -> lmms-eval AnalysisSample
           -> Probe sessions -> artifacts -> AnalysisRecord / manifest
```

lmms-eval 拥有 task 和数据循环；Backbone 通过 semantic binding 暴露 decoder、
Attention、FFN、final norm 和 LM head；Probe session 只实现
`process()`、`finish()` 与 `close()`。详细接口契约见
[`interface_spec.zh.md`](../platform/interface_spec.zh.md)。

### 7.2 推荐阅读顺序

| 顺序 | 文件 | 回答的问题 |
| --- | --- | --- |
| 1 | [`registry/catalog.py`](../../mllm_comp/registry/catalog.py) | 哪些 Probe active，支持哪些 Backbone？ |
| 2 | [`cli/analyze.py`](../../mllm_comp/cli/analyze.py) | CLI override 如何进入配置？ |
| 3 | [`analysis/definition.py`](../../mllm_comp/analysis/definition.py) | CLI override 和配置怎样在模型加载前解析？ |
| 4 | [`analysis/runtime.py`](../../mllm_comp/analysis/runtime.py) | 一个数据循环怎样驱动多个 session？ |
| 5 | [`analysis/bindings.py`](../../mllm_comp/analysis/bindings.py) | Backbone 怎样映射到统一 decoder 语义？ |
| 6 | [`representation_dynamics.py`](../../mllm_comp/analysis/probes/representation_dynamics.py) | 简单 Probe 的完整生命周期是什么？ |
| 7 | [`analysis/artifacts.py`](../../mllm_comp/analysis/artifacts.py) | artifact 和 record 怎样写出？ |

数学计算、hook、干预和聚合位于 `mllm_comp/analysis/common/`。需要开发新 Probe 时
继续阅读[开发 Analysis 方法](2_dev_analysis_method.zh.md)，不要从运行 Guide 推导新的
接口或另建数据集 wrapper。

### 7.3 Backbone 语义来源

如果 decoder 语义与现有模型一致，通常不需要修改七个 Probe，只需提供或确认：

1. 连续的 `llm.layer[i]`、`.attn` 和 `.ffn` structural handles；
2. `llm.final_norm`、`lm_head` 和 Attention head geometry；
3. Token layout 所需的 processor/state metadata；
4. Taylor 使用的 input embedding、`v_proj`、`o_proj` 和 FFN `down_proj`；
5. catalog 支持声明和真实 checkpoint 回归。

DeepStack、AnyRes、GQA/MQA 等模型差异应在 Backbone binding 或共享语义层解决，
不应散落到七个 Probe 中。实际接入步骤见[开发 Backbone](2_dev_backbone.zh.md)。

Qwen2.5-Omni 还会优先读取 runtime 暴露的压缩后有效 Token 布局。Projector/Encoder
压缩使用最终 Thinker 输入布局；LLM Token 压缩按所选 decoder layer 的入口布局
分组，因此 Probe 不会用压缩前的 audio span 去解释压缩后的 hidden state。结构压缩
则使用改写后的连续层索引。

## 8. 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 模型加载前配置校验失败 | 按错误信息检查 Probe 名称、Backbone 支持范围、task、层号和 override。 |
| `does not declare support for backbone` | 当前仅支持表中的四个 Backbone；先完成 semantic binding 和真实验证，不要只改配置名称。 |
| `requires exact eager attention weights` | 为 Attention Flow 或 Taylor 设置 `attn_implementation: eager`；FlashAttention/SDPA 不提供所需的完整精确矩阵。 |
| 输出目录已经存在 | 使用统一脚本生成新目录，或同时指定新的 `--output_dir` 与 `--manifest`。 |
| 显存不足 | 单独运行 Probe、减少 `layers`、保持两个 batch size 为 `1`，并减少视觉 Token；`sample_limit` 主要影响总耗时而非单样本峰值。 |
| 存在 skipped sample | 先读取跳过原因；常见原因是参考答案、Token layout、Attention 权重或语义 handle 缺失。 |
| Taylor 没有可用 loss | task 必须通过 `doc_to_target` 提供参考答案，并使用 `teacher_forced_answer_cross_entropy_v1`。 |

失败重跑不要复用旧 manifest 或非空 `run_dir/analysis`。更换模型权重后也应重新
保存 manifest、binding evidence 和分析结果，不沿用旧验证结论。
