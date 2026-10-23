# 自定义压缩规则

规则在已有 stage 的实际算法步骤中执行，复用位置、预算和基座执行器。

完整方法、对应基座、规则输入输出及限制见 [各方法规则目录](methods.md)。下面两个入门示例分别展示选择和合并；模块替换示例见本文后半部分。

| 示例方法 | 开放的规则 | 基座与位置 |
| --- | --- | --- |
| FastV | `criterion.scorer`、`operation.selector` | Qwen2.5-VL、Qwen3-VL、LLaVA-OneVision；沿用已支持的 LLM Token 边界 |
| 音频 LTBM | `operation.grouper`、`operation.aggregator` | Qwen2.5-Omni；沿用 Projector、Encoder、LLM Token 位置 |

各方法只开放自己的实际步骤。结构删层使用决策函数，学习型变换复用已注册模块，Qwen3-VL 模块替换使用 `factory`；无需每个方法都具备评分、选择、分组、聚合四个接口。

## 配置与运行

- [fastv_feature_norm.yaml](fastv_feature_norm.yaml) 在第 2 层 Attention 残差后、FFN 前，按实际边界的图像特征范数评分，使用原 Top-K 选择和 50% 删除预算。
- [ltbm_weighted_merge.yaml](ltbm_weighted_merge.yaml) 在音频 Encoder 输出后、Projector 前，按邻近位置配对，并将 destination 权重设为 3，保留 50% Token。

准备本地模型和对应任务数据、安装项目及原有 lmms-eval 集成后，可从仓库根运行单样本分析：

```bash
export QWEN25_VL_MODEL_PATH=/absolute/path/to/Qwen2.5-VL
mllm-comp analyze --config examples/compression_rules/fastv_feature_norm.yaml --device cuda:0

export QWEN25_OMNI_MODEL_PATH=/absolute/path/to/Qwen2.5-Omni
mllm-comp analyze --config examples/compression_rules/ltbm_weighted_merge.yaml --device cuda:0
```

示例省略输出字段，每次运行自动生成独立目录。需要指定位置时使用 `--output_dir`；它是精确目录，省略 `--manifest` 时采用该目录下的 `manifest.json`。已写入的目录不能再次使用，仅更改 `--run_id` 不会移动显式目录。这里的任务只演示正式入口，不是质量结论。无需模型下载的小模型验证见 `tests/integration/test_image_fastv_custom_rules_cpu.py` 和 `tests/integration/test_ltbm_custom_rules_cpu.py`。

每个规则描述仅有两个字段：

```yaml
operation:
  rules:
    scorer:
      callable: my_research.rules:score_visual
      params: {power: 2.0}
```

`my_research.rules` 必须在执行进程的 Python 导入路径中；例如把自己的包安装到当前 Dev 环境，或设置 `PYTHONPATH`。规则是普通 Python 函数，调用形式为 `function(request, **params)`。`params` 只接受可记录的有限 JSON 值。读取配置、校验能力、生成 Manifest 不导入用户代码；构造执行器时导入并检查函数签名一次。函数输出的形状和布局约束在实际调用时验证。

推荐在 `operation.rules` 中按方法支持的规则名配置。旧 `criterion.scorer` 和 `operation.selector/grouper/aggregator` 继续有效，并归一到同一执行路径；同一规则不能在新旧位置重复配置。复合方法使用明确的步骤名称，不能将规则放进 `operation.targets` 的位置配置。

未指定的规则继续使用原实现，预算覆盖优先级也不变。可以只替换 scorer 或只替换 aggregator。若要单独研究选择策略，可配置：

```yaml
operation:
  rules:
    selector:
      callable: mllm_comp.compressors.rule_examples:fastv_uniform_select
```

不要把规则描述放进 `stage.config` 或 binding config；它们仍只接受各方法原有数值配置。

## 函数输入与返回

request 是只读约定的 mapping；不得原位修改输入或模型，也不应依赖每次调用变化的外部状态。

**FastV scorer** 接收：

- `hidden_states: [sequence, width]`、`image_indices: [visual_count]`：实际应用边界的完整序列和图像 Token 索引。
- `scorer_query`、`attention_keys`、`attention_key_mask`：原 `prune_layer` 规则捕获的来源上下文；它们可能早于实际应用边界，设备也可能不同。`prune_layer=0` 时 Q/K 为 `None`。
- `keep_count`、`protected_image_positions`：最终图像保留预算及必须保留的 visual-local 索引。
- `prune_layer`、`score_source`、`source_layout`、`result_layout`：来源设置及已有布局描述。

返回有限浮点 Tensor `[visual_count]`。平台对齐到实际值的设备。缺省 selector 按分数稳定降序选取，并优先保留保护项。示例实现见 [`fastv_feature_norm`](../../mllm_comp/compressors/rule_examples.py)。现阶段自定义评分仍复用 FastV 原有捕获路径；这不是省略原 Q/K 捕获的效率实验。

**FastV selector** 接收同一 request，并增加 `scores`。返回恰好 `keep_count` 个无重复的 visual-local 整数索引，必须覆盖保护项且不越界。平台统一恢复源顺序、保留非图像 Token 并构造 `LayoutDelta`。因此 selector 不能直接返回压缩后的值或自行改变预算。

`prune_layer=0` 且未指定 scorer 时，`scores` 为全零；原 seeded-random 排序只供默认选择路径使用。自定义 selector 在该配置下自行决定选择顺序。

**LTBM grouper** 接收当前轮 `values: [tokens, width]`（已 detach）、`source_indices`（偶数位置）、`destination_indices`（奇数位置）、`window_size`、`merge_count`、`round_index`。返回恰好 `merge_count` 个 `(source, destination)` 配对，索引属于当前轮序列。source 不得重复，destination 可以重复；配对必须符合原分区和局部窗口，窗口距离按分区内顺序计算。

**LTBM aggregator** 接收 `values: [group_size, width]`（destination 在第 0 行，保留梯度）、`indices`（同顺序的当前轮索引）、`destination` 和 `round_index`。返回同设备、同 dtype、同宽度的有限浮点向量 `[width]`。未参与合并的 Token 原样保留。多个 source 指向一个 destination 时调用一次聚合；输出仍保留 destination 位置。缺省是原逐轮等权均值，并非最终所有原始成员的均值。

Identity 预算不会调用任何自定义规则。Checkpoint 保存 FastV 的选择或 LTBM 的逐轮配对，重算时仅重放 gather 或相同聚合函数，不重新评分或分组。聚合必须能处理现有参考特征映射的宽度，不能写死模型 hidden size。

LTBM 对参考特征使用已保存配对，再独立执行同一聚合函数。若函数的权重依赖输入值或采用非线性运算，这不等于复用实际 hidden states 上计算的聚合系数；需要固定系数映射的研究方法应明确保存并重放这些系数，不能仅换一个函数就假定两者等价。

## 梯度与实验记录

Token 规则支持 stateless 函数，FastV 保留被选值的梯度，LTBM 聚合在重放中保留值梯度。硬选择与配对本身不求导；普通 scorer/aggregator 函数不注册可训练参数。学习型变换使用既有方法的模块生命周期；下述模块 `factory` 的返回值则通过原结构替换过程注册到模型。

实际规则的导入路径和参数写入 runtime metadata 与 Manifest 的 `rules`。改变规则会标记 `rules_changed`、`development_runtime`，并将原配方证据标为 `canonical_evidence_applicable: false`，保留历史引用。位置覆盖单独记录。即使自定义函数恰好产生相同结果，也不自动继承原方法的 Active、质量或训练结论。复现时还需保存自己的规则源代码及所用版本；框架不为外部 Python 包生成额外代码快照。

## Qwen3-VL 模块替换

`qwen3_vl_attention_replacement` 的六种 Attention 策略和 `qwen3_vl_ffn_replacement` 的三种 FFN 策略均支持 `operation.rules.factory`。原 `config.strategy`、`schedule.targets`、prepare 阶段及默认构造行为保持不变。示例 stage：

```yaml
name: custom_ffn
algorithm: qwen3_vl_ffn_replacement
placement: llm_level
execution_phase: prepare
method: operation_compression
object_family: module_unit
operation:
  type: replace
  rules:
    factory:
      callable: my_research.rules:make_ffn
      params: {width: 128}
schedule:
  targets: [{site: 'llm.layer[2].ffn'}]
config:
  strategy: ffn_two_layer_mlp
```

factory 接收 `original_module`、`strategy`、`target_kind`、`module_path`、`config`（原 action 参数）和无参数 `default_factory`。每个目标模块调用一次；可调用 `default_factory()` 获得原策略模块，也可根据原模块构造研究模块。返回值必须是新的 `torch.nn.Module`，不能直接返回 `original_module`，以免替换标记写入待恢复的原对象。

这两个 Operation 方法与 Token 方法使用同一注册、定义和 Session。旧配方与运行产物中
的 `plugin: qwen3_vl_*_replacement` 在读取时转换为对应 `algorithm`；新配方使用上面的写法。

```python
def make_ffn(request, *, width):
    from torch import nn
    original = request["original_module"]
    weight = original.gate_proj.weight
    hidden_size = weight.shape[1]
    return nn.Sequential(
        nn.Linear(hidden_size, width), nn.SiLU(), nn.Linear(width, hidden_size)
    ).to(device=weight.device, dtype=weight.dtype)
```

FFN 模块遵循原 `[batch, sequence, hidden_size] -> 同形状 Tensor` 合同；Attention 模块须保持原 forward 参数、返回 tuple、mask 和 cache 协议。factory 自行将新参数放在正确设备和 dtype；运行时继承原模块的 train/eval 状态。返回模块通过原 parent attribute 安装，因此参数进入模型的 `parameters()`、优化器和 `state_dict()`，卸载时恢复原模块。沿用 `reapply_config_to_original_checkpoint` 重载约定，需保存 factory 源码与配置。

函数不能在外部偷偷注册其他模块或修改原模型。重新应用同一策略和 factory 配置可复用已安装模块；已有替换上换另一 factory 会明确拒绝，应先结束原 compression session。自定义模块是否保持原 Attention/cache 合同，需用相应基座执行测试验证。
