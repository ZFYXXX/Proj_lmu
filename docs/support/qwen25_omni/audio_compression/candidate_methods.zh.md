# Qwen2.5-Omni 六个音频压缩候选方法

> 生命周期：`target`
> 最后更新：2026-09-16

本页记录六个已有 Algorithm/Audio Variant/Qwen Binding/declared SupportUnit、eval/train
配置和薄脚本的候选方法。满足实际机制合同即可研究运行，唯一身份可由配置加载器补全，
展开后的 `algorithm/variant/binding` 进入生效配置和运行记录；它们不会进入
`ACTIVE_COMPRESSORS`，也不能从配置或脚本存在推导为正式支持。

| 方法 | Plugin | 位置 | 当前已有 | 激活前仍需 |
| --- | --- | --- | --- | --- |
| FastSLM HTA | `fastslm_hta_audio` | Projector Token | 数值与原生小模型测试、两个 projector 位置的权重恢复、下述真实 3B 联合训练闭环 | 其他声明场景的目标规模证据；MMAU/效率另行验证 |
| Uniform Average Pooling（UniAvg） | `uniform_average_pooling_audio` | Projector Token | 算法、数值与原生小模型测试、v2 合同、eval/train 配置 | 真实 3B generate/backward；MMAU/效率另行验证 |
| Uniform Sampling（UniSamp） | `uniform_sampling_audio` | Projector Token | 算法、数值与原生小模型测试、v2 合同、eval/train 配置 | 真实 3B generate/backward；MMAU/效率另行验证 |
| Unsupervised Segmentation（UnSeg） | `unsupervised_segmentation_audio` | Projector Token | 算法、自然峰值预算、数值与原生小模型重放测试、v2 合同、eval/train 配置 | 真实 3B generate/backward；MMAU/效率另行验证 |
| FastAdaSP | `fastadasp_weighted_token_merging` | LLM Token | 论文算法映射、v2 四层合同、eval/train 配置，以及下述真实 3B 压缩、输入梯度与生成代表验证 | 其余声明场景的 Prefill/Decode/cache/M-RoPE、checkpointing backward 与保存恢复证据；MMAU/效率另行验证 |
| HeadRouter | `headrouter_dynamic_audio_pruning` | LLM Token | 官方 3B profile 映射、v2 四层合同、eval/train 配置 | 真实 3B profile 路径、Prefill/Decode/cache/M-RoPE、checkpointing backward、MMAU/效率 |

六个 v2 Algorithm/catalog 条目都声明 `eval` 和 `train` 候选接口，exact SupportUnit
保持 `declared`。这里的 `train` 表示配置可进入
压缩训练链路继续验证，不等于论文提出了训练算法。已有 backward/checkpoint 证据仍须按
具体场景解读，不能推导收敛或训练质量。FastAdaSP、HeadRouter、UniAvg、UniSamp 和 UnSeg 的压缩
规则本身仍是 training-free；候选训练配置是在固定压缩规则下训练 Qwen Thinker。

## 入口与权重

eval 配置位于 `configs/eval/lmms_eval/qwen25_omni_aud_<method>.yaml`，train 配置位于
`configs/train/qwen2_5_omni/train_aud_sft_<method>.yaml`，对应脚本使用相同方法后缀。

FastSLM HTA 与其余五种方法不同：它有独立可学习权重。评测时必须分别提供 Qwen
backbone 和 HTA checkpoint：

```bash
MODEL_PATH=/path/to/Qwen2.5-Omni-3B \
FASTSLM_HTA_CHECKPOINT_PATH=/path/to/Stage3_FastSLM.pt \
  scripts/eval/lmms_eval/qwen25_omni_1gpu_aud_fastslm_hta.sh
```

若平台 full-SFT checkpoint 同时包含 Qwen 与已注册 HTA，可把两个变量指向同一目录。
脚本不再猜测某个带时间戳的本地 run 目录。

## 已验证的 HTA 联合训练场景

2026-09-09 使用真实 Qwen2.5-Omni-3B、单卡 BF16/eager 和两条合成音频样本，
通过公共入口执行 Operation 删去一层、在原 pre-projector 位置安装 HTA，并共同训练
HTA 与末层归一化参数。连续两步与从第一步 checkpoint 恢复后的第二步，参数、Adam、
scheduler 和 RNG 均逐项相等；两组参数均发生实际更新。随后公共恢复入口保持 35 层，
加载训练后的 HTA，生成两个 Token，音频 Token 从 25 个压缩至 1 个。

复现脚本为 `scripts/diagnostics/validate_qwen25_omni_swift_resume.py`。这条证据涵盖
所述完整 Swift checkpoint 恢复，不涵盖全参数、LoRA/DeepSpeed、其他组合或质量收益，
也不自动扩展 catalog 和 SupportUnit 的正式声明范围。

## 已验证的 FastAdaSP 代表场景

[项目进度](../../../progress.md)已记录真实 Qwen2.5-Omni-3B 的一秒音频验证：FastAdaSP
将 25 个音频 Token 合并为 13 个，输入梯度有限且非零，生成不重复选择。该记录补充了
代表场景的真实模型证据，不等于全部配置、缓存/位置场景、完整训练保存恢复或质量/效率
已经验证。其 catalog 生命周期仍为 `target`，exact SupportUnit 仍为 `declared`。

## 激活规则

候选方法按明确声明的工程范围逐项晋升，验证以行为及风险为依据：

1. 方法特定的 schema、独立数值参考、exact-length/identity 和错误边界测试；
2. 固定 Qwen2.5-Omni-3B checkpoint 的真实生成，并由 compression report 对账长度；
3. 声明 `train` 时完成对应 gradient-checkpoint backward；FastSLM 还需保存并重载 HTA；
4. 质量、效率或论文复现结论另做对应实验，baseline/compressed 协议一致；它们不替代工程证据，也不是所有研究运行的门禁；
5. 状态页记录证据后，catalog lifecycle 与对应 exact SupportUnit 分别按证据晋级；不能只改其中一层。

候选有效 v2 运行生成的 Manifest 会把
`compression_stages[].support_status` 写为 `declared`，并保留 exact SupportUnit key；
catalog lifecycle 仍为 `target`。这与正式方法的 active SupportUnit 明确区分。
