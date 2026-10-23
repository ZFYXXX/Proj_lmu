# FastAdaSP：LLM 内部加权音频 Token 合并

> 生命周期：`target`。当前实现与 eval/train 候选配置可供显式验证，但尚未进入正式
> active registry；激活缺口见[六个候选方法状态](candidate_methods.zh.md)。

平台插件名：`fastadasp_weighted_token_merging`。依据 EMNLP 2024 Industry Track 论文
*FastAdaSP: Multitask-Adapted Efficient Inference for Large Speech Language Model*
及官方 `code/FastAdaSP/Qwen-Audio/modeling_qwen.py` 接入。

## 论文与平台映射

| 论文/官方步骤 | 平台实现 |
| --- | --- |
| 在 self-attention residual 后、FFN 前处理音频特征 | 新目标位点 `llm_decoder_layer_post_attention_audio_tokens`；原生 attention/KV cache 执行后先合并，再对短序列运行原生 MLP |
| 相邻 Key 余弦相似度（式 4） | 当前层已应用原生 M-RoPE 的 decoder Key；按音频时间顺序计算相邻相似度 |
| 所有 prompt query、所有 attention head 的累计注意力（式 5） | `full_prompt_received_attention`，显式重算当前层 causal attention，不要求模型返回 attention 矩阵 |
| Top-k 相邻边界组成连续合并簇 | `weighted_adjacent_token_merge()` 稳定选取边界并按 Algorithm 1 合簇 |
| 注意力加权簇平均（式 6） | `attention_weighted_mean`，保留每个簇首 Token 的绝对位置 |
| 稀疏任务单层合并 | `schedule.task_mode: sparse`，`activation_scope.layers` 指定一个零基层 |
| 稠密任务逐层 constant / decay | 从指定层起逐层合并；decay 在最后一层线性降到 0 |
| 只在 Prefill 压缩 | Decode 不重新打分或合并，并按各层入口布局读取原生 `DynamicCache` |

该执行路径不复制官方整份 Qwen-Audio 模型。Qwen2.5-Omni controller 继续统一维护
causal mask、原生 M-RoPE、逐层 KV 物理长度和多音频 span；FastAdaSP 策略只负责论文算法与调度。

## 运行

默认配置是 50% 稀疏任务合并，操作层为零基第 3 层：

```bash
MODEL_PATH=/path/to/Qwen2.5-Omni-3B \
bash scripts/eval/lmms_eval/qwen25_omni_1gpu_aud_fastadasp.sh --plan_only
```

稠密任务可把配置改为：

```yaml
schedule:
  task_mode: constant  # 或 decay
  merge_ratio: 0.1
  minimum_keep_tokens: 1
activation_scope:
  layers: [3]          # constant/decay 的起始层，零基
  lifecycle: prefill_only
```

`merge_ratio` 与官方代码一致，是每个激活层对“当前音频长度”的合并比例；constant/decay
会逐层作用，因此最终保留率不是简单的 `1 - merge_ratio`。

## 已知验证边界

- 论文的 Transfer Entropy 用于离线选择稀疏任务操作层。官方仓库未发布评测流水线或自动 TE 搜索实现；平台要求把离线选出的层显式写入 `activation_scope.layers`，不会伪造自动选层结果。
- v2 Algorithm/catalog 暴露 `eval` 和 `train` 候选接口，exact SupportUnit 保持
  `declared`；这不是正式训练支持声明。论文与官方代码均为 training-free inference，
  train 配置表示在固定 FastAdaSP 合并规则下训练 Qwen Thinker 的候选实验。真实 3B
  代表场景已有压缩、有限非零输入梯度和生成不重复选择的记录，具体范围见
  [候选方法状态](candidate_methods.zh.md#已验证的-fastadasp-代表场景)。其余配置的
  gradient-checkpoint backward、完整训练保存恢复及收敛仍需分别验证。
- 配置可直接用于真实 checkpoint smoke，但方法质量、吞吐和显存结论必须在目标任务与目标硬件上单独测量；代码接入本身不等于复现论文表格数值。
