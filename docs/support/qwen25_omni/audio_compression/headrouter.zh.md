# HeadRouter：LLM 内部动态 Head-Weight 音频 Token 裁剪

> 生命周期：`target`。当前实现与 eval/train 候选配置可供显式验证，但尚未进入正式
> active registry；激活缺口见[六个候选方法状态](candidate_methods.zh.md)。

平台插件名：`headrouter_dynamic_audio_pruning`。论文为
*HeadRouter: Dynamic Head-Weight Routing for Task-Adaptive Audio Token Pruning in Large Audio Language Models*
（arXiv:2604.23717v1），官方实现位于同级工程 `code/HeadRouter/`。

## 平台映射

| 论文/官方步骤 | 平台实现 |
| --- | --- |
| 使用层 `M-1` 的 Q/K，默认 `M=2` | `activation_scope.layers: [1]`，在第 1 层输出后、进入第 2 层前裁剪 |
| 初始 `inputs_embeds` + 层 `M-1` Q/K 的无 RoPE text-to-audio probe | Controller 仅在当前 Prefill 保存初始 embeddings，`HeadRouterLLMStrategy.score_audio_tokens()` 显式忽略 position embedding |
| 逐 head 熵选择性与标准差路由信号 | `compute_selectivity()` 与 `gaussian_route()` |
| semantic / uniform / acoustic 三 profile Gaussian soft mixing | 内置官方 Qwen2.5-Omni-3B layer-1 profile、中心 `0.030/0.036/0.042`、`sigma=0.006` |
| Head-weighted Token importance | `weighted_token_importance()` |
| `floor(N * (1-r))` Top-K，并恢复时间顺序 | `headrouter_output_length()` 与 `select_indices()` |
| 只在 Prefill 选择一次 | 复用 Qwen2.5-Omni `llm_token` controller；Decode 旁路 |

Controller 继续统一维护压缩后的 causal mask、绝对 M-RoPE 位置、逐层 KV 物理布局、训练 labels 与 gradient-checkpoint replay。HeadRouter 策略不复制或替换官方整份 Transformers 模型。

评估配置：`configs/eval/lmms_eval/qwen25_omni_aud_headrouter.yaml`。默认保留 40% 音频 Token，对应官方 quick-start 的 `pruning_ratio=0.6`。

```bash
MODEL_PATH=/path/to/Qwen2.5-Omni-3B \
bash scripts/eval/lmms_eval/qwen25_omni_1gpu_aud_headrouter.sh --plan_only
```

## Profile 边界

仓库内置的 profile 只对应官方公开的 Qwen2.5-Omni-3B、layer 1、16 heads。换用其他 head 数、其他 probe layer 或自行校准的 profile 时，必须在 stage 的 `config` 中提供：

```yaml
config:
  profile_dir: /path/to/profiles       # semantic.json / uniform.json / acoustic.json
  routing_config: /path/to/routing.json
```

Profile head 数与模型不一致时运行会明确失败，不会回退为 uniform profile。评估输出的 compression report 会记录每个样本的 `selectivity_spread` 和三路 routing coefficients。

## 与论文描述的边界

论文消融提到 Frame pre-filter；公开 `configure_headrouter()` 实际设置 `frame=False`，并把 `sss_alpha` 设为 `0.0`。本插件按公开代码路径复现 no-RoPE QK probe、Gaussian routing 和直接 Top-K，不额外加入 Frame 预筛选或 score smoothing。

v2 Algorithm/catalog 的 `eval`/`train` 只表示候选入口存在，exact SupportUnit 保持
`declared`。当前没有真实 3B Prefill/Decode、gradient-checkpoint backward、完整 Swift
checkpoint/Manifest 或目标规模 MMAU 证据，因此不能从配置存在或内置 profile 推导为
正式评测/训练支持。
