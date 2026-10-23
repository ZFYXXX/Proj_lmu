# Analysis examples

这里有三类配置：

- `methods/`：七个分析方法各自独立、可直接运行的 Qwen2.5-VL 与 Qwen2.5-Omni 音频最小示例；
- `*_analysis_matrix.yaml`：四个 active Backbone 的完整七 Probe 矩阵；
- `compression/`：Qwen2.5-Omni 四类代表压缩后的 Representation Dynamics 示例。

直接用 CLI 或 Python 加载这些 YAML 时，每次实验默认生成
`runs/<experiment.name>/<run_id>/`，不需要手动更改输出字段。下文 wrapper 保留其
时间戳目录约定。公共入口和结构化结果见[运行压缩实验](../../docs/guides/1_run_compression_experiment.zh.md)。

## 单方法示例

每个 YAML 只启用一个 Probe，固定 `sample_limit: 1` 和 `layers: [0]`，用于先验证
配置、模型绑定和产物结构。确认最小运行成功后，再增加任务、样本和层数。

| 方法 | 独立配置 | 真实运行脚本 |
| --- | --- | --- |
| Representation Dynamics | [`qwen25_vl_representation_dynamics.yaml`](methods/qwen25_vl_representation_dynamics.yaml) | `bash scripts/analysis/run_analysis_method.sh representation_dynamics` |
| Prediction Dynamics | [`qwen25_vl_prediction_dynamics.yaml`](methods/qwen25_vl_prediction_dynamics.yaml) | `bash scripts/analysis/run_analysis_method.sh prediction_dynamics` |
| Attention Flow | [`qwen25_vl_attention_flow.yaml`](methods/qwen25_vl_attention_flow.yaml) | `bash scripts/analysis/run_analysis_method.sh attention_flow` |
| Residual Stream Decomposition | [`qwen25_vl_residual_stream_decomposition.yaml`](methods/qwen25_vl_residual_stream_decomposition.yaml) | `bash scripts/analysis/run_analysis_method.sh residual_stream_decomposition` |
| Layer Importance | [`qwen25_vl_layer_importance.yaml`](methods/qwen25_vl_layer_importance.yaml) | `bash scripts/analysis/run_analysis_method.sh layer_importance` |
| Late Injection / Early Exit | [`qwen25_vl_late_injection_early_exit.yaml`](methods/qwen25_vl_late_injection_early_exit.yaml) | `bash scripts/analysis/run_analysis_method.sh late_injection_early_exit` |
| Hierarchical Taylor | [`qwen25_vl_hierarchical_taylor.yaml`](methods/qwen25_vl_hierarchical_taylor.yaml) | `bash scripts/analysis/run_analysis_method.sh hierarchical_taylor` |

Qwen2.5-Omni 对应配置采用 `methods/qwen25_omni_audio_<method>.yaml` 命名。它们复用
同一七 Probe 实现，但 Token 组和跨模态边改为 audio 语义；运行时只观察
Thinker/LLM 音频 Token，不进入 Audio Encoder、Talker 或 Token2Wav。例如：

```bash
ANALYSIS_BACKBONE=qwen25_omni \
MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B \
bash scripts/analysis/run_analysis_method.sh layer_importance
```

以 Layer Importance 为例，直接运行最小真实配置：

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-VL-3B-Instruct \
bash scripts/analysis/run_analysis_method.sh layer_importance
```

单方法脚本默认使用 Qwen2.5-VL 教学配置；设置
`ANALYSIS_BACKBONE=qwen25_omni` 后切换到 Qwen 音频配置。输出写入
`outputs/analysis/methods/<backbone>/<method>/<timestamp>/`。服务器已有权重时设置
`MODEL_PATH`、`QWEN25_VL_MODEL_PATH` 或 `QWEN25_OMNI_MODEL_PATH`；未设置时使用
YAML 固定的模型 ID 和 revision。需要查看配置字段含义、计算机制与产物解释时，阅读
[配置字段参考](../../docs/reference/config.zh.md)和
[`1_run_analysis.zh.md`](../../docs/guides/1_run_analysis.zh.md)。

## 完整矩阵

本目录保存可执行 active 分析配置。四份 `*_analysis_matrix.yaml` 分别固定
LLaVA-OneVision、Qwen2.5-VL、Qwen2.5-Omni 和 Qwen3-VL 系列当前最小 checkpoint，并在同一个
lmms-eval 样本循环中运行六个继承 Probe 与 `hierarchical_taylor`。它们统一使用
`attn_implementation: eager`、单样本、单层；Taylor 同时执行 Path、Head、FFN
Neuron、Sublayer 和 Whole Layer 分数，并请求对应因果对照。Whole Layer 的
互斥随机对照需要至少两个所选层，因此正式激活证据另外运行首层加末层配置。
LLaVA 配置还单独固定 checkpoint 所引用的 SigLIP 视觉塔 revision，避免其
`mm_vision_tower` 在真实运行时回退到浮动 `main`。

只诊断 Taylor 时复用对应矩阵配置并传入
`--probe hierarchical_taylor`，无需维护第二份几乎相同的 YAML。

真实执行直接运行：

```bash
bash scripts/analysis/run_analysis_matrix.sh qwen3_vl
```

Qwen 音频矩阵使用：

```bash
bash scripts/analysis/run_analysis_matrix.sh qwen25_omni
```

该脚本仍调用统一的 `mllm-comp analyze` CLI，并为每次运行生成新的输出目录；
所有 Probe 参数继续由本目录 YAML 或转发的 CLI override 拥有。

服务器已有预下载权重时，可额外传入 `--checkpoint /absolute/checkpoint/path`；
示例 YAML 仍保留官方模型 ID 与 revision，避免把某台服务器的目录写入版本库。

## 压缩后分析

`compression/` 提供四个代表路径：Projector Token 的 Segmentwise Top-K、Encoder
Token 的 A-ToME、LLM Token 的 Audio-Pyramid，以及 LLM Operation 的 SPADE。
它们都使用完整的 `algorithm / variant / binding` v2 身份，设置
`compression_mode: compressed` 与 `phase: post_compression`。Audio-Pyramid 的
示例在 layer 8 出口压缩，并观察 layer 9 入口后的表示，避免把压缩边界前后的不同
序列宽度混在一个层记录中。

能力状态和真实验证范围见 `docs/guides/1_run_analysis.zh.md`。使用自动目录的配方可
直接重跑；显式 `--output_dir` 是精确目录，失败重跑需换新目录。未指定
`--manifest` 时自动采用本次目录下的 `manifest.json`。

实验入口在加载 checkpoint 前拒绝已有 Manifest 或非空运行目录；正式执行前保存
`effective_config.json` 和 `result.json`，失败信息写入结果，成功后另写 Manifest。
