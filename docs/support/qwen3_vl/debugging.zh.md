# Qwen3-VL 人工逐层 Debug 与验收手册

> 类型：Debug
> 适用范围：Qwen3-VL 原生 backbone、lmms-eval 与 ms-swift bridge
> 当前状态：Active baseline debugging

本文用于人工确认 MLLM-Comp-Platform 的 Qwen3-VL 适配是否真的正确，而不只是“代码可以 import”或“模型偶然生成了一句话”。验收对象包括平台原生 backbone、lmms-eval 评测桥、ms-swift 训练桥和 Qwen2.5-Omni 回归边界。

## 1. 验收结论的含义

只有以下证据同时存在，才能把 `qwen3_vl` 标记为 active：

| 层级 | 必须证明的内容 |
|---|---|
| 依赖 | Transformers 能导入 Qwen3-VL，版本不低于 4.57 |
| Processor | text/image/video 真实 checkpoint 输出 keys、shape、grid 和 metadata 正确 |
| State | image/video placeholder spans 与 grid 推导 token 数相等 |
| Model | text/image/video 均能在真实权重上生成非空文本 |
| 结构 | language model、vision、merger、DeepStack projectors 可定位 |
| Trace | logits、hidden states、KV cache 按需采集，不请求时不保留 |
| lmms-eval | 原生 CLI 发现平台 entry point，插件在子进程中只加载一份平台 runtime，请求经过 Processor/StateBuilder/Runner |
| ms-swift | 使用 `qwen3_vl` model/template，完成真实一步训练并产生 checkpoint |
| 边界 | audio、Talker、未合法化 token mutation、伪 compressed training 被拒绝 |
| 回归 | Transformers 升级后 Qwen2.5-Omni 全模态与 Talker 仍通过 |

以下结果不能单独证明适配完成：

- YAML 能被解析；
- `AutoProcessor` 能加载；
- 只跑 text，不跑 image/video；
- lmms-eval 选择了上游内置 Qwen wrapper，或请求绕过平台 Processor/StateBuilder/Runner；
- Swift 打印了参数但没有完成 `global_step=1`；
- 压缩只存在于平台内存，而 Swift 实际加载原始 checkpoint；
- 单元测试用假 tensor 通过，但没有真实 checkpoint 证据。

## 2. 能力边界

Qwen3-VL 在本平台中的基座能力是：

- 输入：text、image、video；
- 输出：text；
- position：Interleaved-MRoPE，由 backbone 持有；
- vision：patch size 16、spatial merge size 通常为 2；
- DeepStack：多个视觉 merger 输出进入语言模型不同深度；
- tracing：text logits、hidden states、attentions、KV cache；
- audio：不支持；
- Talker/Token2Wav：不存在；
- streaming：平台完整 `GenerationResult` 路径暂未激活。

不要把 Qwen2.5-Omni 的以下参数传给 Qwen3-VL：

```text
enable_talker
return_audio=true
speaker
talker_max_new_tokens
token2wav_num_steps
use_audio_in_video
enable_audio_output
```

## 3. 代码入口

| 目的 | 文件 |
|---|---|
| capability profile | `mllm_comp/backbones/profiles.py` |
| Processor/Model/State/Runner/Legalizer | `mllm_comp/backbones/qwen3_vl/implementation.py` |
| 原生 invariant | `mllm_comp/backbones/qwen3_vl/validation.py` |
| opt-in trace | `mllm_comp/backbones/qwen3_vl/tracing.py` |
| facade factory | `mllm_comp/backbones/qwen3_vl/backbone.py` |
| runtime loader | `mllm_comp/runtime/loader.py` |
| lifecycle catalog | `mllm_comp/registry/catalog.py` |
| lmms request bridge | `mllm_comp/integrations/lmms_eval/model_bridge.py` |
| lmms executor | `mllm_comp/integrations/lmms_eval/adapter.py` |
| Swift launch/train adapter | `mllm_comp/integrations/ms_swift/adapter.py` |

## 4. 环境与 GPU 安全规则

已验证环境：

```text
Python             3.10.20
PyTorch            2.8.0+cu128
Transformers       4.57.3
qwen-vl-utils      0.0.14
flash-attn         2.8.3
torchcodec         0.7.0
ms-swift           4.4.0
lmms-eval          0.7.2
```

使用当前 lane 分配的 GPU。单卡诊断按下面的方式启动：

1. 进程启动前执行 `nvidia-smi`；
2. 确认本次分配设备的可用显存足够；
3. 设置 `CUDA_VISIBLE_DEVICES="$GPU_ID"`，其中 `GPU_ID` 为分配的设备编号或 UUID；
4. Python 内只应看到一个逻辑设备；
5. 模型使用逻辑 `cuda:0`；
6. 不终止、不迁移其他任务的进程。

`device_map` 使用进程内的逻辑设备。只暴露一张卡后，该卡在进程内就是 `cuda:0`。

## 5. 第零层：依赖和 catalog

首先验证导入，不加载权重：

```bash
python - <<'PY'
import transformers
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
print(transformers.__version__)
print(Qwen3VLForConditionalGeneration.__name__)
print(Qwen3VLProcessor.__name__)
PY
```

期望 Transformers 至少为 4.57。若导入失败，不要使用 `trust_remote_code=true` 绕过；先修正正式依赖。

然后检查 catalog：

```bash
python - <<'PY'
from mllm_comp.registry.backbones import ACTIVE_BACKBONES
from mllm_comp.registry.catalog import CapabilityKind, get_capability_spec
spec = get_capability_spec(CapabilityKind.BACKBONE, "qwen3_vl")
print(spec.status)
print(spec.profile)
print(tuple(ACTIVE_BACKBONES))
PY
```

人工检查：

- `qwen3_vl` 为 active；
- input 是 text/image/video；
- output 只有 text；
- `supports_audio=false`；
- `position_system=m_rope`；
- minimum Transformers 是 4.57；
- catalog 有真实 GPU 1 验证记录。

## 6. 第一层：真实 Processor 合同

该步骤不加载模型权重，也不占用 GPU：

```bash
python scripts/diagnostics/validate_qwen3_vl_processor_contract.py \
  --model_path "$QWEN3_VL_MODEL_PATH" \
  --report_path outputs/qwen3_vl_processor_contract.json \
  --video_reader torchcodec
```

脚本会生成确定性的图片和一秒视频，分别检查 text、image、video。

### 6.1 text

期望 keys：

```text
input_ids
attention_mask
```

不应凭空出现视觉 tensor 或 grid。

### 6.2 image

期望至少出现：

```text
input_ids
attention_mask
pixel_values
image_grid_thw
```

当前 256×256 fixture 的真实证据为：

```text
pixel_values     [256, 1536]
image_grid_thw   [1, 3]
image tokens     64
image span       [15, 79)
```

### 6.3 video

期望至少出现：

```text
input_ids
attention_mask
pixel_values_videos
video_grid_thw
```

`video_metadata_present` 必须为 true。当前 fixture 的真实证据为：

```text
pixel_values_videos  [1152, 1536]
video_grid_thw       [1, 3]
expected video tokens 288
video spans          [21, 165) + [173, 317)
```

两个 span 的长度之和必须为 288。视频出现多个 span 是时间戳/文本对齐序列化的合法结果，不能错误合并为一个连续区间。

### 6.4 关键断点

在 `Qwen3VLProcessorAdapter.prepare_inputs()` 中依次检查：

1. `conversation_input`；
2. `apply_chat_template()` 返回文本；
3. `process_vision_info()` 的三元组；
4. `video_inputs` 是否由 `(video_tensor, video_metadata)` 对组成；
5. `video_kwargs`；
6. processor 返回 keys；
7. `expected_token_counts_by_sample`；
8. `token_ids`；
9. `image_patch_size == 16`；
10. `video_metadata_present`。

不要直接调用模型来“验证 decode”。媒体读取属于 processor/qwen-vl-utils；模型只消费 tensor 和 grid。

## 7. 第二层：ForwardState 与 token spans

在 `Qwen3VLStateBuilder.build()` 后检查：

```python
assert state.native_model_inputs == batch.model_inputs
assert state.video_pixel_values is batch.model_inputs.get("pixel_values_videos")
assert state.token_type_mask is batch.model_inputs.get("mm_token_type_ids")
```

`mm_token_type_ids` 是可选字段；真实 Qwen3-VL-8B-Instruct processor 在本次 text/image/video fixture 中没有返回它。平台必须在字段出现时透传并校验 shape，但不能把它错误设为必需字段。

span 的计算原则：

- image：连续 `<|image_pad|>` token；
- video：连续 `<|video_pad|>` token；
- `start` 包含、`end` 不包含；
- 按 sample index、start、end 排序；
- 每个 sample、每种 modality 的 span 总长度等于 grid 推导 token 数。

如果 invariant 报 `qwen3_vl_placeholder_count`，优先排查：

1. 是否错误使用了 Qwen2.5-VL 的 patch size 14；
2. spatial merge size 是否取错；
3. batch 内 media 顺序是否与 conversation 顺序不同；
4. video 是否被错误合并成单一 span；
5. tokenizer special token id 是否取错。

## 8. 第三层：结构定位和 DeepStack

执行：

```python
handles = facade.model_adapter.structural_handles(model)
for name in sorted(handles):
    print(name, type(handles[name]).__name__)
```

至少应有：

```text
model
llm
lm_head
vision_encoder
vision_projector
vision_deepstack_projectors
vision_deepstack_projector[0]
vision_deepstack_projector[1]
vision_deepstack_projector[2]
llm.layer[0]
llm.layer[0].attn
llm.layer[0].ffn
```

MoE checkpoint 还可能出现 `llm.layer[i].moe.experts`。dense checkpoint 没有 experts 是合法的。

DeepStack 不能只保留最终 `vision_projector`。任何视觉 token 压缩若改变 token 数，还必须回答三个问题：

1. 每个 DeepStack 注入点接收什么 shape；
2. grid 和 placeholder 如何同步；
3. Interleaved-MRoPE position 如何修复。

原生 Backbone 的通用 legalizer 继续拒绝任意 token mutation；已声明的压缩
接入通过对应执行机制修复 DeepStack、grid、placeholder 和 Interleaved-MRoPE。
精确范围以[ImageLLM 状态](../imagellm/status.zh.md)和
[视频压缩状态](../video_compression/status.zh.md)为准。forward、prefill、decode
和完整训练恢复分别需要对应证据。

## 9. 第四层：真实 text/image/video 生成

确认当前 GPU 余量后直接运行诊断；`CUDA_VISIBLE_DEVICES` 由本次环境选择：

```bash
export CUDA_VISIBLE_DEVICES="${GPU_ID:?Set GPU_ID to the assigned GPU}"
nvidia-smi
export QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export QWEN3_VL_SMOKE_REPORT="$PWD/outputs/qwen3_vl_smoke.json"
python scripts/diagnostics/validate_qwen3_vl_gpu_smoke.py \
  --model_path "$QWEN3_VL_MODEL_PATH" \
  --report_path "$QWEN3_VL_SMOKE_REPORT"
```

脚本必须验证：

- Python 只看到一个 CUDA device；
- text 生成非空；
- image 生成非空；
- video 生成非空；
- `result.speech is None`；
- image/video spans 与 processor 合同一致；
- DeepStack handles 存在；
- report 写入成功。

本次真实输出中，video fixture 的回答明确识别为绿色。

## 10. 第五层：Trace

默认不请求 trace：

```python
result = facade.runner.generate(..., max_new_tokens=8, do_sample=False)
assert result.traces is None
```

按需请求：

```python
trace_request = TraceRequest(
    text_logits=True,
    text_hidden_states=True,
    text_kv_cache=True,
)
```

Qwen3-VL 只有统一文本生成分支，因此必须使用 `text_*` trace 字段。把 Qwen2.5-Omni 的 `thinker_*`、`talker_*` 或 `token2wav_*` 字段传给 Qwen3-VL 应立即抛出 invariant error；若请求被静默忽略，则适配失败。

当前 8B checkpoint 的真实证据：

```text
last-step logits   [1, 1, 151936]
hidden-state sets  37
KV cache type      DynamicCache
capture semantics  last_forward_step
```

注意：hook 只保留最后一次 forward 输出。若研究需要逐 token 全轨迹，应新增有显式内存预算的 trace 类型，不要默默把每一步所有层都常驻显存。

## 11. 第六层：lmms-eval 原生 CLI 插件

生产评测不再接收父进程预先构造的模型对象。lmms-eval 负责 CLI、task、dataset、
metric、日志和结果文件；平台父进程只生成启动计划和 plugin spec。模型由 lmms-eval
子进程发现的平台 entry point 加载，正确的数据流是：

```text
mllm-comp eval
  -> LMMSEvalAdapter.build_launch_plan
  -> python -m lmms_eval
  -> lmms_eval.models entry point
  -> MLLMCompChatLMM（默认）或 MLLMCompLMM（显式 force_simple）
  -> load_backbone_runtime
  -> LMMSEvalRequestBridge
  -> Qwen3VLSample
  -> platform ProcessorAdapter / StateBuilder / Runner
  -> lmms-eval metric、result/sample 文件和结果表
  -> 平台读取结果并写 Manifest
```

走读顺序：

1. `mllm_comp/integrations/native_cli.py`
2. `mllm_comp/integrations/lmms_eval/adapter.py`
3. `mllm_comp/integrations/lmms_eval/plugin.py`
4. `mllm_comp/integrations/lmms_eval/model_bridge.py`

先检查完整启动计划，再做一条 MME image smoke：

```bash
export QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
bash examples/eval/eval_qwen3_vl_mme.sh --plan_only
bash examples/eval/eval_qwen3_vl_mme.sh --sample_limit 1
```

必须确认：

- argv 使用 `python -m lmms_eval --model mllm_comp_qwen3_vl`，并包含 `mllm_comp_spec=...`；
- 默认 argv 没有 `--force_simple`，日志实例化 `MLLMCompChatLMM`；
- `plugin._MLLMCompLMMBase.__init__()` 只执行一次 `load_backbone_runtime()`；
- 请求进入 `LMMSEvalRequestBridge`、Qwen3 ProcessorAdapter、StateBuilder 和 Runner；
- 终端保留 lmms-eval 原生日志、进度和结果表，输出目录同时存在原生 result/sample
  与平台 Manifest。

父进程不应加载模型；子进程内也不应再创建上游内置 wrapper 或第二份权重。MME 只
覆盖 image 请求，video 必须另选包含视频样本的 lmms-eval task 验证，不能用 MME
成功推断 video 已通过。

Qwen3-VL lmms bridge 的限制：

- 当前 batch size 为 1；
- 支持 `generate_until`；
- 不支持 `loglikelihood`；
- 不支持 multi-round task；
- audio 和 audio-in-video 显式失败；
- 输出只有文本。

## 12. 第七层：ms-swift

Swift 负责 template、dataset、collator、Trainer、optimizer、LoRA、分布式和 checkpoint。平台只产生并审计启动计划。

### 12.1 `plan_only`

```bash
python -m mllm_comp.cli.main train \
  --config tests/fixtures/qwen3_vl_ms_swift_gpu_smoke.yaml \
  --plan_only
```

期望：

```text
--model       Qwen3-VL checkpoint
--model_type  qwen3_vl
--template    qwen3_vl
environment   {}
trainable_branch language_model
talker_training false
```

`trainable_branch=language_model` 表示 Qwen3-VL 的生成分支边界；Swift 的 `freeze_vit`/`freeze_aligner` 仍决定视觉模块是否训练。不要把该标签解释成框架永远不能训练 vision。

### 12.2 一步训练

```bash
export MLLM_COMP_TRAIN_CONFIG="$PWD/tests/fixtures/qwen3_vl_ms_swift_gpu_smoke.yaml"
export MLLM_COMP_SWIFT_SMOKE_DATASET="$PWD/tests/fixtures/ms_swift_text_smoke.jsonl"
export MLLM_COMP_SWIFT_SMOKE_OUTPUT="$PWD/outputs/qwen3_vl_swift_smoke"
python -m mllm_comp.cli.main train --config "$MLLM_COMP_TRAIN_CONFIG" --max_steps 1
```

必须检查：

- `global_step=1`；
- `logging.jsonl`；
- `checkpoint-1/adapter_model.safetensors`；
- Manifest 的 `train_backend.name=ms_swift`；
- `backbone_name=qwen3_vl`；
- `trainable_branch=language_model`；
- `talker_training=false`；
- `invariants_run` 含 `Qwen3VLLanguageModelTrainingBoundary`。

### 12.3 compressed training 的诚实边界

当前训练入口既支持声明范围内的 compression-aware runtime，也支持相应的
导出产物重载。选择精确方法、位置、阶段和配置后，按[训练指南](../../guides/1_run_training.zh.md)
验证参数归属、实际梯度、保存和恢复；不要沿用早期“只支持导出 checkpoint”的限制。
一步训练、选定 norm 重载与完整 Trainer 恢复的证据分别记录。

## 13. 负向测试

逐项执行并确认失败信息明确：

1. conversation 包含 `audio`；
2. lmms 设置 `use_audio_in_video=true`；
3. runner 设置 `return_audio=true`；
4. runtime loader 设置 `enable_talker=true`；
5. Swift 设置 `trainable_branch=thinker` 或 `talker`；
6. Qwen3 Swift 配置出现 `enable_audio_output`；
7. legalizer 收到非空 `token_actions`；
8. 选择导出产物重载的 compressed Swift 缺少 `checkpoint_in`；
9. compressed Swift 使用当前声明范围外的压缩组合或不匹配的恢复产物；
10. lmms 使用 `runtime_batch_size=2`；
11. lmms task 使用 `loglikelihood`；
12. 同一 processor batch 混用 `enable_thinking=true/false`。

负向测试不是附加项；它证明平台没有把 Qwen2.5-Omni 参数错误泄漏到 Qwen3-VL。

## 14. Qwen2.5-Omni 回归

Qwen3-VL 需要 Transformers 4.57，而 Qwen2.5-Omni 原适配是在 4.52.3 下建立。升级后必须重跑：

```bash
export QWEN25_OMNI_MODEL_PATH=/path/to/Qwen2.5-Omni-3B
export QWEN25_OMNI_SMOKE_REPORT="$PWD/outputs/qwen25_omni_transformers457_regression.json"
export MLLM_COMP_RUN_QWEN25_OMNI_GPU_SMOKE=1
python -m pytest -q tests/integration/test_qwen25_omni_gpu_smoke.py -s
```

验收范围：

- text；
- image；
- audio；
- silent video；
- audio-in-video；
- Thinker text；
- Talker codec tokens；
- Token2Wav 24 kHz waveform；
- Talker/Token2Wav traces。

为了避免 4.57 默认 fast processor 改变 checkpoint 预处理语义，Qwen2.5-Omni adapter 默认 `use_fast=false`；调用方仍可显式覆盖。模型加载参数从弃用的 `torch_dtype` 转换为 `dtype`。

## 15. 常见故障定位

| 现象 | 优先检查 |
|---|---|
| 无法导入 Qwen3VL 类 | Transformers 是否低于 4.57 |
| `decord 0.6.0 is not supported` | 不要依赖该 wheel；设置 torchcodec reader |
| 找不到 ffmpeg | 检查 `Path(sys.executable).with_name("ffmpeg")` |
| image token 数不一致 | patch size 是否错误用了 14 |
| video token 数不一致 | metadata、fps、temporal patch、多个 span |
| `mm_token_type_ids` 缺失 | 它是可选字段，不应强制要求 |
| greedy generation flags 警告 | runner 是否把 temperature/top-p/top-k 显式置为 None |
| lmms 显存翻倍 | 是否同时启动了多个评测进程，或插件初始化/`load_backbone_runtime()` 被执行多次 |
| Swift 成功但无 checkpoint | 是否真的完成 step，检查版本化 output 子目录 |
| compressed training 实际未压缩 | 是否加载了导出 checkpoint 或注入真实训练 hook |
| Qwen3 返回 speech | 严重边界错误，立即失败 |

## 16. 最终证据清单

每次声称完成适配前至少保留：

- processor contract JSON；
- native GPU smoke JSON；
- native GPU smoke 日志中的启动前 `nvidia-smi`；
- lmms GPU smoke 日志；
- Swift `plan_only` JSON；
- Swift 完整训练日志；
- `checkpoint-1` 文件清单；
- 平台 Manifest；
- Qwen2.5-Omni 升级回归 JSON 和日志；
- 本地与远端全量测试结果；
- `git diff --check` 和 Python compile 结果。

只有这些证据互相一致，才能说明平台、框架和基座边界都正确。

## Operation 实际激活参考

各策略统一通过现有 Python 入口选择；`--help` 列出 Attention、FFN 和 cache 参数。
例如低秩 FFN：

```bash
python scripts/diagnostics/validate_qwen3_vl_attention_replacement.py \
  --model_path "$QWEN3_VL_MODEL_PATH" --strategy ffn_low_rank --layer 0 --rank 256 \
  --report_path outputs/qwen3_vl_ffn_low_rank.json \
  --trace_path outputs/operator_traces/qwen3_vl_ffn_low_rank.pt
```

复现原 NHA smoke 的 16-token 窗口时，显式指定 `--window_size 16`；共用 CLI 的
默认窗口为 64，原 LoLCATs smoke 使用该默认值：

```bash
python scripts/diagnostics/validate_qwen3_vl_attention_replacement.py \
  --model_path "$QWEN3_VL_MODEL_PATH" --strategy attn_nha --window_size 16 \
  --report_path outputs/qwen3_vl_nha.json \
  --trace_path outputs/operator_traces/qwen3_vl_nha.pt
```

该诊断比较真实替换模块的激活与独立数学参考；LoLCATs/NHA 还检查适用的 decode
历史状态。无需再为每个策略维护一个只转发环境变量的 Shell 脚本。
