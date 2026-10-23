# Qwen2.5-Omni Backbone Pipeline 人工 Debug 与准确性验收手册

> 类型：Debug
> 适用范围：未压缩 Qwen2.5-Omni baseline pipeline
> 最后更新：2026-08-04

本文用于人工验证 MLLM-Comp-Platform 中 Qwen2.5-Omni backbone 是否忠实执行 Transformers 原生契约。手册从真实运行入口开始，沿外部框架、media decode、processor、state、模态 encoder、Thinker、可选 Talker/Token2Wav、结果与 validator 顺序检查。

本文验证的是未压缩 backbone baseline。压缩执行族的架构、插入点和故障定位分别见 [Projector-Level](projector_level.zh.md)、[Encoder-Level](encoder_level.zh.md)、[LLM-Token / Audio-Pyramid](llm_token_audio_pyramid.zh.md)和 [LLM-Operation / SPADE](llm_operation_spade.zh.md)架构与 Debug 手册；自动评估与 backward gate 见 [八方法 Runbook](../eight_methods_runbook.zh.md)。

## 1. 先看清完整 Qwen2.5-Omni Pipeline

### 1.1 平台外层调用链

当前最适合熟悉 audio + Thinker 文本路径的真实入口是未压缩 MMAU 单样本评估：

~~~text
mllm-comp eval
  -> EvalConfig / lmms-eval launch plan
  -> lmms-eval worker
  -> _MLLMCompLMMBase
  -> load_backbone_runtime
  -> LMMSEvalRequestBridge
  -> Qwen25OmniProcessorAdapter
  -> Qwen25OmniStateBuilder
  -> Qwen25OmniRunner.generate
  -> Qwen2_5OmniForConditionalGeneration.generate
  -> GenerationResult
  -> Qwen25OmniNativeValidator
  -> lmms-eval answer / Manifest
~~~

Talker 和 Token2Wav 不经过 lmms-eval 质量入口；需要通过本手册第 5 节的直接 runner 路径或第 8 节 GPU smoke 验收。

### 1.2 模型内部多模态路径

~~~text
conversation
  -> qwen_omni_utils.process_mm_info
       ├─ image decode
       ├─ audio decode / resample
       └─ video decode，可选提取视频音轨
  -> Qwen2_5OmniProcessor
       ├─ input_ids / attention_mask
       ├─ pixel_values / image_grid_thw
       ├─ pixel_values_videos / video_grid_thw / video_second_per_grid
       └─ input_features / feature_attention_mask
  -> ProcessedBatch
  -> ForwardState + modality spans
  -> Qwen2_5OmniThinkerForConditionalGeneration
       ├─ image/video
       │    -> vision encoder
       │    -> vision projector
       ├─ audio
       │    -> Conv1 + Conv2
       │    -> positional embedding
       │    -> Audio Encoder layers
       │    -> avg_pooler
       │    -> ln_post
       │    -> audio_tower.proj
       └─ modality embeddings 替换对应 placeholders
  -> get_rope_index / M-RoPE
  -> Thinker LLM prefill
  -> Thinker LLM autoregressive decode
  -> Thinker text token IDs / text
~~~

如果 return_audio=True，还会继续：

~~~text
Thinker reply representation
  -> Talker
  -> codec token IDs
  -> 同一个 codec tensor 交给 Token2Wav
  -> 24 kHz waveform
  -> SpeechGenerationOutput
~~~

### 1.3 第一次熟悉时的推荐范围

第一遍只走下面这条主线：

~~~text
audio request
  -> Processor
  -> ProcessedBatch
  -> ForwardState/audio span
  -> Audio Tower
  -> pooler/projector
  -> Thinker prefill/decode
  -> text GenerationResult
  -> NativeValidator
~~~

第一遍可以跳过 image、video、Talker、Token2Wav、trace、故障注入和完整 benchmark。主线理解后，再按第 4–8 节扩展。

## 2. 从真实入口启动

### 2.1 固定 Lane、版本、环境和设备

先确认当前源码及其对应环境；本项目在所属 Dev lane 中开发与验证，不设固定 Review 工作区。按 resolver 返回的实际 lane 和角色选择对应源码、环境与 tmux；并行验证其他版本时使用独立 Dev lane。然后记录：

~~~bash
git rev-parse HEAD
git status --short
python -c "import mllm_comp; print(mllm_comp.__file__)"
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__)"
nvidia-smi
~~~

必须确认：

- mllm_comp editable import 指向当前 lane 的远端 workspace；
- QWEN25_OMNI_MODEL_PATH 指向 Qwen2.5-Omni-3B；
- Transformers 为当前绑定版本 4.57.3；
- Talker 文本基线关闭，只有语音分支显式启用；
- 在进程启动前通过 CUDA_VISIBLE_DEVICES 只暴露已确认空闲的 GPU；
- 进程内部使用映射后的 cuda:0，不根据物理编号写死 cuda:1。

推荐环境变量：

~~~bash
export QWEN25_OMNI_MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B
export CUDA_VISIBLE_DEVICES=<free_physical_gpu>
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
~~~

首次 Debug 不要使用 device_map="auto" 跨卡切分，也不要终止其他 lane 或用户的 GPU 进程。

### 2.2 本地契约测试

~~~bash
python -m pytest -q \
  tests/unit/test_qwen25_omni_runtime.py \
  tests/unit/test_lmms_eval_integration.py

python -m mllm_comp.cli.main eval \
  --config examples/compression/qwen25_omni_audio_segmentwise_topk.yaml \
  --plan_only

git diff --check
~~~

这些检查覆盖 processor/state/runner/trace 的轻量契约和 lmms-eval bridge，但不能替代真实 checkpoint。

### 2.3 先检查启动计划

~~~bash
bash scripts/eval/lmms_eval/qwen25_omni_2gpu_aud.sh --plan_only
~~~

预期：

- backbone.name 为 qwen25_omni；
- compression_mode 为 none，pipeline 没有 enabled compression stage；
- backend 为 lmms_eval，task 为 mmau_test_mini；
- enable_talker 为 false；
- checkpoint 来自 QWEN25_OMNI_MODEL_PATH；
- 原生命令进入平台 lmms-eval plugin。

plan_only 不加载模型，不能作为 backbone 正确性的运行证据。

### 2.4 启动单卡单样本主线

~~~bash
CUDA_VISIBLE_DEVICES=<free_physical_gpu> \
python -m mllm_comp.cli.main eval \
  --config configs/eval/lmms_eval/qwen25_omni_aud.yaml \
  --num_processes 1 \
  --sample_limit 1
~~~

使用支持 Python subprocess auto-attach 的 IDE 跟进 lmms-eval worker。不要在多 rank job 中使用交互式 pdb。若只用终端，可通过第 4 节直接 driver 进入同一 backbone pipeline。

## 3. 沿真实评估 Pipeline 逐步 Debug

### 3.1 B0：CLI、配置与 lmms-eval Launch Plan

断点：

- mllm_comp/cli/eval.py::main
- lmms-eval adapter 的 build_launch_plan

检查：

~~~python
config.backbone.name
config.eval.backend
config.eval.compression_mode.value
config.eval.runtime_mode.value
config.eval.tasks
config.backbone.config
~~~

预期：

- backbone 是 qwen25_omni；
- compression_mode 是 none；
- runtime mode 是 default；
- Talker 未启用；
- use_audio_in_video=False；
- generation 的 temperature=0.0，MMAU task 继续拥有 max_new_tokens 等任务参数。

通过后进入 lmms-eval worker。

### 3.2 B1：加载 Facade、Processor、Model 与 Runtime

断点：

- mllm_comp/integrations/lmms_eval/plugin.py::_MLLMCompLMMBase.__init__
- mllm_comp/runtime/loader.py::load_backbone_runtime
- Qwen25OmniModelAdapter.load_model()

在 runtime 返回后检查：

~~~python
runtime.model_path_or_id
runtime.device
runtime.dtype
type(runtime.processor).__name__
type(runtime.model).__name__
runtime.model.training
runtime.metadata
~~~

预期：

- model 与 processor 来自同一 checkpoint；
- model 处于 eval mode；
- dtype/device 与配置一致；
- Talker baseline 未加载或未启用；
- runtime metadata 的 backbone、checkpoint 与 launch plan 一致。

再检查稳定结构句柄：

~~~python
handles = facade.model_adapter.structural_handles(runtime.model)
required = {
    "thinker",
    "thinker.llm",
    "thinker.lm_head",
    "vision_encoder",
    "vision_projector",
    "audio_encoder",
    "audio_projector",
}
assert required <= handles.keys()
assert any(name.startswith("llm.layer[") for name in handles)
~~~

如果句柄缺失，先检查 checkpoint 类型和 Transformers 版本，不要通过猜测内部路径修改 compressor 或 probe。

### 3.3 B2：lmms-eval Request 与 Media Decode

断点：

- LMMSEvalRequestBridge.build_chat_sample()
- LMMSEvalRequestBridge._generate_sample()
- qwen_omni_utils.process_mm_info() 前后

检查：

~~~python
sample.sample_id
sample.conversation
sample.use_audio_in_video
audios
images
videos
~~~

预期：

- task/split/doc_id 映射为唯一 sample ID；
- conversation 使用 Qwen chat schema；
- audio 被实际读取，采样数据非空；
- audio-only 请求没有 image/video；
- use_audio_in_video=False 在 request、processor 和 generate 三处一致。

Media decode 失败时不要先改 processor；先验证文件存在、音轨、解码后对象和采样率。

### 3.4 B3：Qwen Processor 输出

断在 Qwen25OmniProcessorAdapter.prepare_inputs() 中 outputs = processor(...) 之后。

检查：

~~~python
processed.sample_ids
processed.model_inputs.keys()
processed.model_inputs["input_ids"].shape
processed.model_inputs["attention_mask"].shape
processed.model_inputs["input_features"].shape
processed.model_inputs["feature_attention_mask"].shape
processed.prompt_lengths
processed.metadata["processor_output_contract"]
processed.metadata["expected_token_counts_by_sample"]
~~~

audio-only 主线预期：

- input_ids 为 rank 2、int64；
- attention_mask 与 input_ids 同 shape；
- input_features 与 feature_attention_mask 成对存在；
- prompt_lengths 等于每行 attention_mask.sum()；
- expected audio placeholder count 大于 0；
- image/video 字段不存在。

ProcessorAdapter 最强检查是和 Transformers 原生 Processor 逐 tensor 对齐，具体代码见第 4.3 节。

### 3.5 B4：ProcessedBatch 转为 ForwardState

断在 Qwen25OmniStateBuilder.build() 返回 ForwardState 前，以及 Qwen25OmniNativeValidator.validate_state()。

~~~python
state.sample_ids
state.input_ids.shape
state.attention_mask.shape
state.audio_feature_lengths
[(s.sample_index, s.modality, s.start, s.end, s.media_unit)
 for s in state.modality_token_spans]
state.modality_metadata["expected_token_counts_by_sample"]
~~~

预期：

- state.sample_ids 与 processed.sample_ids 完全一致；
- native_model_inputs 中每个 tensor 与 ProcessedBatch 保持对象身份；
- audio span 为左闭右开区间 [start, end)；
- span 只覆盖 audio placeholder；
- span 总长度等于 processor 计算的 expected audio count；
- audio_feature_lengths 与 feature_attention_mask 一致；
- validate_state() 返回空列表。

这是平台状态层与 Qwen 原生输入之间最关键的契约边界。

### 3.6 B5：Runner 整理原生 Model Inputs

断在 Qwen25OmniRunner.generate() 调用 model.generate() 之前。

检查：

~~~python
model_inputs.keys()
model_inputs["input_ids"].shape
model_inputs["attention_mask"].shape
model_inputs["input_features"].shape
return_audio
speaker
use_audio_in_video
thinker_max_new_tokens
thinker_do_sample
~~~

预期：

- tensor 被移动到 runtime.device；
- floating tensor 使用 runtime.dtype，整数 tensor 保持原 dtype；
- baseline state 未被 legalizer 改写；
- audio feature 仍完整；
- 主线 return_audio=False；
- thinker_do_sample=False；
- use_audio_in_video 与 processor metadata 一致；
- native validator 在 generate 前没有 violation。

### 3.7 B6：进入 Qwen Thinker 与 Audio Tower

在 Qwen2_5OmniThinkerForConditionalGeneration.get_audio_features() 和 thinker.audio_tower.forward() 逐步检查：

~~~text
input_features / feature_attention_mask
  -> padded_and_mask_function
  -> Conv1 + Conv2
  -> positional embedding
  -> 原生 attention chunks
  -> Audio Encoder layers
  -> avg_pooler
  -> ln_post
  -> audio_tower.proj
  -> LLM hidden-size audio embeddings
~~~

重点检查：

- Conv 后 Token 长度与 feature mask 推导一致；
- cu_seqlens 从 0 严格递增，末项等于当前 encoder Token 总数；
- 32 层 Audio Encoder baseline 不改变序列长度；
- avg_pooler 后的每音频 Token 数等于 processor audio placeholder 数；
- projector 只改变 hidden size，不改变 Token 数；
- audio embedding 数与 audio placeholder 数严格相等；
- dtype/device 与 Thinker LLM 输入兼容。

这里理解清楚后，再看压缩手册会很直观：A-ToMe 改变 Audio Encoder 中间长度，Segmentwise 在 ln_post 后、projector 前改变长度；baseline 两处都不改变 Token。

### 3.8 B7：Embedding 替换、M-RoPE 与 Thinker Prefill

在 Thinker forward 中检查 modality embedding 替换和 get_rope_index()：

~~~python
input_ids.shape
attention_mask.shape
inputs_embeds.shape
position_ids.shape
~~~

预期：

- audio placeholders 被 projector 输出逐一替换；
- sequence 总长度不变；
- attention_mask 与 inputs_embeds 的序列维一致；
- M-RoPE position 覆盖完整 prompt；
- 第一次 Thinker LLM forward 是 prefill；
- prefill 长度等于有效 prompt；
- audio encoder 只在 prefill 执行一次。

如果 embedding 数和 placeholder 数不一致，应立即报错，不应通过 reshape、截断 feature 或减少 placeholder 掩盖问题。

### 3.9 B8：Thinker Decode 与 Raw Output

断在 Qwen25OmniRunner.generate() 的 raw_output = model.generate(...) 之后。

~~~python
type(raw_output)
raw_output.shape if hasattr(raw_output, "shape") else None
state.input_ids.shape[1]
~~~

预期：

- return_audio=False 时得到文本生成序列；
- prompt 只切除一次；
- new token IDs 非空；
- decode 每步通常只有一个新位置；
- decode 不重新运行 audio encoder；
- greedy 参数固定时结果可复核。

平台新增 Token 必须与同模型、同 native inputs、同生成参数的 Transformers 原生 model.generate() 完全一致，比较方法见第 4.5 节。

### 3.10 B9：GenerationResult、Validator 与 lmms-eval 返回

断点：

- Runner 构造 GenerationResult 后
- Qwen25OmniNativeValidator.validate_generation()
- LMMSEvalRequestBridge._generate_sample() 返回前

检查：

~~~python
result.text.token_ids
result.text.text
result.speech
result.traces
violations
sample.sample_id
~~~

audio + Thinker 主线预期：

- text token IDs 和解码文本非空；
- speech is None；
- 未请求 trace 时 traces is None；
- token batch 维与 sample ID 一致；
- validate_generation() 返回空列表；
- lmms-eval 接收的是本次 result.text，而不是旧缓存；
- task/split/doc_id、checkpoint 和输出目录可追溯；
- 最终 Manifest 对应本次 baseline run。

至此，第一遍 audio + Thinker backbone pipeline 已走完。随后可选择第 4 节做原生 tensor/token 精确对齐、第 5 节检查 Talker，或直接进入音频压缩手册。

## 4. 直接 Driver：做 Tensor 与原生 Transformers 精确对齐

CLI 路径最接近真实评估，直接 driver 更适合停在单个 tensor。两者必须使用同一 checkpoint、dtype 和生成参数。

### 4.1 加载 Facade、Processor 与 Model

~~~python
import os
from pathlib import Path

import torch

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen25_omni import (
    DEFAULT_AUDIO_SYSTEM_PROMPT,
    Qwen25OmniSample,
    create_qwen25_omni_backbone,
)

MODEL_PATH = Path(os.environ["QWEN25_OMNI_MODEL_PATH"])
assert MODEL_PATH.is_dir()
assert torch.cuda.device_count() == 1

facade = create_qwen25_omni_backbone()
processor = facade.processor_adapter.load_processor(
    str(MODEL_PATH),
    local_files_only=True,
)
model = facade.model_adapter.load_model(
    str(MODEL_PATH),
    train=False,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},
    enable_talker=False,
)
runtime = BackboneRuntime(
    model=model,
    processor=processor,
    model_path_or_id=str(MODEL_PATH),
    device=torch.device("cuda:0"),
    dtype=torch.bfloat16,
    metadata={"speech_sample_rate": 24000},
)

assert not model.training
assert runtime.device == torch.device("cuda:0")
~~~

### 4.2 构造确定性 Audio Sample

~~~python
def conversation(content):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_AUDIO_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]

sample = Qwen25OmniSample(
    sample_id="manual-audio",
    conversation=conversation([
        {"type": "audio", "audio": "/absolute/path/to/tone.wav"},
        {"type": "text", "text": "Briefly describe the sound."},
    ]),
)
batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="eval")
state = facade.state_builder.build(batch, phase="eval")
assert facade.native_validator.validate_state(runtime, state, batch) == []
~~~

建议使用 16 kHz、1 秒、440 Hz 单声道 WAV，并记录文件 SHA-256。首次准确性验收不要使用网络 URL、随机音频或会变化的媒体。

### 4.3 ProcessorAdapter 与原生 Processor 逐 Tensor 对齐

~~~python
from qwen_omni_utils import process_mm_info

native_text = processor.apply_chat_template(
    sample.conversation,
    tokenize=False,
    add_generation_prompt=True,
)
audios, images, videos = process_mm_info(
    sample.conversation,
    use_audio_in_video=False,
)
native_batch = processor(
    text=native_text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=False,
)

assert set(native_batch.keys()) == set(batch.model_inputs)
for key in native_batch:
    left = native_batch[key]
    right = batch.model_inputs[key]
    if torch.is_tensor(left):
        assert left.dtype == right.dtype, key
        assert left.shape == right.shape, key
        assert torch.equal(left, right), key
~~~

通过这一步才能证明 adapter 没有改写原生 token、grid、feature 或 mask。

### 4.4 ForwardState Span 对账

~~~python
from collections import Counter

actual = Counter()
for span in state.modality_token_spans:
    actual[span.modality] += span.end - span.start
    print(
        span.sample_index,
        span.modality,
        span.start,
        span.end,
        span.media_unit,
        span.encoder_origin,
    )

expected = batch.metadata["expected_token_counts_by_sample"][0]
for modality in ("image", "video", "audio"):
    assert actual[modality] == expected[modality], (modality, actual, expected)

for key, value in batch.model_inputs.items():
    assert state.native_model_inputs[key] is value
~~~

### 4.5 平台 Thinker Token 与原生 Generate 对齐

~~~python
text_result = facade.runner.generate(
    runtime,
    state,
    processed_batch=batch,
    return_audio=False,
    thinker_max_new_tokens=16,
    thinker_do_sample=False,
)

def move_native(value):
    if not hasattr(value, "to"):
        return value
    value = value.to(runtime.device)
    if callable(getattr(value, "is_floating_point", None)) and value.is_floating_point():
        value = value.to(dtype=runtime.dtype)
    return value

native_inputs = {
    name: move_native(value)
    for name, value in state.native_model_inputs.items()
}
native_sequences = model.generate(
    **native_inputs,
    return_audio=False,
    speaker="Chelsie",
    use_audio_in_video=batch.metadata["use_audio_in_video"],
    thinker_max_new_tokens=16,
    thinker_do_sample=False,
)
prompt_width = state.input_ids.shape[1]
native_new_tokens = native_sequences[:, prompt_width:]

assert torch.equal(
    native_new_tokens.detach().cpu(),
    text_result.text.token_ids.detach().cpu(),
)
native_text = processor.batch_decode(
    native_new_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
assert native_text == text_result.text.text
~~~

不一致时依次检查 sampling、max_new_tokens、use_audio_in_video、prompt slicing、model train/eval mode、dtype 和 device。

## 5. 多模态分支与 Talker/Token2Wav

### 5.1 五类输入的 Processor 字段矩阵

主线通过后，逐个样本运行 prepare_inputs()，不要第一次就混成 batch：

| 输入 | 必须出现的附加字段 | 必须出现的 span |
| --- | --- | --- |
| text | 无 | 无 |
| image | pixel_values、image_grid_thw | image |
| audio | input_features、feature_attention_mask | audio |
| silent video | pixel_values_videos、video_grid_thw、video_second_per_grid | video |
| AV video | video 三字段 + audio 两字段 | video、audio |

推荐确定性 fixture：

| Fixture | 建议内容 |
| --- | --- |
| image | 224×224 纯蓝 PNG |
| audio | 16 kHz、1 秒、440 Hz 单声道 WAV |
| silent video | 224×224、4 fps、1 秒纯蓝 MP4 |
| AV video | 224×224、4 fps、1 秒纯绿 MP4 + 440 Hz 音轨 |

额外不变量：

- image_grid_thw 与 pixel_values 成对出现；
- video_grid_thw 与 video_second_per_grid 第一维一致；
- input_features 与 feature_attention_mask 成对出现；
- AV video 的 video span 与 audio span 不重叠；
- use_audio_in_video=True 同时传给 process_mm_info、processor 和 model.generate；
- silent video 不得伪造 audio span。

如果日志提示 decord wheel 不支持当前平台，但实际显示 qwen-vl-utils 使用 torchcodec 且 video tensor 正常生成，该 decord 警告不是本 pipeline 的失败原因。

### 5.2 启用 Talker 的 Runtime

语音输出要求 batch size 1、默认 audio system prompt、有效 speaker，并在加载模型时 enable_talker=True。直接 driver 中将第 4.1 节的加载参数改为：

~~~python
model = facade.model_adapter.load_model(
    str(MODEL_PATH),
    train=False,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},
    enable_talker=True,
)
assert model.has_talker
print(sorted(model.speaker_map))
~~~

当前已验证 speaker 包括 Chelsie 和 Ethan。不要绕过 model.speaker_map 校验。

Token2Wav 使用 fp32，因此出现 FlashAttention 2 不支持 fp32、Token2Wav 回退到 SDPA 的日志是预期行为；Thinker/Talker 仍按加载配置运行。

结构句柄还应包含：

~~~python
handles = facade.model_adapter.structural_handles(model)
required_speech = {
    "talker",
    "talker.model",
    "talker.thinker_projector",
    "talker.lm_head",
    "token2wav",
    "token2wav.dit",
    "token2wav.vocoder",
}
assert required_speech <= handles.keys()
assert any(name.startswith("talker.layer[") for name in handles)
~~~

### 5.3 Talker 到 Token2Wav 的正式链路

~~~python
from mllm_comp.backbones.base import TraceRequest

trace_request = TraceRequest(
    thinker_reply_part=True,
    talker_codec_token_ids=True,
    token2wav_intermediates=True,
)
speech_result = facade.runner.generate(
    runtime,
    state,
    processed_batch=batch,
    trace_request=trace_request,
    return_audio=True,
    speaker="Chelsie",
    thinker_max_new_tokens=8,
    thinker_do_sample=False,
    talker_max_new_tokens=64,
    talker_do_sample=False,
    token2wav_num_steps=2,
)

assert speech_result.text is not None
assert speech_result.speech is not None
assert speech_result.speech.sample_rate == 24000
assert speech_result.speech.waveform.numel() > 0
assert torch.isfinite(speech_result.speech.waveform).all()
assert speech_result.speech.codec_token_ids is not None
assert speech_result.speech.codec_token_ids.numel() > 0
assert speech_result.speech.speaker == "Chelsie"

assert speech_result.traces is not None
assert speech_result.traces.thinker_reply_part is not None
assert (
    speech_result.traces.talker_codec_token_ids
    is speech_result.speech.codec_token_ids
)
assert (
    speech_result.traces.token2wav_intermediates["code"]
    is speech_result.speech.codec_token_ids
)
~~~

关键契约是：

~~~text
Thinker hidden/reply representation
  != Talker codec token IDs
  -> 同一个 codec tensor 进入 Token2Wav
  -> 24 kHz waveform
~~~

shape 不需要彼此相同，对象身份和阶段关系必须正确。

### 5.4 默认无 Trace 的语音路径

~~~python
plain_speech = facade.runner.generate(
    runtime,
    state,
    processed_batch=batch,
    return_audio=True,
    speaker="Chelsie",
    thinker_max_new_tokens=8,
    thinker_do_sample=False,
    talker_max_new_tokens=64,
    talker_do_sample=False,
    token2wav_num_steps=2,
)
assert plain_speech.traces is None
assert plain_speech.speech.codec_token_ids is not None
~~~

这证明 codec IDs 属于正式 SpeechGenerationOutput，而不是只有启用 trace 才存在。

### 5.5 保存 Waveform 试听

~~~python
import soundfile as sf

waveform = speech_result.speech.waveform.detach().float().cpu().numpy()
sf.write("/tmp/qwen25_omni_manual_debug.wav", waveform, 24000)
~~~

试听只作为补充证据。还要检查 waveform 非空、有限、不是全零，文件时长与 sample 数/24 kHz 一致，且没有严重削波。

## 6. Trace 的 Opt-in 语义

默认所有 trace 字段关闭：

~~~python
no_trace = facade.runner.generate(
    runtime,
    state,
    processed_batch=batch,
    return_audio=False,
    thinker_max_new_tokens=4,
    thinker_do_sample=False,
)
assert no_trace.traces is None
~~~

只开启一个 Thinker 字段：

~~~python
one_trace = facade.runner.generate(
    runtime,
    state,
    processed_batch=batch,
    trace_request=TraceRequest(thinker_hidden_states=True),
    return_audio=False,
    thinker_max_new_tokens=4,
    thinker_do_sample=False,
)
assert one_trace.traces is not None
assert one_trace.traces.thinker_hidden_states is not None
assert one_trace.traces.talker_logits is None
assert one_trace.traces.token2wav_intermediates == {}
~~~

Qwen2.5-Omni 必须使用 thinker_* trace 字段。通用 text_* 字段属于非 Omni backbone，传给 Qwen2.5-Omni 时应立即抛出 invariant error。

大 tensor trace 只用于短序列 Debug。不要在长视频、长输出或正式评测中同时开启 logits、hidden states、attentions 和 KV cache。

## 7. 断点定位与故障注入

### 7.1 按函数名定位，不保存固定行号

~~~bash
rg -n "def (load_backbone_runtime|prepare_inputs|build|generate|validate_state|validate_generation)" \
  mllm_comp/runtime/loader.py \
  mllm_comp/backbones/qwen25_omni/implementation.py \
  mllm_comp/backbones/qwen25_omni/validation.py

rg -n "def (build_chat_sample|_generate_sample|generate_until|generate_until_chat)" \
  mllm_comp/integrations/lmms_eval/model_bridge.py

rg -n "def (token2wav_pre_hook|capture_qwen25_omni_traces)" \
  mllm_comp/backbones/qwen25_omni/tracing.py
~~~

推荐顺序就是 B0–B9。每个断点只打印 keys、shape、dtype、device、有限性和少量切片，不打印完整 logits、attention、waveform 或 KV cache。

### 7.2 故障注入：证明 Validator 不是空检查

故障注入只在临时对象或临时副本中执行。

attention mask 错位：

~~~python
from dataclasses import replace

bad_state = replace(state, attention_mask=state.attention_mask[:, :-1])
violations = facade.native_validator.validate_state(runtime, bad_state, batch)
assert "qwen_attention_shape" in {item.name for item in violations}
~~~

modality tensor/grid 不成对：

~~~python
bad_state = replace(state, image_grid_thw=None)
violations = facade.native_validator.validate_state(runtime, bad_state, batch)
assert "qwen_modality_pair" in {item.name for item in violations}
~~~

span 数量错误：

~~~python
bad_spans = list(state.modality_token_spans)
bad_spans[0] = replace(bad_spans[0], end=bad_spans[0].end - 1)
bad_state = replace(state, modality_token_spans=bad_spans)
violations = facade.native_validator.validate_state(runtime, bad_state, batch)
assert "qwen_placeholder_count" in {item.name for item in violations}
~~~

codec 缺失：

~~~python
bad_speech = replace(speech_result.speech, codec_token_ids=None)
bad_result = replace(speech_result, speech=bad_speech)
violations = facade.native_validator.validate_generation(
    runtime, state, batch, bad_result
)
assert "qwen_talker_codec_required" in {item.name for item in violations}
~~~

Talker 到 Token2Wav handoff 被复制：

~~~python
bad_intermediates = dict(speech_result.traces.token2wav_intermediates)
bad_intermediates["code"] = speech_result.speech.codec_token_ids.clone()
bad_traces = replace(
    speech_result.traces,
    token2wav_intermediates=bad_intermediates,
)
bad_result = replace(speech_result, traces=bad_traces)
violations = facade.native_validator.validate_generation(
    runtime, state, batch, bad_result
)
assert "qwen_token2wav_codec_handoff" in {item.name for item in violations}
~~~

预处理与生成阶段的 use_audio_in_video 不一致：

~~~python
from mllm_comp.core.errors import InvariantError

try:
    facade.runner.generate(
        runtime,
        state,
        processed_batch=batch,
        use_audio_in_video=not batch.metadata["use_audio_in_video"],
        return_audio=False,
    )
except InvariantError as error:
    assert "use_audio_in_video" in str(error)
else:
    raise AssertionError("runner should reject inconsistent use_audio_in_video")
~~~

只有正常路径通过且上述负例被精确拒绝，才能证明 validator 真正执行了契约。

## 8. 完整真实 Checkpoint GPU Smoke

### 8.1 启动前

先用 resolver 确认当前 lane、remote workspace、tmux 和 environment；再检查 GPU，无权终止其他任务：

~~~bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader

export CUDA_VISIBLE_DEVICES=<free_physical_gpu>
export QWEN25_OMNI_MODEL_PATH=/absolute/path/to/Qwen2.5-Omni-3B
export MLLM_COMP_RUN_QWEN25_OMNI_GPU_SMOKE=1
export QWEN25_OMNI_SMOKE_REPORT="$PWD/outputs/qwen25_omni_smoke.json"
~~~

验证进程只看到预期设备：

~~~bash
python -c "import os, torch; print(os.environ['CUDA_VISIBLE_DEVICES']); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
~~~

### 8.2 运行 Smoke

~~~bash
python -m pytest -q \
  tests/integration/test_qwen25_omni_gpu_smoke.py -s --pdb
~~~

该 smoke 覆盖 text、image、audio、silent video、audio-in-video、Thinker 文本、Talker codec 和 Token2Wav waveform。

最终结构化报告至少记录：

- commit、Transformers/PyTorch 版本；
- checkpoint config hash；
- CUDA_VISIBLE_DEVICES、逻辑 device、GPU 名称和 dtype；
- fixture SHA-256；
- 五类输入的 processor 字段、shape、dtype 和 spans；
- Thinker token shape 与非空文本；
- thinker_reply_part、Talker codec、Token2Wav code 和 waveform shape；
- speech sample rate 和 speaker；
- validator 结果。

历史基线曾观察到：

~~~text
text/image/audio/silent-video/audio-in-video: non-empty Thinker output
waveform: [30240]
Talker codec: [1, 63]
Token2Wav code: [1, 63]
thinker_reply_part: [1, 8, 2048]
sample rate: 24000
speaker: Chelsie
~~~

生成长度和自然语言内容不是跨版本固定常量。只在同 checkpoint、依赖、fixture 和 generation 参数下比较可复核结果。

如果远端环境没有 pytest，No module named pytest 只表示测试工具未安装，不代表 backbone 错误。获得许可后安装测试依赖，或使用第 4–7 节的直接 driver；不能把未运行测试写成通过。

## 9. 常见问题与固定定位顺序

| 现象 | 首先回到 | 不应采取的做法 |
| --- | --- | --- |
| No module named mllm_comp | 第 2.1 节，editable import 与 PYTHONPATH | 修改包内 import 绕过 |
| checkpoint/model 类型错误 | B1、config.json、Transformers 版本 | 猜测内部模块路径 |
| media decode 失败 | B2，文件/音轨/解码对象 | 直接修改 Processor |
| Processor 字段缺失 | B3，process_mm_info 和原生 processor | 手工补 tensor |
| qwen_modality_pair | B3/B4，tensor 与 grid/mask 配对 | 删除 validator 检查 |
| qwen_placeholder_count | B4/B6，token ID、span 和 encoder 输出 | 手工伪造 expected count |
| audio embedding 数不等 | B6/B7，pool/projector 与 placeholders | 截断 input_features |
| audio-in-video 缺 audio span | B2/B3，视频音轨和三阶段参数 | 把 silent video 当 AV video |
| 文本 Token 不对齐 | B5–B8，sampling、prompt slicing、dtype | 只比较最终字符串 |
| decode 重复运行 Audio Tower | B6–B8，prefill/decode 边界 | 忽略重复调用 |
| codec 为 None | 第 5 节，Talker 启用和 speaker | 把 waveform 当 Talker 输出 |
| Token2Wav 回退 SDPA | Token2Wav fp32，是预期行为 | 强制 fp16 FlashAttention |
| waveform 空或 NaN | codec handoff、Token2Wav 输入、dtype | 只检查 WAV 能否打开 |
| GPU 看到多张卡 | CUDA_VISIBLE_DEVICES 设置时机 | 在代码中改用物理 cuda 编号 |

固定定位顺序：

~~~text
CLI/config
  -> runtime/model/processor
  -> request/media decode
  -> Processor tensors
  -> ForwardState/spans
  -> Runner model inputs
  -> modality encoder
  -> projector/placeholder/M-RoPE
  -> Thinker prefill/decode
  -> optional Talker/Token2Wav
  -> GenerationResult
  -> NativeValidator
  -> external framework result/Manifest
~~~

不要从最终回答或可播放 WAV 反推中间 pipeline 正确。

## 10. 最终人工验收清单

### Audio + Thinker 主线

- [ ] 当前 lane、commit/diff、environment、checkpoint、GPU 和输入 hash 已记录。
- [ ] 从真实 mllm-comp eval 入口按 B0–B9 顺序走完。
- [ ] ProcessorAdapter 与原生 Processor 逐 tensor 一致。
- [ ] input_features、feature_attention_mask、audio placeholders 和 spans 对账。
- [ ] validate_state() 返回空列表。
- [ ] Audio Tower、pooler、projector 和 LLM placeholder 数量对齐。
- [ ] M-RoPE 覆盖完整 prompt。
- [ ] Audio Tower 只在 prefill 执行，decode 不重复编码音频。
- [ ] 平台 Thinker 新增 Token 与原生 model.generate() 完全一致。
- [ ] return_audio=False 只返回 text。
- [ ] validate_generation() 返回空列表。

### 其他模态

- [ ] text/image/audio/silent-video/AV-video 字段矩阵正确。
- [ ] image/video/audio spans 与 processor expected counts 一致。
- [ ] use_audio_in_video 在 decode、processor 和 generate 三阶段一致。

### Talker 与 Token2Wav

- [ ] return_audio=True 同时返回 text、codec 和 waveform。
- [ ] Talker codec 与 Token2Wav code 是同一个 handoff 对象。
- [ ] waveform 非空、有限、采样率 24 kHz。
- [ ] 未请求 trace 时 result.traces is None。
- [ ] 请求字段存在，未请求字段没有被保留。

### 负向与回归

- [ ] mask/grid/span/codec/handoff 故障注入均被 validator 拒绝。
- [ ] 完整 GPU smoke 退出码为 0，并生成结构化报告。
- [ ] 未将模型成功加载、非空文本或可播放 WAV 单独当作正确性证据。

只有对应范围的 checklist 全部通过，才能声明该范围的 Qwen2.5-Omni backbone 集成已验收。第一次学习只需完成 Audio + Thinker 主线；Talker 和其他模态可以在后续需要时补验。
