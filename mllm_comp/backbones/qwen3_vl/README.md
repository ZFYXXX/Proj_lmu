# Qwen3-VL backbone

This package is the platform-native Qwen3-VL boundary. It uses the Transformers
implementation and does not vendor Qwen model source code.

Owned here:

- processor conversion for text, image, and video conversations;
- the Qwen3-VL `image_patch_size=16` and video-metadata contract;
- native `ForwardState`, image/video token spans, and invariant checks;
- dense/MoE loading plus language, vision, merger, and DeepStack handles;
- prepare-time, shape-preserving replacement of selected language-model
  attention and FFN modules through the platform runtime session;
- text generation and opt-in logits/hidden-state/attention/KV-cache traces via
  the generic `text_*` fields, with explicit rejection of Omni-only trace fields;
- explicit rejection of audio, Talker output, streaming, and unlegalized token
  mutation.

## Video compression boundary

The independent video bridge is pinned to the Transformers 4.57.3 provider
shape. It accepts batch 1 with exactly one video and no simultaneous image,
then exposes the native post-merger main features plus every DeepStack feature
branch through one shared video payload. A callback must return a
`Qwen3VLVideoCompressionResponse`: its main `VideoCompressionResult` and all
DeepStack branches must declare the exact same source groups and result
cardinality. A main-stream-only result is rejected.

The original `video_grid_thw` and provider timing kwargs remain untouched for
vision encoding and native M-RoPE construction. Compressed features are first
scattered at representative source placeholders; the language boundary then
gathers those native M-RoPE anchors and compacts placeholders, attention mask,
ignored labels, physical cache positions, and `visual_pos_masks` together.
Every DeepStack branch keeps one row per `True` visual position. Generation
publishes an exact rope delta after prefill and repairs dense decode inputs by
the same physical deletion count. Text and image calls delegate unchanged, and
the image and video provider bridges cannot patch the same model instance.

This facade/legalizer contract alone does not activate an algorithm SupportUnit.
The nine separate video variants have their own real-checkpoint inference and
minimal training/reload evidence; see the [video status](../../../docs/support/video_compression/status.zh.md)
for the validated scope. These checks do not establish paper-quality reproduction,
convergence, or benchmark improvement.

Not owned here:

- benchmark datasets or metrics, which stay in lmms-eval;
- templates, collators, Trainer, optimizer, LoRA, or checkpoint lifecycle, which
  stay in ms-swift;
- Qwen3-VL model implementation, which stays in Transformers;
- video decoding implementation, which stays in `qwen-vl-utils` and its selected
  reader (torchcodec on the validated remote environment).

The project validates and pins Transformers 4.57.3 and `qwen-vl-utils==0.0.14`;
newer versions require their own compatibility checks.
It has text output only; Qwen2.5-Omni Talker controls must never be forwarded to
this backbone.

## LLM Operation replacement

The two plugins target semantic handles such as `llm.layer[0].attn` and
`llm.layer[0].ffn`. Their YAML stages use `execution_phase: prepare`: the runtime
session installs the structural replacement after checkpoint loading and before
`generate()` or training `forward()`. Uninstalling the session restores the
original modules.

Available attention strategies are `attn_linear_attention`, `attn_cnn`,
`attn_identity`, `attn_lolcats_linear_window`, `attn_nha`, and
`attn_sparse_window`. Available FFN strategies are `ffn_low_rank`,
`ffn_identity`, and `ffn_two_layer_mlp`. All preserve the hidden-state I/O shape,
but none of these replacements implies baseline numerical equivalence.
The identity strategies return their module input unchanged; the decoder still adds
its residual, so this is not the original attention/FFN computation or a zero-output skip.

Validate a definition and inspect its launch plan without loading a model, for example:

```bash
python -m mllm_comp.cli.main eval --plan_only \
  --config configs/eval/lmms_eval/qwen3_vl_linear_attention.yaml

python -m mllm_comp.cli.main eval --plan_only \
  --config configs/eval/lmms_eval/qwen3_vl_cnn_attention.yaml

python -m mllm_comp.cli.main eval --plan_only \
  --config configs/eval/lmms_eval/qwen3_vl_ffn_low_rank.yaml
```

Run the real-image generation smoke, including prefill and cached decode:

```bash
export QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export CUDA_VISIBLE_DEVICES=0

python scripts/diagnostics/validate_qwen3_vl_attention_replacement.py \
  --model_path "$QWEN3_VL_MODEL_PATH" --strategy attn_linear_attention \
  --report_path outputs/qwen3_vl_linear_attention.json

python scripts/diagnostics/validate_qwen3_vl_attention_replacement.py \
  --model_path "$QWEN3_VL_MODEL_PATH" --strategy attn_nha --window_size 64 \
  --report_path outputs/qwen3_vl_nha.json
```

The direct diagnostic records the installed class, target handle, first-call
input/output tensors, an independent reference comparison, generated text, and
peak GPU memory. The lmms-eval wrapper accepts any of the nine strategy names:

```bash
bash scripts/eval/lmms_eval/qwen3_vl_attention_replacement.sh attn_sparse_window
```

The `.shared_llm` replacements are reapplied from YAML when loading the original
checkpoint; they are not exported as standalone Transformers checkpoints.
Benchmark quality must be measured separately for every replacement strategy.

The same plugins also expose `.image_llm` variants for Qwen2.5-VL and Qwen3-VL,
with separate image/text execution and a dedicated save/reload path; see
[modality split](../../modality_split/README.md). Qwen3-VL also has bindings for
[14 new comparison method families](../../../docs/support/imagellm/comparison_methods.zh.md).
These additions retain `declared` SupportUnits and do not inherit the historical
image-method training evidence. Comparison token mappings also update DeepStack.
