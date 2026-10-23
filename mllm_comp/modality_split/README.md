# Dense Qwen modality split

This package installs differentiable physical gather/operator/scatter wrappers on
selected decoder attention and FFN modules. It supports the Transformers 4.57.3
dense Qwen2.5-VL and Qwen3-VL models, including their text-model classes.

## Platform pipeline entry

The existing `qwen3_vl_attention_replacement` and `qwen3_vl_ffn_replacement`
algorithms now also expose `.image_llm` variants on both Qwen families. Their
`prepare` bindings are compiled together by the platform into one modality-split
session. Use `modalities: [image]` in the selected stages; the `.shared_llm`
variants remain a separate Qwen3-VL-only whole-module replacement path.

Runnable training recipes are provided for [Qwen2.5-VL full SFT](../../configs/modality_split/qwen25_vl_sft.yaml),
[Qwen3-VL full SFT](../../configs/modality_split/qwen3_vl_sft.yaml) and
[Qwen3-VL LoRA](../../configs/modality_split/qwen3_vl_lora_sft.yaml).
They use the public `mllm-comp train --config ...` entry and native ms-swift;
the optional [shell wrapper](../../scripts/train/modality_split_sft.sh) forwards
arguments to that same entry. Set each recipe's model, dataset and output variables first.

The exact `.image_llm` SupportUnits remain `declared`. Recipe availability and
the CPU training-bridge tests below do not certify a complete GPU Swift CLI
training/resume run. For direct Python use, install the following wrappers
before creating the optimizer.

## Direct Python entry

```python
from mllm_comp.modality_split import (
    install_modality_split, native_state_dict,
    save_modality_split, load_modality_split,
)

spec = {
    "schema_version": 1,
    "backbone": "qwen3_vl",  # or qwen25_vl
    "attention": {"layers": [0], "image_operator": "window", "config": {}},
    "ffn": {"layers": [0], "image_operator": "two_layer_mlp",
            "config": {"bottleneck_dim": 1024}},
}
session = install_modality_split(model, spec)  # BEFORE creating optimizer
```

Both branches are optional, but at least one selected layer is required. Image
attention: `full`, `skip`, `window`, `cnn`, `linear`, `lolcats`, `nha`. Image FFN:
`native`, `skip`, `low_rank`, `two_layer_mlp`. Config keys for non-native operators
are those of the existing Qwen3 replacement constructors. Window accepts only
`local_left` (default 127) and `debug`: causality, current-token inclusion, and
absence of prefix/stride/global extra keys cannot be changed.

Text attention uses the original checkpoint's projections, Qwen3 Q/K norms
(absent for Qwen2.5), rotary coordinates, scale, causal rule and dropout, with
PyTorch SDPA. Image full uses that same calculation over the image subsequence.
All other image mixers call the unchanged existing Qwen3 numerical kernels.
Qwen2.5's multimodal rotary channel sections are explicitly assembled into
standard cosine/sine tensors before those kernels are called. Original sequence
positions are never replaced by compact indices.

Each sample and modality is gathered independently; no padding is projected.
Skip returns zero branch delta without image projections or image cache slots.
The decoder owns residual additions. Cross-modal edges are excluded **at selected
attention modules**; native/unselected layers can still mix modalities. Previous
images in the same sample are part of the causal image subsequence.

## Inputs and training

Ordinary conditional-generation calls infer image positions from `input_ids ==
model.config.image_token_id`. A conditional-generation model called with
`inputs_embeds` requires `modality_image_mask` of shape `[batch, current_length]`,
even if the inputs are all text. Embedded-only calls also remain responsible for
the underlying model's required position IDs/padding metadata. Direct text-model
calls without the image-token config default to all text, or accept the same
explicit image mask. Pass optional arguments by keyword.

Only unpacked samples with absent or binary 2D padding masks are supported.
Video tokens/pixels/grids, packing metadata, reset text positions, and 4D/custom
attention masks fail closed. Do not concatenate packed sequences while hiding
their boundaries: such boundaries cannot be inferred from arbitrary embeddings.

Layout is cloned per forward and passed through decoder kwargs. Each decoder
forward binds its own `ContextVar`, including during gradient-checkpoint
recomputation. There is no mutable last-mask state. Registered new image weights
are visible to optimizers. Full-SFT uses `use_cache=False`; cached recurrent
training is not supported. Train/eval mode is inherited from the original model.

## Generation and caching

Dynamic generation caches maintain a global original-sequence clock and separate
compact per-layer/per-sample/per-modality caches. Native layers keep their normal
KV storage. CNN/linear/LoLCATs/NHA reuse original kernel caches/recurrent owners
inside each compact branch. Greedy and beam generation are tested. Start from no
cache or an empty `DynamicCache`; a nonempty unrelated or static cache is rejected.
Speculative/cropped generation and post-cache batch expansion are unsupported.
`cache.reset()` clears compact/recurrent state, rebuilds empty native storage,
and resets the global clock; reuse then behaves like a fresh cache. Native
sliding-attention configurations are rejected at installation because truncated
native cache offsets are not supported by this protocol.

Generation with ordinary `input_ids` is supported. The explicit
`modality_image_mask` route is for direct forwards/training only: HF generation
currently rejects that keyword during argument validation, so embedded-only
generation with an explicit mask is not supported.

## Checkpoints

```python
model.save_pretrained(directory, state_dict=native_state_dict(model))
processor.save_pretrained(directory)
save_modality_split(model, directory)

restored = NativeQwenClass.from_pretrained(directory)
session = load_modality_split(restored, directory)
```

`native_state_dict` exports **trained** native weights under original HF names,
excluding newly registered image operators. `modality_split.json` is the exact
bare spec; `modality_split.pt` stores the entire state of each selected wrapper,
including trained native and image replacement weights. The sidecar alone does
not contain unselected modules or the visual encoder; save the native checkpoint
as shown for complete model restoration. Loading installs before strictly
restoring selected states. `session.uninstall()` reinstates the trained original
modules and their original forwards; image-only parameters are removed.

## Validation boundaries

`tests/integration/test_modality_split_cpu.py` covers both Qwen families with
native tiny models: all-text identity, window attention with low-rank image FFNs,
pixel-input backward, gradient-checkpoint replay parity, trained native/sidecar
save and reload, mixed-input incremental decoding, cache reset, greedy/beam
generation, and the four-stage sample pipeline's training bridge and restoration.
Direct incremental forwards must supply native `cache_position`; `generate`
manages it automatically.

This CPU suite does not certify all operator combinations, CUDA/BF16 numerical
parity, full pretrained benchmarks, or complete native Swift CLI training/resume.
Quantized, tensor-parallel, compiled and distributed execution are unverified.
