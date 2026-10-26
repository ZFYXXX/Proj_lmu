# Bifocal Stage 1 / Stage 1+2 on Qwen2.5-VL-7B

Two separately named controls are added at the user's request. All inference
and scoring go through the native MLLM-Comp eval CLI, with the same checkpoint,
revision/content identity, BF16/eager, native dynamic resolution, JPEG85, seed,
task prompts and generation parameters as the existing three-task evaluation.

| Version | Placement | Budget | Interpretation |
|---|---|---|---|
| bifocal_stage1_k128 | post-merger, pre-LLM | min(128,N) throughout all LLM layers | Stage-1 component control; not full Bifocal |
| bifocal_stage12_r25 | post-merger plus decoder boundary 15 | first 15/28 layers round(0.4185185 N), later 13 layers max(8,round(N*32/576)) | Two-stage adaptation, approximately 25% layer-average budget |

The 25% criterion applies ONLY to this two-stage Bifocal reference. It does not
change the fixed K128 budgets of our methods or the other projector controls.
The rows do not isolate Stage 2 at a matched initial or aggregate budget. Report
actual source, Stage-1, Stage-2 and layer-average counts rather than giving the
two-stage version a misleading single K128 label.

Stage 1 retains query-focused fine tokens, image-intrinsic fine tokens and
pooled coarse context. The new optional target_count overrides only its output
budget; its default None preserves historical fractional configurations and
positional argument compatibility. Fixed K matches its original per-image
fractional implementation with s1_frac=K/N on tested rectangular grids.

Stage 2 ranks remaining visual tokens using the last prompt-text query's
head-averaged, visual-key-softmax attention, multiplied by 3x3 occupancy density.
The implementation follows the inspected Qwen reference, with original M-RoPE
coordinates retained. Early-layer KV caches intentionally remain longer than
late-layer caches; each layer uses its own physical cache offset and causal
mask during decode. No base weights, tokenizer, global Transformers source or
other user's files are changed.

The new binding `bifocal_two_stage.image.qwen25_vl` is classified LLM-level,
explicitly reports its projector+decoder behavior, supports eval-time GENERATE
only, and fails on unsupported padding, sliding attention, batch size, full-state
capture or training. Its source-specific language loop is instance-local and
reversible; identity Stage 2 matches the native Stage-1 generation in tests.

References inspected:

- https://arxiv.org/html/2610.18399 (paper and Stage 2).
- https://github.com/EIT-NLP/Bifocal (public release, currently LLaVA-oriented).
- `/home/YinPeiran/vlm_model/qwen2_5_vl/modeling_qwen2_5_vl.py`: read-only Qwen
  schedule and Stage-2 operator reference; its file SHA is recorded in the run.

CPU gates: fixed budgets, native identity, heterogeneous-cache generation,
first/subsequent decode M-RoPE, restoration, density occupancy, and exact
Stage-2 selection against the read-only Qwen reference in FP32 and BF16.
Twenty-three original Stage-1 parity tests cannot run without their distinct
old local reference. They are recorded as skipped, never counted as passed.
This remains a controlled adaptation, not a claim to reproduce paper numbers.

Only this experiment uses `MLLM_COMP_METHOD_PROFILE=full`, because the default
projector-only registry intentionally hides decoder-level bindings. The default
workspace profile and all existing external-control runs remain unchanged.
Native v2 stage-contract validation covers conditioning, preserved modalities,
coupling and spatial alignment for both stages before queueing GPU work.

The scheduler waits for the existing external-control rerun to finish before
taking its GPUs. It runs 8 smoke requests/task/version (48 total) and only after
all pass starts ChartQA 2500, OCRBench 1000 and AI2D 3088 for both versions
(13176 full requests). Every failure stops subsequent queued cells, preserves
artifacts, and marks failure. Existing dense/our-method predictions are reused
only after prompt/target and task-contract checks; no old full result is overwritten.
