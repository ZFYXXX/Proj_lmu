# Matched soft versus straight-through Top-K, Qwen2.5-VL-7B

Purpose: separate the training forward-gate choice from scorer architecture,
initialization, data/order, optimization, and hard-deployment budget. This is NOT
a clean comparison of “Top-K operator versus no Top-K.”

Both arms freshly initialize the same LN(D)-Linear(D,256)-GELU-Linear(256,1)
scorer, seed42; frozen base; same original 2560-image manifest, global batch8,
320steps, AdamW2e-4, weight decay.01, warmup16, cosine, clip1. Both use CE +
0.5 teacher KL + BCE on genuine continuous probabilities (weight.1→3).

* soft: forward applies sigmoid-threshold probabilities of mass min(128,N).
* ste: forward applies exact hard 0/1 top-min(128,N); backward uses exactly the
  same sigmoid-threshold Jacobian via `hard + (soft - soft.detach())`.

Both train with the entire N-position sequence. Zero embeddings still have
positions and participate in attention, so this is NOT hard physical pruning.
Both deploy the existing identical hard source-order original-token selection,
with unchanged sparse visual M-RoPE, text positions, placeholder/mask/cache.
No old selector code, format, hash rule, or checkpoint is changed.

Execution: isolated hash-frozen uncommitted worktree, no commits/tags; GPU0
sequential arms, preserving the existing GPU1 experiment. Each arm first passes
real7B identity/decode and uninterrupted4 versus restarted2+2 exact parameter /
optimizer checks. Initialization digests and all 320 batches must match.

After fixed step320: native framework smoke8 per task, development250 independent
images per task, then full ChartQA2500 and OCRBench1000. An initial-selector
reference is also evaluated. No tuning or checkpoint selection on these data;
development cohorts were already inspected in earlier experiments. Evaluation
uses MLLM-Comp CLI, existing native prompts, dynamic resolution, greedy decoding,
repetition penalty1.05, ChartQA16 / OCRBench128 maximum generation tokens.

Results: native task scores, soft−STE, each trained arm−initial; paired
10,000-resample image-level bootstrap, matching prompts/targets and source token
counts. Full ChartQA repeats images, so headline question-weighted differences
and equal-image-weight CI estimands are explicitly distinguished. Single-seed
exploration cannot establish training-seed robustness. Overlapping intervals
are not evidence of equivalence.

Interpretation:

* Both beat initial similarly: supports learning a selector, no detected extra
  benefit from the soft forward gate under this recipe.
* Soft reliably beats STE: supports this continuous-forward training recipe,
  not every differentiable Top-K algorithm or a standalone innovation claim.
* STE beats soft: suggests hard-forward alignment helps; zero-mask versus true
  removal remains untested.
* Only training loss improves: training/deployment mismatch; no task-quality
  benefit established.

Runner: `scripts/train/qwen25_vl/run_topk_mask_ablation.py prepare|pipeline`.
Use `MLLM_TOPK_HASH_LOCAL_SOURCE=1 MLLM_TOPK_NORMAL_GPU=1`, the pinned Python,
and a new owned output root. `pipeline --gpu 0` resumes completed hash-validated
stages; no silent source/data changes or missing predictions are accepted.
