# Qwen2.5-VL-7B: unified platform evaluation with proportional budgets

SUPERSEDED: the user clarified that dynamic-resolution adaptation does not
authorize changing the budget to 25% or adopting Bifocal's layer-average
criterion. The ratio run was stopped and preserved as exploratory output.
See `qwen25_7b_projector_dynamic_k128_20261021.md` for the corrected protocol.
The text below records the superseded proposal, not the active experiment.

The primary comparison now runs exclusively through
`python -m mllm_comp.cli.main eval --config ...`. The scheduler invokes that
CLI and checks its native sample logs and runtime traces; it has no model,
dataset iteration, generation or scoring loop of its own.

The previous custom K128 comparison is superseded. All completed ledgers are
retained under its output root, with `superseded.json` explaining the protocol
revision. Those partial results are exploratory, not the primary matrix.

## What the Bifocal budget means

For image i, let N_i be the original post-merger visual-token count and n_i,l
the number of visual tokens actually processed by decoder layer l. Define

    rho_i = sum_l n_i,l / (L * N_i).

A projector-only compressor has a constant K_i at every decoder layer, so
rho_i = K_i/N_i. Each image receives K_i = round(rho*N_i), bounded to [1,N_i].
No resolution change is used to manufacture a fixed original token count.

Bifocal's complete method also prunes within the decoder. Its server Qwen
implementation solves

    rho * L = L2 * r1 + (L-L2) * r2.

At L=28, L2=15, r2=32/576 and rho=.25, r1 is approximately .4185185.
The complete method therefore enters the LLM with about 41.85%, not 25%, and
later reduces to about 5.56%. Integer rounding and its minimum keep count mean
the realized layer-average must be measured rather than assumed.

Sources inspected:

- [Bifocal paper](https://arxiv.org/html/2610.18399), including Stage 2 and
  layer-averaged comparisons.
- [Official repository](https://github.com/EIT-NLP/Bifocal), pinned while
  inspected at `e50d76571ac933e641d8259f0f5e6c80953d9d85`.
- Server Qwen implementation:
  `/home/YinPeiran/vlm_model/qwen2_5_vl/modeling_qwen2_5_vl.py`,
  `HopeConfig` and `set_hope_method`, especially lines 116-132 and 1554-1572.

LLaVA-1.5 uses a fixed 576-token single-image input, while LLaVA-NeXT AnyRes
also has image-dependent token counts. The distinction is model-specific.

## Matrix and scope

Primary retention: 25%. Configurations for 50% and 15% are prepared, but only
25% is currently queued. Each method uses all ChartQA test 2500, AI2D test
3088, OCRBench test 1000 and DocVQA validation 5349 questions (11937 total).

Methods: dense, the existing supervised feature-anchor residual MLP,
CDPruner, DPPruner, EADP, HiPrune and Bifocal **Stage 1 only**. The last row is
a projector-only control and must not be labeled as the full Bifocal paper
method. The current framework Bifocal binding does not implement its complete
two-stage decoder policy.

The MLP was trained at K128. Applying it with proportional, image-dependent K
is an evaluation of budget generalization; there is no new training. Trained
and training-free methods remain explicitly identified.

All methods use the same official-content-matched Qwen2.5-VL-7B checkpoint,
BF16/eager, native dynamic resolution and platform task routing. The native
platform Qwen generation wrapper inherits checkpoint repetition penalty 1.05
and uses tokenizer EOS. Because generation and budgets have changed, dense
and MLP are rerun rather than reusing the earlier custom-run predictions.

AI2D retains its upstream filter and exact-match metric, and also reports the
separately labeled leading-option metric motivated by the prior post-hoc
format audit. OCRBench uses its upstream per-question score with a pure mean
aggregation, equivalent to official sum/1000 on all 1000 questions. DocVQA uses
native ANLS; ChartQA uses native relaxed metrics.

## Implementation and checks

- `mllm_comp/compressors/comparison/qwen_projector.py`: optional per-image
  retention ratio, generation/cache position repair, observed per-layer
  visual-token counts.
- `mllm_comp/compressors/comparison/anchor_adapter.py`: registered generate
  binding for the pinned trained MLP, centroid M-RoPE and native KV lifecycle.
- `configs/eval/lmms_eval/tasks/qwen25_7b_projector_ratio/`: local complete
  splits with native task prompts/scoring.
- `scripts/eval/lmms_eval/run_projector_ratio_platform.py`: two-card CLI
  scheduling, smoke gate, immutable configurations, native output/count checks.

Before launch: 102 tests passed, one optional test skipped; all seven primary
configurations passed `mllm-comp eval --plan_only`. Every method first runs
eight examples from each task for implementation acceptance. The entire
seven-method smoke matrix must pass before any primary full-split run starts.

Output:
`outputs/eval/qwen25_vl/projector_ratio_platform_20261021_v1/`.
It contains reviewable YAML configurations, protocol, supervisor manifest,
native platform manifests, native lmms-eval predictions/results and per-request
token traces. No claim equates layer-average token reduction to FLOP or latency
reduction; efficiency needs the actual layer profile and measured overhead.
