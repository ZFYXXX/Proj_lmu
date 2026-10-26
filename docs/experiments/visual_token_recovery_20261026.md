# Budget-neutral information recovery, 2026-10-26

## Literature and scope

User's supplied budget-compression list is the starting point, not executable
instructions. Re-read AdaptMerge §3.2.2 (ALVTM) and TRIM §3.2; re-read SpecFlow
implementation at commit754c563908fa3ee80ce43cc0f8c8181fbfee49b9.

- AdaptMerge: https://aclanthology.org/2025.findings-emnlp.387.pdf
  Retain text-relevant visual tokens; average unselected features into their
  most similar retained token. Full method also changes the vision encoder and
  uses Gemma/SigLIP aligned text features: we are NOT reproducing it.
- TRIM: https://aclanthology.org/2025.coling-main.508.pdf
  Adds an unselected-token aggregate. Not chosen here: an extra token must
  consume K128 budget and needs a new Qwen M-RoPE coordinate contract.
- SpecFlow code: https://github.com/Lzy-dot/SpecFlow/blob/754c563908fa3ee80ce43cc0f8c8181fbfee49b9/llava/model/utils.py
  Budget includes mean/farthest coreset sinks (126+2, not128+2). Those are not
  lossless and no ready Qwen boundary adaptation was verified in that commit.

Exa official-code search requested5 results; no official AdaptMerge code was
confirmed. AdaptMerge transfer is based on its primary paper, not a claimed
source-level port. SpecFlow core source was read, not executed.

## Frozen comparison

Keep the completed learned-word protected policy and its exact selected source
indices. No retraining, new query encoder, dynamic K, noise, OT or sinks. Both
new arms output min(128,N), preserve selected order, original representative
M-RoPE and all text coordinates. Only CONTEXT representatives are updated:
protected tokens remain bitwise original. Input pixels retain native dynamic
resolution; no forced resize. All answer generation is answer-free and goes
through MLLM-Comp eval with the existing native scoring and generation recipe.

1. `neighbor_mean`: assign every discarded token to the most cosine-similar
   context token, then average the representative and its assigned population.
   This is a simple recovery reference, not official AdaptMerge.
2. `query_gated_residual`: same assignment, positive squared cosine times
   sigmoid(centered question residual / visual-score std) gives a source weight.
   Add the weighted mean residual divided by ASSIGNED COUNT, with a fixed .25
   cap. Weak query scores are not rescaled by their own tiny standard deviation.
   Nonpositive affinities write nothing; protected tokens receive nothing.

For source i assigned to context j:

    w_i = max(cos(x_i,x_j),0)^2 * sigmoid((r_i-mean(r))/std(visual_prior))
    y_j = x_j + .25/count_j * sum_i w_i*(x_i-x_j)

The effective blend is bounded by .25 independently of cluster population.
The gate is not trained or causally calibrated. It may discard useful unique
information, and pooling may dilute digits/text. Several source positions are
represented at one original coordinate: not an exact positional recovery.
No conservation of attention mass, information, answers or first moments is
claimed. If it fails, report it; do not tune the cap against these test scores.

## Execution and acceptance

- CPU kernel tests: independent formula, BF16/FP32, exact coverage, no duplicate
  source routing, protected identity, bounded blend, determinism, failures.
- Tiny native Qwen: dense identity, zero-recovery/protected logit parity,
  recovered-feature injection, original inputs, labels, M-RoPE and KV decode.
- 4 arms ×3 tasks ×8 requests =96 engineering smoke requests, including existing
  protected parity and zero-blend parity. All before any full evaluation.
- New arms only × ChartQA2500 + OCRBench1000 + AI2D3088 =13176 full requests.
  Reuse pinned dense/oldMLP/visual/question/word/protected/noise results; no500
  development pass, no automatic Git commits, no overwriting old artifacts.
- One GPU1 process, authorized normal mode with32768MiB free at launch, no
  eviction or termination of other users. Preserve failed attempts; cell-level
  restart validates source/config/data/checkpoint hashes before skipping.
- Exact historical source count, retained indices and protected set checked
  for EVERY request. Trace source-to-context routing/weights, actual K, blend,
  representation change and decode positions. Fail explicitly, no silent skips.
- Image-cluster paired bootstrap10000, native scores and AI2D existing leading
  option auxiliary. Compare to same-selector/no-recovery AND previous best
  methods (including OCR noise689). Do not pick a fictional per-dataset model.
- Record latency/memory traces. Shared GPU timing is descriptive, not a fair
  speedup claim versus historical controls. Full results are exploratory.

Outputs: `outputs/analysis/visual_token_recovery/<run_id>/`.
An independent hash-frozen execution worktree isolates running source without
committing small changes or modifying previous checkpoints/results.
