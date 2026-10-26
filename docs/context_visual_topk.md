# Question-conditioned selector v1 (2026-10-24)

## Scope and prior art

Implement question conditioning first. No token recovery, dynamic K, decoder
pruning, Qwen3.5 adaptation, or new Top-K operator ablations in this experiment.
Retain K=min(128,N), native dynamic resolution, original token contents/order
and sparse M-RoPE. Old aggregation MLP and visual selector files/checkpoints
remain unchanged. No automatic Git commits; execution uses a hash-pinned
isolated worktree including all uncommitted source changes.

These are existing motivations, not a novelty claim:

| Source | Relevant idea | What this experiment does NOT reproduce |
|---|---|---|
| [MMTok](https://arxiv.org/html/2508.18264v2), [code](https://github.com/Ironieser/MMTok) | Cheap instruction/visual coverage using embeddings | Its multimodal coverage optimization and complete official pipeline |
| [CDPruner](https://arxiv.org/abs/2506.10967) | Instruction relevance alongside visual diversity | Conditional DPP or its selection objective |
| [Task-Related Token Compression](https://arxiv.org/html/2506.01097) | Task-dependent relevance can guide a trained selector | First-layer attention, explanation labels, or its CNN |
| [Hi-Lo Prune](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Hi-Lo_Prune_Look_at_What_Youll_Lose_before_Pruning_with_CVPR_2026_paper.html), [code](https://github.com/sealost/Hi-Lo_Prune) | Absorb information before removal | Early-decoder attention fusion; recovery is a separate future experiment |

Research via Exa primary-paper search/fetch on 2026-10-24. A new MLP built from
these motivations is not equivalent to reproducing any of the above papers.
Question conditioning and information recovery alone are not new contributions.

## Architecture

Use the frozen LLM input embedding table to embed only the last complete user
turn, after removing image sentinels. Questions include legitimate task prompt
formatting/options; assistant answers, system turns and generated text are not
inputs. Training extracts from the separate answer-free prompt. Batch size one,
single image only; malformed or empty user spans fail explicitly.

For post-merger visual x and prompt embeddings q:

```
h_i = Linear_1(LayerNorm(x_i))
c   = tanh(W_context LayerNorm(mean(q)))
s_i = Linear_2(GELU(h_i + c))
```

The visual scorer is warm-started from the old fixed step320 soft selector.
W_context starts at zero: both arms initially match its scores exactly.
Context LayerNorm has no affine parameters. At D=3584, width=256 this adds
917,504 parameters to 925,185, for 1,842,689 trainable parameters. There is
one extra D-by-256 projection per image/question, not per visual token; no
extra decoder forward is required. Measured wall time is still the authority.
All context features detach from the frozen backbone.

The matched image-context control replaces mean(q) with mean(x), leaving
architecture, initialization, parameters, optimizer and extra training budget
identical. The direct comparison therefore tests question versus image-global
conditioning, not every possible query-conditioned architecture. The original
frozen visual selector and old aggregation MLP are also retained as references.

## Fixed training and execution

Both arms start from the same previously trained visual checkpoint and receive
one additional pass over the SAME 2560-image manifest, in the same order:
TextVQA-train 2048 plus ScienceQA-train 512. Validate content-hash nonoverlap
against ChartQA, OCRBench AND AI2D; any overlap stops preparation, no resampling.
New optimizer, seed42, batch8, 320 additional steps, AdamW 2e-4/weight-decay .01,
16-step warmup/cosine, clipping1, same CE + .5 teacher KL + annealed mask BCE.
No selection of checkpoints using test results. Existing continuous Top-K soft
full-sequence training and hard source-order compaction remain unchanged.

CPU gates test gradients, answer exclusion, changing question/request isolation,
identity logits, budget, decode coordinates, strict independent checkpoint
format, exact optimizer/RNG restoration, and native config resolution.
Real 7B gates compare uninterrupted4 with process-restarted2+2 bitwise, including
optimizer moments and scheduler. Each arm runs on one available GPU, normal
unthrottled mode; no termination or modification of other users' processes.
Only owned output/cache paths are written; shared models/data stay read-only.

## Evaluation and boundaries

All generation/scoring uses `mllm-comp eval`, not a separate custom loop.
Tasks: ChartQA2500, OCRBench1000, AI2D3088. Development:500 independent
content-hash-selected images per task, without targets/predictions in sampling.
Eight-request-per-task smoke precedes development and full evaluation.
Groups: dense, old MLP, frozen visual selector, image context, question context.
Greedy, repetition penalty1.05; max generation tokens16/128/16 respectively.
AI2D uses upstream multi-choice extraction and exact_match; leading-option
accuracy remains a separate supplementary metric. OCRBench reports its native
normalized score, with full split sum interpretable on the /1000 scale.

Record predictions, native scores, original/retained token counts, selection
indices, question token count/hash, generation time/memory, source/config/data
hashes, base identity and checkpoint provenance. Require every request and
trace; failed attempts remain and do not silently count as complete.
Paired bootstrap:10,000 independent image-cluster resamples, question-weighted
estimand for BOTH headline difference and confidence interval (repeated QAs are
not independent images). This is single-seed exploratory method development;
development overlaps full evaluation, not independent confirmation.

A null result may indicate this cheap mean-embedding conditioner is weak. It
does not invalidate question conditioning generally. After results, separately
test discarded-information recovery at the same total K, so its benefit is not
confounded by conditioning, increased token count, or additional training.
