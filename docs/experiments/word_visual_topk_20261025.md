# Word-level question matching residual, v1

Keep Qwen2.5-VL-7B, original dynamic-resolution processor bounds, K=min(128,N),
native prompt/scoring/generation, original visual content/order/M-RoPE and cache
legalizer. No OT, noise, recovery, dynamic budget or Qwen3.5.

## Model

Freeze the previously trained visual-only scorer (fixed step320). Reuse its
256-dimensional hidden states h and raw prior scores s. Project h and each
frozen question-word embedding to 128 dimensions using trainable bias-free
linear layers. Normalize projected vectors, form cosine similarities [N,T],
then aggregate each visual row with temperature .1 log-MEAN-exp. Center the
relevance without dividing by its own variance; multiply by detached prior
score std and a learned positive mix initialized at1, and add to the prior.
No per-query min-max or z-score amplification, no transport matrix.

The two projections and scalar mix add491,521 trainable parameters at D3584.
The925,185 visual-prior parameters and the complete backbone remain frozen.
This is token-specific matching, NOT a contextual question encoder: word order
and relation understanding remain limitations, not solved claims.

Strip only exact known answer-formatting SUFFIX token sequences. Preserve
negation, relations, digits, units, and options. Extract only last-user,
answer-free tokens; fail on empty cleaned questions. Cleaning patterns are
stored in the separate checkpoint format and pinned with the source hash.

## Training and absent semantic labels

Same2560 independent-image training manifest (TextVQA2048 + ScienceQA512),
same order, seed42, batch8, AdamW2e-4, weight decay.01, warmup16, cosine,
gradient clip1, fixed320 steps. Only the new matching parameters update.
Retain existing continuous Top-K full-physical-sequence training with answer
CE + .5teacherKL + the historical .1→3 self-hard-mask BCE. That BCE is NOT
verified token-importance supervision. No forced different-question selection
loss is added.

An optional evidence loss accepts ONLY verified human/dataset positive and
negative region boxes, with reviewer, image hash, normalized coordinates and
disjoint native-grid token supports. Empty evidence manifest means this loss
is inactive (weight contribution0), visibly recorded in the run. At preparation,
no independent verified annotations are currently available; do not synthesize
them from attention/own scores or present this as evidence-supervised training.

CPU tests verify per-branch gradients, prior freeze, cardinality/ties/source
order, exact prior-only parity, question-specific toy control, answer isolation,
cleaning, checkpoint integrity/CPU optimizer-RNG restore, native generation,
soft forward/backward and registered placement. Real7B gates compare dense
identity, fixed K decode, and uninterrupted4 vs restart2+2 optimizer/parameters.

## Queue and evaluation

1. Native7B eight-request smoke/task for prior-only control and initial word
   head. Prior-only must reproduce historical frozen-visual token sets/answers.
2. Real training/save/resume gates; fail on invalid or zero branch gradients,
   backbone gradients, frozen-prior changes, position errors or input drift.
3. Independent fixed320-step training, then eight-request smoke/task.
4. NEW trained method only: ChartQA2500, OCRBench1000, AI2D3088, through
   mllm-comp eval. No500-image dev pass, no old/full/random re-runs.

Reuse pinned dense, old aggregation MLP, visual selector and global question
selector predictions for quality comparisons, with image-cluster paired
bootstrap10000. Historical latency is not a matched speed claim. New native
requests retain timing/memory/token metadata. On every prefill, run cheap
selector-only wrong/generic-question audits on the same visual features; do
not regenerate answers for those inputs or replace the actual LLM question.

Export native same-image question pairs for semantic review. These are valid
dataset questions but NOT automatically different-evidence pairs or verified
paraphrases. More exchanged tokens is not a success criterion; semantic
coverage/paraphrase conclusions remain pending independent review. Native
task-quality results are exploratory while that review is pending.

## Isolation

Independent model format/checkpoint/run/source worktree. No automatic Git
commit, no old output overwrite. OnlyGPU1 requested; normal authorized
execution with >=32GiB free launch headroom and >=8GiB step headroom, without
terminating or modifying other users' processes. Shared weights/data readonly.

Prior-art motivation: MMTok https://arxiv.org/html/2508.18264 (word coverage),
CDPruner https://arxiv.org/html/2506.10967v2 (instruction relevance),
SparseVLM https://arxiv.org/html/2410.04417v4 (relevant prompt tokens),
Task-Related Token Compression https://arxiv.org/html/2506.01097 (supervision).
These components do not establish novelty or reproduce those complete methods.
