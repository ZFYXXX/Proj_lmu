# Fixed-K128 protected-evidence diagnostic

This is the first step after the completed word-level residual experiment.
It does not retrain a model or implement mediator/recovery/dynamic budgets.

## Frozen experiment

Qwen2.5-VL-7B, BF16, eager attention, native dynamic resolution and existing
task prompts/generation/scorers remain unchanged. Both arms load exactly the
completed word-level step320 checkpoint and its frozen visual prior.

- `learned_word`: pure learned question residual is the relevance priority.
- `raw_cosine`: maximum cosine between original post-merger visual features
  and cleaned answer-free question word embeddings is the relevance priority.

In both cases, first protect a relevance-ranked set. Start with20 tokens,
compare later candidates to the SAME initial set, count novelty>.2 events
cumulatively, stop on the third event or at64 tokens. Allocate the remaining
K128 budget by softmax(frozen visual scores) times minimum cosine distance to
the current retained set. Ties use source order; restore native order at output.
For N<=128, preserve all N original tokens without padding or forced changes.

This is an AnchorPrune-inspired diagnostic, NOT the official method: the
global prior is our learned visual scorer, not the official vision-attention
prior. The policy changes protection and complementary selection together;
it cannot isolate those two effects individually. There is no novelty claim.

## Checks and execution

1. Numerical unit tests, tiny native Qwen forward/labels/generation/cache and
   identity checks; original model inputs and checkpoint parameters preserved.
2. Same eight requests/task: historical unified-word parity, learned protection,
   raw protection. Nine smoke cells,72 requests. All must pass before any full.
3. Two new arms × ChartQA2500 + OCRBench1000 + AI2D3088 =13176 requests.
   Existing dense/oldMLP/visual/global-question/word-unified results are pinned
   and reused, never re-evaluated at full scale. No500-image development pass.
4. GPU1 only, one process, normal authorized memory-headroom gate. No eviction,
   no GPU0 use. Preserve logs and failed states; never silently skip examples.

All generation/scoring uses MLLM-Comp's eval entry point. Original source visual
content, source order, text coordinates and sparse visual M-RoPE are retained
through the existing legalizer. Trace every anchor, context count, retained
index, actual budget and first decode coordinate.

Question audits run only the selector with wrong/generic questions; the actual
LLM always receives the original question. Token exchange is not proof of
semantic evidence coverage; no verified region labels are presently available.

## Results and interpretation

Report native task scores, per-image clustered paired bootstrap10000 vs pinned
controls, anchor sizes and question sensitivity. OCRBench points are1000 times
the mean native score. Old speed measurements are not matched timings.

If protection improves task scores, investigate evidence-versus-context
competition before enlarging the network. If raw matching outperforms learned
matching, investigate learned relevance/supervision. These are hypotheses,
not causal conclusions based on score changes alone. No test-driven tuning
of the frozen20/.2/3/64 settings or checkpoint selection is planned.

Code, input and configuration hashes plus actual import paths are recorded in
an isolated execution worktree. No automatic Git commit; no old artifact edits.
