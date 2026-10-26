# What does the question-conditioned selector actually use?

Inference-only diagnosis on the original question-context step320 checkpoint,
NOT the noise-gate or OT model. Preserve all old checkpoints/hash formats,
historical outputs and frozen source worktrees. Do not train or commit changes.

## Intervention

The real LLM always sees the original image, question, options and task prompt.
Only the selector's text embedding input changes. All conditions retain
min(128,N) original post-merger tokens, source order, original visual/text
M-RoPE, the existing placeholder/mask/label/cache lifecycle.

- True: original last-user embedding tokens. Reuse the existing 6588 full
  predictions; three new eight-request smokes must reproduce predictions and
  selected indices exactly.
- Matched wrong question: a frozen seed42 permutation within each dataset;
  OCRBench additionally matches its question_type. Require a different image,
  maximize different question token sequences, then minimize total token-length
  mismatch with seeded tie-breaking (linear assignment). Do not use answers or
  model performance in matching. This is not uniform random permutation.
- Generic: one predetermined instruction per dataset. No entities/options from
  the actual question enter the selector. These conditions are not length
  matched, and the AI2D selector loses options too; do not call the contrast a
  pure entity/order intervention.

Some OCR questions are identical within a task subtype. Such swaps are marked
non-informative, NOT counted as evidence that question information is unused.
Singleton-image strata are explicitly unchanged. Retain all full samples and
report the changed-question subgroup separately.

Native samples supply prompt strings; every re-tokenized original question must
match its historical selector question-ID hash before any GPU job. Manifest
contains only token IDs, donor/image IDs, task/stratum and matching flags, never
answers, predictions or scores. Request identity is scoped/reset with ContextVar
so evaluation sort order and repeated identical prompts cannot mix up donors.

## Metrics and interpretation

ChartQA2500 / OCRBench1000 / AI2D3088. Same Qwen2.5-VL-7B BF16/eager/native
resolution/JPEG85, native task prompts, greedy, repetition penalty1.05 and
generation limits16/128/16. All generation/scoring uses `mllm-comp eval`.
No 500-example development pass. Smoke:72 requests; new full:13176 requests,
true full6588 reused. Use a separate hash-frozen execution worktree on GPU1.

Record native score differences with 10k image-cluster paired bootstrap,
selection overlap/Jaccard, changed selected sets, answer changes, rescue/harm,
and unchanged-selection/changed-question subgroups. Check every treatment's
diagnostic true-condition selected set against the historical reference. If
the same selected tokens produce different answers, fail rather than attribute
the difference to question conditions. Emit all per-image QA case records;
do not store full hidden states or tune a new method using these results.

Interpret cautiously:

- Different tokens do not by themselves imply better task grounding.
- Same tokens can be appropriate for different questions on the same evidence.
- True beats matched wrong supports useful current-question conditioning, not
  proof of object/relationship reasoning.
- Similar generic performance can reflect weak use, redundant evidence, an
  insensitive benchmark, or limited question variation; report intervals.
- Original question input is a mean of frozen raw embeddings and is
  mathematically permutation invariant (apart from numeric roundoff), not a
  contextual question encoder. This architectural fact alone does not prove
  that word-order loss causes the observed task failures.
- This is one fixed model and one matching seed on previously inspected full
  datasets; no new independent-confirmation or cross-seed stability claim.
