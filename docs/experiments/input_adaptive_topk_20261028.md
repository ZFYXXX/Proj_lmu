# Input-adaptive relevance gate, Qwen2.5-VL-7B

The repository already contains token-wise question matching in `word_visual_topk`.
That head uses a single learned global relevance strength for every input. This
experiment adds an image/question-conditioned two-layer gate to that existing
head, avoiding a repeat of the word-matching experiment. Its output still has
K=min(128,N) original tokens, source order and original sparse M-RoPE. The gate
does not predict a token budget, semantic task label, or measured complexity.

## Exact computation and training

Keep the frozen step320 visual scorer and the exact original word-head
initialization. Project its visual hidden features and individual question-word
embeddings into 128 dimensions. As in the old word head, normalized pairwise
similarities use temperature-.1 log-mean-exp pooling over question tokens.

The new gate is Linear(256,64), GELU, Linear(64,1), then 2*sigmoid. Its input is
the concatenation of the mean projected visual tokens and mean projected question
tokens. Initialize the final linear layer to zero, so the gate is exactly one
and all shared initial scores match the old word head bitwise. The final score is
the frozen image prior plus the old learned relevance residual times this gate.
Only the 128-dimensional matching projections, scalar mix and gate are trained.
The gate adds 16,513 parameters, for 508,034 trainable parameters at hidden=3584.
Raw word embeddings do not encode word order; do not claim relational or negation
understanding from this architecture. A changing gate alone is not a useful
method result: it must improve task outputs.

Train on the unchanged 2,560-image manifest (TextVQA-train 2,048,
ScienceQA-train 512), seed42, batch8, 320 steps, AdamW2e-4, weight decay.01,
warmup16/cosine, gradient clip1. Use the same CE + .5 teacher KL + annealed
soft-mask BCE as the fixed-word reference. No evidence annotations are invented.
Continuous Top-K training retains full physical sequence length; deployment
hard-compacts tokens. These two execution modes are recorded separately.
The fixed-word reference received the same data and schedule from the same shared
initial parameters; its complete checkpoint is reused. Old heads and checkpoint
formats remain unchanged.

## Evaluation

Run native MLLM-Comp eval on OCRBench1000, MMStar1500 and AI2D3088 after
8-request-per-task smoke. No extra development pass or test-selected checkpoint.
Each selected dataset is checked for image-content overlap with training before
launch; any overlap stops preparation. Preserve the current dynamic resolution,
BF16/eager, prompts, greedy generation and repetition penalty1.05. Max new
tokens128 on OCRBench and16 on MMStar/AI2D.

Compare dense, old aggregation MLP, visual-only selector, image-global context,
question-mean context, fixed-word relevance, and the new input gate. Audit and
reuse compatible complete results using checkpoint/processor/method/task and
generation contracts plus source/config/artifact hashes. Evaluate only the new
three cells and missing fixed-word MMStar. New outputs include source/retained
counts, exact indices, prompt-only question hash, gate/mix, cache-coordinate
checks, per-request timing, native score and completion status.

AI2D headline uses the previously corrected leading-option accuracy. Preserve
strict whole-output exact matching as a separately labelled diagnostic only.
MMStar's headline is native detailed-category macro average, not micro accuracy;
recompute category denominators in every image-cluster bootstrap draw.
OCRBench's normalized fraction is also shown on its /1000 full-split scale.
All comparisons use 10,000 image-cluster paired resamples. This is single-seed
method development and the tasks have already been inspected; no independent
confirmation or innovation claim is implied. Shared-GPU and historical timings
cannot establish a matched end-to-end acceleration advantage.

## Execution and recovery

Freeze all source changes in a detached local worktree and record its dirty-tree
hash; do not create a Git commit. Output lives under the owner's main repository
at `outputs/train/qwen25_vl/input_adaptive_topk_20261028_v2/`. The interrupted v1
preflight is preserved and superseded because its primary AI2D metric was the
historical strict match. Every attempt gets
a separate log/directory. Checkpoint contracts include source, data, config,
base identity, optimizer/scheduler and RNG state. Real 7B gates compare four
continuous steps against a process restart after two steps before full training.
Train on GPU0 and fill the missing fixed-word MMStar evaluation on GPU1, then
distribute new full evaluations across both GPUs. Check available memory at
launch and during training. Other users' processes and files are not modified.

Related motivations are question-guided selection in
[QG-VTC](https://arxiv.org/abs/2504.00654), joint coverage in
[MMTok](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4f8745c0ca5c951a92844ad7668cfca5-Abstract-Conference.html),
and learning query-dependent relevance in
[Task-Related Token Compression](https://proceedings.iclr.cc/paper_files/paper/2026/file/b74c141c9f7bbac5f250f73b0c84a478-Paper-Conference.pdf).
This small gate is an experimental extension of our existing implementation,
not a reproduction of those complete pipelines.
