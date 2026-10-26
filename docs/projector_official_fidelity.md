# External projector fidelity correction

The 2026-10-24 real-feature audit found BF16 selected-set deviations hidden by
the previous FP32 tests. Earlier full external results remain preserved, but
should not be presented as faithful reproductions or used to certify superiority.

Corrections:

- EADP visual normalization and text averaging follow released dtype order.
- MMTok normalizes in input dtype before computing FP32 coverage.
- PruneSID computes NMS thresholds in input dtype, sorts the full score matrix,
  preserves semantic-group output order for K<=150, and source order for K>150.
- Its actual decoder gather now preserves selector order for embeddings, original
  M-RoPE coordinates, mask and labels. Compact physical cache positions remain
  sequential. Degenerate duplicate outputs fail rather than being silently repaired.

Official ratio budgets are not substituted for the controlled fixed K128 budget.
PCA randomness stays request-local at seed 0. JPEG85, native dynamic resolution,
Qwen2.5-VL-7B weights, task prompts, generation and native scoring stay unchanged.
This is a controlled adaptation, not end-to-end paper-number reproduction.

Validation: official CPU FP32/BF16 parity, numerical/layout/cache/identity tests,
then 24 real 7B requests (8/task) comparing sets and PruneSID order, then native
MLLM-Comp compressed generation smoke (8/task/method). Only after every smoke
cell passes does the runner start all 9 full external method/task cells.

`scripts/train/qwen25_vl/rerun_projector_official_fidelity.py` stores an immutable
source identity, protocol and hashes, linked previous predictions, failure records,
native outputs, K128/cache checks and paired image-bootstrap comparisons. New
source is copied to a separate detached worktree, hash-frozen without committing.
No old source snapshot, checkpoint, evaluation artifact or shared input is changed.

Datasets: ChartQA 2500 questions, OCRBench 1000 questions, AI2D 3088 questions.
Methods: EADP, PruneSID, MMTok. Dense and our existing methods are not rerun;
their matched-protocol quality results are reused, not their latency measurements.
