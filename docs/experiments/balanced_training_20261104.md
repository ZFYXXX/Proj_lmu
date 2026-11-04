# R1 training-data comparison, 2026-11-04

## Active revision v2: NeXT subset plus explicit supplements

The v1 preparation stopped before training: 43 original ScienceQA training
images also occur in ScienceQA test. Historical weights/results/manifests are
preserved. V2 replaces those 43 control entries with distinct, non-evaluation
ScienceQA training images and records every removed/replacement pair. The
control remains 2048 TextVQA + 512 ScienceQA; it is a cleaned control, not the
exact historical sample list.

The active candidate contains 1792 examples from pinned NeXT shards: OCR-VQA
320, ChartQA 320, DocVQA 256, GQA 384, COCO instruction QA 256, AI2D 256.
ScienceQA 256 and RefCOCO expression-to-box 512 are explicit supplements. They
are not labeled as NeXT data. This retains the registered domain totals without
inventing ScienceQA/RefCOCO annotations in the NeXT release. NeXT's VG first
turns are region-to-description tasks, not the same grounding supervision.
Only selected image row groups are fetched, not the whole 92 GB image pool.

Source profile: `configs/workflow/profiles/p6000_next_subset.json`.
Data: `outputs/training_data/next_subset2560_20261104_v2/`.
Training: `outputs/train/r1_next_subset_20261104_v2/`.
The preparation command below should add the profile above and use these v2
paths. Both arms still use the same scratch initialization and frozen scorer.

The scorer retains its prior training exposure in both arms. Cleaning the MLP
training list does not erase that shared exposure; ScienceQA full-test scores
must not be described as entirely unseen-image results for the whole pipeline.

The failed v1 preparation logs remain in their original directory. The active
protocol and commands below describe v2.

This study changes the training mixture while fixing Qwen2.5-VL-7B, the
image-context scorer, importance-guided FPS, K128, ordinary cluster means and
the residual MLP architecture. Both arms initialize the MLP from the same seed
with its final linear layer zeroed. Neither arm loads the old MLP weights.
Existing weights and all previous run directories are read-only references.

| Arm | Unique training images | Trainable parameters |
|---|---:|---|
| R1_clean_control_scratch | Cleaned TextVQA-train 2048 + ScienceQA-train 512 | MLP only |
| R1_balanced_data_scratch | New balanced mixture below | MLP only |

| Source | Images |
|---|---:|
| OCR-VQA from NeXT training mixture | 320 |
| ChartQA from NeXT training mixture | 320 |
| DocVQA from NeXT training mixture | 256 |
| GQA from NeXT training mixture | 384 |
| COCO instruction QA from NeXT | 256 |
| RefCOCO train, expression to pixel-space bbox | 512 |
| ScienceQA-IMG train | 256 |
| AI2D from NeXT training mixture | 256 |

The grounding category uses 512 genuine RefCOCO training examples. It does not
include VG region captions masquerading as expression-to-box supervision. The
other categories total 35%, 25%, and 20%, respectively. This is an exploratory
allocation, not an established optimal ratio.

The NeXT release is pinned to `c8aef391ce214167c4ebc7f06bc05a50dee2e75f`,
RefCOCO annotations to `9fba8200c5326e996f789191f095bd464ef1d09e`.
Only declared train-shard candidate pools are accessed. The manifest records
their restricted scope rather than claiming a uniform sample of the entire
NeXT mixture. Raw JSON and parquet have different lengths: match each sample by
its ID and first question, not a global row offset. Text-only rows are excluded.
Each selected image is unique across domains, with one self-contained QA.

Every locally available evaluation image from the workflow task catalog is
excluded by decoded RGB hash (GQA images are read from its image split).
VQAv2 validation COCO IDs additionally exclude resized/recompressed versions
of those images. The original control is audited too; v2 records each
authorized overlap replacement in a separate manifest. The exact
exclusion sources are recorded in `evaluation_exclusion.json`. This checks
exact image matches and COCO identities, not all possible near duplicates.

Targets for grounding use the current native-Qwen evaluation prompt and boxes
in the pixel space of the processed image. Raw bbox metadata and processed
image size are retained per example. The prompt never includes the target.

Both arms run one pass: global batch 8, 320 optimizer steps, seed 42, AdamW
learning rate 1e-4, weight decay .01, warmup 16, cosine decay, gradient clipping
1, answer CE + .5 dense-teacher KL. The balanced arm mixes the four domains in
40-example blocks with exact 14/10/8/8 counts. The old arm preserves its
historical data order. Results therefore test the new data-and-order recipe;
they do not isolate the contribution of domain batching alone.

Only fixed step320 is evaluated; test results do not select checkpoints.
Automatic full evaluations: ChartQA, OCRBench, AI2D, GQA, ScienceQA-IMG,
RefCOCO+ testA and testB, through the existing MLLM-Comp eval queue. Historical
dense scores are reused only where the existing queue accepts their identity
and protocol; missing baselines are shown as missing, not recomputed ratios.
Frozen R1 remains a reference, not a freshly trained matched-data control.

## Execution

```bash
HF_HUB_DISABLE_XET=1 /home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python -u \
  -m mllm_comp.workflow.training_data \
  --profile configs/workflow/profiles/p6000_next_subset.json \
  --output /home/ZhouFengyu/fastv-repro/worktrees/mlp-next/outputs/training_data/next_subset2560_20261104_v2 \
  --launch-training
```

Preparation writes only inside this worktree. It automatically creates the
paired config, freezes the source and input hashes, queues training for free
GPUs, runs the existing two-step save/resume smoke, then trains and evaluates.
Do not run a second preparation process while its owner-scoped lock is held.

Data preparation log:
`outputs/training_data/next_subset2560_20261104_v2/preparation.log`.
Training and evaluations:
`outputs/train/r1_next_subset_20261104_v2/`.

The preliminary targeted CPU checks cover scratch initialization identity,
preservation of old weights, normalized sample order, streaming annotations,
first-turn handling, domain quotas, checkpoint/resume and existing workflow
compatibility. Actual GPU smoke is a prerequisite for each formal arm.
