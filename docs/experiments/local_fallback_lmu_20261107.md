# Local4 + cosine-regret fallback: LMU handoff

This is a frozen-weight, 15%-retention exploratory ablation, not new training.
Keep the same importance-FPS anchors, mean pooling, centroid M-RoPE policy,
image-context scorer and balanced2560 step320 MLP. For each non-anchor token:

1. Find the four spatially nearest anchors on the native post-merger grid.
2. Find the best cosine match among those candidates, and the global best match.
3. If `global_cosine - local_cosine > 0.10`, assign globally; otherwise assign locally.

Anchors own their clusters. The gap is computed before fallback, in FP32. The
threshold is one fixed exploratory preset, not tuned on benchmark results and
not a claim that any cosine value identifies objects or background. No grid
coverage anchors are added in this revision. Original global and hard-local4
settings remain available and unchanged when the new option is absent.

Method: `importance_fps_balanced_local4_gap01_r15`.
Options: `assignment_neighbors=4`, `assignment_max_regret=0.1`.
Budget: `min(N, max(1, round(0.15*N)))` per image, without a hidden 128-token cap.
Traces include fallback token indices/count/fraction and the pre-fallback cosine
regret mean/max, alongside cluster membership and spatial diagnostics.

## Dataset order and defaults

RefCOCO testB, RefCOCO+ testB, RefCOCO testA, RefCOCO+ testA, RefCOCOg;
then OCRBench, DocVQA, TextVQA;
then MMStar, AI2D, POPE, GQA, ScienceQA-IMG, OK-VQA, MMMU single-image,
HallusionBench (generation only, official judging deferred).

There are 16 actual tasks. ChartQA, VQAv2 and multi-image MMMU are excluded.
Four resident workers share locked 512-request chunks. Task order is a priority,
not a global barrier: free workers may claim the next task when earlier chunks
are already claimed, rather than idling. Every task retains its native prompt,
generation settings and scoring configuration from the unified eval entrypoint.

The default Slurm launch evaluates only the NEW fallback method. It does not
retrain or rerun dense/original/local4 controls automatically. The experiment
also lists `importance_fps_balanced_r15` and `importance_fps_balanced_local4_r15`
for an explicitly requested matched-control run via `FALLBACK_METHODS`.
Missing dense results remain missing, not 100% or zero; the bundled audited
references are imported where available. Compare remaining raw scores with the
previously recorded same-model dense scores, not with another backbone.

## Paths and model weights

Edit only `configs/workflow/profiles/lmu_local_fallback.json` if LMU paths differ.
It uses the existing `zfy-mllm-comp` interpreter and shared Qwen2.5-VL-7B,
dataset root and `Proj` checkpoint package. The image-context selector comes
from `Proj/checkpoints/image_context_selector`. The small balanced MLP already
ships at `artifacts/checkpoints/r1_balanced2560_step320`; no replacement weight
is required. Model identity and original checkpoint implementation hashes are
still enforced. No local P6000 output paths are required on LMU.

## Manual LMU launch

Run code sync on an allocated CPU node under the group's rules. No agent or
inference process runs on the login node. Preserve local changes before switching
branches; do not restore over a dirty checkout.

```bash
cd /home/hk-project-p0022189/tum_yvc3016/haowu/Fengyu/Proj_lmu
git fetch origin codex/lmu-local-fallback-r15
git worktree add --detach ../Proj_lmu_local_fallback FETCH_HEAD
cd ../Proj_lmu_local_fallback
mkdir -p logs/slurm
bash -n scripts/eval/lmms_eval/lmu_qwen25_local_fallback.sbatch
sbatch scripts/eval/lmms_eval/lmu_qwen25_local_fallback.sbatch
```

The separate worktree preserves local edits in the existing checkout. If that
directory already exists, inspect it before attempting another worktree or update;
do not overwrite it or reset it automatically.
No environment activation is necessary: the script invokes the absolute Python
interpreter. Slurm allocates one node, four GPUs, 32 CPUs, 23h50m, job prefix 28,
mail NONE, offline W&B/HF. There is no automatic submission from P6000.

The job first creates a content-hashed full source snapshot, runs two-request
smoke checks on all selected tasks on one allocated GPU, checks that source/config
did not change, then starts the full four-GPU queue from that snapshot. Smoke
failure prevents full evaluation. Exact completed chunks resume on resubmission
with unchanged source/config/weights and the same `FALLBACK_OUTPUT`; changed
definitions require a new output directory. No automatic extra job is submitted.

Default output:
`/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/outputs/Fengyu/qwen25_keep15_local4_fallback_v1`.
Keep `workflow.json`, `workflow_status.json`, `plan.json`, `scores.csv`, per-request
predictions/traces, attempt logs and failure records. A generated-only Hallusion
cell is not a completed official benchmark score. Shared P6000 smoke timings are
not exclusive-GPU speed measurements.

## P6000 validation before publication

On 2026-11-07, the local/guarded assignment, native tiny-model generation and
shared-queue/workflow checks passed 38 CPU tests, including the isolated release
source tree. Real Qwen2.5-VL-7B smoke passed all 16 tasks with two requests each
(32 requests), worker exit 0, no failed cells. This validates execution only;
it does not establish benchmark improvement or A100-specific compatibility.
The LMU batch script repeats smoke on its allocated A100 before starting full
evaluation. Existing P6000 full experiments continue from their earlier frozen
snapshots; no active experiment was edited or restarted for this release.
