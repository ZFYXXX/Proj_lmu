# Three-arm local assignment study: fallback, grid coverage, both

Common reference: the existing hard-local4 method, not the original global-only
assignment. All arms freeze the same balanced2560 step320 MLP and image-context
scorer, retain exactly `min(N, max(1, round(0.15*N)))` tokens, use ordinary means
and the same centroid M-RoPE rule. No retraining or additional output tokens.

| Arm | Alias | Grid floor | Regret fallback |
|---|---|---|---|
| A | `importance_fps_balanced_local4_gap01_r15` | off | on, gap > 0.10 |
| B | `importance_fps_balanced_grid4_local4_r15` | on | off |
| AB | `importance_fps_balanced_grid4_gap01_r15` | on | on, gap > 0.10 |

For each token, the gate computes exactly
`delta = max_global cosine - max_local4 cosine` in FP32, then falls back globally
when `delta > 0.10`. Delta is input-dependent; **0.10 is a fixed exploratory
decision threshold**, not an absolute cosine similarity floor, learned parameter,
dynamic budget or benchmark-tuned optimum. Anchor tokens keep their own clusters.

The grid floor preserves the original first/highest-importance anchor. It divides
the native rectangular post-merger grid into up to four nonempty 2x2 cells and
adds a spatial-center representative from each still-uncovered cell as FPS seeds.
The rest of the SAME K slots come from the original importance-weighted FPS.
Equal-distance ties use smaller source index. When K is below the number of
nonempty cells, a deterministic farthest-cell subset fits K and the trace marks
coverage infeasible; K is never increased. A grid seed need not be background or
an object boundary. This tests coarse regional coverage, not semantic coverage.

Keep existing global and hard-local4 arms for score comparisons; their options
and old results remain unchanged. To estimate the grid/fallback interaction,
the hard-local4 reference is the fourth factorial cell. Its existing P6000
results should not be claimed as matched-A100 latency measurements. Positive
score changes alone do not establish the proposed background-contamination cause.

## LMU: one four-GPU job, all three arms, strict phases

Use `scripts/eval/lmms_eval/lmu_qwen25_local_factorial.sbatch`, not the older
single-arm fallback launcher. It reuses that launcher's smoke/resume logic with
`FALLBACK_EXPERIMENT` and `FALLBACK_TASKS` filters, without copying model forward
code. It schedules 48 method/task cells in three output directories:

1. `ref/`: RefCOCO B, RefCOCO+ B, RefCOCO A, RefCOCO+ A, RefCOCOg (3 x 5).
2. `ocr/`: OCRBench, DocVQA, TextVQA (3 x 3).
3. `other/`: MMStar, AI2D, POPE, GQA, ScienceQA-IMG, OK-VQA, MMMU single-image,
   HallusionBench (3 x 8).

All three arms finish the Ref phase before OCR starts, and OCR finishes before
other tasks start. Within each phase, four resident workers share locked chunks
and steal unfinished work. Each phase runs two-request smoke on all selected
cells before full evaluation; a failure stops the pipeline. HallusionBench is
generation-only until separately judged. No ChartQA, VQAv2 or multi-image MMMU.

Paths remain in `configs/workflow/profiles/lmu_local_fallback.json`. Weights are
unchanged; the small new MLP is already in the repository artifacts and the image
scorer uses the existing `Proj` package. No new weight upload is necessary.
Default output root is
`/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/outputs/Fengyu/qwen25_keep15_local_factorial_v1`.
Set `FACTORIAL_OUTPUT` to use a new experiment root. Changed code/config cannot
resume an old signature; do not overwrite earlier experiment directories.
The job requests one node, four GPUs, 32 CPUs, 23h50m, prefix 28 and mail NONE.
The larger three-arm run may need resubmission after the time limit; the same
source/config/root reuses completed chunks. No follow-up jobs are auto-submitted.

On an allocated CPU node for code synchronization, create a new worktree so an
older LMU run or local edits are not changed:

```bash
cd /home/hk-project-p0022189/tum_yvc3016/haowu/Fengyu/Proj_lmu
git fetch origin codex/lmu-local-fallback-r15
git worktree add --detach ../Proj_lmu_local_factorial FETCH_HEAD
cd ../Proj_lmu_local_factorial
mkdir -p logs/slurm
sbatch scripts/eval/lmms_eval/lmu_qwen25_local_factorial.sbatch
```

Existing directories are not reset or deleted. LMU resource allocation and
submission are manual; no agent connects to LMU. Old single-arm fallback v1
remains available under its original launcher and output root.

## Validation before updating GitHub

48 CPU tests passed, including native tiny-model generation for both grid modes
and tests of exact K, unique membership, region coverage, unchanged default
behavior and shared chunk claims. The isolated release source tree passed the
same test selection. A CPU-only shell simulation checked Ref/OCR/other phase
filters, all three arm names and smoke-before-full wiring; it did not submit a
Slurm job or execute GPUs. Real Qwen2.5-VL-7B smoke passed the two new arms on
all 16 tasks, two requests per task (32 cells / 64 requests), exit 0 and no failed
cells. A was already validated in the prior 16-task release. These are execution
checks, not benchmark improvement evidence or direct A100 validation; LMU repeats
smoke on its allocated hardware before full evaluation.
