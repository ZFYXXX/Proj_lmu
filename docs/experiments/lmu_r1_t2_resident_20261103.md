# R1 / T2: portable Qwen2.5-VL-7B continuation

This release schedules `importance_fps` (R1), `importance_fps_pool` (T2),
and nine missing OK-VQA/DocVQA comparison cells. R1/T2 reuse the existing frozen image-context scorer and aggregation MLP.
There is no new training or weight upload. Each image retains min(128,N)
post-merger tokens with native dynamic resolution and centroid M-RoPE.

## Tasks

| Category | Tasks |
|---|---|
| General | GQA, OK-VQA |
| Reasoning | ScienceQA-IMG, AI2D, MMStar, single-image MMMU |
| Hallusion | POPE, HallusionBench |
| OCR / charts | TextVQA val, DocVQA val, OCRBench, ChartQA |
| Grounding | RefCOCO testA/testB, RefCOCO+ testA/testB, RefCOCOg test |

VQAv2 val and test-dev are excluded from this job. MMMU remains the single-image
subset, with exclusions recorded; do not label it the full official validation
score. HallusionBench retains predictions and judge payloads; official scoring
is deferred. A generation completion is not a HallusionBench accuracy.

There are 43 target method/task cells. R1/T2 on ChartQA, OCRBench and AI2D are
already complete and included in the portable result bundle, so six cells can
be reused after protocol/model/checkpoint/data checks. The remaining 37 cells
are scheduled unless another specified result source already contains them.

Edit `configs/eval/lmms_eval/qwen25_r1t2_completion.json` to select experiments.
Its `method_tasks` mapping lists each method's tasks, not a Cartesian product.
DocVQA additionally requests PruneSID, visual selector, image-context selector,
question-context selector and protected selector. OK-VQA additionally requests
PruneSID, MMTok, question-context selector and protected selector.
Existing method bindings and checkpoint loading are reused. Changing which
supported methods/tasks run needs only this JSON; a new algorithm needs one
framework binding before it can be selected here.

`EXPERIMENT_CONFIG` selects another JSON file; optional `METHODS` and `TASKS`
environment variables filter its cells (e.g. `TASKS=ai2d` selects just R1/T2).
Resolved cells and the config hash are frozen in `plan.json`. Unrequested table
cells are marked `not_requested` in CSV and excluded from completion counts.
Ref/ref+ are arithmetic means of TestA/TestB accuracy, not extra splits;
the report and summary include raw means and their ratio to the dense mean.

## Reuse and scores

The bundled `artifacts/eval/qwen25_p6000_reuse_20261103.json.gz` contains native
scores and compact prediction evidence for 14 completed cells: six R1/T2 cells
and eight dense references. It contains no images, models, training states or
hidden states. Export compares native task functions, generation settings,
processor settings and checkpoint content. AI2D uses corrected
`leading_option_accuracy`, including in reused dense scores.

The two earlier LMU directories in `SEED_RUN_DIRS` supply other completed dense
references. Only requested methods and dense references are imported; other
methods are never scheduled. Full-cell reuse does not depend on chunk size.
Partial results can only be imported with their original chunk size.

The historical DocVQA diagnostic baseline has a different result format and is
not automatically certified as this native-eval reference. If no compatible
dense reference exists, R1/T2 still finish and report raw DocVQA ANLS; its
dense-relative score remains pending. No dense job is silently started.

`report.md` contains raw and dense-relative tables. `scores.csv` contains
unscaled `raw_score`, display value/unit, metric, count and normalized percent.
`summary.json` retains the same values and category membership. Raw OCRBench
is displayed on the /1000 scale; other scores are percentages, with DocVQA
ANLS multiplied by 100. Relative score is method/dense*100. Missing scores are
never zero. Category means equally weight the declared tasks and require all
of that category's selected scores; they are not a separately measured metric.

Every new chunk retains native sample JSONL, compression traces, timing/peak
memory records, source/task metadata and failure status. Reused evidence and
its source are recorded separately. A smoke run cannot seed a full run.

## Four-GPU execution

One Slurm job requests one node, four A100 GPUs, 32 CPUs, 23h50m, ID-28 job
name and mail-type NONE. Each GPU runs one resident worker. Workers claim
unfinished chunks from a shared pool and prefer their already loaded method.
The model survives chunk and dataset boundaries; changing method reloads the
model and reinstalls the compressor. Each chunk gets a fresh request bridge,
native task dictionary and output traces. Each worker's actual load/reuse
counts appear in `resident_worker_N.json`.

Chunks default to 1024 requests and are checkpoint boundaries, not model-load
boundaries. At most four chunks remain in flight near the job deadline. Resume
uses the same run directory, code/config and chunk size. A failed cell stops
the job and preserves its traceback; inspect the failure before restarting.
The submit helper refuses another pending/running ID-28 job.

## LMU commands

Use an allocated CPU node for code synchronization. Do not install an agent on
LMU. The environment and model/checkpoints already exist.

```bash
salloc --account=hk-project-p0024767 --partition=dev_cpuonly \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
  --time=00:30:00 --job-name=28-code-sync

cd /home/hk-project-p0022189/tum_yvc3016/haowu/Fengyu
git clone --branch codex/lmu-resident-r1-t2 --single-branch \
  https://github.com/ZFYXXX/Proj_lmu.git Proj_lmu_r1t2
cd Proj_lmu_r1t2
```

Use this separate checkout to preserve local changes in the previous checkout.
Edit only `scripts/eval/lmms_eval/lmu_qwen25_resident.env.sh` if paths differ.
It holds the dataset/model/adapter/environment paths, previous result paths,
run directory and experiment-config path. R1/T2 use the same two existing head
checkpoints; the script selects the correct Python executable explicitly.

After the CPU allocation ends, submit from the project root on the login node:

```bash
cd /home/hk-project-p0022189/tum_yvc3016/haowu/Fengyu/Proj_lmu_r1t2
env -u METHODS -u TASKS -u RUN_DIR -u EXPERIMENT_CONFIG \
  bash scripts/eval/lmms_eval/submit_lmu_qwen25_resident.sh
```

This command submits the GPU allocation; no separate `salloc` is needed for
the full evaluation. Repeating it after a time-limited job resumes completed
chunks. Preserve the old LMU result directories until the continuation finishes.
The v2 default output is `qwen25_resident_bench_20261103_v2`. Clearing the four
old environment overrides avoids unintentionally retaining v1's R1/T2-only
selection. Cancel the pending v1 job before replacing it and updating that
checkout; do not change sources used by a running job.

Inspect `logs/slurm/28-q25-resident_JOBID_output.log`, its error log, and
`$RUN_DIR/logs/worker_*.log`. The shared `report.md` updates after completed
method/task cells. `parallel_status.json` counts only the 43 requested target cells;
pending dense references do not falsely mark their generation incomplete.

## P6000 validation

The resident path uses the existing MLLM-Comp adapter and native lmms-eval CLI,
prompts, generation, metrics and sample logger. Only model construction is
cached. A two-chunk GPU smoke of dense/R1/T2 completed six chunks with three
model loads and three reuses. A four-process CPU test verifies shared claims
and exactly-once chunk execution. Cross-task R1/T2 GPU smoke and release-source
checks are recorded in the accompanying validation file. These checks do not
claim a physical four-A100 test on the two-GPU P6000 host.
