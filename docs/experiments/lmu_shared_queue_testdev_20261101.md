# LMU shared chunk queue, Qwen2.5-VL-7B

Run from the LMU project root:

```bash
bash scripts/eval/lmms_eval/submit_lmu_qwen25_dynamic_queue.sh
```

The submitter requests one node with four A100 GPUs, 32 CPUs and a 24-hour
limit. It uses the `zfy-mllm-comp` environment by absolute Python path. Paths
remain configurable through `DATA_ROOT`, `MODEL_PATH`, `CKPT_ROOT`, `RUN_DIR`
and `SEED_RUN_DIR`. It refuses submission while another `28-` job is pending
or running.

The four GPU workers share one queue of 512-request chunks. A worker claims
the next available chunk as soon as it finishes. Long datasets can therefore
use all four GPUs. A compressed chunk becomes eligible after the corresponding
dense chunk is complete; its source metadata and document hashes must match.
File locks prevent duplicate work and are released when a process exits.
Startup, model loading, aggregation, and the final few chunks can still leave
GPUs temporarily idle; this does not promise constant 100% GPU utilization.

The default task order is GQA, ScienceQA-IMG, single-image MMMU, HallusionBench,
TextVQA, RefCOCO testA/testB, RefCOCOg, then VQAv2 test-dev. Each task has the
nine existing methods: dense, EADP, PruneSID, MMTok, aggregation MLP, visual
selector, image-context selector, question-context selector, protected selector.

Previous results are imported from
`qwen25_remaining_benchmarks_20261031/worker_*/cells/` by completion records,
including completed chunks of an interrupted cell. The importer checks model
identity, data root, checkpoint hashes, chunk size, limits and subset flags.
Raw predictions remain at their original paths, so retain the old run directory.
Existing complete cells are skipped. The new default run directory is
`$WS/outputs/qwen25_remaining_dynamic_testdev_20261101`.

VQAv2 reads **only `VQAv2/data/testdev-*.parquet`**. The P6000 copy has 36
shards and 107,394 questions; LMU uses its own validated source row counts.
Test-dev has no public gold answers. The native lmms-eval test submission
processor is used, with the original short-answer prompt and 16-token limit.
After all chunks of a method finish, the queue merges and validates a submission
at `cells/<method>/vqav2_testdev/vqav2_testdev_submission.json`. Each record has
only `question_id` and `answer`; duplicate IDs or empty predictions fail the run.
Its score remains null with `pending_official_server` until official scoring.
HallusionBench scoring also remains deferred. Successful generation does not
claim that these two benchmark scores have been obtained.

Validation on P6000: Python/shell syntax; four real CPU processes claiming
disjoint chunks with dense dependencies; two actual local test-dev examples
loaded through lmms-eval and passed through its CPU dummy model to verify native
submission logging. Dummy responses are format fixtures, not model predictions
or benchmark results. No new GPU evaluation was launched for this check.
