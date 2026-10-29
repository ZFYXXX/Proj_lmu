# Qwen2.5 evaluation interruption, 2026-10-28 (UTC+8)

The first Oct27 interruption remains unexplained: both original POPE workers
ended around22:20 without a Python traceback. The Oct28 restart has a different,
explicitly logged failure and should not be described as the same proven cause.

## Evidence for the restart failure

- 00:38:56: recovered visual-selector POPE generation started on GPU1.
- 00:53:58: recovered dense POPE generation started on GPU0, after training ended.
- 00:53:59: the independent adaptive pipeline also started MMStar on GPU1.
- 00:54:02: the adaptive pipeline started OCRBench on GPU0.
- 00:55:20 / 00:55:21: visual POPE and adaptive MMStar both failed with
  `Shared GPU headroom fell below 8 GiB; stopping evaluation`.
- 01:05:49: dense POPE failed with the same explicit exception.
- Completed request counts were3503/9000,173/1500,2557/9000 respectively.
- Both independent dispatchers checked available memory at launch but did not
  reserve the physical GPU across dispatchers. The memory guard was working;
  concurrent launches were the scheduling error.
- Last completed MMStar request reserved44300238848 bytes in its allocator;
  visual POPE independently reserved18062770176 bytes. Other-user workloads were
  present. These are per-process samples, not simultaneous global measurements.
- Adaptive training completed320 steps. OCRBench1000/1000 and AI2D3088/3088
  passed native result/trace checks and have complete markers. Their artifacts
  must be reused. No valid complete result exists for the failed POPE cells.

## Recovery

`supervise_qwen25_recovery.py` executes the adaptive pipeline first (only missing
MMStar evaluation), then POPE/RefCOCO+, then OK-VQA. Pipelines may use both cards,
but independent pipelines cannot run concurrently through this supervisor.
The supervisor holds owner-local GPU file locks and checks for other same-owner
CUDA processes before each stage, so old workers which lack those locks are
also detected. It writes a heartbeat every10 seconds including process PID,
native pipeline state and free memory. Known resource failures wait and retry;
numerical/data/provenance failures remain explicit failures. The8GiB safety
guard is retained. No other user's process is stopped.

The active adaptive source/checkpoint remains in its existing frozen worktree.
Recovered POPE/OK-VQA scripts run from a new detached source snapshot, with a
preserved before/after protocol amendment. Checkpoints, task pins and dependency
hashes must still match. Existing completed cells are validated and skipped;
incomplete attempts are retained but their cells restart because they contain
no complete native sample/aggregate artifact.

Validation:3 CPU tests check resource-error classification, owner-specific GPU
process filtering, serialized stages, retry, and completed-stage skipping.
There is no guarantee of a completion deadline under changing external GPU
load; a running launch alone does not establish end-to-end completion.
