# Learned visual Top-K v1: isolated exploratory experiment

## Scope

The new `learned_visual_topk` branch is **not** a replacement for either the
old feature-anchor mean or its trained aggregation MLP. It scores original
post-merger Qwen2.5-VL-7B visual tokens with
`LayerNorm(D) -> Linear(D,256) -> GELU -> Linear(256,1)`.
No question input, dynamic count, discarded-information pooling, or Qwen3.5 is
introduced in this experiment. EADP, PruneSID, and MMTok remain training-free.

Only VisionSelector's continuous cardinality operator is ported; its model,
training loop, and Transformer scorer are not reproduced. Operator provenance:

- repository: https://github.com/JulietChoo/VisionSelector
- commit: `0336cd58ddc29c443c7ad8d2f45f2911b37ab0e1`
- file: `qwen-vl-finetune/compression_method/selector_model.py`
- SHA256: `613a5c31962af13634c81147509040c382b097cc270326f6fe9c69dfa532d820`
- CPU parity checks compare both forward weights and backward gradients against
  that local source, in addition to FP64 gradcheck. No license file was present
  in this checkout; do not infer permission to redistribute the entire upstream
  repository from its availability. This experiment is local, with no push.

## Frozen protocol

Train on the existing ordered 2560-image manifest: TextVQA-train 2048 and
ScienceQA-train 512. The original checkpoint and historical adapter format are
unchanged. All original inputs are read-only; new outputs must resolve below
`/home/ZhouFengyu`.

The frozen backbone is BF16/eager at native dynamic resolution
(`min_pixels=200704`, `max_pixels=1605632`); scorer parameters and Top-K statistics
are FP32. Soft training multiplies all N visual embeddings by continuous
sigmoid weights summing to `min(128,N)` and retains the **full physical sequence**.
It does not provide training-time token-count savings. Deployment selects
`min(128,N)` original tokens, restores source order, and preserves their original
visual M-RoPE and all text coordinates. Placeholder, mask, labels, and cache
offsets are compacted together. N<K is an identity path, not duplication.

Only the scorer is trained: seed42, global batch8, AdamW lr2e-4, decay.01,
320 optimizer steps / one pass, warmup16, cosine decay, clipping1. Answer CE +
0.5 teacher KL + mask BCE. BCE coefficient increases **linearly** from .1 at
step0 to3 at step319; detached hard Top-K mask is its target. Step320 is the
only evaluated trained checkpoint; development/test scores never select it.

One or two ranks are supported. The global sample batch is always manifest
indices `[8*step:8*(step+1)]`; two ranks take alternate indices, accumulate4
each, and explicitly average scorer gradients before clipping. One rank
accumulates8. The frozen base is replicated, not wrapped in DDP. World size is
part of the strict restore contract and cannot be changed during resume.

## Versioning and launch

Baseline: `workspace-baseline-20261023` (`9c20afa`). Development branch:
`codex/learned-visual-topk-v1`. Source/config/tests are tracked; datasets,
checkpoints, caches, and results are not. Third-party checkouts are not nested
commits. The runtime fingerprints the actual editable lmms-eval source tree,
package versions, base identity, data files, cohort, and imported source path.

Before preparing an experiment, commit a clean version and create a detached
execution worktree. An RC tag indicates CPU readiness only. The final
`learned-visual-topk-v1.0.0` tag is created automatically **only after real-7B
gates pass and formal training is requested**, at the same pinned commit.
Never move either an old tag or an old result directory.

From the detached worktree, using the existing Python environment:

```bash
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/train/qwen25_vl/run_learned_visual_topk.py prepare --root /home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main/outputs/train/qwen25_vl/learned_visual_topk_v1_20261023_v1
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/train/qwen25_vl/run_learned_visual_topk.py gates --root /home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main/outputs/train/qwen25_vl/learned_visual_topk_v1_20261023_v1 --gpus 0 1
/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python scripts/train/qwen25_vl/run_learned_visual_topk.py train --root /home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main/outputs/train/qwen25_vl/learned_visual_topk_v1_20261023_v1 --gpus 0 1
```

`prepare` hashes actual RGB image contents in **all** ChartQA2500/OCRBench1000
examples and checks overlap against the fixed training manifest. Any overlap is
a hard stop with explicit fingerprints; no silent filtering/resampling. Each
development set is hash-selected250 independent images (one question per image),
without looking at answers or scores. Subsequent commands verify hashes again.

`gates` compares uninterrupted4 steps with2 steps + process restart +2 steps;
parameters, optimizer moments, and schedule must match exactly. Each fresh run
also checks real7B dense/identity logits, strict hard budget, and first decode
position. CPU tests cover the operator, soft gradients, frozen base, native
generate/cache, exact RNG/data/state resume, and strict mismatch rejection.

Both GPU0 and GPU1 are authorized as of the latest user instruction. Only idle
GPUs are selected; other users' compute processes are never terminated or
borrowed. With no idle device, the command exits75 and records
`blocked_resource.json`: **this is not a launched or queued training job**.
Rerun when resources become available. If only one card is free, gates/train
can use one card; training must use the world size that passed its gates.
An advisory lock prevents overlapping orchestrators for one experiment.

Resume formal training with the same command plus `--resume`. Checkpoints
contain scorer weights, optimizer, scheduler, completed step, and Python,
NumPy, CPU/CUDA random states for every rank. The completion marker and checksums
reject partial/corrupt artifacts. A new launch gets a new log and new checkpoint
attempt names; interrupted logs/checkpoints are preserved. Code/config/data/base/
dependency/world-size changes reject direct continuation.

## Unified evaluations and report

Invoke the same runner with `eval --phase smoke`, then `eval --phase dev`, then
`eval --phase full` (and the same `--root`, `--gpus 0 1` arguments). Each method
uses only `python -m mllm_comp.cli.main eval --config ...`, not a separate
generation/data loop. Methods are distributed across idle GPUs, one process
per card, sequentially within each card. Every phase must complete all groups
before the next phase. Completed jobs are checked and reused only within the
same experiment and source/config provenance. Failed attempts are retained;
retry creates a new attempt directory.

Groups: dense; feature-anchor mean K128; old trained aggregation MLP; random
pruning seeds11/23/47; untrained scorer seed42; fixed-step320 scorer; EADP;
PruneSID; MMTok. External methods are Qwen adaptations under this protocol, not
certified reproductions of the papers' reported numbers.

ChartQA uses native relaxed scores, greedy/max16. OCRBench uses the native
per-question score, greedy/max128, reported as its full sum out of1000.
Both explicitly fix repetition_penalty1.05, checkpoint/revision, processor,
prompt, native resolution, and ordering. Each group gets the same samples.
Smoke is8 requests/task, development250 independent images/task, full2500+1000
questions. Development results do not tune hyperparameters or checkpoints.

After all full jobs, run `report --root ...`. Preserve native predictions,
targets, compression request traces, actual N/K, synchronized generation time,
peak allocated/reserved memory, and source/config provenance. Per-request
performance recording is opt-in and disabled for historical runs by default.
It checks finite logits before argmax; timing includes this validation overhead
and actual variable-length answers, not model load/CPU processing. Do not call
these fixed-length speed benchmarks.

Report paired image-cluster bootstrap95% CIs (10000 resamples) and per-question
wins/losses. Native headline accuracy is question-weighted; the bootstrap's
mean paired effect is explicitly **image-weighted**, averaging same-image
questions first. Never treat multiple questions or random seeds as independent
images. Soft training loss improving without hard generation quality is not a
success. A trained scorer beating its untrained counterpart supports this
selector/training combination, not the Top-K operator alone or novelty.

## CPU regression audit

The image-only feature/regression suite passes104 tests, including official
operator parity, exact state/RNG restoration, native cache lifecycles, fixed
configs, and rejection of invalid output traces. This does not certify a real
7B training run: the real-checkpoint gates remain separately required.

The expanded legacy lmms-eval integration suite has4 pre-existing failures:
the removed `segmentwise_topk_audio` capability and three archived audio YAML
references. These were independently reproduced in the baseline worktree at
`9c20afa` (35 pass,4 fail); they are not silently counted as passing or repaired
by restoring unrelated audio methods. Keep their result separate from this
feature's image-only regression suite.
