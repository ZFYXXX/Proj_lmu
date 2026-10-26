# Qwen3.5-9B K128 transfer — 2026-10-26

## Scope

This is a new same-data experiment, not a transfer of Qwen2.5 head weights.
The native checkpoint is `/home/WuHao/checkpoints/ckpts-vlm/Qwen3.5-9B`.
Qwen3.5 uses 4096-dimensional text features and hybrid linear/full attention;
the old Qwen2.5-7B heads are dimensionally incompatible.

Run directory:
`outputs/eval/qwen35_vl/transfer_k128_20261026_v1/`

Frozen execution source:
`/home/ZhouFengyu/fastv-repro/worktrees/qwen35-transfer-20261026-v1/`

No automatic commit, remote write, model modification, or other-user process
termination was performed. The dirty source snapshot and all input files are
content pinned in `protocol.json`. Keep that execution snapshot immutable.

## Matrix

All methods run ChartQA 2500 questions, OCRBench 1000, and AI2D 3088 through
the MLLM-Comp native eval entry, with native task prompts/scorers, BF16/eager,
dynamic resolution, greedy decoding, repetition penalty 1.05, and thinking off.
Generation limits are 16 / 128 / 16 respectively. AI2D strict and the existing
leading-option auxiliary metric remain distinct.

- Dense baseline.
- EADP, PruneSID, MMTok, Bifocal Stage1 controlled adaptations.
- Retrained anchor aggregation MLP, visual selector, global-question selector,
  and word-relevance-plus-protection selector.

There are 27 full cells and 59,292 generation requests, excluding smoke.
No noise gate, OT, dynamic budget, or new appended-token recovery is included.
Bifocal Stage1+2 is NOT included: the Qwen2.5 decoder rewrite is not portable
to Qwen3.5 hybrid caches without a separate implementation and validation.

## Training and gates

Freeze the base. Reuse the exact 2560-image training manifest (2048 TextVQA
train + 512 ScienceQA train), verify content hashes and no overlap with all
three evaluation cohorts. Each head trains seed42, global batch8, 320 steps,
AdamW 2e-4 / weight decay .01, warmup16 / cosine / gradient clip1. Selectors
use the existing continuous Top-K CE + .5 teacher KL + annealed mask BCE.
The MLP uses CE + .5 teacher KL. Question and word heads warm-start the same
completed Qwen3.5 visual prior. Use fixed step320; no test-set model selection.

Each GPU first runs a real9B gate: all eight compression methods, strict K128,
first-decode M-RoPE validation, native hybrid cache, original-prefix restoration,
and identity-versus-dense generation equality. Each trained head then passes
a two-step training smoke with an actual disk checkpoint reload and exact
next-step parameter replay before its formal training. Each evaluation method
passes eight examples per task before its full three-task run.

GPU0 is the training/own-method lane; GPU1 is the dense/external lane. Each
lane is sequential and file-locked; no other user's process is modified.
Launch memory gates are 64GiB for training and 40GiB for eval, with an 8GiB
runtime headroom check. There is no shared-compute throttling. CUDA fallback
linear-attention kernels are valid but slower; these runs are not optimized
or matched end-to-end speed benchmarks.

## Inspect and resume

Use `manifest.json`, `worker_training.json`, `worker_external.json`, per-head
`training.jsonl` / `latest.json`, and each eval cell's `complete.json`.
`coordinator_process.json` records the detached process and log.
Do not infer running status from a prepared configuration or an old timestamp.

Re-run the coordinator from the identical frozen source/runtime only after
checking its PID and locks; it validates completed artifacts and resumes
checkpointed training. Failed attempts and logs are retained. Any change to
the frozen source/protocol requires a new run identity, not in-place edits.

The isolated runtime is Transformers5.19 with torch2.8; its historical path
contains `qwen38` but the loaded checkpoint identity must be Qwen3.5-9B.
No system-wide package replacement is used. A 9B-versus-7B comparison is not
a clean architecture-only causal comparison.
