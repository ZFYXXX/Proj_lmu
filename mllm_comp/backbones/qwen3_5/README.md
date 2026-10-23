# Qwen3.5 (Qwen3.8) image/video adapter

This is a separate Transformers 5.x path; it does not alter the Qwen2.5-VL or
Qwen3-VL adapters. The image path has ToMe, DivPrune, VisionZip, OTPrune, and
PruMerge+ integrations. OTPrune has passed a two-sample real-checkpoint MME
smoke; PruMerge+ uses the shared mean-received attention capture and fixed-budget
adaptation, not the original CLS-based paper configuration.
The video path
initially supports the framework's fixed-K VQToken kernel only; audio, training,
analysis probes, and benchmark claims are not enabled.

The `tome.image.qwen3_5` and `divprune.image.qwen3_5` bindings compress or
select post-merger image features before LLM scatter. The
`visionzip.image.qwen3_5` binding captures final-layer vision Q/K statistics,
uses mean-received attention (Qwen3.8 has no CLS token), pools patch statistics
over native contiguous spatial-merger groups, and runs the shared VisionZip
kernel. Attention is accumulated in query chunks instead of storing a dense
patch-by-patch matrix; Qwen rotary keys are retained for key-similarity merging.
Registration and CPU layout/generation contracts pass, and a two-sample real
GPU smoke completed. Its support unit remains declared pending broader lifecycle
validation; the smoke is not a benchmark-quality result. The
`otprune.image.qwen3_5` binding selects post-merger image features using the
released OTPrune-style `I + gamma*S*S^T` kernel and greedy Cholesky log-det
updates, then compacts the image placeholders, masks, and M-RoPE positions in
the same adapter path. The smoke uses `gamma=0.01` and 50% retention to match
the existing ToMe/DivPrune/VisionZip integration comparison; the OTPrune paper's
reference setting retains about 9.8%, so this is not a paper-result comparison.
Its config is `configs/eval/lmms_eval/qwen3_5_mme_smoke_otprune.yaml`.
PruMerge+ uses the final Qwen3.8 vision attention averaged over query tokens
and pooled rotary keys. Its fixed output budget includes the global residual
token, so a 50% target can be compared directly with the other framework
adaptations. The MME/GQA/MMBench configs use local evaluation tasks and batch
size 1; they are not paper reproduction settings.

MMTok uses the platform's text-vision coverage kernel at the same boundary.
The Qwen3.8 adapter pools final pre-merger patch states into one reference
feature per merger token and scores them against prompt input embeddings. Its
50% retention setting is the fixed cross-method comparison budget.

GAP is registered as a visual-attention top-k selection adaptation. Because
Qwen3.8 has no CLS token, the adapter pools final-layer mean-received attention
over native merger groups and keeps tokens in their original spatial order.

VisPruner combines mean-received attention saliency with iterative diversity
selection. The adapter uses final-layer pre-merger patch states pooled to the
merger-token grid as selection features, and the same attention capture used by
the other visual-cue methods.

AIM merges projected merger-token features by alternating-source bipartite
matching at the same pre-LLM boundary; its smoke uses an exact 50% output budget.

RESTORE uses the shared VisPruner saliency/diversity selector for its pristine
subset, then merges residual tokens into anchors while retaining representative
layout and token-mass metadata.

The `vqtoken.video.qwen3_5` binding clusters post-merger video features before
LLM scatter, retaining the canonical 32 tokens per video. DivPrune uses cosine
greedy max-min selection; its Qwen3.5 recipe targets 50% feature-token
retention to match the existing ToMe comparison budget, rather than the generic
DivPrune kernel default. Both pass compact features through the model's native
`mm_encoder_outputs` argument and compacts image placeholders, attention masks,
`mm_token_type_ids`, M-RoPE positions, and decode-time rope delta together. The
ToMe binding is a framework-integration adaptation, not a ToMe paper
reproduction.

Use the isolated Transformers 5.19 runtime and local checkpoint. The smoke YAMLs
are `configs/eval/lmms_eval/qwen3_5_mme_smoke.yaml` and
`configs/eval/lmms_eval/qwen3_5_tome_mme_smoke.yaml`; they run one locally cached
MME sample with batch size 1. Set `HF_HOME` to the user cache containing the
official `kernels-community/finegrained-fp8` kernel, keep dataset cache in a
writable project directory, and use `HF_DATASETS_OFFLINE=1` to prevent dataset
downloads. Generation currently uses the Transformers PyTorch fallback for
Gated DeltaNet when FLA/causal-conv1d are absent.

The VisionZip GPU smoke configuration is
`configs/eval/lmms_eval/qwen3_5_mme_smoke_visionzip.yaml` (two local MME samples,
batch size 1). The completed run is under
`outputs/qwen3_5-eval/visionzip-smoke-gpu1-r2-20261012`; it uses the same 50%
retention target as the ToMe/DivPrune comparison.

The single-video LMMS-Eval smoke configs are
`configs/eval/lmms_eval/qwen3_5_video_smoke_base.yaml` and
`configs/eval/lmms_eval/qwen3_5_video_smoke_vqtoken.yaml`. They use the local
MVBench action-sequence task with `sample_limit: 1`; this validates integration,
not benchmark quality.

The LMMS-Eval plugin registers a process-exit cleanup callback for the platform
runtime. Image and video compression activations have explicit idempotent
`uninstall()` methods used by `CompressionSession` rollback and lifecycle tests.
