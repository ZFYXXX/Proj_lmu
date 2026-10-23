# Additional projector-boundary baselines, Qwen2.5-VL-7B

Only GPU 0 is authorized. GPU 1 is reserved and never selected by this queue.
No training, checkpoint change, image resizing policy change or percentage
budget is introduced. The original dynamic rectangular grids are retained;
the actual post-merger output is min(N,128).

All dataset loading, prompts, generation, scoring and sample logs go through
`python -m mllm_comp.cli.main eval --config ...`. The scheduler has no model
or generation loop. Tasks remain ChartQA 2500, AI2D 3088, OCRBench 1000 and
DocVQA validation 5349: 11937 questions per full method.

## Queue and interpretation

1. DivPrune K128: farthest-first diversity selection, using the framework's
   shared kernel and native post-merger Qwen binding.
2. VisionZip K128: 108 dominant and 20 contextual tokens with mean-received
   visual attention and native vision keys. This is explicitly a Qwen
   post-merger adaptation, not the original pre-projector placement.
3. AIM input merge K128: only the input bipartite merging component. It is
   not the complete AIM method with decoder pruning.
4. GAP attention K128: the framework's received-vision-attention Top-K with
   preserved positions, not a certified complete paper reproduction. The
   registry itself describes its selector as a bundled baseline.
5. HiPrune Qwen L16 K128: explicit object_layer=16, alpha=.1, vision_layer=-1.
6. EADP beta2 K128: explicit alpha=.5 and beta=2, matching the official Qwen
   run-script setting. Native resolution is intentionally retained rather
   than adopting its published fixed-square preprocessing.

Original reference repositories:

- https://github.com/vbdi/divprune
- https://github.com/JIA-Lab-research/VisionZip
- https://github.com/Danielement321/HiPrune/tree/82781005a7e72a6be9ede58fd77473efa72b5e4f
- https://github.com/SJTU-DeepVisionLab/EADP/tree/e1a08801461c181871b9ff1cb0803b8e9966f083

PruMerge+ is not reintroduced, respecting the earlier exclusion. ToMe and
SAINT modify encoder execution; decoder pruners and complete two-stage
Bifocal are not relabeled projector-only methods. Merely existing in the
registry does not establish real-checkpoint quality or official equivalence.

## Mandatory execution gates

Four new generate bindings use the existing native projector-only lifecycle.
New CPU tests use actual small Transformers Qwen models and verify exact
cardinality, native cache lifecycle and first-decode M-RoPE, as well as
identity-budget token/logit agreement. Public registry phases are tested too.
Previously supported multi-image *forward* is preserved; the new generate
scope remains one image, batch size one.

Verification: 128 related CPU regressions pass; all 16 native eval plans pass.
Four pre-existing exact-inventory assertions in `test_imagellm_registry.py`
still fail (variant counts, binding counts, support counts and conflict sets).
The same four fail after restoring the prior AIM/GAP phase declarations in
memory. These are recorded separately, not suppressed or relabeled as passes.

Before full evaluation, each of the four new bindings runs a 32-request real
7B identity-budget check against the previous platform dense logged outputs.
The logged AI2D response is filtered, so this gate alone does not certify raw
AI2D generation identity. Each of all six methods also runs the same 32-request
compressed smoke. These 320 requests are implementation gates, not statistical
evidence of task quality. All gates must pass before any full run begins.

The native ledgers are checked for question/target agreement, complete sample
IDs, finite scores, actual K, source token counts against the previous run,
explicit HiPrune/EADP parameters, and verified native projector decode telemetry.
Any failed gate stops the queue and preserves its outputs; no silent skipping.

## Reproducibility and outputs

The new protocol pins code and configuration hashes, model identity and prior
baseline/MLP provenance. Prior full results are preserved and never rewritten.
The baseline and trained MLP references use identical task/generation settings;
training-free and trained results remain labeled separately. Shared-GPU timing
is not presented as a controlled speed comparison. AI2D primary and auxiliary
leading-option metrics remain distinct.

Entry: `scripts/eval/lmms_eval/run_qwen25_projector_extension.py`.
Output: `outputs/eval/qwen25_vl/projector_extension_k128_platform_20261022_v3/`.
The v1 preparation directory is retained; its plan-only gate discovered that
AIM's declared generate phase also needed promotion alongside the implemented
provider lifecycle. No real model evaluation was launched in that directory.
The v2 directory preserves a completed first 32-request identity evaluation and
its validator failure. Native sample IDs include an execution ordinal, which
changes with length sorting and sample limits. The v3 validator matches the
stable task/split/document ID instead, while retaining duplicate and source-N
checks. Rechecking those v2 outputs passed all 32 dense responses and all 32
decode-position checks. The v3 queue starts with the corrected validation and
does not reuse or rewrite the historical predictions.

Outputs include protocol, configs, supervisor manifest, native request traces,
native scores/predictions, per-job checks, and a summary/report after all full
methods complete. Source/identity/cache checks do not by themselves establish
equivalence to the authors' complete implementations or published scores.
