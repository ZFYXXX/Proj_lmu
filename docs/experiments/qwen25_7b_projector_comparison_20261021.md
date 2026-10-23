# Qwen2.5-VL-7B projector comparison, 2026-10-21

Evaluate the existing supervised feature-anchor residual MLP against the
framework's CDPruner, DPPruner, EADP and HiPrune implementations. These are
Qwen2.5 adaptations of the paper algorithms; no claim is made that the
original paper's backbone, data processing or published scores are reproduced.

Framework classification and implementation boundaries:
[comparison methods](../support/imagellm/comparison_methods.zh.md).

## Fixed experiment

| Split | Questions | Independent image contents | Generation limit |
|---|---:|---:|---:|
| ChartQA test | 2500 | 1509 | 16 |
| AI2D test | 3088 | 814 | 16 |
| OCRBench test | 1000 | 930 | 128 |
| DocVQA validation | 5349 | 1285 | 32 |

All new arms emit exactly 128 post-merger visual tokens. Original token values
and three-axis M-RoPE are retained; text positions are unchanged. CDPruner and
EADP read only the current prompt's user instruction. DPPruner and HiPrune do
not read the question. HiPrune uses native shallow/window and deep/global
vision attention, with window ordering restored by the framework's Qwen hooks.

Checkpoint, BF16 eager attention, processor bounds, prompts, greedy generation,
EOS policy, repetition penalty 1.0 and sample order match the completed 7B
runs. Dense and trained-MLP predictions are reused with checked per-sample
identity and provenance. No new training is performed. The supervised MLP
must be labeled as trained alongside the untrained paper algorithms.

Fixed K128 is a much smaller *retention fraction* on DocVQA (mean dense
1951.6 tokens, about 6.6%) than on AI2D (510.2, about 25.1%). Compare methods
within each dataset and do not equate these rows to a paper's fixed 25% result.

## Verification before full execution

1. CPU direct-kernel/cached-decode versus native binding/generate parity,
   including prompt-only selection and original position preservation.
2. Existing native Qwen projector training/generate tests.
3. All 16 historical smoke images from each dataset (64 total), each with
   native versus historical dense generation parity and four native-binding
   versus direct-kernel generation comparisons.
4. Separately record a standard lmms-eval generation-setting arm, inheriting
   repetition penalty 1.05 and tokenizer EOS, for implementation/protocol
   auditing. It does not change the primary K128 protocol after results are seen.

The five local model-weight SHA256 values were checked against the official
Hugging Face repository's LFS metadata and all match revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. This resolves the weight-identity
question, not the unresolved difference from the Bifocal OCRBench baseline.
The run saves its own API verification record. See
[official Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

## Outputs and execution

Entry: `scripts/eval/lmms_eval/run_anchor7b_projector_comparison.py`.

Output root:
`outputs/eval/qwen25_vl/anchor7b_projector_comparison_k128_20261021_v2/`.

The initial v1 smoke found a native comparison binding generation-position
bug before any full comparison ran: after pruning 160 tokens, a native first
decode position was 212 instead of 52. Selection indices already agreed.
GenerationMixin's original sequence offset was being combined with the
compaction-adjusted RoPE delta. The binding now repairs generated positions
using the actual compact KV length. Four added regression cases detect this
even when tiny random models happen to generate unchanged token IDs. v1
failure records are retained; v2 starts with a new frozen implementation.

The two GPU workers shard by the same image-content rule as the original
evaluation, one owned worker per GPU. They wait for at least 40 GiB of free
GPU memory and leave other processes running. Full evaluation starts only
after both smoke shards pass. An owned worker failure stops its sibling and
preserves all completed ledgers for recovery under the same frozen config.

Reports include native scores, deltas versus dense and MLP, image-cluster
paired bootstrap intervals (10000 draws), OCR categories, source-token bins,
paired wins/ties/losses, selection indices and trace. Timing is diagnostic
on shared GPUs; reused dense/MLP timing is not a matched speed benchmark.

Original source repositories:

- [CDPruner / DPPruner](https://github.com/Theia-4869/CDPruner)
- [EADP](https://github.com/SJTU-DeepVisionLab/EADP)
- [HiPrune](https://github.com/Danielement321/HiPrune)

Interpretation: failure to beat a trained MLP is not an equal-training-budget
test of the aggregation design. A large MLP deficit against untrained selectors
would be strong motivation to revisit grouping/selection. Similar failure
across all K128 methods would motivate a separate, frozen larger-budget study.

## AI2D output-format audit

Inspection of historical predictions found that the upstream AI2D filter only
extracts a leading `A.`-style response. A leading `C` followed by additional
text is otherwise compared as a whole response. This can confound claims of
better visual understanding with better adherence to answer format.

An explicitly **post-hoc** alternate-parser audit extracts an unambiguous leading
A-D followed by punctuation/whitespace/end. The original dense/MLP scores
74.45%/77.95% become 81.06%/79.60% under this alternate rule. Both sets are
retained. The original AI2D gain cannot establish better visual preservation.
This audit does not overwrite or modify the primary full-comparison scoring.

Reproduce the audit with `scripts/eval/lmms_eval/audit_ai2d_answer_format.py`,
passing `--source-run` and `--output`. It accepts the original completed run
or the new comparison run after its entire AI2D split is complete, and reports
all available methods under both rules with paired image-level intervals.
