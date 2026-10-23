# Corrected scope: native dynamic resolution, fixed K128 output

The user clarified that compatibility with Qwen dynamic resolution does not
mean changing the compression budget to 25%, nor adopting Bifocal's
layer-averaged budget. The ratio experiment is superseded and its existing
outputs are preserved. Only its owned scheduler and workers were stopped.

For each image, use the native processor resolution and its actual post-merger
rectangular grid. Compress to K_i=min(N_i,128), using the already trained K128
adapter. Dynamic resolution concerns input geometry and sequence lifecycle;
it does not dictate a retention rule. Record N_i, K_i and the realized K_i/N_i
as descriptive statistics, not optimization or matching targets. No equality
between token reduction, FLOPs and wall-clock speed is assumed.

The corrected matrix uses the actual `mllm-comp eval` CLI for baseline,
feature-anchor MLP, CDPruner, DPPruner, EADP and HiPrune. The platform's Bifocal
Stage 1 binding has no exact K128 interface and is excluded from this fixed-K
matrix; it is not silently converted to a ratio or labeled a full reproduction.
The existing task splits, processor, prompt, generation and scoring are held
constant across methods. Baseline is rerun under the same platform protocol.

Full splits: ChartQA 2500, AI2D 3088, OCRBench 1000, DocVQA validation 5349.
Each method first completes eight smoke requests per task, then the full
matrix proceeds with one owned worker per GPU. No training is started.

Output: `outputs/eval/qwen25_vl/projector_dynamic_k128_platform_20261021_v1/`.
