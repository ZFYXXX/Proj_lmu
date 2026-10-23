# OneVision comparison source port

Source supplied by the project owner: `OV对比方法.zip`, directory
`Comp-Analysis-Toolkit-fuxain-comparison-methods/LLaVA-NeXT/llava/model/token_compression/`.
The accompanying LLaVA-NeXT Apache-2.0 license is reproduced in `LICENSE`.
The original source module docstrings identify paper-derived implementations,
released-code adaptations, calibration assumptions and reproduction limits.

Imported primitives: `cdpruner.py`, `eadp.py`, `hiprune.py`, `vlm_pruner.py`,
`twigvlm.py`, `core_matching.py`, `dop.py`, `shortv.py`, `skip_vision.py`,
`redundancy_lens.py`, `yopo.py`, `adallava.py`, `short_lvlm.py`,
`structural_pruning.py`, and `qwen2_flops.py`. The YOPO reference JSON is from
the same source's `assets/yopo_qwen2vl_mask14.json`; it is an explicit
Qwen2-VL reproduction option, not a mask calibrated for every supported model.

Platform changes (2026-10-03): native activation/registry integration; removable
module patches; checkpoint-safe weighted token replay; rectangular spatial
grids and window-aware HiPrune capture; native Qwen MRoPE/QK normalization and
DeepStack synchronization; reversible SCP input projection; scheduler/summary
sidecar persistence. The unused source in-place SCP materializer was removed.
VLM-Pruner exposes fusion weights to replay the exact selected aggregation.
Skip-Vision exposes its mixing weights to synchronize native DeepStack taps.

TwigVLM remains **initialization_only**. Structural-Pruning covers the supplied
ShortGPT depth branch. DOP exposes both depth-only and DOP-V with native
VisPruner selection, including grouped pre-merger features on Qwen backbones.
YOPO requires an explicit calibration source; its reference mask is opt-in.
Standalone source-loader fallbacks were removed in favor of package imports.
The source's OV calibration policies and missing binary calibration weights
are not represented as Qwen2.5-VL/Qwen3-VL calibration evidence.

See `docs/support/imagellm/comparison_methods.zh.md` for the supported execution
envelope, configuration examples, adaptation details and evidence status.
