# Balanced R1 step320 MLP

This is the completed `R1_balanced_data_scratch` arm from `r1_next_subset_20261104_v2`.
Qwen2.5-VL-7B; importance-guided FPS; K128; ordinary mean pooling; residual aggregation MLP.
Only the 1,846,016-parameter MLP was trained for 320 steps on 2,560 independent images.
The existing image-context anchor scorer remained frozen. The MLP used scratch initialization with a zeroed final residual layer.

`projector.safetensors` and `projector_config.json` are the complete inference checkpoint.
Use the original same-content Qwen2.5-VL-7B model and the existing `image_context_selector` from the Proj checkpoint package.
Check the files with `sha256sum -c SHA256SUMS`.

The full evaluation is still in progress. This checkpoint is a new training arm, not a claim of improved performance over the frozen R1 or T2 variants.
Optimizer/scheduler/RNG state remains in the private Hugging Face transfer repository and is not included in Git.

To submit a four-GPU evaluation on LMU from the project root:

```bash
bash scripts/eval/lmms_eval/submit_lmu_balanced_mlp_eval.sh
```

This entrypoint evaluates the bundled checkpoint and needs no training data download or retraining.
