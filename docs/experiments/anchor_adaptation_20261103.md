# K128 warm-start adaptation

Four config-selected arms share the original 2560-image manifest, seed42,
global batch8, 320 additional optimizer steps, and CE + 0.5 dense-teacher KL.
All start from exactly the old step320 aggregation MLP, with a fresh optimizer.
The learning rate is 1e-4 for every arm, half the historical from-scratch rate;
AdamW weight decay .01, warmup16, cosine schedule and clipping1 are shared.
The original model, merger, LLM, image-context anchor scorer and files are frozen.

| Arm | Anchors | Pooling | Updated parameters |
|---|---|---|---|
| T0_continue | Historical cosine FPS | Mean | Old MLP |
| Train_R1 | Fixed importance FPS | Mean | Old MLP |
| Train_T2 | Fixed importance FPS | Original normalized probabilities | Old MLP |
| Train_T2_pool | Same fixed importance FPS | Independent scorer copy's normalized probabilities | MLP + pool scorer copy |

T0 controls for 320 more training steps and a second pass over the same data.
Compare each adapted arm to its own frozen counterpart, and compare changes
against T0's continued-training change. These are additional steps after the
historical MLP's 320, not a new from-scratch 320-step training comparison.
There is no zero-init arm or test-score-based checkpoint selection in this batch.

The pool-only scorer copy starts identically to the frozen image-context scorer.
Its gradient flows through continuous within-cluster weights, with discrete
anchors/membership chosen by the separate, unchanged original scorer. It
cannot change which anchors are selected. This separates pool adaptation
from learning a different discrete partition. No answer text enters either scorer.
The same normalized-sigmoid probabilities as frozen T2 are used at initialization;
they are not silently replaced by score softmax. No mask BCE is used.

The student physically processes K128 visual tokens. The frozen dense teacher
still processes the original visual sequence for KL. Freezing the LLM's weights
does not remove student backpropagation through the LLM to the MLP/pool weights.
All positions, placeholders, masks and answer labels use existing native compact
helpers. This batch keeps centroid positions and does not introduce query
conditioning, dynamic K or a new token-recovery mechanism.

Each arm first completes a two-step GPU smoke with a checkpoint at step1 and
resume to step2, in a separate directory. The formal optimizer restarts from
the common old MLP after smoke. Fixed step320 then enters full native ChartQA
2500, OCRBench1000 and corrected AI2D3088 evaluation. Original dense references
are imported only after the existing content/protocol checks; changed trained
heads cannot reuse frozen R1/T2 predictions. The pool-tuned arm explicitly loads
its separately saved pool scorer during deployment.

Artifacts retain frozen source/config hashes, manifest checksum, identities,
MLP/scorer inputs, optimizer/scheduler/RNG state, per-step sample IDs and losses,
gradient norms, original/fine-tuned checkpoints, native predictions and metrics.
Shared machine elapsed times are descriptive, not an acceleration benchmark.

The local supervisor launches one arm per free selected GPU; two free cards can
run different arms concurrently, each using accumulation8 to preserve global
batch8. It does not put additional training on GPUs with active compute processes,
and does not stop other users. If both cards are occupied, it records
`waiting_for_gpu` and persists after the terminal disconnects. A failed owned
worker stops this family and preserves its traceback; it is not silently skipped.

```bash
bash workflow.sh train-plan --profile configs/workflow/profiles/p6000.json \
  --training-config configs/workflow/training/anchor_adaptation.json
bash workflow.sh train --profile configs/workflow/profiles/p6000.json \
  --training-config configs/workflow/training/anchor_adaptation.json \
  --launch --wait-for-gpu
bash workflow.sh status --run-dir /absolute/path/to/training_family
```

LMU submission and GitHub publishing are outside this launch. Historical
checkpoints, results and other worktrees remain read-only.
