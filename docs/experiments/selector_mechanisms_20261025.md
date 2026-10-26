# Mechanism supplement: not another score-only comparison

Do NOT modify/restart the ongoing question-condition full run. Prepare a new
hash-frozen source snapshot and queue this supplement after that run completes.
No training, no parameter change, no automatic Git commits. Qwen2.5-VL-7B,
the original question-context selector checkpoint, K128, native resolution,
prompts/scorers/generation settings remain unchanged.

## 1. Internal path

Measure true versus matched-wrong-question logits on identical visual states:
common scalar shift versus centered differential change, exact score-energy
closure, tie-aware Spearman/rank displacement, K/K+1 margin, exchanged tokens
and concentration near the cutoff. Record 8x8 spatial histograms on the actual
rectangular post-merger grid. Recompute the real shared-condition -> GELU ->
score path and measure its first-order residual. The first-order approximation
is NOT an exact attribution identity; the common/centered energy identity is.
No complete hidden states or raw score arrays are saved by default.

## 2. Candidate-token causality

Sample250 independent images/task, one hash-selected question/image,750 globally
deduplicated images total. Sampling does not inspect answers, predictions or
success/failure. Keep all selected images and report inapplicability explicitly.

Let A be the true-question selected set, B the wrong-question set, L=A\B and
G=B\A. Select up to8 target lost/gained tokens by the question-induced score
change. Select disjoint equally sized controls from the SAME L/G populations,
matched by spatial quadrant and feature norm where available, with deterministic
sampling among nearest8. Every quadrant relaxation is recorded. If fewer than2
exchanged tokens exist, no disjoint-control mediation claim is possible.

Four crossed restorations `B - G_(t/c) + L_(t/c)` and four crossed removals
`A - L_(t/c) + G_(t/c)` separate adding back candidates from removing distractors.
Every state has exactly K original token contents/positions. Run observe plus
eight variants through native `mllm-comp eval`:6750 full requests plus216 smoke
requests. Observe must reproduce the prior wrong-question predictions and all
true-condition selected sets must match the historical fixed selector.

Report native scores and image-paired intervals for candidate-addition versus
control, distractor-removal versus control, and reverse removal. These tests
identify consequences of the specified token replacements, not unique semantic
causes. Targets are scorer-affected CANDIDATES, never automatically called GT.
Coverage/norm/rank correlations alone do not establish a mechanism.

## 3. Evidence identity and review

The earlier1000-image regional interaction study used Qwen2.5-VL-3B and concluded
insufficient confirmatory evidence. Its report is retained as background, not
converted into evidence labels for the current7B selector.

Produce a blind annotation task list for the independently selected images,
without predictions or selected-token overlays. Evidence schema requires image
fingerprint, normalized-original-image coordinates, multiple named box groups,
verified status and human/dataset provenance/reviewer. Until independent review,
coverage is `pending_review`, not zero/false and not a saliency pseudo-label.
The automatic run may finish while semantic grounding remains pending. Verified
annotations require a separately pinned extension, not editing a frozen run.
`render_selector_evidence_review.py --root RUN` exports an original-image,
answer/selection-blind local page (100 images/task by hash by default). A reviewer
explicitly confirms boxes and exports JSON; nothing is automatically verified.
After observe cells finish, `run_selector_mechanisms.py review-report --root RUN
--annotations EXPORT.json` validates provenance/coordinates and writes a new
`semantic_extensions/<annotation SHA>/` coverage report. It never changes the
GPU-run manifest or silently promotes score-based candidates into semantic GT.

## Products and boundaries

`cohort.json`, label-free condition manifest, empty verified-evidence manifest,
blind review tasks, per-request scorer/path/spatial/swap trace, native predictions,
case JSONL, image-paired contrasts, and explicit annotation-completeness status.
All failures stop the corresponding queue; old outputs/weights remain intact.
No small demonstration is treated as the full population, and no result is
called a root cause merely because a final task score moved.
