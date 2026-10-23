"""The semantic recorder observes real shared calls and rejects false evidence."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mllm_comp.compressors.image import decoder_selection
from mllm_comp.compressors.image.pyramid_drop import PyramidDropSpec
from tests.image_compression_semantic_audit import (
    SemanticAuditRecorder,
    _pyramid_audit,
    _visionzip_audit,
)


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("corruption", [None, "group_mean", "bin_center"])
def test_visionzip_audit_uses_released_target_sampling_and_merge(corruption):
    tokens = torch.arange(0, 12, 2, dtype=torch.float32).reshape(6, 1)
    attention = torch.tensor([[6., 5., 4., 3., 2., 1.]]).repeat(6, 1)
    keys = torch.tensor([[1., 0.]]).repeat(6, 1)
    # Highest attention selects token 0. Prefix sampling selects token 1;
    # its value is 2 + mean(4, 6, 8, 10) = 9, not the whole-group mean 6.
    target, merged = 1, 9.
    if corruption == "group_mean":
        merged = 6.
    elif corruption == "bin_center":
        target, merged = 3, 12.
    result = SimpleNamespace(
        tokens=torch.tensor([[0.], [merged]]),
        indices=torch.tensor([0, target]),
        source_groups=((0,), (1, 2, 3, 4, 5)),
        metadata={"attention_mode": "mean_received", "protected_indices": (),
                  "dominant_indices": (0,), "contextual_indices": (target,)},
    )
    if corruption:
        with pytest.raises(AssertionError, match="assigned mean|prefix-stride"):
            _visionzip_audit(tokens, attention, keys, {}, result)
    else:
        assert _visionzip_audit(tokens, attention, keys, {}, result)["maximum_value_error"] == 0


def _inputs():
    hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    image = torch.tensor([1, 2, 3, 4])
    keys = torch.tensor([
        [[2., 1.], [1., 2.]], [[1., 0.], [0., 1.]],
        [[0., 1.], [1., 0.]], [[-1., 0.], [0., -1.]],
        [[0., -1.], [-1., 0.]], [[4., 3.], [3., 4.]],
    ])
    signals = dict(
        instruction_query=torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]]]),
        image_keys=keys[image], attention_keys=keys, image_count=4,
        image_key_indices=image,
        attention_key_mask=torch.tensor([[True, True, True, True, True, False], [True] * 6]),
    )
    return hidden, image, signals


def _install(monkeypatch):
    # PyramidDrop is deliberately absent from the native adapter: only the
    # shared scoring/selection symbols execute after the refactor.
    adapter = SimpleNamespace(tome_merge_image=lambda *a, **kw: None,
                              visionzip_image=lambda *a, **kw: None)
    recorder = SemanticAuditRecorder()
    recorder.install(monkeypatch, adapter)
    return recorder


@pytest.mark.parametrize("precomputed", [True, False])
def test_recorder_audits_shared_scoring_and_selection_without_retaining_tensors(monkeypatch, precomputed):
    hidden, image, signals = _inputs()
    recorder = _install(monkeypatch)
    mark = recorder.mark()
    if precomputed:
        kwargs = {"image_attention_scores": decoder_selection.pyramid_attention_scores(**signals)}
    else:
        kwargs = {key: value for key, value in signals.items() if key not in {"image_count", "image_key_indices"}}
    decoder_selection.pyramid_drop(hidden, image_indices=image, spec=PyramidDropSpec(),
                                   protected_indices=(4,), **kwargs)
    summary = recorder.since(mark)
    assert len(summary["pyramid_attention"]) == len(summary["pyramid_drop"]) == 1
    assert summary["pyramid_attention"][0]["instruction_query_count"] == 2
    assert summary["pyramid_attention"][0]["attention_scores_match_reference"]
    assert summary["pyramid_drop"][0]["layout_matches_reference_selection"]
    assert summary["pyramid_drop"][0]["maximum_value_error"] == 0
    json.dumps(vars(recorder))


def test_score_audit_rejects_incorrect_native_attention_probabilities(monkeypatch):
    _, _, signals = _inputs()
    monkeypatch.setattr(decoder_selection, "pyramid_attention_scores", lambda **kwargs: torch.zeros(4))
    _install(monkeypatch)
    with pytest.raises(AssertionError, match="visible-attention score"):
        decoder_selection.pyramid_attention_scores(**signals)


@pytest.mark.parametrize("corruption", ["budget", "selection", "values", "layout"])
def test_selection_audit_rejects_wrong_budget_tokens_values_and_layout(corruption):
    hidden, image, signals = _inputs()
    kwargs = dict(image_indices=image, image_attention_scores=decoder_selection.pyramid_attention_scores(**signals),
                  spec=PyramidDropSpec(), protected_indices=(4,))
    result = decoder_selection.pyramid_drop(hidden, **kwargs)
    if corruption == "budget":
        kwargs["spec"] = PyramidDropSpec(keep_ratio=0.75)
    elif corruption == "selection":
        # Keep the protected token and budget, but substitute a dropped image
        # for the score-selected token so the counterexample cannot be a no-op.
        wrong_kept = sorted((int(result.dropped_indices[0]), *kwargs["protected_indices"]))
        assert len(wrong_kept) == result.kept_image_indices.numel()
        assert set(wrong_kept) != set(result.kept_image_indices.tolist())
        result = replace(result, kept_image_indices=torch.tensor(wrong_kept))
    elif corruption == "values":
        result = replace(result, hidden_states=result.hidden_states + 1)
    else:
        delta = SimpleNamespace(**vars(result.layout_delta))
        delta.old_to_new = {**dict(delta.old_to_new), 0: None}
        result = replace(result, layout_delta=delta)
    with pytest.raises(AssertionError):
        _pyramid_audit(hidden, kwargs, result)
