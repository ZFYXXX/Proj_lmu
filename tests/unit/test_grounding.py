import pytest

from mllm_comp.integrations.lmms_eval.grounding import (
    box_scores, normalize_prediction, parse_box, qwen25_input_size, rescore_rows,
)


@pytest.mark.parametrize("raw", [
    '[100, 200, 500, 600]', '[{"bbox_2d": [100, 200, 500, 600]}]',
    '```json\n{"box": [100, 200, 500, 600]}\n```',
    'The box is (100, 200), (500, 600).',
])
def test_explicit_bbox_formats(raw):
    assert parse_box(raw) == [100, 200, 500, 600]


@pytest.mark.parametrize("raw", ['object 1 is at 2 with 3 of 4', '[true, 2, 3, 4]', '[NaN, 2, 3, 4]'])
def test_invalid_boxes_do_not_become_predictions(raw):
    assert parse_box(raw) is None
    pred, status = normalize_prediction(raw, "qwen3_vl")
    assert status == "parse_error"
    assert all(value == 0 for value in box_scores(pred, [0, 0, 1, 1]).values())


def test_coordinate_protocol_and_processed_size():
    raw = '[100, 200, 500, 600]'
    assert normalize_prediction(raw, "qwen3_vl")[0] == [0.1, 0.2, 0.5, 0.6]
    assert normalize_prediction(raw, "qwen25_vl", input_size=(500, 800))[0] == [0.2, 0.25, 1, 0.75]
    assert qwen25_input_size(560, 280, 784, 39200) == (280, 140)
    with pytest.raises(ValueError, match="verified processed"):
        normalize_prediction(raw, "qwen25_vl")


def test_rescore_preserves_raw_and_uses_normalized_iou():
    row = {"doc_id": 1, "filtered_resps": ['[100, 200, 500, 600]'],
           "refcoco_IoU": {"answer": "cat", "ann_id": 7, "bbox": [0.1, 0.2, 0.5, 0.6]},
           "unrelated": "retained"}
    corrected, metrics, invalid = rescore_rows([row], "qwen3_vl")
    assert corrected[0]["filtered_resps"] == row["filtered_resps"]
    assert corrected[0]["unrelated"] == "retained"
    assert metrics["refcoco_IoU,none"] == 1 and invalid == 0
    assert "pred" not in row["refcoco_IoU"]


def test_scorer_install_updates_model_context_without_stacking(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from mllm_comp.integrations.lmms_eval.grounding import install_grounding_scorer
    module = ModuleType("lmms_eval.api.task")
    class Task:
        config = SimpleNamespace(task="refcoco_bbox_rec_testB")

        def process_results(self, doc, results):
            return {"native": results}

    module.ConfigurableTask = Task
    monkeypatch.setitem(sys.modules, "lmms_eval.api.task", module)
    task = Task()
    doc = {"answer": "cat", "question_id": 1, "bbox": [0.1, 0.2, 0.5, 0.6]}
    raw = ['[100, 200, 500, 600]']
    install_grounding_scorer("qwen3_vl", 784, 39200)
    assert task.process_results(doc, raw)["refcoco_IoU"]["pred"] == doc["bbox"]
    task.config = SimpleNamespace(task="mmstar")
    assert task.process_results(doc, raw) == {"native": raw}
    task.config = SimpleNamespace(task="refcoco_bbox_rec_testB")
    install_grounding_scorer("qwen25_omni", 784, 39200)
    assert task.process_results(doc, raw) == {"native": raw}
