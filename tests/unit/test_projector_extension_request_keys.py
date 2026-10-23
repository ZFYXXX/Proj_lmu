import pytest

from scripts.eval.lmms_eval.run_qwen25_projector_extension import request_key


def test_request_ordinal_is_not_part_of_document_identity():
    smoke = "qwen25_7b_ratio_chartqa:test:7:24"
    full = "qwen25_7b_ratio_chartqa:test:7:8421"
    assert request_key(smoke) == request_key(full)
    assert request_key(smoke) != request_key("qwen25_7b_ratio_chartqa:test:8:24")


@pytest.mark.parametrize("value", [
    "qwen25_7b_ratio_chartqa:validation:7:24",
    "qwen25_7b_ratio_docvqa_val:test:7:24",
    "unknown:test:7:24",
    "qwen25_7b_ratio_chartqa:test:-1:24",
    "qwen25_7b_ratio_chartqa:test:2500:24",
    "qwen25_7b_ratio_chartqa:test:7:abc",
    "qwen25_7b_ratio_chartqa:test:7",
])
def test_request_key_does_not_hide_invalid_task_split_or_document(value):
    with pytest.raises(ValueError):
        request_key(value)
