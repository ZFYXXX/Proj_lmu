import pytest
from mllm_comp.analysis.common.condition_ledger import append_json, records


def test_ledger_roundtrip_partial_tail_and_mismatch(tmp_path):
    path = tmp_path / "conditions.jsonl"
    row = {"sample_id": "x", "variant": "dense", "config_hash": "abc"}
    append_json(path, row)
    with path.open("ab") as stream:
        stream.write(b'{"sample_id":')
    assert records(path, "abc") == {("x", "dense"): row}
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(ValueError):
        records(path, "other")
    append_json(path, row)
    with pytest.raises(ValueError):
        records(path, "abc")
