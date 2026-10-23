import json
from pathlib import Path

import pytest

from scripts.diagnostics import organize_projector_workspace as cleanup


@pytest.mark.parametrize("path", ["/home/WuHao/file", "/home/lixiaocong/file", "/tmp/file", "/", "/home/ZhouFengyu"])
def test_other_users_and_broad_paths_are_out_of_scope(path):
    with pytest.raises(ValueError, match="scope"):
        cleanup.guarded(Path(path), must_exist=False)


def test_actual_archive_manifest_never_leaves_owner_scope():
    path = cleanup.ARCHIVE / "moves.json"
    if not path.exists():
        pytest.skip("Operational archive not present in this installation")
    data = json.loads(path.read_text())
    for row in data["moves"]:
        assert row["status"] == "archived"
        assert Path(row["source"]).is_relative_to(cleanup.WORKSPACE)
        assert cleanup.guarded(Path(row["destination"])).is_relative_to(cleanup.ARCHIVE)
        assert not Path(row["source"]).exists()
        assert not any(part in {"outputs", "models", "datasets", "checkpoints"}
            for part in Path(row["source"]).relative_to(cleanup.WORKSPACE).parts)


def test_restore_refuses_existing_destinations(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "WORKSPACE", tmp_path)
    source, archived = tmp_path / "original", tmp_path / "archived"
    source.mkdir()
    archived.mkdir()
    ledger = tmp_path / "ledger.json"
    from mllm_comp.core.serialization import atomic_write_json
    atomic_write_json(ledger, {"moves": [{"source": str(source), "destination": str(archived)}]})
    monkeypatch.setattr("sys.argv", ["cleanup", "--restore", str(ledger)])
    with pytest.raises(ValueError, match="overwrite"):
        cleanup.main()
