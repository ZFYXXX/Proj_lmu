import json
from pathlib import Path

import pytest

from scripts.eval.lmms_eval.run_qwen35_transfer import METHODS,config,child_env,COUNTS


def test_method_matrix_excludes_noise_and_does_not_claim_bifocal_stage2():
    assert len(METHODS)==9
    assert "bifocal_stage1" in METHODS
    assert not any("noise" in x or "stage12" in x for x in METHODS)
    assert sum(COUNTS.values())==6588


def test_dense_native_config_and_external_lifecycle(tmp_path):
    identity=tmp_path/"identity.json"
    identity.write_text(json.dumps({"checkpoint":"/models/Qwen3.5-9B","revision":"local-sha256:test"}))
    protocol={"identity_path":str(identity)}
    dense=config(tmp_path,protocol,"dense","full","chartqa",tmp_path/"run")
    assert dense["backbone"]["name"]=="qwen3_5"
    assert dense["pipeline"]["stages"]==[]
    assert dense["eval"]["sample_limit"] is None
    assert dense["eval"]["generation"]["repetition_penalty"]==1.05
    for method in ("eadp","prunesid","mmtok","bifocal_stage1"):
        cfg=config(tmp_path,protocol,method,"smoke","ai2d",tmp_path/"run")
        stage,=cfg["pipeline"]["stages"]
        assert stage["execution_phase"]=="generate"
        assert stage["config"]["target_count"]==128
        assert cfg["eval"]["sample_limit"]==8


def test_untrained_head_cannot_enter_full_eval(tmp_path):
    identity=tmp_path/"identity.json"
    identity.write_text(json.dumps({"checkpoint":"/models/Qwen3.5-9B","revision":"test"}))
    with pytest.raises(FileNotFoundError):
        config(tmp_path,{"identity_path":str(identity)},"protected_selector","full","chartqa",tmp_path/"run")
