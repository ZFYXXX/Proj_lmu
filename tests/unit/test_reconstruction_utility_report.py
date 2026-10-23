import importlib.util
import json
from pathlib import Path

import pytest


def load_report():
    file = Path(__file__).parents[2] / "scripts/analysis/report_reconstruction_utility.py"
    spec = importlib.util.spec_from_file_location("reconstruction_report_test", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_rows():
    common = {"config_hash": "hash", "sample_id": "task:1", "placeholder_mask_labels_alignment_passed": True,
              "text_positions_preserved": True, "gold_answer_in_decode_context": False,
              "retained_visual_tokens": 128, "candidate_answer_nll": 1., "primary_task_score": .5}
    names = ["dense", "summary96", "reference/feature_mean_native128", "reference/feature_representative128",
             "reference/feature_mean_centroid128", "reference/prumerge_plus128"]
    names += [f"hybrid/{name}" for name in ["high_residual", "low_residual", "attention", "pca_residual", "diversity"]]
    names += [f"hybrid/{kind}_{seed}" for kind in ["random", "spatial_random"] for seed in [11, 23, 47]]
    names += [f"spectral/{fraction:g}/{name}" for fraction in [.05, .10] for name in ["low", "high", "random_11", "random_23", "random_47"]]
    names += [f"tile/{i}" for i in range(16)]
    result = []
    for name in names:
        row = {**common, "request_id": name}
        if name == "dense" or name.startswith(("spectral/", "tile/")):
            row["retained_visual_tokens"] = 256
        if name.startswith("spectral/"):
            row["frequency_trace"] = {"actual_ac_fraction": float(name.split("/")[1])}
        if name.startswith("tile/"):
            index = int(name.split("/")[1])
            row.update({"tile_residual_score": index, "tile_pca_score": -index, "tile_attention_score": 1.,
                        "tile_feature_norm": index, "candidate_answer_nll": 1. + index / 10})
        if name == "hybrid/high_residual":
            row["primary_task_score"] = 1.
        if name == "hybrid/random_47":
            row["primary_task_score"] = .2
        result.append(row)
    return result


def test_seed_average_and_tile_correlations_are_image_level(tmp_path):
    module = load_report()
    path = tmp_path / "conditions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in fixture_rows()) + "\n")
    summary = {"condition_path": str(path), "condition_count": 43, "config_hash": "hash", "sample_id": "task:1",
               "image_fingerprint": "image", "task": "task", "source_visual_tokens": 256}
    values, _ = module.image_metrics(summary)
    assert values["effects"]["high_residual_minus_random"] == pytest.approx(.6)
    assert values["effects"]["high_residual_minus_spatial_random"] == pytest.approx(.5)
    assert values["correlations"]["tile_residual_score"]["nll_damage"] == pytest.approx(1.)
    assert values["correlations"]["tile_pca_score"]["nll_damage"] == pytest.approx(-1.)
    assert values["correlations"]["tile_attention_score"]["nll_damage"] is None
    assert values["correlations"]["tile_residual_score"]["score_damage"] is None
    rows = fixture_rows()
    rows[6]["retained_visual_tokens"] = 129
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="budget"):
        module.image_metrics(summary)


def test_partial_confirmation_never_runs_formal_inference(tmp_path):
    module = load_report()
    (tmp_path / "cohort.json").write_text(json.dumps({"entries": [{"sample_id": "t:1", "split": "confirmation"}]}))
    result = module.report(tmp_path, "confirmation")
    assert result["status"] == "incomplete"
    assert result["inference_permitted"] is False
    assert "confirmatory_tests" not in result
