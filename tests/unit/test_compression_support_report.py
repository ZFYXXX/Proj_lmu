from __future__ import annotations

from mllm_comp.compressors.support_report import (
    builtin_imagellm_rows,
    render_support_matrix_markdown,
)


def test_builtin_report_contains_every_generate_cell() -> None:
    rows = builtin_imagellm_rows()

    # 63 ImageLLM cells + the declared bifocal.image.qwen25_vl cell.
    assert len(rows) == 64
    assert len({(row.algorithm, row.backbone) for row in rows}) == 64
    assert {row.target_modality for row in rows} == {"image"}
    assert {row.phase for row in rows} == {"generate"}
    assert {row.status for row in rows} == {"declared", "active"}
    assert sum(row.status == "active" for row in rows) == 42
    assert all(
        row.missing_evidence == ()
        if row.status == "active"
        else "real_checkpoint" in row.missing_evidence
        for row in rows
    )


def test_forward_report_marks_only_evidence_backed_cells_active() -> None:
    rows = builtin_imagellm_rows(phase="forward")

    assert len(rows) == 73
    assert {row.phase for row in rows} == {"forward"}
    assert {row.status for row in rows} == {"declared", "active"}
    assert sum(row.status == "active" for row in rows) == 36
    active = tuple(row for row in rows if row.status == "active")
    assert all(row.missing_evidence == () for row in active)
    assert all(
        row.passing_evidence
        == ("kernel", "cpu_contract", "real_checkpoint", "train_step")
        for row in active
    )


def test_markdown_report_is_stable_and_names_bindings() -> None:
    rendered = render_support_matrix_markdown(builtin_imagellm_rows())

    assert rendered.startswith("| Algorithm | Backbone |")
    assert rendered.count("\n|") == 65
    assert "bifocal.image.qwen25_vl" in rendered
    assert "visionzip.image.qwen3_vl" in rendered
    assert "pyramid_drop.image.llava_onevision" in rendered
