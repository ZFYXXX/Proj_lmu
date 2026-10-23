"""Public record contracts and state/operation invariants."""

from __future__ import annotations

from mllm_comp.analysis.records import AnalysisSample
from mllm_comp.backbones.structural_handles import StructuralHandles
from mllm_comp.core import OperationAction, OperationApplyResult, OperationExportRecord
from mllm_comp.core.invariants import ForwardStateInvariant, OperationPlanInvariant
from mllm_comp.core.manifest import BackendRecord, CompressionRunManifest
from mllm_comp.core.serialization import to_jsonable
from mllm_comp.core.state import ForwardState, TokenSpan


class _TensorStub:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def test_analysis_sample_has_required_keys() -> None:
    assert AnalysisSample.__required_keys__ == frozenset(
        {"sample_id", "task_name", "processor_outputs"}
    )


def test_operation_result_and_manifest_export_records() -> None:
    export_record = OperationExportRecord(
        action_id="a",
        op="skip",
        target_kind="layer",
        target_scope="llm",
        module_path="llm.layer[0]",
        attention_routes=[],
        target_indices=None,
        original_shape=None,
        pruned_shape=None,
        export_signature="sig",
    )
    result = OperationApplyResult(export_records=[export_record])

    manifest = CompressionRunManifest(
        run_id="run",
        created_at="now",
        config_path="config.yaml",
        config_hash="hash",
        source_commit=None,
        env_lockfile_hash=None,
        backbone_name="qwen25_omni",
        backbone_version=None,
        backbone_code_origin="transformers_native",
        backbone_source_ref="source",
        model_path_or_id="model",
        compressor_name="segmentwise_topk_audio",
        compressor_version=None,
        compressor_config={},
        train_backend=BackendRecord(name="ms_swift", version="unknown"),
        eval_backend=None,
        dataset_id="dataset",
        dataset_commit=None,
        eval_tasks=[],
        operation_export_records=result.export_records,
    )

    payload = to_jsonable(manifest)
    assert payload["operation_export_records"][0]["export_signature"] == "sig"
    assert payload["eval_backend"] is None


def test_forward_state_invariant_rejects_unsorted_overlapping_and_unbounded_spans() -> None:
    state = ForwardState(
        input_ids=_TensorStub((1, 10)),
        sample_ids=["sample-0"],
        modality_token_spans=[
            TokenSpan(sample_index=0, start=5, end=7, modality="audio"),
            TokenSpan(sample_index=0, start=1, end=2, modality="video"),
            TokenSpan(sample_index=0, start=6, end=11, modality="audio"),
        ],
    )

    violations = ForwardStateInvariant().check(state)

    assert any(
        violation.name == "modality_span_sorted_by_sample_and_start"
        for violation in violations
    )
    assert any(violation.name == "modality_span_non_overlapping" for violation in violations)
    assert any(violation.name == "modality_span_bounds" for violation in violations)


def test_structural_handles_iterates_layers_in_index_order() -> None:
    handles = StructuralHandles({"llm.layer[1]": object(), "llm.layer[0]": object()})
    assert [idx for idx, _ in handles.iter_indexed("llm.layer")] == [0, 1]


def test_operation_plan_invariant_signature_and_checks() -> None:
    action = OperationAction(
        action_id="skip_0",
        op="skip",
        target_kind="layer",
        target_scope="llm",
        module_path="llm.layer[0]",
        apply_mode="runtime_gate",
        shape_effect="preserve_shape",
        original_shape=None,
        pruned_shape=None,
        requires_checkpoint_export=False,
    )

    violations = OperationPlanInvariant().check(
        actions=[action],
        structural_handles=StructuralHandles({"llm.layer[0]": object()}),
        composition_policy=None,
    )

    assert violations == []


def test_operation_plan_invariant_requires_export_signature_when_exporting() -> None:
    action = OperationAction(
        action_id="prune_0",
        op="prune",
        target_kind="ffn",
        target_scope="llm",
        module_path="llm.layer[0].ffn",
        apply_mode="structural_rewrite",
        shape_effect="preserve_io_shape",
        original_shape=None,
        pruned_shape=None,
        requires_checkpoint_export=True,
        export_signature=None,
    )

    violations = OperationPlanInvariant().check(
        actions=[action],
        structural_handles=StructuralHandles({"llm.layer[0].ffn": object()}),
        composition_policy=None,
    )

    assert any(
        violation.name == "operation_export_signature_required" for violation in violations
    )


def test_operation_plan_invariant_rejects_cycles() -> None:
    first = OperationAction(
        action_id="a",
        op="skip",
        target_kind="layer",
        target_scope="llm",
        module_path="llm.layer[0]",
        apply_mode="runtime_gate",
        shape_effect="preserve_shape",
        original_shape=None,
        pruned_shape=None,
        requires_checkpoint_export=False,
        depends_on=["b"],
    )
    second = OperationAction(
        action_id="b",
        op="skip",
        target_kind="layer",
        target_scope="llm",
        module_path="llm.layer[1]",
        apply_mode="runtime_gate",
        shape_effect="preserve_shape",
        original_shape=None,
        pruned_shape=None,
        requires_checkpoint_export=False,
        depends_on=["a"],
    )

    violations = OperationPlanInvariant().check(
        actions=[first, second],
        structural_handles=StructuralHandles({"llm.layer[0]": object(), "llm.layer[1]": object()}),
        composition_policy=None,
    )

    assert any(violation.name == "operation_dependency_acyclic" for violation in violations)


def test_operation_plan_invariant_treats_expert_as_width_wise() -> None:
    action = OperationAction(
        action_id="prune_expert_0",
        op="prune",
        target_kind="expert",
        target_scope="llm",
        module_path="llm.layer[0].moe.experts",
        apply_mode="structural_rewrite",
        shape_effect="change_num_experts",
        original_shape=None,
        pruned_shape=None,
        target_indices=[1],
    )

    violations = OperationPlanInvariant().check(
        actions=[action],
        structural_handles=StructuralHandles({"llm.layer[0].moe.experts": object()}),
        composition_policy=None,
    )

    assert any(violation.name == "operation_coupled_modules_required" for violation in violations)
