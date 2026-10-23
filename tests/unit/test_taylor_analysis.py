from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.analysis.bindings import (
    layer_batches,
    require_taylor_semantics,
    resolve_analysis_binding,
)
from mllm_comp.analysis.common.sample import (
    analysis_processed_batch,
    teacher_forced_objective,
)
from mllm_comp.analysis.common.taylor import (
    FrozenModelParameters,
    activation_magnitude_reference,
    modality_neuron_partition,
    neuron_taylor_reference,
    path_block_taylor_reference,
    scalar_taylor_reductions,
    spearman_average_ties,
)
from mllm_comp.analysis.common.taylor_aggregate import (
    TaylorTaskAccumulator,
    add_bottom_up_mass_views,
    ffn_modality_cross_check,
    flatten_hierarchical_taylor,
    partition_taylor_task_scores,
)
from mllm_comp.analysis.common.rank import aggregate_task_percentiles
from mllm_comp.analysis.common.taylor_validation import (
    select_causal_validation_candidates,
    validate_causal_validation_config,
)
from mllm_comp.backbones.teacher_forcing import OBJECTIVE_NAME
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.analysis.probes import HierarchicalTaylor


def test_taylor_reductions_keep_normalized_and_direct_removal_distinct() -> None:
    report = scalar_taylor_reductions([1.0, -1.0, 2.0, -2.0])

    assert report["normalized_score"] == pytest.approx(1.5)
    assert report["direct_removal_score"] == pytest.approx(0.0)
    assert report["signed_sum"] == pytest.approx(0.0)
    assert report["element_count"] == 4


def test_taylor_reduction_reports_non_finite_without_emitting_a_rank_score() -> None:
    report = scalar_taylor_reductions([1.0, float("nan")])

    assert report["status"] == "non_finite"
    assert report["finite_count"] == 1
    assert report["non_finite_count"] == 1
    assert report["normalized_score"] is None
    assert report["direct_removal_score"] is None


def test_path_reference_uses_attention_block_value_contribution_per_query() -> None:
    reports = path_block_taylor_reference(
        attention=[[[1.0, 0.0], [0.5, 0.5]]],
        values=[[[2.0], [4.0]]],
        gradients=[[[3.0], [1.0]]],
    )

    assert len(reports) == 1
    assert reports[0]["signed_sum"] == pytest.approx(9.0)
    assert reports[0]["normalized_score"] == pytest.approx(4.5)
    assert reports[0]["direct_removal_score"] == pytest.approx(9.0)
    assert reports[0]["query_count"] == 2


def test_ffn_neuron_reference_preserves_token_and_gate_views() -> None:
    reports = neuron_taylor_reference(
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
    )
    activation = activation_magnitude_reference([[1.0, -2.0], [3.0, 4.0]])

    assert reports[0]["normalized_score"] == pytest.approx(2.0)
    assert reports[0]["direct_removal_score"] == pytest.approx(2.0)
    assert reports[1]["normalized_score"] == pytest.approx(3.0)
    assert reports[1]["direct_removal_score"] == pytest.approx(2.0)
    assert activation == pytest.approx([2.0, 3.0])


def test_modality_partition_and_spearman_use_deterministic_tie_semantics() -> None:
    partition = modality_neuron_partition(
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0],
        bottom_fraction=0.5,
    )

    assert partition["counts"] == {
        "both_redundant": 0,
        "vision_redundant_text_important": 2,
        "text_redundant_vision_important": 2,
        "both_important": 0,
    }
    assert spearman_average_ties([1.0, 1.0, 3.0], [2.0, 2.0, 4.0]) == pytest.approx(
        1.0
    )
    assert spearman_average_ties([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(
        -1.0
    )
    assert math.isnan(spearman_average_ties([1.0, 1.0], [2.0, 3.0]))


def test_audio_modality_partition_uses_audio_identity() -> None:
    partition = modality_neuron_partition(
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0],
        bottom_fraction=0.5,
        modality_name="audio",
    )

    assert partition["counts"] == {
        "both_redundant": 0,
        "audio_redundant_text_important": 2,
        "text_redundant_audio_important": 2,
        "both_important": 0,
    }


def test_teacher_forced_objective_is_required_and_uses_full_sequence_length() -> None:
    sample = {
        "sample_id": "sample",
        "task_name": "task",
        "processor_outputs": {"input_ids": [[1, 2]]},
        "teacher_forced_processor_outputs": {
            "input_ids": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]],
            "labels": [[-100, -100, 3, 4]],
        },
        "objective": {
            "objective": OBJECTIVE_NAME,
            "ignore_index": -100,
            "normalization": "mean_over_answer_tokens",
            "answer_token_counts": [2],
            "answer_token_positions_by_sample": [(2, 3)],
        },
        "metadata": {"prompt_lengths": [2]},
    }

    objective = teacher_forced_objective(sample, probe_name="Taylor")
    processed = analysis_processed_batch(
        sample,
        probe_name="Taylor",
        processor_outputs_field="teacher_forced_processor_outputs",
    )

    assert objective["objective"] == OBJECTIVE_NAME
    assert processed.prompt_lengths == [4]
    assert processed.model_inputs["labels"] == [[-100, -100, 3, 4]]

    with pytest.raises(BackendIntegrationError, match="requires objective"):
        teacher_forced_objective(
            {**sample, "objective": {"objective": "surrogate"}},
            probe_name="Taylor",
        )

    with pytest.raises(BackendIntegrationError, match="does not match its labels"):
        teacher_forced_objective(
            {
                **sample,
                "objective": {
                    **sample["objective"],
                    "answer_token_counts": [1],
                },
            },
            probe_name="Taylor",
        )


def test_taylor_binding_requires_explicit_projection_and_embedding_handles() -> None:
    v_proj = object()
    o_proj = object()
    down_proj = object()
    attention = SimpleNamespace(
        v_proj=v_proj,
        o_proj=o_proj,
        num_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        config=SimpleNamespace(hidden_size=16),
    )
    ffn = SimpleNamespace(down_proj=down_proj)
    layer = SimpleNamespace(self_attn=attention, mlp=ffn)
    embeddings = object()
    language_model = SimpleNamespace(layers=[layer], norm=object())
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=language_model, visual=None),
        lm_head=object(),
        get_input_embeddings=lambda: embeddings,
    )
    from mllm_comp.backbones.qwen25_vl import Qwen25VLModelAdapter

    backbone = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_vl"),
        model_adapter=Qwen25VLModelAdapter(),
    )

    binding = resolve_analysis_binding(backbone, model)
    require_taylor_semantics(binding, binding.layers)

    assert binding.input_embeddings is embeddings
    assert binding.layers[0].attention_value_projection is v_proj
    assert binding.layers[0].attention_output_projection is o_proj
    assert binding.layers[0].ffn_down_projection is down_proj


def test_frozen_parameter_context_restores_original_flags() -> None:
    class Parameter:
        def __init__(self, state: bool) -> None:
            self.requires_grad = state

        def requires_grad_(self, state: bool) -> "Parameter":
            self.requires_grad = state
            return self

    parameters = [Parameter(True), Parameter(False)]
    model = SimpleNamespace(parameters=lambda: iter(parameters))

    with FrozenModelParameters(model):
        assert [parameter.requires_grad for parameter in parameters] == [False, False]

    assert [parameter.requires_grad for parameter in parameters] == [True, False]


def test_hierarchical_flattening_keeps_component_and_estimator_identity() -> None:
    report = _hierarchical_report()
    add_bottom_up_mass_views(report)
    flattened = flatten_hierarchical_taylor(report)

    assert flattened["normalized_taylor"]["layer:0/path:v2v/head:1"] == 0.2
    assert flattened["direct_removal_taylor"][
        "layer:0/ffn_neuron:0/group:vision"
    ] == 1.0
    assert flattened["activation_magnitude"][
        "layer:0/ffn_neuron:1/group:text"
    ] == 4.0
    bottom_up = report["layers"]["0"]["bottom_up_mass"]["vision"]
    path_to_head = report["layers"]["0"]["path_to_head"]
    assert bottom_up["head_normalized_abs_child_mass"] == pytest.approx(0.3)
    assert bottom_up["ffn_neuron_normalized_abs_child_mass"] == pytest.approx(0.3)
    assert bottom_up["direct_whole_layer_removal_score"] == pytest.approx(3.0)
    assert path_to_head["declared_path_coverage"] == "partial"
    assert path_to_head["measured_path_types"] == ["v2v"]
    assert path_to_head[
        "measured_path_normalized_abs_child_mass_by_head"
    ] == pytest.approx([0.1, 0.2])
    assert path_to_head["direct_head_removal_score"] == pytest.approx([0.4, 0.5])
    assert flattened["normalized_taylor"][
        "layer:0/path_child_mass/head:1"
    ] == pytest.approx(0.2)
    assert "layer:0/path_child_mass/head:1" not in flattened[
        "direct_removal_taylor"
    ]


def test_path_to_head_view_rejects_inconsistent_head_geometry() -> None:
    report = _hierarchical_report()
    report["layers"]["0"]["path"]["t2v"] = {
        "status": "finite",
        "normalized_score": [0.1],
        "direct_removal_score": [0.2],
    }

    with pytest.raises(ValueError, match="query-head count"):
        add_bottom_up_mass_views(report)


def test_taylor_ranks_are_partitioned_by_comparable_granularity() -> None:
    report = _hierarchical_report()
    add_bottom_up_mass_views(report)
    scores = {
        family: {"task": values}
        for family, values in flatten_hierarchical_taylor(report).items()
    }
    partitioned = partition_taylor_task_scores(scores)

    path_scores = partitioned["normalized_taylor"]["path"]["task"]
    neuron_scores = partitioned["normalized_taylor"]["ffn_neuron"]["task"]
    assert path_scores
    assert neuron_scores
    assert not set(path_scores) & set(neuron_scores)
    ranked_paths = aggregate_task_percentiles({"task": path_scores})
    assert set(ranked_paths["components"]) == set(path_scores)


def test_taylor_layer_batches_bound_selected_capture_width() -> None:
    assert layer_batches((0, 1, 2, 3, 4), 2) == (
        (0, 1),
        (2, 3),
        (4,),
    )


def test_streaming_task_aggregation_and_ffn_cross_check_preserve_raw_tasks() -> None:
    first = _hierarchical_report()
    second = _hierarchical_report(scale=2.0)
    accumulator = TaylorTaskAccumulator()
    accumulator.update("task-a", first)
    accumulator.update("task-a", second)
    accumulator.update("task-b", first)

    scores = accumulator.scores_by_family()
    vectors = accumulator.vector_scores_by_family()
    vector_id = "layer:0/ffn_neuron/group:vision"
    assert vectors["normalized_taylor"]["task-a"][vector_id][0] == pytest.approx(
        0.15
    )
    assert vectors["normalized_taylor"]["task-b"][vector_id][0] == pytest.approx(
        0.1
    )
    assert not any(
        "/ffn_neuron:" in component
        for component in scores["normalized_taylor"]["task-a"]
    )
    cross_check = ffn_modality_cross_check(
        scores,
        bottom_fraction=0.5,
        vector_scores_by_family=vectors,
    )

    assert cross_check["layers"]["0"]["taylor"]["neuron_count"] == 2
    assert cross_check["layers"]["0"]["activation"]["partition"][
        "selected_per_modality"
    ] == 1


def test_hierarchical_taylor_writes_distinct_raw_score_and_rank_artifacts(
    tmp_path: Path,
) -> None:
    attention = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            _attn_implementation="eager",
        ),
        num_heads=2,
        num_key_value_heads=1,
        head_dim=2,
    )
    layer = SimpleNamespace(self_attn=attention, mlp=object())
    handles = {
        "llm": SimpleNamespace(),
        "llm.input_embeddings": object(),
        "llm.layer[0]": layer,
        "llm.layer[0].attn": attention,
        "llm.layer[0].attn.v_proj": object(),
        "llm.layer[0].attn.o_proj": object(),
        "llm.layer[0].ffn": layer.mlp,
        "llm.layer[0].ffn.down_proj": object(),
        "llm.final_norm": object(),
        "lm_head": object(),
    }
    backbone = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_vl"),
        model_adapter=SimpleNamespace(structural_handles=lambda _model: handles),
    )
    model_source = {
        "checkpoint": {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "revision": "revision",
            "resolved_source": "D:/hf-cache/checkpoint",
        },
        "dependencies": {},
    }
    runtime = SimpleNamespace(
        model=SimpleNamespace(_mllm_comp_source_evidence=model_source),
        model_path_or_id="Qwen/Qwen2.5-VL-3B-Instruct",
        revision="revision",
    )
    session = HierarchicalTaylor().open(
        backbone=backbone,
        runtime=runtime,
        config={"name": "hierarchical_taylor", "layers": [0]},
        phase="pre_compression",
        output_dir=str(tmp_path),
    )
    session.accumulator.update("task", _hierarchical_report())
    session.sample_summaries.append(
        {
            "sample_id": "sample",
            "task_name": "task",
            "loss": 1.0,
            "finite_component_scores": 12,
        }
    )

    record = session.finish()
    session.close()

    assert record.sample_count == 1
    assert set(record.artifact_paths) == {"task_scores", "task_ranks"}
    scores = json.loads(
        Path(record.artifact_paths["task_scores"]).read_text(encoding="utf-8")
    )
    ranks = json.loads(
        Path(record.artifact_paths["task_ranks"]).read_text(encoding="utf-8")
    )
    assert scores["artifact_kind"] == "task_raw_scores"
    assert scores["payload"]["binding_evidence"]["model_source"] == model_source
    assert scores["payload"]["score_storage"] == (
        "compact_ffn_layer_group_vectors_v2"
    )
    assert scores["payload"]["score_vectors_by_family"][
        "normalized_taylor"
    ]["task"]["layer:0/ffn_neuron/group:vision"] == pytest.approx([0.1, 0.2])
    assert ranks["artifact_kind"] == "cross_task_percentile_ranks"
    assert ranks["payload"]["aggregation"] == (
        "mean_task_percentile_average_ties_by_granularity_v2"
    )
    assert ranks["payload"]["partitioning"] == (
        "score_family_and_granularity_v2"
    )
    assert ranks["payload"]["score_vector_families"]["normalized_taylor"][
        "vectors"
    ]["layer:0/ffn_neuron/group:vision"]["length"] == 2


def test_causal_validation_selection_is_lowest_ranked_and_disjoint_random() -> None:
    config = validate_causal_validation_config(
        {
            "causal_validation": {
                "enabled": True,
                "component_count": 1,
                "kinds": ["path", "head", "ffn_neuron", "sublayer", "layer"],
                "seed": 7,
            }
        }
    )
    selections = select_causal_validation_candidates(
        _hierarchical_report(),
        kinds=config["kinds"],
        component_count=config["component_count"],
        seed=config["seed"],
        sample_id="sample",
    )

    assert selections["path"]["selected"][0]["score"] == pytest.approx(0.1)
    assert selections["path"]["selected"][0]["id"] != (
        selections["path"]["random_control"][0]["id"]
    )
    assert selections["head"]["status"] == "ready"
    assert selections["ffn_neuron"]["status"] == "ready"
    assert selections["sublayer"]["status"] == "ready"
    assert selections["layer"]["status"] == "unavailable"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"causal_validation": {"enabled": "true"}}, "enabled must be a boolean"),
        ({"causal_validation": {"kinds": [1]}}, "entries must be strings"),
        ({"layers": [True]}, "layers must be 'all'"),
        ({"save_raw": "false"}, "save_raw must be a boolean"),
    ],
)
def test_hierarchical_taylor_rejects_ambiguous_config_types(
    config: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HierarchicalTaylor().validate_config(config)


def test_hierarchical_taylor_capture_runs_one_backward_on_toy_model() -> None:
    torch = pytest.importorskip("torch")

    from mllm_comp.analysis.bindings import AttentionGeometry, DecoderLayerBinding
    from mllm_comp.analysis.common.taylor_capture import HierarchicalTaylorCapture
    from mllm_comp.analysis.common.token_mask import TokenGroupLayout
    from mllm_comp.analysis.hooks import HookSession

    class ToyAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v_proj = torch.nn.Linear(4, 2, bias=False)
            self.o_proj = torch.nn.Linear(4, 4, bias=False)
            self.last_attention = None

        def forward(self, hidden: object) -> tuple[object, object]:
            batch, sequence, _ = hidden.shape
            values = self.v_proj(hidden).view(batch, sequence, 1, 2)
            values = values.permute(0, 2, 1, 3).repeat_interleave(2, dim=1)
            causal = torch.tril(torch.ones(sequence, sequence, device=hidden.device))
            attention = causal / causal.sum(dim=-1, keepdim=True)
            attention = attention.view(1, 1, sequence, sequence).repeat(batch, 2, 1, 1)
            self.last_attention = attention
            heads = torch.matmul(attention, values)
            merged = heads.permute(0, 2, 1, 3).reshape(batch, sequence, 4)
            return self.o_proj(merged), attention

    class ToyFFN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = torch.nn.Linear(4, 3, bias=False)
            self.down_proj = torch.nn.Linear(3, 4, bias=False)

        def forward(self, hidden: object) -> object:
            return self.down_proj(torch.tanh(self.up(hidden)))

    class ToyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = ToyAttention()
            self.mlp = ToyFFN()

        def forward(self, hidden: object) -> object:
            attention, _ = self.self_attn(hidden)
            hidden = hidden + attention
            return hidden + self.mlp(hidden)

    embedding = torch.nn.Embedding(8, 4)
    layer = ToyLayer()
    binding = DecoderLayerBinding(
        0,
        layer,
        layer.self_attn,
        layer.mlp,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.down_proj,
    )
    capture = HierarchicalTaylorCapture(
        [binding],
        AttentionGeometry(2, 1, 2, 4),
    )
    layout = TokenGroupLayout(
        labels_by_sample=(("image", "text", "text"),),
        evidence={},
    )
    with HookSession() as hooks:
        capture.install(hooks, input_embeddings=embedding)
        hidden = embedding(torch.tensor([[1, 2, 3]]))
        output = layer(hidden)
        capture.bind_layout(layout)
        loss = output.square().mean()
        loss.backward()
        report = capture.finish({0: layer.self_attn.last_attention})

    assert len(report["layers"]["0"]["head"]["all"]["normalized_score"]) == 2
    assert len(
        report["layers"]["0"]["ffn_neuron"]["taylor"]["vision"][
            "normalized_score"
        ]
    ) == 3
    assert report["layers"]["0"]["path"]["t2v"]["query_count"] == 2
    capture.clear()

    audio_capture = HierarchicalTaylorCapture(
        [binding],
        AttentionGeometry(2, 1, 2, 4),
        target_modality="audio",
    )
    audio_layout = TokenGroupLayout(
        labels_by_sample=(("audio", "text", "text"),),
        evidence={},
    )
    with HookSession() as hooks:
        audio_capture.install(hooks, input_embeddings=embedding)
        hidden = embedding(torch.tensor([[1, 2, 3]]))
        output = layer(hidden)
        audio_capture.bind_layout(audio_layout)
        output.square().mean().backward()
        audio_report = audio_capture.finish({0: layer.self_attn.last_attention})

    assert audio_report["component_groups"] == ["all", "audio", "text"]
    assert audio_report["path_types"] == ["a2a", "t2a", "t2t"]
    assert len(
        audio_report["layers"]["0"]["ffn_neuron"]["taylor"]["audio"][
            "normalized_score"
        ]
    ) == 3
    audio_capture.clear()


def _hierarchical_report(scale: float = 1.0) -> dict[str, object]:
    def vector(values: list[float]) -> list[float]:
        return [value * scale for value in values]

    component = {
        "status": "finite",
        "normalized_score": 0.5 * scale,
        "direct_removal_score": 1.5 * scale,
    }
    return {
        "layers": {
            "0": {
                "path": {
                    "v2v": {
                        "status": "finite",
                        "normalized_score": vector([0.1, 0.2]),
                        "direct_removal_score": vector([0.5, 0.6]),
                    }
                },
                "head": {
                    "all": {
                        "status": "finite",
                        "normalized_score": vector([0.1, 0.2]),
                        "direct_removal_score": vector([0.4, 0.5]),
                    },
                    "vision": {
                        "status": "finite",
                        "normalized_score": vector([0.1, 0.2]),
                        "direct_removal_score": vector([0.4, 0.5]),
                    }
                },
                "ffn_neuron": {
                    "taylor": {
                        "all": {
                            "status": "finite",
                            "normalized_score": vector([0.1, 0.2]),
                            "direct_removal_score": vector([1.0, 2.0]),
                        },
                        "vision": {
                            "status": "finite",
                            "normalized_score": vector([0.1, 0.2]),
                            "direct_removal_score": vector([1.0, 2.0]),
                        },
                        "text": {
                            "status": "finite",
                            "normalized_score": vector([0.2, 0.1]),
                            "direct_removal_score": vector([2.0, 1.0]),
                        },
                    },
                    "activation_magnitude": {
                        "all": {
                            "status": "finite",
                            "mean_abs_activation": vector([1.0, 2.0]),
                        },
                        "vision": {
                            "status": "finite",
                            "mean_abs_activation": vector([1.0, 2.0]),
                        },
                        "text": {
                            "status": "finite",
                            "mean_abs_activation": vector([3.0, 4.0]),
                        },
                    },
                },
                "sublayer": {
                    "attention": {
                        "all": dict(component),
                        "vision": dict(component),
                    },
                    "ffn": {
                        "all": dict(component),
                        "vision": dict(component),
                    },
                },
                "layer": {
                    "all": {
                        "status": "finite",
                        "normalized_score": 1.0 * scale,
                        "direct_removal_score": 3.0 * scale,
                    },
                    "vision": {
                        "status": "finite",
                        "normalized_score": 1.0 * scale,
                        "direct_removal_score": 3.0 * scale,
                    }
                },
            }
        }
    }
