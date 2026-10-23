"""Small real-Transformers CPU integration for encoder-level A-ToMe."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.compressors.encoder_level.adjacent_token_merging import merge_adjacent_hidden_states, merge_ragged_adjacent_hidden_states
from mllm_comp.backbones.qwen25_omni.encoder_compression import (
    Qwen25OmniEncoderCompressionController,
)
from mllm_comp.backbones.base import ProcessedBatch
from mllm_comp.backbones.qwen25_omni.token_debug import (
    Qwen25OmniTokenDebugCollector,
)
from mllm_comp.core.state import ForwardState, TokenSpan
from tests.fixtures.qwen25_omni_audio import (
    _MiniThinker, encoder_stage, encoder_extension_stage,
)


def test_tensor_merging_preserves_order_and_chunk_boundaries() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    keys = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
    )

    merged, starts, _ = merge_adjacent_hidden_states(
        hidden,
        keys,
        merge_ratio=0.25,
    )
    assert starts.tolist() == [0, 4]
    assert merged.tolist() == [
        [1.0, 2.0],
        [4.0, 5.0],
        [6.0, 7.0],
        [9.0, 10.0],
        [12.0, 13.0],
        [14.0, 15.0],
    ]

    ragged, lengths, reports = merge_ragged_adjacent_hidden_states(
        hidden,
        keys,
        [4, 4],
        merge_ratio=0.25,
        collect_details=True,
    )
    assert ragged.shape[0] == 6
    assert lengths == [3, 3]
    assert [item["merged_pairs"] for item in reports] == [1, 1]


def test_real_audio_encoder_runs_layerwise_atome_and_repairs_lengths() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.3":
        pytest.skip("This version-specific integration requires Transformers 4.57.3.")
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoder,
    )

    config = Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=6,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        d_model=16,
        max_source_positions=64,
        n_window=16,
        output_dim=12,
    )
    config._attn_implementation = "eager"
    audio_tower = Qwen2_5OmniAudioEncoder(config).eval()
    thinker = _MiniThinker(audio_tower)
    model = SimpleNamespace(thinker=thinker)
    controller = Qwen25OmniEncoderCompressionController(
        model,
        encoder_stage(),
        profile_enabled=True,
        debug_enabled=True,
    )
    controller.install()

    audio_token_id = 12
    input_ids = torch.tensor([[1, *([audio_token_id] * 16), 2]])
    attention_mask = torch.ones_like(input_ids)
    input_features = torch.randn(8, 64)
    raw_feature_lengths = torch.tensor([64], dtype=torch.long)
    state = ForwardState(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_values=input_features,
        audio_feature_lengths=raw_feature_lengths,
        native_model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
        },
        modality_token_spans=[
            TokenSpan(sample_index=0, start=1, end=17, modality="audio")
        ],
        modality_metadata={
            "token_ids": {"audio": audio_token_id},
            "pad_token_id": 0,
            "padding_side": "right",
            "expected_token_counts_by_sample": [{"audio": 16}],
        },
    )

    try:
        compact_state, plan = controller.prepare_state(state)
        assert compact_state.input_ids.shape == (1, 16)
        with controller.request(plan):
            aftercnn_lengths, compressed_output_lengths = (
                audio_tower._get_feat_extract_output_lengths(raw_feature_lengths)
            )
            output = audio_tower(
                input_features,
                feature_lens=raw_feature_lengths,
                aftercnn_lens=aftercnn_lengths,
            ).last_hidden_state

        assert aftercnn_lengths.tolist() == [32]
        assert compressed_output_lengths.tolist() == [14]
        assert output.shape == (14, 12)
        assert controller.last_report is not None
        assert controller.last_report["active_layers"] == [2, 5]
        assert controller.last_report["initial_encoder_tokens"] == 32
        assert controller.last_report["final_encoder_tokens"] == 28
        assert controller.last_report["score_seconds"] is not None
        assert controller.last_report["merge_seconds"] is not None
        assert [
            item["output_tokens"] for item in controller.last_report["layers"]
        ] == [30, 28]
    finally:
        controller.uninstall()


@pytest.mark.parametrize(
    "plugin,score_source,factor",
    [
        ("adjacent_token_merging_audio", None, 2),
        ("encoder_attention_topk_audio", None, 2),
        ("progressive_attention_pruning_audio", None, 2),
        ("uniform_sampling_audio", None, 2),
        ("uniform_average_pooling_audio", None, 2),
        ("local_temporal_bipartite_merging_audio", None, 2),
        ("segmentwise_topk_audio", "uniform", 2),
        ("segmentwise_topk_audio", "random", 2),
        ("uniform_sampling_audio", None, 1),
        ("fastslm_hta_audio", None, 2),
    ],
)
@pytest.mark.parametrize("boundary,operation,source_layers,attention_lengths,ffn_lengths,raw_length", [
    ("mid", {}, [0, 2], [32, 16, 16, 8], [16, 16, 8, 8], 64),
    ("full", {}, [0, 2], [32, 16, 16, 8], [32, 16, 16, 8], 64),
    ("mid", {"layers": [1, 3]}, [0, 2], [32, 32, 16, 16], [32, 16, 16, 8], 64),
    ("full", {"layers": [1, 3]}, [0, 2], [32, 32, 16, 16], [32, 32, 16, 16], 64),
    ("mid", {"layers": [1, 2]}, [0, 1], [32, 32, 16, 8], [32, 16, 8, 8], 64),
    ("full", {"layers": [1, 2]}, [0, 1], [32, 32, 16, 8], [32, 32, 16, 8], 64),
    ("full", {"layers": [0, 2], "layer_edge": "before"}, [0, 2], [16, 16, 8, 8], [16, 16, 8, 8], 64),
    ("full", {"layers": [0, 2], "layer_edge": "before"}, [0, 2], [16, 16, 8, 8], [16, 16, 8, 8], 62),
    ("full", {"layers": [1, 2]}, [0, 1], [31, 31, 16, 8], [31, 31, 16, 8], 62),
], ids=["default-mid", "default-full", "later-mid", "later-full", "overlap-mid", "overlap-full", "initial", "initial-ragged", "overlap-ragged"])
def test_real_audio_encoder_positions_and_checkpoint_gradients(
    plugin, score_source, factor, boundary, operation, source_layers, attention_lengths, ffn_lengths, raw_length,
) -> None:
    import copy

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.3":
        pytest.skip("This version-specific integration requires Transformers 4.57.3.")
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoder,
    )

    config = Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=4,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        d_model=16,
        max_source_positions=64,
        n_window=16,
        output_dim=12,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(23)
    original = Qwen2_5OmniAudioEncoder(config).train()
    features = torch.randn(8, raw_length)
    stage = encoder_stage() if plugin == "adjacent_token_merging_audio" else encoder_extension_stage(plugin)
    if plugin == "adjacent_token_merging_audio":
        stage.schedule = {"merge_ratio": 0.5}
        from mllm_comp.core.vocabulary import ExecutionPhase
        stage.execution_phase = ExecutionPhase.FORWARD
    if plugin.startswith("uniform_"):
        stage.schedule = {"compression_factor": factor}
    if score_source:
        stage.criterion = {"score_source": score_source, "num_segments": 3}
    stage.activation_scope = {"layers": source_layers}
    target_site = (
        "audio_encoder_layer_post_mhsa_pre_ffn" if boundary == "mid"
        else "audio_encoder_layer_output_post_block"
    )
    stage.operation["target_site"] = target_site
    stage.operation.update(operation)
    # Top-K floors each chunk's budget; adjacent merging keeps the odd tail.
    feature_only = plugin.startswith("uniform_") or plugin in {"local_temporal_bipartite_merging_audio", "segmentwise_topk_audio", "fastslm_hta_audio"}
    stage_lengths = [15, 7] if raw_length == 62 and plugin not in {"adjacent_token_merging_audio", "uniform_sampling_audio"} else [16, 8]
    if stage_lengths == [15, 7]:
        attention_lengths = [{16: 15, 8: 7}.get(length, length) for length in attention_lengths]
        ffn_lengths = [{16: 15, 8: 7}.get(length, length) for length in ffn_lengths]
    if factor == 1:
        stage_lengths = [raw_length // 2] * 2
        attention_lengths = ffn_lengths = [raw_length // 2] * 4
    projected_tokens = stage_lengths[-1] // 2
    results = []
    runs = [("none", False), ("nonreentrant", False), ("reentrant", False)]
    if boundary == "full" and not operation:
        # The equivalent next-layer entry must use the same physical hook once.
        runs.append(("none", True))
    for checkpointed, equivalent_entry in runs:
        if equivalent_entry:
            stage.operation.update({"layers": [1, 3], "layer_edge": "before"})
        audio_tower = copy.deepcopy(original)
        if checkpointed != "none":
            audio_tower.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": checkpointed == "reentrant"}
            )
        ffn_widths = [[] for _ in audio_tower.layers]
        attention_widths = [[] for _ in audio_tower.layers]
        normalized_inputs = [[] for _ in audio_tower.layers]
        handles = []
        for index, layer in enumerate(audio_tower.layers):
            handles.append(layer.fc1.register_forward_pre_hook(
                lambda module, args, index=index: ffn_widths[index].append(int(args[0].shape[0]))
            ))
            handles.append(layer.self_attn.register_forward_pre_hook(
                lambda module, args, kwargs, index=index: attention_widths[index].append(
                    int(kwargs["hidden_states"].shape[0])
                ), with_kwargs=True,
            ))
            handles.append(layer.self_attn_layer_norm.register_forward_hook(
                lambda module, args, output, index=index: normalized_inputs[index].append(output.detach().clone())
            ))
        if plugin == "fastslm_hta_audio":
            torch.manual_seed(81)
        controller = Qwen25OmniEncoderCompressionController(
            SimpleNamespace(thinker=_MiniThinker(audio_tower)),
            stage,
            profile_enabled=True,
            debug_enabled=True,
        )
        controller.install()
        probes = []
        score_inputs = []
        original_probe = getattr(controller.strategy, "probe", None)

        def observe_probe(layer, normalized, lengths):
            probes.append((list(audio_tower.layers).index(layer), normalized.shape[0]))
            score_inputs.append(normalized.detach().clone())
            return original_probe(layer, normalized, lengths)

        if not feature_only:
            controller.strategy.probe = observe_probe
        audio_token_id = 12
        original_tokens = raw_length // 4
        input_ids = torch.tensor([[1, *([audio_token_id] * original_tokens), 2]])
        raw_feature_lengths = torch.tensor([raw_length], dtype=torch.long)
        input_features = features.clone().requires_grad_()
        state = ForwardState(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            audio_values=input_features,
            audio_feature_lengths=raw_feature_lengths,
            native_model_inputs={"input_ids": input_ids, "input_features": input_features},
            modality_token_spans=[
                TokenSpan(sample_index=0, start=1, end=original_tokens + 1, modality="audio")
            ],
            modality_metadata={
                "token_ids": {"audio": audio_token_id},
                "pad_token_id": 0,
                "padding_side": "right",
                "expected_token_counts_by_sample": [{"audio": original_tokens}],
            },
        )
        try:
            compact, plan = controller.prepare_state(state)
            assert compact.input_ids.shape == (1, projected_tokens + 2)
            with controller.request(plan):
                aftercnn_lengths, output_lengths = (
                    audio_tower._get_feat_extract_output_lengths(raw_feature_lengths)
                )
                output = audio_tower(
                    input_features,
                    feature_lens=raw_feature_lengths,
                    aftercnn_lens=aftercnn_lengths,
                ).last_hidden_state
            # Checkpoint replay must work after the request has released its
            # active state, without re-probing or retaining scoring features.
            output.square().mean().backward()

            assert output_lengths.tolist() == [projected_tokens]
            assert output.shape == (projected_tokens, 12)
            assert [values[0] for values in attention_widths] == attention_lengths
            assert [values[0] for values in ffn_widths] == ffn_lengths
            assert probes == ([] if feature_only else list(zip(source_layers, [raw_length // 2, stage_lengths[0]])))
            assert controller.runtime_metadata()["target_site"] == target_site
            assert controller.last_report["plugin"] == plugin
            assert controller.last_report["active_layers"] == (stage.operation["layers"] if feature_only else source_layers)
            assert [report["output_tokens"] for report in controller.last_report["layers"]] == stage_lengths
            reports = controller.last_report["layers"]
            for index, source in enumerate([] if feature_only else source_layers):
                expected = normalized_inputs[source][0]
                if expected.shape[0] != score_inputs[index].shape[0]:
                    # Overlapping sources: independently apply the first event
                    # to the source snapshot captured before that event.
                    mapped = []
                    for chunk, detail in zip(expected.split(reports[0]["input_chunk_lengths"]), reports[0]["chunks"]):
                        if plugin == "adjacent_token_merging_audio":
                            pairs = set(detail["selected_pair_starts"])
                            mapped.append(torch.stack([
                                (chunk[i] + chunk[i + 1]) / 2 if i in pairs else chunk[i]
                                for i in range(chunk.shape[0]) if i - 1 not in pairs
                            ]))
                        else:
                            mapped.append(chunk[detail["kept_indices"]])
                    expected = torch.cat(mapped)
                torch.testing.assert_close(score_inputs[index], expected, rtol=0, atol=0)
            assert [item["score_layer_index"] for item in reports] == ([None] * 2 if feature_only else source_layers)
            assert controller.runtime_metadata()["score_layers_zero_based"] == ([] if feature_only else source_layers)
            assert [item["application_position"] for item in reports] == controller.runtime_metadata()["application_positions"]
            if operation.get("layer_edge") == "before":
                assert reports[0]["layer_index"] is None
            module_gradients = []
            if plugin == "fastslm_hta_audio":
                module_gradients = [p.grad for p in controller.strategy_runtime_module.parameters() if p.requires_grad]
                assert all(g is not None and torch.isfinite(g).all() for g in module_gradients)
                assert any(torch.count_nonzero(g) for g in module_gradients)
                assert controller.strategy_runtime_module_name.endswith("_encoder_token")
            results.append((output.detach(), input_features.grad, *[
                layer.fc1.weight.grad for layer in audio_tower.layers
            ], *module_gradients))
        finally:
            for handle in handles:
                handle.remove()
            controller.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("initial", [False, True])
def test_debug_collector_observes_every_encoder_layer_after_merging(initial) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.3":
        pytest.skip("This version-specific integration requires Transformers 4.57.3.")
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoder,
    )

    config = Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=6,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        d_model=16,
        max_source_positions=64,
        n_window=16,
        output_dim=12,
    )
    config._attn_implementation = "eager"
    audio_tower = Qwen2_5OmniAudioEncoder(config).eval()
    thinker = _MiniThinker(audio_tower)
    thinker.model = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))
    model = SimpleNamespace(thinker=thinker)
    stage = encoder_stage()
    if initial:
        stage.activation_scope = {"layers": [0, 3]}
        stage.operation.update({
            "target_site": "audio_encoder_layer_output_post_block",
            "layers": [0, 3], "layer_edge": "before",
        })
    controller = Qwen25OmniEncoderCompressionController(
        model,
        stage,
        debug_enabled=True,
    )
    controller.install()
    collector = Qwen25OmniTokenDebugCollector(model)
    collector.install()

    audio_token_id = 12
    input_ids = torch.tensor([[1, *([audio_token_id] * 16), 2]])
    attention_mask = torch.ones_like(input_ids)
    input_features = torch.randn(8, 64)
    raw_feature_lengths = torch.tensor([64], dtype=torch.long)
    state = ForwardState(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_values=input_features,
        audio_attention_mask=torch.ones(1, 64, dtype=torch.long),
        audio_feature_lengths=raw_feature_lengths,
        native_model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": torch.ones(1, 64, dtype=torch.long),
        },
        modality_token_spans=[
            TokenSpan(sample_index=0, start=1, end=17, modality="audio")
        ],
        modality_metadata={
            "token_ids": {"audio": audio_token_id},
            "pad_token_id": 0,
            "padding_side": "right",
            "expected_token_counts_by_sample": [{"audio": 16}],
        },
    )
    batch = ProcessedBatch(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": torch.ones(1, 64, dtype=torch.long),
        },
        sample_ids=["debug-audio"],
        prompt_lengths=[18],
        metadata={
            "modality_counts_by_sample": [
                {"audio": 1, "image": 0, "video": 0}
            ],
            "expected_token_counts_by_sample": [{"audio": 16}],
        },
    )

    try:
        compact_state, plan = controller.prepare_state(state)
        with collector.request(
            original_state=state,
            active_state=compact_state,
            processed_batch=batch,
        ):
            with controller.request(plan):
                aftercnn_lengths, _ = audio_tower._get_feat_extract_output_lengths(
                    raw_feature_lengths
                )
                audio_tower(
                    input_features,
                    feature_lens=raw_feature_lengths,
                    aftercnn_lens=aftercnn_lengths,
                )
        collector.apply_compression_report(controller.last_report)

        assert collector.last_report is not None
        encoder_layers = [
            record
            for record in collector.last_report["layers"]
            if record["stage"] == "audio_encoder"
        ]
        assert [
            record["encode"]["input"]["token_counts"]["audio"]
            for record in encoder_layers
        ] == ([30, 30, 30, 28, 28, 28] if initial else [32, 32, 32, 30, 30, 30])
        assert [
            record["encode"]["output"]["token_counts"]["audio"]
            for record in encoder_layers
        ] == ([30, 30, 28, 28, 28, 28] if initial else [32, 32, 30, 30, 30, 28])
        modules = {
            record["stage"]: record for record in collector.last_report["modules"]
        }
        assert modules["audio_pooler"]["encode"]["input"]["total_tokens"] == 28
        assert modules["audio_pooler"]["encode"]["output"]["total_tokens"] == 14
    finally:
        collector.uninstall()
        controller.uninstall()


@pytest.mark.parametrize("score_source", ["uniform", "random"])
@pytest.mark.parametrize("raw_length,ratio,encoder_count,pooled_count", [(18, 0.5, 3, 1), (14, 1.0, 6, 3)])
def test_segmentwise_encoder_budgets_before_pooling_and_honors_floor_at_ratio_one(score_source, raw_length, ratio, encoder_count, pooled_count):
    import copy
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder

    torch.manual_seed(62)
    config = Qwen2_5OmniAudioEncoderConfig(num_mel_bins=4, encoder_layers=2, encoder_attention_heads=4,
        encoder_ffn_dim=32, d_model=16, max_source_positions=32, n_window=16, output_dim=12)
    config._attn_implementation = "eager"
    original = Qwen2_5OmniAudioEncoder(config).train()
    features = torch.randn(4, raw_length)
    stage = encoder_extension_stage("segmentwise_topk_audio")
    stage.criterion = {"score_source": score_source, "num_segments": 3}
    stage.schedule = {"keep_ratio": ratio, "allocation": "paper_floor"}
    stage.operation.update(layers=[0], layer_edge="before")
    results = []
    for checkpoint in (None, False, True):
        tower = copy.deepcopy(original)
        if checkpoint is not None:
            tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        controller = Qwen25OmniEncoderCompressionController(SimpleNamespace(thinker=_MiniThinker(tower)), stage)
        values = features.clone().requires_grad_()
        lengths = torch.tensor([raw_length])
        count = raw_length // 4
        ids = torch.tensor([[1, *([12] * count), 2]])
        state = ForwardState(input_ids=ids, attention_mask=torch.ones_like(ids), audio_values=values, audio_feature_lengths=lengths,
            native_model_inputs={"input_ids": ids, "input_features": values},
            modality_token_spans=[TokenSpan(0, 1, count + 1, modality="audio")],
            modality_metadata={"token_ids": {"audio": 12}, "pad_token_id": 0, "padding_side": "right",
                               "expected_token_counts_by_sample": [{"audio": count}]})
        widths = []
        handle = tower.layers[0].fc1.register_forward_pre_hook(lambda module, args: widths.append(args[0].shape[0]))
        controller.install()
        try:
            compact, plan = controller.prepare_state(state)
            assert compact.input_ids.shape[1] == pooled_count + 2
            assert "requested_audio_tokens" not in plan.this_stage_metadata.stats
            with controller.request(plan):
                aftercnn, output_lengths = tower._get_feat_extract_output_lengths(lengths)
                output = tower(values, feature_lens=lengths, aftercnn_lens=aftercnn).last_hidden_state
            output.square().mean().backward()
            assert widths[0] == encoder_count
            assert output_lengths.tolist() == [pooled_count]
            assert output.shape == (pooled_count, 12)
            results.append((output.detach(), values.grad, *[layer.fc1.weight.grad for layer in tower.layers]))
        finally:
            handle.remove()
            controller.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("position", ["last-input", "last-output", "last-mid", "two-events", "single-layer-initial"])
def test_segmentwise_encoder_keeps_last_layer_attention_source(position):
    import copy
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
    from mllm_comp.compressors.projector_level.segmentwise_topk import segmentwise_topk_indices

    depth = 1 if position == "single-layer-initial" else 4
    torch.manual_seed(68)
    config = Qwen2_5OmniAudioEncoderConfig(num_mel_bins=4, encoder_layers=depth, encoder_attention_heads=4,
        encoder_ffn_dim=32, d_model=16, max_source_positions=32, n_window=16, output_dim=12)
    config._attn_implementation = "eager"
    original = Qwen2_5OmniAudioEncoder(config).train()
    features = torch.randn(4, 62)
    stage = encoder_extension_stage("segmentwise_topk_audio")
    stage.criterion = {"num_segments": 3}  # Keep the standard encoder_last_layer_attention source.
    stage.operation["layers"] = [2, 3] if position == "two-events" else [2 if position == "last-input" else depth - 1]
    if position == "last-mid":
        stage.operation["target_site"] = "audio_encoder_layer_post_mhsa_pre_ffn"
    if position == "single-layer-initial":
        stage.operation["layer_edge"] = "before"
    results = []
    cases = [(None, False), (False, False), (True, False)]
    if position == "last-input":
        cases.append((None, True))
    for checkpoint, before_alias in cases:
        if before_alias:
            stage.operation.update(layers=[3], layer_edge="before")
        tower = copy.deepcopy(original)
        if checkpoint is not None:
            tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        controller = Qwen25OmniEncoderCompressionController(SimpleNamespace(thinker=_MiniThinker(tower)), stage, debug_enabled=True)
        probes = []
        probe = controller.strategy.probe
        def observe_probe(layer, values, lengths):
            probes.append((list(tower.layers).index(layer), values.detach().clone()))
            return probe(layer, values, lengths)
        controller.strategy.probe = observe_probe
        selections = []
        compress = controller.strategy.rule.compress
        def observe_selection(values, **kwargs):
            output, indices, details = compress(values, **kwargs)
            expected = segmentwise_topk_indices(kwargs["scores"].tolist(), keep_ratio=kwargs["keep_count"] / values.shape[0],
                num_segments=3, minimum_keep_tokens=kwargs["keep_count"])
            assert indices.tolist() == expected
            torch.testing.assert_close(output.detach(), values.detach()[expected], rtol=0, atol=0)
            selections.append(indices.detach())
            return output, indices, details
        controller.strategy.rule.compress = observe_selection
        normalized = []
        handles = [tower.layers[-1].self_attn_layer_norm.register_forward_hook(
            lambda module, args, output: normalized.append(output.detach().clone()))]
        ffn_widths = [[] for _ in range(depth)]
        attention_widths = [[] for _ in range(depth)]
        for index, layer in enumerate(tower.layers):
            handles.append(layer.fc1.register_forward_pre_hook(lambda module, args, index=index: ffn_widths[index].append(args[0].shape[0])))
            handles.append(layer.self_attn.register_forward_pre_hook(lambda module, args, kwargs, index=index: attention_widths[index].append(kwargs["hidden_states"].shape[0]), with_kwargs=True))
        values = features.clone().requires_grad_()
        lengths = torch.tensor([62])
        ids = torch.tensor([[1, *([12] * 15), 2]])
        state = ForwardState(input_ids=ids, attention_mask=torch.ones_like(ids), audio_values=values, audio_feature_lengths=lengths,
            native_model_inputs={"input_ids": ids, "input_features": values}, modality_token_spans=[TokenSpan(0, 1, 16, modality="audio")],
            modality_metadata={"token_ids": {"audio": 12}, "pad_token_id": 0, "padding_side": "right", "expected_token_counts_by_sample": [{"audio": 15}]})
        controller.install()
        try:
            compact, plan = controller.prepare_state(state)
            with controller.request(plan):
                aftercnn, output_lengths = tower._get_feat_extract_output_lengths(lengths)
                output = tower(values, feature_lens=lengths, aftercnn_lens=aftercnn).last_hidden_state
            output.square().mean().backward()
            expected_count = 3 if position == "two-events" else 7
            assert output_lengths.tolist() == [expected_count]
            assert output.shape == (expected_count, 12)
            assert controller.runtime_metadata()["score_layers_zero_based"] == [depth - 1]
            stages = 2 if position == "two-events" else 1
            assert [i for i, _ in probes] == [depth - 1] * stages
            assert len(selections) == 2 * stages
            assert [row["score_layer_index"] for row in controller.last_report["layers"]] == [depth - 1] * stages
            torch.testing.assert_close(probes[0][1], normalized[0], rtol=0, atol=0)
            attn = [31] * depth
            ffn = [31] * depth
            if position in {"last-input", "two-events"}:
                attn[-1] = ffn[-1] = 15
            if position == "last-mid":
                ffn[-1] = 15
            if position == "single-layer-initial":
                attn = ffn = [15]
            assert [row[0] for row in attention_widths] == attn
            assert [row[0] for row in ffn_widths] == ffn
            results.append((output.detach(), values.grad, *[layer.fc1.weight.grad for layer in tower.layers]))
        finally:
            for handle in handles:
                handle.remove()
            controller.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)


def test_segmentwise_encoder_rejects_application_before_fixed_scoring_source():
    from mllm_comp.core.errors import InvariantError

    stage = encoder_extension_stage("segmentwise_topk_audio")
    stage.criterion = {}
    stage.operation.update(layers=[0], layer_edge="before")
    model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=SimpleNamespace(layers=[object()] * 4)))
    with pytest.raises(InvariantError, match="precedes scoring layer 3"):
        Qwen25OmniEncoderCompressionController(model, stage)
