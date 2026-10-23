"""Small real-Transformers CPU integration for the audio compression hooks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.backbones.qwen25_omni.audio_compression import (
    Qwen25OmniAudioCompressionController,
)
from mllm_comp.core.state import ForwardState, TokenSpan
from tests.fixtures.qwen25_omni_audio import _MiniThinker, projector_stage


@pytest.mark.parametrize("explicit_generation_config", [False, True])
def test_dynamic_generation_overrides_training_checkpoint_cache_default(
    explicit_generation_config,
):
    from mllm_comp.backbones.base import BackboneRuntime, ProcessedBatch
    from mllm_comp.backbones.qwen25_omni import create_qwen25_omni_backbone
    from mllm_comp.pipeline.runtime_session import CompressionActivationRecord, CompressionSession
    from tests.fixtures.qwen25_omni_audio import _create_controller, _hta_audio_state, _tiny_model

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(32)
    thinker = _tiny_model(torch).thinker.eval()
    # Swift checkpoints retain this training setting; a restored Thinker derives
    # its generation configuration from the nested text configuration.
    thinker.config.text_config.use_cache = False
    thinker.generation_config = transformers.GenerationConfig.from_model_config(thinker.config)
    assert thinker.generation_config.use_cache is False

    facade = create_qwen25_omni_backbone()
    processor = SimpleNamespace(batch_decode=lambda rows, **kwargs: [
        " ".join(map(str, row.tolist())) for row in rows
    ])
    runtime = BackboneRuntime(model=thinker, processor=processor)
    raw = _hta_audio_state(torch)
    batch = ProcessedBatch(
        model_inputs={key: value for key, value in raw.native_model_inputs.items() if key != "labels"},
        sample_ids=["audio"], prompt_lengths=[raw.input_ids.shape[1]],
        metadata=raw.modality_metadata,
    )
    state = facade.state_builder.build(batch, phase="eval")
    stage = projector_stage("uniform_sampling_audio")
    controller = _create_controller(thinker, stage)
    session = CompressionSession([CompressionActivationRecord(
        name=stage.name, binding=stage.binding, activation=controller,
    )])
    runtime.metadata["_compression_session"] = session
    projector_calls = []
    handle = thinker.audio_tower.proj.register_forward_hook(
        lambda module, args, output: projector_calls.append(output.shape[0])
    )
    generation_kwargs = {"thinker_min_new_tokens": 3, "thinker_max_new_tokens": 3,
                         "thinker_do_sample": False}
    if explicit_generation_config:
        generation_kwargs["thinker_generation_config"] = transformers.GenerationConfig(use_cache=True)
    session.install()
    try:
        with torch.no_grad():
            result = facade.runner.generate(runtime, state, processed_batch=batch, **generation_kwargs)
        assert result.text.token_ids.shape == (1, 3)
        assert projector_calls == [4]
        assert controller.last_report["original_audio_tokens"] == 8
        assert controller.last_report["kept_audio_tokens"] == 4
        assert thinker.config.text_config.use_cache is False
    finally:
        session.uninstall()
        handle.remove()


def test_real_audio_encoder_hooks_prune_two_audio_units_before_projector() -> None:
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
        encoder_layers=2,
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
    controller = Qwen25OmniAudioCompressionController(model, projector_stage())
    controller.install()

    audio_token_id = 12
    input_ids = torch.tensor(
        [[1, *([audio_token_id] * 8), 2, *([audio_token_id] * 12), 3]]
    )
    attention_mask = torch.ones_like(input_ids)
    input_features = torch.randn(8, 80)
    state = ForwardState(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_values=input_features,
        native_model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
        },
        modality_token_spans=[
            TokenSpan(sample_index=0, start=1, end=9, modality="audio"),
            TokenSpan(sample_index=0, start=10, end=22, modality="audio"),
        ],
        modality_metadata={
            "token_ids": {"audio": audio_token_id},
            "pad_token_id": 0,
            "padding_side": "right",
            "expected_token_counts_by_sample": [{"audio": 20}],
        },
    )

    try:
        compact_state, plan = controller.prepare_state(state)
        assert compact_state.input_ids.shape == (1, 13)
        assert compact_state.audio_values is input_features

        raw_feature_lengths = torch.tensor([32, 48], dtype=torch.long)
        with controller.request(plan):
            rope_args, rope_kwargs = thinker.get_rope_index(
                compact_state.input_ids,
                None,
                None,
                compact_state.attention_mask,
                False,
                raw_feature_lengths,
                None,
            )
            assert rope_args[5].tolist() == [16, 24]
            assert rope_kwargs == {}
            aftercnn_lengths, compressed_output_lengths = (
                audio_tower._get_feat_extract_output_lengths(raw_feature_lengths)
            )
            output = audio_tower(
                input_features,
                feature_lens=raw_feature_lengths,
                aftercnn_lens=aftercnn_lengths,
            ).last_hidden_state

        assert aftercnn_lengths.tolist() == [16, 24]
        assert compressed_output_lengths.tolist() == [4, 6]
        assert output.shape == (10, 12)
        assert controller.last_report is not None
        assert controller.last_report["original_audio_tokens"] == 20
        assert controller.last_report["kept_audio_tokens"] == 10
        assert len(controller.last_report["items"]) == 2
        assert all(
            item["indices_chronological"]
            for item in controller.last_report["items"]
        )
        assert all(
            abs(item["score_sum"] - item["original_tokens"]) < 1e-4
            for item in controller.last_report["items"]
        )
    finally:
        controller.uninstall()


@pytest.mark.parametrize(
    "plugin",
    [
        "segmentwise_topk_audio",
        "local_temporal_bipartite_merging_audio",
        "similarity_attention_merge_prune_audio",
        "uniform_sampling_audio",
        "uniform_average_pooling_audio",
        "unsupervised_segmentation_audio",
        "fastslm_hta_audio",
    ],
)
@pytest.mark.parametrize("target_site", [
    "audio_encoder_output_pre_projector",
    "audio_projector_output_post_projector",
])
def test_real_audio_encoder_projector_positions(plugin: str, target_site: str, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.3":
        pytest.skip("This version-specific integration requires Transformers 4.57.3.")
    # SA-MAP intentionally rejects threshold groupings smaller than the final
    # budget, so this integration fixture must not sample that valid rejection
    # nondeterministically while testing the encoder forward/backward path.
    torch.manual_seed(0)
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoder,
    )

    config = Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=2,
        encoder_attention_heads=6 if plugin == "fastslm_hta_audio" else 4,
        encoder_ffn_dim=32,
        d_model=24 if plugin == "fastslm_hta_audio" else 16,
        max_source_positions=64,
        n_window=16,
        output_dim=16 if plugin == "fastslm_hta_audio" else 12,
    )
    config._attn_implementation = "eager"
    audio_tower = Qwen2_5OmniAudioEncoder(config).train()
    thinker = _MiniThinker(audio_tower)
    stage = projector_stage(plugin)
    stage.operation["target_site"] = target_site
    if plugin == "fastslm_hta_audio":
        stage.config["target_site"] = stage.operation.pop("target_site")
    controller = Qwen25OmniAudioCompressionController(
        SimpleNamespace(thinker=thinker),
        stage,
    )
    controller.install()
    projector_input_lengths = []
    handle = audio_tower.proj.register_forward_pre_hook(
        lambda module, args: projector_input_lengths.append(int(args[0].shape[0]))
    )
    audio_token_id = 12
    input_ids = torch.tensor([[1, *([audio_token_id] * 16), 2]])
    raw_feature_lengths = torch.tensor([64], dtype=torch.long)
    input_features = torch.randn(8, 64, requires_grad=True)
    state = ForwardState(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        audio_values=input_features,
        audio_feature_lengths=raw_feature_lengths,
        native_model_inputs={"input_ids": input_ids, "input_features": input_features},
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

    planning_widths = []
    if plugin == "unsupervised_segmentation_audio":
        original_probe = controller._probe_projector_hidden_states

        def capture_planning_width(state):
            tensors = original_probe(state)
            planning_widths.extend(int(tensor.shape[-1]) for tensor in tensors)
            return tensors

        monkeypatch.setattr(controller, "_probe_projector_hidden_states", capture_planning_width)
    try:
        compact, plan = controller.prepare_state(state)
        keep_count = int(plan.token_actions[0].metadata["keep_count"])
        assert compact.input_ids.shape == (1, keep_count + 2)
        if plugin != "unsupervised_segmentation_audio":
            assert keep_count == 8
        else:
            assert planning_widths == [12 if target_site == "audio_projector_output_post_projector" else 16]
        projector_input_lengths.clear()
        with controller.request(plan):
            aftercnn_lengths, output_lengths = (
                audio_tower._get_feat_extract_output_lengths(raw_feature_lengths)
            )
            output = audio_tower(
                input_features,
                feature_lens=raw_feature_lengths,
                aftercnn_lens=aftercnn_lengths,
            ).last_hidden_state
            output.square().mean().backward()

        assert output_lengths.tolist() == [keep_count]
        assert output.shape == (keep_count, config.output_dim)
        assert projector_input_lengths == [
            16 if target_site == "audio_projector_output_post_projector" else keep_count
        ]
        assert input_features.grad is not None
        assert torch.isfinite(input_features.grad).all()
        assert torch.count_nonzero(input_features.grad) > 0
        assert controller.last_report is not None
        assert controller.last_report["plugin"] == plugin
        assert controller.last_report["kept_audio_tokens"] == keep_count
        assert controller.runtime_metadata()["target_site"] == target_site
        if plugin == "fastslm_hta_audio":
            assert controller.runtime_metadata()["num_attention_heads"] == (
                4 if target_site == "audio_projector_output_post_projector" else 6
            )
            gradients = [parameter.grad for parameter in controller.strategy_runtime_module.parameters()]
            assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in gradients)
    finally:
        handle.remove()
        controller.uninstall()


def test_hta_rejects_checkpoint_from_different_projector_width(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.projector_level.fastslm_hta import (
        build_fastslm_hta,
        load_fastslm_hta_checkpoint,
    )
    from mllm_comp.core.errors import InvariantError

    options = dict(num_attention_heads=4, stage_query_tokens=4,
                   output_query_tokens=8, window_tokens=16, detail_recovery_depth=1)
    before = build_fastslm_hta(hidden_size=16, **options)
    checkpoint = tmp_path / "hta-before-projector.pt"
    torch.save(before.state_dict(), checkpoint)
    after = build_fastslm_hta(hidden_size=12, **options)
    with pytest.raises(InvariantError, match="tensor shapes do not match"):
        load_fastslm_hta_checkpoint(after, str(checkpoint), strict=False)
    with pytest.raises(InvariantError, match="attention heads must divide"):
        build_fastslm_hta(hidden_size=16, **{**options, "num_attention_heads": 6})
