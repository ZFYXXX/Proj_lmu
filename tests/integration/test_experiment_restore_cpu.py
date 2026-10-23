"""Public run restoration reloads learned compression state on native models."""

from copy import deepcopy

import pytest

from mllm_comp.backbones.qwen25_omni.implementation import Qwen25OmniModelAdapter
from mllm_comp.config import config_to_dict
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.serialization import atomic_write_json, to_jsonable
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.restoration import restore_config
from tests.fixtures.qwen25_omni_audio import projector_stage
from tests.fixtures.qwen25_omni_audio import _create_controller, _tiny_model


@pytest.mark.parametrize("target", [
    "audio_encoder_output_pre_projector", "audio_projector_output_post_projector",
])
def test_restore_run_reconstructs_hta_at_original_position_and_resumes_update(tmp_path, target):
    torch = pytest.importorskip("torch")
    torch.manual_seed(731)
    model = _tiny_model(torch).train()
    stage = projector_stage("fastslm_hta_audio")
    stage.config["target_site"] = target
    features = torch.randn(4, 16)

    def step(thinker, controller, optimizer):
        optimizer.zero_grad(set_to_none=True)
        ids = torch.tensor([[1, 99, 99, 99, 99, 2]])
        lengths = torch.tensor([16])
        state = ForwardState(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            audio_values=features, audio_feature_lengths=lengths,
            native_model_inputs={"input_ids": ids, "input_features": features},
            modality_token_spans=[TokenSpan(0, 1, 5, modality="audio")],
            modality_metadata={"token_ids": {"audio": 99}, "pad_token_id": 0,
                               "padding_side": "right", "expected_token_counts_by_sample": [{"audio": 4}]},
        )
        _, plan = controller.prepare_state(state)
        with controller.request(plan):
            aftercnn, output_lengths = thinker.audio_tower._get_feat_extract_output_lengths(lengths)
            output = thinker.audio_tower(features, feature_lens=lengths, aftercnn_lens=aftercnn).last_hidden_state
            assert output.shape == (2, 12) and output_lengths.tolist() == [2]
            loss = output.square().mean()
            loss.backward()
        module = controller.strategy_runtime_module
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() for p in module.parameters())
        optimizer.step()
        return loss.detach()

    controller = _create_controller(model, stage)
    controller.install()
    checkpoint = tmp_path / "checkpoint"
    try:
        optimizer = torch.optim.SGD(controller.strategy_runtime_module.parameters(), lr=0.05, momentum=0.9)
        before = deepcopy(controller.strategy_runtime_module.state_dict())
        step(model.thinker, controller, optimizer)
        saved = deepcopy(controller.strategy_runtime_module.state_dict())
        assert any(not torch.equal(saved[name], before[name]) for name in saved)
        model.thinker.save_pretrained(checkpoint)
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        expected_loss = step(model.thinker, controller, optimizer)
        expected = deepcopy(controller.strategy_runtime_module.state_dict())
    finally:
        controller.uninstall()

    run = tmp_path / "run"
    config = experiment_config_from_dict({
        "experiment": {"name": "learned_hta"},
        "backbone": {"name": "qwen25_omni", "model_path": "original_model"},
        "pipeline": {"stages": [to_jsonable(stage)]},
        "train": {"compression_mode": "compressed"},
    })
    atomic_write_json(run / "effective_config.json", config_to_dict(config))
    atomic_write_json(run / "result.json", {"task": "train", "status": "succeeded",
                                           "output": {"checkpoint_path": str(checkpoint)}})
    restored_config = restore_config(run, task="analyze")
    restored = Qwen25OmniModelAdapter().load_model(
        restored_config.backbone.model_path, train=True, **restored_config.backbone.config,
    )
    restored_controller = _create_controller(restored, restored_config.pipeline.stages[0])
    restored_controller.install()
    try:
        assert restored_controller.runtime_metadata()["target_site"] == target
        for name, value in restored_controller.strategy_runtime_module.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
        optimizer = torch.optim.SGD(restored_controller.strategy_runtime_module.parameters(), lr=0.05, momentum=0.9)
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", weights_only=True))
        actual_loss = step(restored, restored_controller, optimizer)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for name, value in restored_controller.strategy_runtime_module.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    finally:
        restored_controller.uninstall()
