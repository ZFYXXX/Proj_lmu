"""Native tiny-model coverage for the imported modality-split implementation."""
import copy

import pytest

from tests.fixtures.native_image import inputs, qwen_model


def split_spec(backbone):
    return {
        "schema_version": 1, "backbone": backbone,
        "attention": {"layers": [0], "image_operator": "window", "config": {"local_left": 1}},
        "ffn": {"layers": [0], "image_operator": "low_rank", "config": {"rank": 4}},
    }


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl"])
def test_text_identity_and_uninstall(backbone):
    torch = pytest.importorskip("torch")
    from mllm_comp.modality_split import install_modality_split
    torch.manual_seed(71)
    model = qwen_model(backbone).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits
        session = install_modality_split(model, split_spec(backbone))
        actual = model(input_ids=ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        session.uninstall()
        torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, expected)


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl"])
def test_pixel_training_checkpoint_replay_and_save_restore(backbone, tmp_path):
    torch = pytest.importorskip("torch")
    from mllm_comp.modality_split import (
        install_modality_split, load_modality_split, native_state_dict, save_modality_split,
    )
    torch.manual_seed(72)
    plain = qwen_model(backbone)
    replay = copy.deepcopy(plain)
    for model in (plain, replay):
        install_modality_split(model, split_spec(backbone))
        model.train()
    replay.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sample = inputs(torch, backbone)
    labels = torch.full_like(sample["input_ids"], -100)
    labels[:, -1] = 2
    outputs = []
    for model in (plain, replay):
        output = model(**sample, labels=labels, use_cache=False)
        output.loss.backward()
        outputs.append(output)
    torch.testing.assert_close(outputs[0].logits, outputs[1].logits)
    for (name, p), (_, q) in zip(plain.named_parameters(), replay.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)
    image_parameters = [p for name, p in plain.named_parameters() if ".mlp.image_op." in name]
    assert image_parameters and any(p.grad is not None and p.grad.count_nonzero() for p in image_parameters)
    torch.optim.SGD(plain.parameters(), lr=0.02).step()
    plain.eval()
    plain.save_pretrained(tmp_path, state_dict=native_state_dict(plain))
    save_modality_split(plain, tmp_path)
    restored = type(plain).from_pretrained(tmp_path).eval()
    load_modality_split(restored, tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(**sample, use_cache=False).logits,
            plain(**sample, use_cache=False).logits,
        )


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl"])
def test_mixed_prefill_incremental_cache_and_generation(backbone):
    torch = pytest.importorskip("torch")
    from mllm_comp.modality_split import install_modality_split
    torch.manual_seed(73)
    model = qwen_model(backbone).eval()
    install_modality_split(model, split_spec(backbone))
    sample = inputs(torch, backbone)
    ids = sample["input_ids"]
    full = {**sample, "input_ids": torch.cat([ids, torch.tensor([[3]])], dim=1),
            "attention_mask": torch.ones((1, ids.shape[1] + 1), dtype=torch.long)}
    with torch.no_grad():
        expected = model(**full, use_cache=False).logits[:, -1]
        prefill = model(**sample, use_cache=True)
        cache = prefill.past_key_values
        actual = model(input_ids=torch.tensor([[3]]), cache_position=torch.tensor([ids.shape[1]]),
                       attention_mask=full["attention_mask"],
                       past_key_values=cache, use_cache=True).logits[:, -1]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        assert cache.get_seq_length() == full["input_ids"].shape[1]
        cache.reset()
        torch.testing.assert_close(model(**sample, past_key_values=cache, use_cache=True).logits, prefill.logits)
        for beams in (1, 2):
            generated = model.generate(**sample, max_new_tokens=2, do_sample=False,
                                       num_beams=beams, eos_token_id=None, pad_token_id=0)
            assert generated.shape == (1, ids.shape[1] + 2)


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl"])
def test_pipeline_training_sidecars_and_restore(backbone, tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    from mllm_comp.config import load_config
    from mllm_comp.modality_split.pipeline import compile_pipeline_split
    from mllm_comp.integrations.ms_swift.split_lifecycle import (
        install_training_split, inspect_split_checkpoint, native_save_state, save_split_checkpoint,
    )
    for name in ("QWEN25_VL_MODEL_PATH", "QWEN3_VL_MODEL_PATH", "MLLM_COMP_TRAIN_DATASET", "MLLM_COMP_TRAIN_RUN_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    config = load_config(Path(__file__).parents[2] / "configs/modality_split" / f"{backbone}_sft.yaml")
    # Preserve the real sample pipeline while scaling its FFN to the tiny model.
    config.pipeline.stages[2].config["bottleneck_dim"] = 8
    model = qwen_model(backbone).train()
    plan = compile_pipeline_split(config, model=model)
    events = []
    recorder = SimpleNamespace(record=lambda event, **kwargs: events.append(event))
    install_training_split(model, plan, recorder=recorder)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sample = inputs(torch, backbone)
    labels = torch.full_like(sample["input_ids"], -100)
    labels[:, -1] = 2
    length = labels.shape[1]
    output = model(**sample, labels=labels, cu_seq_lens_q=torch.tensor([0, length]),
                   cu_seq_lens_k=torch.tensor([0, length]), max_length_q=length, max_length_k=length)
    output.loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert all(events.count(event) == 4 for event in ("install", "forward", "backward"))
    state = native_save_state(model)
    model.save_pretrained(tmp_path, state_dict=state)
    save_split_checkpoint(model, tmp_path)
    assert inspect_split_checkpoint(tmp_path) == plan.contract
    restored = type(model).from_pretrained(tmp_path)
    install_training_split(restored, plan, recorder=recorder, checkpoint=tmp_path)
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**sample).logits, model(**sample).logits)
