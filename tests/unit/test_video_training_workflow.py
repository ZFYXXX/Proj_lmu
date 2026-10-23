"""Shared training mechanics tested without replacing algorithm/layout validators."""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.diagnostics import video_training_checkpoint_common as common


def test_training_state_preserves_inputs_and_uses_teacher_forcing_without_cache() -> None:
    events = []
    sample = object()
    prompt, teacher = object(), object()

    @dataclass
    class State:
        native_model_inputs: dict

    original = State({"labels": "target", "pixel_values": "video", "use_cache": True})

    def prepare(processor, value, **kwargs):
        assert value is sample and kwargs == {"phase": "train"}
        events.append("prompt")
        return prompt

    def forcing(processor, value, **kwargs):
        assert value is sample and kwargs == {"target_text": "D", "prompt_batch": prompt, "phase": "train"}
        events.append("teacher")
        return teacher

    def build(batch, **kwargs):
        assert batch is teacher and kwargs == {"raw_batch": sample, "phase": "train"}
        events.append("state")
        return original

    facade = SimpleNamespace(
        processor_adapter=SimpleNamespace(prepare_inputs=prepare, prepare_teacher_forced_inputs=forcing),
        state_builder=SimpleNamespace(build=build),
    )
    batch, state = common.training_state(facade=facade, processor=object(), sample=sample, target="D")
    assert batch is teacher
    assert events == ["prompt", "teacher", "state"]
    assert state.native_model_inputs == {"labels": "target", "pixel_values": "video", "use_cache": False, "return_dict": True}
    assert original.native_model_inputs["use_cache"] is True


@pytest.mark.parametrize("device_map", ["cuda:0", {"": 0}])
@pytest.mark.parametrize("train", [True, False])
def test_loader_reuses_the_same_checkpoint_contract_for_training_and_reload(device_map, train) -> None:
    events = []
    model = SimpleNamespace(gradient_checkpointing_disable=lambda: events.append("disable_checkpointing"))
    processor = object()

    def load_processor(path, **kwargs):
        assert path == "checkpoint" and kwargs == {"local_files_only": True}
        events.append("processor")
        return processor

    def load_model(path, **kwargs):
        assert path == "checkpoint"
        assert kwargs == {"train": train, "local_files_only": True, "torch_dtype": "bf16", "attn_implementation": "sdpa", "device_map": device_map}
        events.append("model")
        return model

    facade = SimpleNamespace(processor_adapter=SimpleNamespace(load_processor=load_processor), model_adapter=SimpleNamespace(load_model=load_model))
    loaded = common.load_training_runtime(Path("checkpoint"), facade=facade, torch=SimpleNamespace(bfloat16="bf16", device=str), train=train, device_map=device_map)
    assert loaded[:3] == (facade, processor, model)
    assert loaded[3].model is model and loaded[3].processor is processor
    assert loaded[3].metadata["compression_mode"] == "compressed"
    assert events == ["processor", "model", "disable_checkpointing"]


@pytest.mark.parametrize("missing", [None, "loss", "logits"])
def test_training_step_propagates_forward_failure_before_parameter_update(missing) -> None:
    events = []
    loss = SimpleNamespace(backward=lambda: events.append("backward"))
    logits = object()
    output = {"loss": loss, "logits": logits}
    if missing:
        output.pop(missing)

    def forward(runtime, state):
        events.append("forward")
        return SimpleNamespace(**output)

    kwargs = {
        "facade": SimpleNamespace(runner=SimpleNamespace(forward_train=forward)),
        "runtime": object(), "state": object(),
        "optimizer": SimpleNamespace(
            zero_grad=lambda **kwargs: events.append("zero_grad"),
            step=lambda: events.append("step"),
        ),
        "selected": [],
        "torch": SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: events.append("synchronize"))),
    }
    if missing:
        with pytest.raises(RuntimeError, match="omitted loss or logits"):
            common.training_forward_step(**kwargs)
        assert events == ["zero_grad", "forward"]
    else:
        assert common.training_forward_step(**kwargs) == (loss, logits, [])
        assert events == ["zero_grad", "forward", "backward", "step", "synchronize"]


@pytest.mark.parametrize("optimizer_steps", [3, 5])
def test_final_norm_training_checkpoint_reloads_exactly_for_unlabelled_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, optimizer_steps: int
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def model_from_base():
        torch.manual_seed(117)
        return torch.nn.Sequential(OrderedDict([
            ("projection", torch.nn.Linear(3, 3)),
            ("final_norm", torch.nn.LayerNorm(3)),
            ("head", torch.nn.Linear(3, 2)),
        ]))

    @dataclass
    class State:
        native_model_inputs: dict

    inputs = torch.tensor([[1.0, -2.0, 0.5]])
    labels = torch.tensor([1])
    eval_source = State({"input_values": inputs, "use_cache": True})
    observed = []

    def forward(runtime, state):
        values = state.native_model_inputs
        assert values["use_cache"] is False and values["return_dict"] is True
        observed.append((runtime.model.training, torch.is_grad_enabled(), "labels" in values))
        logits = runtime.model(values["input_values"])
        loss = torch.nn.functional.cross_entropy(logits, values["labels"]) if "labels" in values else None
        return SimpleNamespace(loss=loss, logits=logits)

    def prepare(processor, sample, *, phase):
        assert phase == "eval"
        return inputs

    def build(batch, *, raw_batch, phase):
        assert batch is inputs and phase == "eval"
        return eval_source

    facade = SimpleNamespace(
        model_adapter=SimpleNamespace(structural_handles=lambda model: {"llm.final_norm": model.final_norm}),
        processor_adapter=SimpleNamespace(prepare_inputs=prepare),
        state_builder=SimpleNamespace(build=build),
        runner=SimpleNamespace(forward_train=forward),
    )
    model = model_from_base().train()
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    selected, optimizer, optimizer_config = common.final_norm_optimizer(
        facade, model, learning_rate=0.01, torch=torch,
    )
    assert [name for name, _ in selected] == ["final_norm.weight"]
    assert [name for name, parameter in model.named_parameters() if parameter.requires_grad] == ["final_norm.weight"]
    state = State({"input_values": inputs, "labels": labels, "use_cache": False, "return_dict": True})
    runtime = SimpleNamespace(model=model)
    progress_path = tmp_path / "progress.jsonl"
    records = []
    for step in range(1, optimizer_steps + 1):
        loss, logits, parameters = common.training_forward_step(
            facade=facade, runtime=runtime, state=state, optimizer=optimizer, selected=selected, torch=torch,
        )
        assert torch.isfinite(logits).all()
        checks = common.training_step_checks(loss=loss, parameters=parameters, stable_layout=True, torch=torch)
        assert all(checks.values())
        common.record_training_step(progress_path, records, {
            "event": "optimizer_step", "step": step, "loss": float(loss.detach()),
            "trainable_parameters": parameters, "checks": checks,
        })
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), original[name]) is (name != "final_norm.weight")

    args = SimpleNamespace(output_dir=tmp_path, checkpoint=tmp_path / "base", code_revision="fixture")
    scope = {"algorithm": "fixture", "variant": "fixture.video", "backbone": "tiny", "binding": "fixture.video.tiny"}
    checkpoint = common.save_training_checkpoint(
        args, records=records, selected=selected, scope=scope, canonical_config={},
        optimizer=optimizer_config, optimizer_steps=optimizer_steps, torch=torch,
    )
    common.checkpoint_saved(progress_path, checkpoint, optimizer_steps=optimizer_steps)
    assert Path(checkpoint["path"]).name == f"checkpoint-{optimizer_steps}"
    saved_state = json.loads(Path(checkpoint["training_state_path"]).read_text(encoding="utf-8"))
    assert saved_state["global_step"] == optimizer_steps
    fresh = model_from_base()
    assert torch.equal(fresh.final_norm.weight, original["final_norm.weight"])
    reloaded = common.load_lightweight_checkpoint(fresh, checkpoint, torch=torch)
    assert reloaded["all_exact"] and reloaded["parameter_records"][0]["exact_reload"]
    assert torch.equal(fresh.final_norm.weight, model.final_norm.weight)
    _, eval_state = common.inference_state(facade=facade, processor=object(), sample=object())
    assert "labels" not in eval_state.native_model_inputs
    assert eval_source.native_model_inputs["use_cache"] is True
    with torch.inference_mode():
        expected_logits = model.eval()(inputs)
    actual_logits = common.inference_forward(
        facade=facade, runtime=SimpleNamespace(model=fresh), state=eval_state, torch=torch,
    )
    assert torch.equal(actual_logits, expected_logits) and not actual_logits.requires_grad
    assert observed == [(True, True, True)] * optimizer_steps + [(False, False, False)]
    common.record_post_reload(progress_path, {"checks": {"exact_logits": True}, "reload": reloaded})
    events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in events] == ["optimizer_step"] * optimizer_steps + ["checkpoint_saved", "post_reload_inference"]
    assert events[-2]["global_step"] == optimizer_steps


@pytest.mark.parametrize("failure", ["nonfinite_loss", "missing_target", "no_update", "changed_layout"])
def test_training_step_checks_reject_failed_training_evidence(failure: str) -> None:
    torch = pytest.importorskip("torch")
    loss = torch.tensor(float("nan") if failure == "nonfinite_loss" else 1.0)
    parameters = [{
        "gradient_present": True, "gradient_finite": True, "gradient_nonzero": True,
        "max_parameter_delta": 0.0 if failure == "no_update" else 0.01,
    }]
    checks = common.training_step_checks(
        loss=loss, parameters=[] if failure == "missing_target" else parameters,
        stable_layout=failure != "changed_layout", torch=torch,
    )
    failed_check = {
        "nonfinite_loss": "finite_loss", "missing_target": "exact_single_intended_target",
        "no_update": "target_parameter_updated", "changed_layout": "layout_stable_across_steps",
    }[failure]
    assert checks[failed_check] is False


def test_failed_training_step_is_persisted_before_success_summary(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.jsonl"
    records = []
    record = {"event": "optimizer_step", "step": 1, "loss": 1.0, "trainable_parameters": [], "checks": {"exact_single_intended_target": False}}
    with pytest.raises(RuntimeError, match="Training step 1 failed"):
        common.record_training_step(progress_path, records, record)
        common.print_training_step(record, gradient_norm=record["trainable_parameters"][0]["gradient_norm"])
    assert records == [record]
    assert json.loads(progress_path.read_text(encoding="utf-8")) == record


def test_failed_post_reload_is_persisted_before_error(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.jsonl"
    report = {"checks": {"layout_matches_training": False}, "layout_records": []}
    with pytest.raises(RuntimeError, match="Post-reload compressed inference failed"):
        common.record_post_reload(progress_path, report)
    assert json.loads(progress_path.read_text(encoding="utf-8")) == {"event": "post_reload_inference", **report}


@pytest.mark.parametrize("fails", [False, True])
def test_training_main_publishes_final_or_failure_evidence(tmp_path: Path, fails: bool) -> None:
    output = tmp_path / "run"
    parser = argparse.ArgumentParser()
    parser.set_defaults(output_dir=output, learning_rate=0.01, physical_gpu=0, code_revision="fixture")
    scope = {"algorithm": "fixture", "binding": "fixture.video"}

    def run(args):
        assert args.output_dir == output
        common.append_jsonl(output / "progress.jsonl", {"event": "completed_step"})
        if fails:
            raise RuntimeError("layout failed")
        return {"status": "passed", "scope": scope, "checked": True}

    kwargs = dict(parser=parser, run=run, scope=scope, gpu_snapshot=lambda gpu: {"gpu": gpu})
    if fails:
        with pytest.raises(RuntimeError, match="layout failed"):
            common.training_main([], **kwargs)
    else:
        assert common.training_main([], **kwargs) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == ("failed" if fails else "passed")
    assert report["scope"]["binding"] == scope["binding"]
    if fails:
        assert report["error_type"] == "RuntimeError"
        assert report["progress_sha256"] == common.sha256(output / "progress.jsonl")
        assert report["gpu_after"] == {"gpu": 0}
    else:
        assert report["checked"] is True


def test_llava_vqtoken_command_resolves_checkpoint_and_persists_early_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.diagnostics import validate_llava_vqtoken_video_training_checkpoint as command

    output = tmp_path / "run"
    checkpoint = tmp_path / "models" / ".." / "checkpoint"
    observed = []
    failure = RuntimeError("checkpoint unavailable")

    def inspect_checkpoint(path):
        observed.append(path)
        raise failure

    monkeypatch.setattr(command, "prepare_checkpoint_environment", lambda gpu: str(gpu))
    monkeypatch.setattr(command, "_checkpoint_record", inspect_checkpoint)
    monkeypatch.setattr(command, "_gpu_snapshot", lambda gpu: {"gpu": gpu})
    with pytest.raises(RuntimeError, match="checkpoint unavailable") as caught:
        command.main([
            "--checkpoint", str(checkpoint),
            "--nextqa_parquet", str(tmp_path / "nextqa.parquet"),
            "--video_root", str(tmp_path / "videos"),
            "--output_dir", str(output),
            "--code_revision", "fixture-revision",
            "--physical_gpu", "0",
        ])

    assert caught.value is failure
    assert observed == [checkpoint.resolve()]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert report["error"] == "checkpoint unavailable"
    assert report["scope"] == {
        "algorithm": "vqtoken", "variant": "vqtoken.video",
        "backbone": "llava_onevision", "binding": "vqtoken.video.llava_onevision",
        "execution_phase": "forward",
    }
    assert report["code_revision"] == "fixture-revision"
    assert report["gpu_after"] == {"gpu": 0}
    assert not (output / "progress.jsonl").exists()
