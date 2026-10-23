"""Real Swift CLI: Operation + HTA train/save/resume, then public inference.

Run from the resolved Dev environment on one visible GPU, after checking space
for three full model checkpoints. This diagnostic trains only the original HTA
module and the final LLM norm. It does not validate full-parameter convergence,
LoRA/DeepSpeed restoration, or answer quality.

    python -m scripts.diagnostics.validate_qwen25_omni_swift_resume \
        --model_path /local/Qwen2.5-Omni-3B --output_dir /fresh/diagnostic --device cuda:0

The same file is a native Swift external plugin registering one observational
callback. Model loading, training, optimizer construction and saves stay native.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any


_CALLBACK = "mllm_comp_joint_resume_diagnostic"
_CALLBACK_FILE = "joint_resume_observation.jsonl"
_HTA_PREFIX = "thinker._mllm_comp_fastslm_hta_audio."
_NORM = "thinker.model.norm.weight"
_TRAINABLE_REGEX = r"^thinker\.(?:_mllm_comp_fastslm_hta_audio\..+|model\.norm\.weight)$"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _register_callback() -> None:
    import torch
    from swift.callbacks import TrainerCallback, callbacks_map

    class ObserveJointTraining(TrainerCallback):
        def _record(self, event: str, **values: Any) -> None:
            path = Path(self.args.output_dir) / _CALLBACK_FILE
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": event, **values}, sort_keys=True) + "\n")

        def on_train_begin(self, args, state, control, **kwargs):
            self.selected = {
                name: parameter for name, parameter in self.trainer.model.named_parameters()
                if parameter.requires_grad
            }
            expected = {
                name for name, _ in self.trainer.model.named_parameters()
                if re.search(_TRAINABLE_REGEX, name)
            }
            optimizer = kwargs["optimizer"]
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            checks = {
                "exact_trainable_names": set(self.selected) == expected,
                "hta_present": any(name.startswith(_HTA_PREFIX) for name in self.selected),
                "llm_norm_present": _NORM in self.selected,
                "optimizer_owns_exact_trainables": optimizer_ids == {id(p) for p in self.selected.values()},
                "resume_has_optimizer_state": state.global_step == 0 or bool(optimizer.state),
            }
            self._record(
                "train_begin", global_step=state.global_step, checks=checks,
                parameters={name: {"shape": list(p.shape), "dtype": str(p.dtype), "numel": p.numel()}
                            for name, p in self.selected.items()},
                optimizer_state_count=len(optimizer.state),
            )
            if not all(checks.values()):
                raise RuntimeError(f"Native Swift trainable/optimizer ownership failed: {checks}")

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            groups = {}
            for group, names in (
                ("hta", [name for name in self.selected if name.startswith(_HTA_PREFIX)]),
                ("llm_norm", [_NORM]),
            ):
                gradients = [self.selected[name].grad for name in names if self.selected[name].grad is not None]
                groups[group] = {
                    "gradient_tensors": len(gradients),
                    "finite": bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients),
                    "nonzero": any(bool(torch.count_nonzero(g)) for g in gradients),
                }
            self._record("before_optimizer_step", step=state.global_step + 1, groups=groups)
            if not all(value["finite"] and value["nonzero"] for value in groups.values()):
                raise RuntimeError(f"HTA and LLM norm both require finite nonzero gradients: {groups}")

    callbacks_map[_CALLBACK] = ObserveJointTraining


def _training_config(model_path: Path, output: Path, dataset: Path, device: str):
    from mllm_comp.config import load_config

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/train/qwen2_5_omni/train_aud_sft_fastslm_hta.yaml")
    operation = load_config(root / "configs/train/qwen2_5_omni/train_aud_sft_spade_operation_only.yaml").pipeline.stages[0]
    source = _json(model_path / "config.json")
    depth = int(source["thinker_config"]["text_config"]["num_hidden_layers"])
    if depth < 2:
        raise ValueError("This structural diagnostic requires at least two source decoder layers.")
    operation.activation_scope["layers"] = [depth // 2]
    config.pipeline.stages.append(operation)  # Dispatcher must order PREPARE first.
    config.backbone.model_path = str(model_path)
    config.backbone.config.update({
        "device": device, "torch_dtype": "bfloat16", "attn_implementation": "eager",
        "enable_talker": False,
    })
    config.experiment.name = "swift_operation_hta_resume"
    config.experiment.seed = 731
    config.train.max_steps = 2
    config.train.args = {
        "datasets": [str(dataset)], "trainable_branch": "thinker", "enable_audio_output": False,
        "tuner_type": "full", "freeze_parameters_regex": ".*",
        "trainable_parameters_regex": _TRAINABLE_REGEX,
        "freeze_llm": True, "freeze_vit": True, "freeze_aligner": True,
        "torch_dtype": "bfloat16", "attn_impl": "eager", "device_map": device,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
        # A diagnostic update must exceed BF16 rounding around norm weights=1.
        "learning_rate": 0.01, "lr_scheduler_type": "constant", "warmup_ratio": 0,
        "optim": "adamw_torch", "weight_decay": 0, "max_length": 512,
        "gradient_checkpointing": True, "gradient_checkpointing_kwargs": '{"use_reentrant": false}',
        "vit_gradient_checkpointing": False, "packing": False, "padding_free": False,
        "seed": 731, "data_seed": 731, "full_determinism": True,
        "dataset_shuffle": False, "train_dataloader_shuffle": False,
        "split_dataset_ratio": 0, "dataset_num_proc": 1,
        "dataloader_num_workers": 0, "load_from_cache_file": False, "ignore_data_skip": False,
        "logging_steps": 1, "save_steps": 1, "save_total_limit": 2,
        "save_only_model": False, "safe_serialization": True, "report_to": ["none"],
        "external_plugins": [str(Path(__file__).resolve())], "callbacks": [_CALLBACK],
    }
    config.data.dataset_id = str(dataset)
    config.output.run_dir = str(output)
    config.output.manifest_path = None
    return config, depth


def _selected_checkpoint_tensors(checkpoint: Path, names: set[str]) -> dict[str, Any]:
    from safetensors import safe_open

    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        weight_map = _json(index)["weight_map"]
        shards = {name: weight_map[name] for name in names}
    else:
        shards = {name: "model.safetensors" for name in names}
    values = {}
    for filename in sorted(set(shards.values())):
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as stream:
            for name in names:
                if shards[name] == filename:
                    values[name] = stream.get_tensor(name)
    return values


def _compare(left: Any, right: Any) -> dict[str, Any]:
    """Exact comparison for this deterministic, same-device continuation."""
    import numpy as np
    import torch

    errors = []
    maximum = 0.0

    def visit(a, b, path):
        nonlocal maximum
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            equal = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            if a.shape == b.shape and a.numel():
                delta = float((a.double() - b.double()).abs().max())
                maximum = max(maximum, delta)
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            equal = a.dtype == b.dtype and np.array_equal(a, b)
        elif isinstance(a, dict) and isinstance(b, dict):
            equal = a.keys() == b.keys()
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, type(a)):
            equal = len(a) == len(b)
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{index}]")
        else:
            equal = type(a) is type(b) and a == b
        if not equal:
            errors.append(path)

    visit(left, right, "state")
    return {"equal": not errors, "max_abs_error": maximum, "mismatched_paths": sorted(errors),
            "rtol": 0, "atol": 0}


def _observations(checkpoint: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (checkpoint.parent / _CALLBACK_FILE).read_text(encoding="utf-8").splitlines()]


def _infer(run_dir: Path, device: str, audio: Path, expected_depth: int) -> dict[str, Any]:
    import torch
    from mllm_comp.backbones.qwen25_omni import Qwen25OmniSample
    from mllm_comp.restoration import restore_config
    from mllm_comp.runtime.loader import load_backbone_runtime

    config = restore_config(run_dir, task="eval")
    facade, runtime = load_backbone_runtime(config, device=device, enable_talker=False)
    try:
        thinker = getattr(runtime.model, "thinker", runtime.model)
        actual_depth = len(thinker.model.layers)
        if actual_depth != expected_depth:
            raise RuntimeError(f"Restored inference has {actual_depth} decoder layers; expected {expected_depth}.")
        sample = Qwen25OmniSample(sample_id="joint-resume", conversation=[{
            "role": "user", "content": [
                {"type": "audio", "audio": str(audio)},
                {"type": "text", "text": "Briefly describe the sound."},
            ],
        }])
        batch = facade.processor_adapter.prepare_inputs(runtime.processor, sample, phase="eval")
        state = facade.state_builder.build(batch, phase="eval")
        with torch.no_grad():
            generated = facade.runner.generate(
                runtime, state, processed_batch=batch, thinker_min_new_tokens=2, thinker_max_new_tokens=2,
                do_sample=False, use_cache=True, return_audio=False,
            )
        compression = generated.metadata["compression"]
        if not compression or compression["kept_audio_tokens"] >= compression["original_audio_tokens"]:
            raise RuntimeError("Restored inference did not execute the learned audio compression.")
        return {"decoder_layers": actual_depth, "text": generated.text.text,
                "token_ids": generated.text.token_ids.tolist(), "compression": compression}
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    import torch
    from mllm_comp.core.serialization import atomic_write_json
    from mllm_comp.experiments import run_experiment
    from mllm_comp.restoration import restore_config
    from tests.integration.test_qwen25_omni_audio_compression_gpu_smoke import _create_audio

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; existing diagnostic artifacts are not overwritten.")
    if not args.device.startswith("cuda") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("This diagnostic requires one visible GPU and its CUDA device name.")
    if any(int(os.environ.get(name, "1")) > 1 for name in ("WORLD_SIZE", "NPROC_PER_NODE")):
        raise ValueError("This exact-continuation diagnostic runs one native Swift worker.")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "running", "scope": "native Swift full-tuner HTA+norm, one GPU, two steps",
        "comparison_policy": "Exact saved values: same device/dtype, eager attention, full_determinism, fixed sequential data and identical max_steps=2.",
    }
    try:
        audio = output / "tone.wav"
        _create_audio(audio)
        dataset = output / "train.jsonl"
        rows = [{"messages": [
            {"role": "user", "content": f"<audio>{prompt}"},
            {"role": "assistant", "content": answer},
        ], "audios": [str(audio)]} for prompt, answer in (
            ("Briefly describe the sound.", "A steady synthetic tone."),
            ("Is this speech or a tone?", "This is a continuous tone, not speech."),
        )]
        dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        config, source_depth = _training_config(args.model_path.resolve(), output / "continuous", dataset, args.device)
        continuous = run_experiment(config, task="train")
        report["continuous"] = continuous.to_dict()
        if continuous.status != "succeeded":
            raise RuntimeError("Continuous native Swift training failed; inspect its result and logs.")
        continuous_backend = _json(Path(continuous.paths.manifest))["train_backend"]["args"]
        report["continuous_argv"] = continuous_backend["argv"]
        report["continuous_compression_evidence"] = continuous_backend["compression_training"]
        reference = Path(continuous.output["checkpoint_path"])
        first = reference.parent / "checkpoint-1"
        if _json(first / "trainer_state.json")["global_step"] != 1:
            raise RuntimeError("The uninterrupted run did not save its first native training step.")
        restored = restore_config(continuous.paths.run_dir, task="train", checkpoint=first)
        restored.output.run_dir = str(output / "resumed")
        resumed = run_experiment(restored, task="train")
        report["resumed"] = resumed.to_dict()
        if resumed.status != "succeeded":
            raise RuntimeError("Native Swift full restoration failed; inspect its result and logs.")
        resumed_backend = _json(Path(resumed.paths.manifest))["train_backend"]["args"]
        report["resumed_argv"] = resumed_backend["argv"]
        report["resumed_compression_evidence"] = resumed_backend["compression_training"]
        actual = Path(resumed.output["checkpoint_path"])
        observations = _observations(reference)
        resumed_observations = _observations(actual)
        beginning = next(item for item in observations if item["event"] == "train_begin")
        resumed_beginning = next(item for item in resumed_observations if item["event"] == "train_begin")
        if resumed_beginning["global_step"] != 1 or resumed_beginning["parameters"] != beginning["parameters"]:
            raise RuntimeError("Swift did not resume the same trainable parameters at global_step=1.")
        names = set(beginning["parameters"])
        before = _selected_checkpoint_tensors(first, names)
        after = _selected_checkpoint_tensors(reference, names)
        recovered = _selected_checkpoint_tensors(actual, names)
        deltas = {name: float((after[name].float() - before[name].float()).abs().max()) for name in names}
        updates = {
            "hta": max(value for name, value in deltas.items() if name.startswith(_HTA_PREFIX)),
            "llm_norm": deltas[_NORM],
        }
        comparisons = {"parameters": _compare(after, recovered)}
        for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            # These are trusted artifacts produced above by our own native CLI.
            comparisons[filename] = _compare(
                torch.load(reference / filename, map_location="cpu", weights_only=False),
                torch.load(actual / filename, map_location="cpu", weights_only=False),
            )
        steps = {name: _json(path / "trainer_state.json")["global_step"]
                 for name, path in (("continuous", reference), ("resumed", actual))}
        report.update({"observations": observations, "resumed_observations": resumed_observations,
                       "checkpoint1_to_2_max_parameter_delta": updates, "comparisons": comparisons,
                       "global_steps": steps, "source_decoder_layers": source_depth})
        if not all(value > 0 for value in updates.values()) or not all(item["equal"] for item in comparisons.values()) or set(steps.values()) != {2}:
            raise RuntimeError("Saved joint updates or exact native optimizer continuation differ; inspect numerical evidence.")
        del before, after, recovered
        report["inference"] = _infer(Path(resumed.paths.run_dir), args.device, audio, source_depth - 1)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        atomic_write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
elif os.environ.get("MLLM_COMP_TRAIN_COMPRESSION_SPEC"):
    _register_callback()
