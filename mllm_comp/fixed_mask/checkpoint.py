"""Persist mask semantics with native full-weight and PEFT checkpoints."""
from __future__ import annotations

from functools import wraps
import json
from pathlib import Path

from .spec import normalize_spec


def inspect_checkpoint(directory):
    from .runtime import SIDECAR, CONFIG_KEY
    directory = Path(directory)
    sidecar = directory / SIDECAR
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    config = directory / "config.json"
    if config.is_file():
        return json.loads(config.read_text(encoding="utf-8")).get(CONFIG_KEY)
    return None


def validate_checkpoint(directory, backbone, stages, *, compression_mode="compressed"):
    saved = inspect_checkpoint(directory)
    if saved is None:
        return None
    selected = [s for s in stages if s.enabled and s.algorithm == "fixed_mask"]
    if compression_mode != "compressed" or len(selected) != 1:
        raise ValueError("A fixed-mask checkpoint requires its fixed_mask stage and compression_mode=compressed")
    expected = normalize_spec(selected[0].config)
    actual = normalize_spec(saved["spec"])
    # Generation provenance is explanatory; behavior is determined by selections.
    expected.pop("provenance")
    actual.pop("provenance")
    if saved["backbone"] != backbone or expected != actual:
        raise ValueError("Checkpoint fixed masks differ from the configured fixed_mask stage")
    return saved


def patch_swift_save(trainer_class=None):
    if trainer_class is None:
        from swift.trainers.mixin import SwiftMixin as trainer_class
    if vars(trainer_class).get("_fixed_mask_save_hook", False):
        return
    original = trainer_class._save_model

    @wraps(original)
    def save(trainer, output_dir=None, state_dict=None):
        from .runtime import session_for, save_fixed_mask
        result = original(trainer, output_dir, state_dict)
        for module in trainer.model.modules():
            session = session_for(module)
            if session is None:
                continue
            directory = output_dir or trainer.args.output_dir
            save_fixed_mask(trainer.model, directory)
            recorder = getattr(module, "_mllm_comp_training_recorder", None)
            if recorder is not None:
                recorder.record("save", executor_family="image_fixed_mask", output_dir=str(directory),
                                fixed_mask=session.payload())
            break
        return result

    trainer_class._save_model = save
    trainer_class._fixed_mask_save_hook = True
