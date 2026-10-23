"""Explicit modality-split runtime lifecycle, separate from legacy stages."""
from __future__ import annotations

import json
from pathlib import Path

from mllm_comp.core.errors import BackendIntegrationError


def activate_split(runtime, backbone, spec=None, checkpoint=None):
    from mllm_comp.modality_split import install_modality_split, load_modality_split

    if isinstance(spec, str):
        import yaml
        spec = yaml.safe_load(Path(spec).read_text())
    if spec is not None and (not isinstance(spec, dict) or spec.get("backbone") != backbone):
        raise BackendIntegrationError("modality_split backbone does not match runtime backbone.")
    if checkpoint is not None:
        native_source = getattr(runtime, "model_path_or_id", None)
        if native_source is None or Path(native_source).resolve() != Path(checkpoint).resolve():
            raise BackendIntegrationError(
                "Trained modality_split sidecar requires native weights loaded from the same checkpoint directory.")
        saved_spec = json.loads((Path(checkpoint) / "modality_split.json").read_text())
        if saved_spec.get("backbone") != backbone:
            raise BackendIntegrationError("Saved modality_split backbone does not match runtime.")
        if spec is not None and saved_spec != spec:
            raise BackendIntegrationError("Requested modality_split differs from saved checkpoint.")
        session = load_modality_split(runtime.model, checkpoint)
    else:
        session = install_modality_split(runtime.model, spec)
    runtime.metadata["_modality_split_session"] = session
    runtime.metadata["modality_split"] = session.metadata()
    runtime.metadata["compression_mode"] = "compressed"
    return session
