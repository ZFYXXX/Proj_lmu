"""Saved strong-anchor control evaluated through the existing analysis lifecycle."""
import hashlib
import json
from pathlib import Path
import time

from mllm_comp.analysis.common.compression_training_check import (
    anchor_base, compact_anchor, evaluate_state, implementation_hash, load_anchor,
)
from mllm_comp.analysis.common.learned_hybrid_runtime import CHECKPOINT, REVISION, prepare_state
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.sample import prepare_teacher_forced_analysis_forward_state
from mllm_comp.analysis.common.visual_token_generation import primary_task_score
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.analysis.probes.visual_evidence_interaction import _Session as EvidenceSession
from mllm_comp.core.serialization import to_jsonable


class AnchorTrainingEvaluation:
    name = "anchor_training_evaluation"
    version = "1.0.0"
    requires_teacher_forcing = True

    def validate_config(self, config):
        if config.get("stage") not in {"smoke", "development", "confirmation"} or not config.get("checkpoint_path"):
            raise ValueError("Explicit phase and saved anchor checkpoint required")
        if config.get("save_raw", False):
            raise ValueError("No complete hidden-state dumps")

    def open(self, *, backbone, runtime, config, phase, output_dir):
        self.validate_config(config)
        return Session(self, backbone, runtime, config, phase, output_dir)


class Session(EvidenceSession):
    def __post_init__(self):
        if self.backbone.capabilities().name != "qwen25_vl" or self.runtime.revision != REVISION:
            raise ValueError("Frozen native Qwen2.5-VL revision required")
        self.root, self.stage = Path(self.config["artifact_root"]), self.config["stage"]
        self.code_hash = implementation_hash()
        checkpoint = Path(self.config["checkpoint_path"])
        self.module = load_anchor(checkpoint, self.runtime.device).eval()
        self.checkpoint_hash = hashlib.sha256((checkpoint / "projector.safetensors").read_bytes()).hexdigest()
        self.config_hash = digest(json.dumps(self.config, sort_keys=True) + self.code_hash + self.checkpoint_hash +
            hashlib.sha256(Path(self.config["cohort_path"]).read_bytes()).hexdigest())
        self.completed = []
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _process(self, sample, entry, folder):
        import torch
        from mllm_comp.analysis.common.condition_ledger import append_json, records
        started = time.perf_counter()
        fingerprint = media_fingerprint(sample["media"])
        if fingerprint != entry["image_fingerprint"]:
            raise ValueError("Held-out image mismatch")
        handles = self.backbone.model_adapter.structural_handles(self.runtime.model)
        bundle = {"facade": self.backbone, "runtime": self.runtime, "model": self.runtime.model,
                  "processor": self.runtime.processor, "llm": handles["llm"], "vision": handles["vision_encoder"], "device": self.runtime.device}
        teacher, _ = prepare_teacher_forced_analysis_forward_state(backbone=self.backbone, runtime=self.runtime,
            sample=sample, probe_name=self.plugin.name)
        state = prepare_state(bundle, sample["processor_outputs"], teacher.native_model_inputs)
        base = anchor_base(state)
        ledger = folder / "conditions.jsonl"
        recorded = records(ledger, self.config_hash)
        for name in ["dense", "feature_anchor_untrained", "feature_anchor_adapter"]:
            key = (sample["sample_id"], name)
            if key in recorded:
                continue
            output, compact = None, None
            if name != "dense":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = self.module(base) if name == "feature_anchor_adapter" else base
                compact = compact_anchor(state, output)
            result = evaluate_state(bundle, state, compact, max_new=16 if "chartqa" in sample["task_name"] else 32,
                native_check=self.stage == "smoke" and name == "dense")
            prediction = result["prediction"]
            stops = sample.get("generation_stops", [])
            for stop in ([stops] if isinstance(stops, str) else stops):
                if stop:
                    prediction = prediction.split(stop, 1)[0].strip()
            metrics = sample["score_prediction"](prediction)
            strict = None
            if "chartqa" in sample["task_name"]:
                answers = sample["raw_doc"]["answer"]
                answers = answers if isinstance(answers, list) else [answers]
                strict = float(any(prediction.casefold() == str(a).strip().casefold() for a in answers))
            row = {**result, "sample_id": sample["sample_id"], "task": sample["task_name"], "stage": self.stage,
                   "image_fingerprint": fingerprint, "config_hash": self.config_hash, "variant": name,
                   "status": "completed", "checkpoint": CHECKPOINT, "revision": REVISION,
                   "projector_sha256": self.checkpoint_hash if name == "feature_anchor_adapter" else None,
                   "prediction": prediction, "primary_task_score": primary_task_score(sample["task_name"], metrics),
                   "strict_score": strict, "native_task_metrics": metrics,
                   "compression_trace": None if output is None else {k: output[k] for k in ["indices", "members", "position_policy"]}}
            append_json(ledger, row)
            recorded[key] = row
        if len(recorded) != 3:
            raise ValueError("Incomplete or duplicate anchor conditions")
        return {"sample_id": sample["sample_id"], "task": sample["task_name"], "stage": self.stage,
                "image_fingerprint": fingerprint, "config_hash": self.config_hash, "implementation_hash": self.code_hash,
                "condition_path": str(ledger), "condition_count": 3, "elapsed_s": time.perf_counter()-started,
                "native_generation_identity_passed": recorded[(sample["sample_id"], "dense")].get("native_generation_identity_passed")}
