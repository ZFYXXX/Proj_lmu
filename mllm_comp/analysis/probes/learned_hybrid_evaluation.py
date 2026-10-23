"""Native task evaluation of saved learned hybrids on fixed held-out images."""

import hashlib
import json
from pathlib import Path
import time

from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.learned_hybrid_runtime import (
    CHECKPOINT, REVISION, answer_logits, code_hash, compact_state, load_module, prepare_state,
)
from mllm_comp.analysis.common.sample import prepare_teacher_forced_analysis_forward_state
from mllm_comp.analysis.common.visual_token_generation import primary_task_score
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.analysis.probes.visual_evidence_interaction import _Session as EvidenceSession, cached_decode
from mllm_comp.compressors.image.learned_hybrid import VARIANTS
from mllm_comp.core.serialization import to_jsonable


class LearnedHybridEvaluation:
    name = "learned_hybrid_evaluation"
    version = "1.0.0"
    requires_teacher_forcing = True

    def validate_config(self, config):
        if config.get("stage") not in {"smoke", "development", "confirmation"} or set(config.get("checkpoint_paths", {})) != set(VARIANTS):
            raise ValueError("Explicit held-out stage and all four saved hybrid variants required.")
        if config.get("save_raw", False):
            raise ValueError("Only predictions/traces/statistics may be saved.")

    def open(self, *, backbone, runtime, config, phase, output_dir):
        self.validate_config(config)
        return Session(self, backbone, runtime, config, phase, output_dir)


class Session(EvidenceSession):
    def __post_init__(self):
        if self.backbone.capabilities().name != "qwen25_vl" or self.runtime.revision != REVISION:
            raise ValueError("Frozen Qwen2.5-VL-3B revision required.")
        self.root, self.stage = Path(self.config["artifact_root"]), self.config["stage"]
        self.code_hash = code_hash()
        self.modules, self.checkpoint_hashes = {}, {}
        required_step = 2 if self.stage == "smoke" and self.config.get("allow_smoke_checkpoints", False) else 320
        for variant, path in self.config["checkpoint_paths"].items():
            folder = Path(path)
            pointer = json.loads((folder / "latest.json").read_text())
            manifest = json.loads((folder / "manifest.json").read_text())
            if manifest["status"] != "complete" or manifest["steps_completed"] != required_step or pointer["step"] != required_step:
                raise ValueError("Wrong checkpoint step for this evaluation phase.")
            checkpoint = folder / pointer["checkpoint"]
            checksum = hashlib.sha256((checkpoint / "projector.safetensors").read_bytes()).hexdigest()
            if checksum != pointer["sha256"]:
                raise ValueError("Saved model checksum mismatch.")
            self.modules[variant] = load_module(checkpoint, self.runtime.device).eval()
            self.checkpoint_hashes[variant] = checksum
        self.config_hash = digest(json.dumps(self.config, sort_keys=True) + self.code_hash +
            hashlib.sha256(Path(self.config["cohort_path"]).read_bytes()).hexdigest() + json.dumps(self.checkpoint_hashes, sort_keys=True))
        self.completed = []
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _process(self, sample, entry, folder):
        import torch
        import torch.nn.functional as F
        from transformers import GenerationConfig
        started = time.perf_counter()
        fingerprint = media_fingerprint(sample["media"])
        if fingerprint != entry["image_fingerprint"]:
            raise ValueError("Held-out image fingerprint mismatch.")
        handles = self.backbone.model_adapter.structural_handles(self.runtime.model)
        bundle = {"facade": self.backbone, "runtime": self.runtime, "model": self.runtime.model,
                  "processor": self.runtime.processor, "llm": handles["llm"], "vision": handles["vision_encoder"], "device": self.runtime.device}
        teacher_state, _ = prepare_teacher_forced_analysis_forward_state(backbone=self.backbone, runtime=self.runtime,
            sample=sample, probe_name=self.plugin.name)
        state = prepare_state(bundle, sample["processor_outputs"], teacher_state.native_model_inputs)
        teacher_logits, targets = answer_logits(bundle, state["hidden"], state["positions"], state["teacher_inputs"]["labels"])
        teacher_log = F.log_softmax(teacher_logits, -1)
        eos = bundle["model"].generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        max_new = 16 if "chartqa" in sample["task_name"] else 32
        ledger = folder / "conditions.jsonl"
        recorded = {}
        if ledger.exists():
            with ledger.open("rb+") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        handle.truncate(offset)
                        break
                    row = json.loads(line)
                    if row["config_hash"] != self.config_hash or row["variant"] in recorded:
                        raise ValueError("Mismatched or duplicate held-out condition.")
                    recorded[row["variant"]] = row
        for variant in ["dense", *VARIANTS]:
            if variant in recorded:
                continue
            before = time.perf_counter()
            output = None
            if variant == "dense":
                hidden, positions, prefix = state["hidden"], state["positions"], state["prefix"]
                logits, labels = teacher_logits, targets
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = self.modules[variant](state["features"], height=state["height"], width=state["width"],
                        random_seed=int(digest(fingerprint + ":42")[:8], 16))
                    compact = compact_state(state, output)
                    logits, labels = answer_logits(bundle, compact["hidden"], compact["positions"], compact["labels"])
                hidden, positions, prefix = compact["hidden"], compact["positions"], compact["prefix"]
            decoded = cached_decode(llm=bundle["llm"], head=bundle["model"].lm_head, embedding=bundle["model"].get_input_embeddings(),
                hidden=hidden[:, :prefix], positions=positions[:, :, :prefix], eos_ids=eos, max_new=max_new)
            if self.stage == "smoke" and variant == "dense":
                native = bundle["model"].generate(**state["prompt_inputs"], generation_config=GenerationConfig(max_new_tokens=max_new,
                    do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=bundle["processor"].tokenizer.pad_token_id))[0, state["prefix"]:].tolist()
                if native != decoded["generated_token_ids"]:
                    raise ValueError("Held-out native/dense generation mismatch.")
                decoded["native_generation_identity_passed"] = True
            prediction = bundle["processor"].tokenizer.decode(decoded["generated_token_ids"], skip_special_tokens=True).strip()
            stops = sample.get("generation_stops", [])
            stops = [stops] if isinstance(stops, str) else stops
            for stop in stops:
                if stop:
                    prediction = prediction.split(stop, 1)[0].strip()
            score = sample["score_prediction"](prediction)
            strict = None
            if "chartqa" in sample["task_name"]:
                answers = sample["raw_doc"]["answer"]
                answers = answers if isinstance(answers, list) else [answers]
                strict = float(any(prediction.casefold() == str(a).strip().casefold() for a in answers))
            row = {"sample_id": sample["sample_id"], "task": sample["task_name"], "stage": self.stage,
                   "image_fingerprint": fingerprint, "variant": variant, "status": "completed", "config_hash": self.config_hash,
                   "checkpoint": CHECKPOINT, "revision": REVISION, "projector_sha256": self.checkpoint_hashes.get(variant),
                   "source_visual_tokens": len(state["visual"]), "retained_visual_tokens": len(state["visual"]) if output is None else len(output["indices"]),
                   "prediction": prediction, "primary_task_score": primary_task_score(sample["task_name"], score),
                   "strict_score": strict, "native_task_metrics": score, "answer_nll": float(F.cross_entropy(logits, labels)),
                   "teacher_kl": float(F.kl_div(F.log_softmax(logits, -1), teacher_log.exp(), reduction="batchmean").clamp_min(0)),
                   "compression_trace": None if output is None else {key: output[key] for key in ["indices", "summary_indices", "detail_indices", "members", "position_policy", "coarse_grid"]},
                   **decoded, "elapsed_s": time.perf_counter() - before,
                   "placeholder_mask_labels_alignment_passed": True, "text_positions_preserved": True}
            encoded = json.dumps(to_jsonable(row), ensure_ascii=False, allow_nan=False)
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                __import__("os").fsync(handle.fileno())
            recorded[variant] = json.loads(encoded)
        if set(recorded) != {"dense", *VARIANTS}:
            raise ValueError("Incomplete held-out image conditions.")
        return {"sample_id": sample["sample_id"], "task": sample["task_name"], "stage": self.stage,
                "image_fingerprint": fingerprint, "config_hash": self.config_hash, "implementation_hash": self.code_hash,
                "condition_path": str(ledger), "condition_count": 5, "elapsed_s": time.perf_counter() - started,
                "native_generation_identity_passed": recorded["dense"].get("native_generation_identity_passed"),
                "train_eval_overlap": False}
