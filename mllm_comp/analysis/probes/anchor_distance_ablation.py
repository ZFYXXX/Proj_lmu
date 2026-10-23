"""Standalone opt-in probe, deliberately not added to a live frozen registry."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from mllm_comp.analysis.common.anchor_distance_ablation import cluster, compare, specifications
from mllm_comp.analysis.common.compression_training_check import anchor_base, compact_anchor, evaluate_state, implementation_hash as parent_hash
from mllm_comp.analysis.common.condition_ledger import append_json, records
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.learned_hybrid_runtime import CHECKPOINT, REVISION, prepare_state
from mllm_comp.analysis.common.sample import prepare_teacher_forced_analysis_forward_state
from mllm_comp.analysis.common.visual_token_generation import primary_task_score
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.analysis.probes.visual_evidence_interaction import _Session as EvidenceSession


def implementation_hash():
    from mllm_comp.analysis.common import anchor_distance_ablation
    paths = [Path(__file__), Path(anchor_distance_ablation.__file__),
             Path(__file__).parents[3]/"scripts/analysis/qwen25_vl/run_anchor_distance_ablation.py"]
    h = hashlib.sha256(parent_hash().encode())
    for p in paths:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def timed_evaluate(bundle, state, compact, *, max_new, native_check):
    """Separate first decode prefill from answer-NLL and subsequent decode work."""
    import torch
    calls, events = [0], []
    def before(module, args):
        if calls[0] == 1:  # answer_logits is call zero, cached_decode starts at one
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); events.append((start, end))
    def after(module, args, output):
        if calls[0] == 1:
            events[0][1].record()
        calls[0] += 1
    a = bundle["llm"].register_forward_pre_hook(before)
    b = bundle["llm"].register_forward_hook(after)
    torch.cuda.synchronize(bundle["device"])
    started = time.perf_counter()
    try:
        result = evaluate_state(bundle, state, compact, max_new=max_new, native_check=native_check)
        torch.cuda.synchronize(bundle["device"])
    finally:
        a.remove(); b.remove()
    if len(events) != 1:
        raise ValueError("Could not locate cached prefill call")
    result["decoder_prefill_gpu_ms"] = events[0][0].elapsed_time(events[0][1])
    result["nll_and_generation_ms"] = 1000*(time.perf_counter()-started)
    return result


class AnchorDistanceAblation:
    name = "anchor_distance_ablation"
    requires_teacher_forcing = True


class Session(EvidenceSession):
    def __post_init__(self):
        if self.runtime.revision != REVISION or Path(self.runtime.model_path_or_id).resolve() != Path(CHECKPOINT).resolve():
            raise ValueError("This is the isolated 3B experiment, not a 7B training modification")
        self.root, self.stage = Path(self.config["artifact_root"]), self.config["stage"]
        self.code_hash = implementation_hash()
        self.config_hash = digest(json.dumps(self.config, sort_keys=True)+self.code_hash+
            hashlib.sha256(Path(self.config["cohort_path"]).read_bytes()).hexdigest())
        self.completed = []
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _process(self, sample, entry, folder):
        import torch
        started = time.perf_counter()
        fingerprint = media_fingerprint(sample["media"])
        if fingerprint != entry["image_fingerprint"]:
            raise ValueError("Frozen image fingerprint mismatch")
        handles = self.backbone.model_adapter.structural_handles(self.runtime.model)
        bundle = {"facade": self.backbone, "runtime": self.runtime, "model": self.runtime.model,
            "processor": self.runtime.processor, "llm": handles["llm"], "vision": handles["vision_encoder"], "device": self.runtime.device}
        teacher, _ = prepare_teacher_forced_analysis_forward_state(backbone=self.backbone, runtime=self.runtime,
            sample=sample, probe_name=self.plugin.name)
        state = prepare_state(bundle, sample["processor_outputs"], teacher.native_model_inputs)
        reference = cluster(state["features"], height=state["height"], width=state["width"])
        old = anchor_base(state)
        if old["members"] != tuple(tuple(g) for g in reference["members"]) and [list(g) for g in old["members"]] != reference["members"]:
            raise ValueError("Cosine control is not the existing deployed grouping")
        if old["indices"] != reference["indices"] or not torch.equal(old["tokens"], reference["tokens"]):
            raise ValueError("Cosine control values/layout differ from deployed baseline")
        ledger = folder/"conditions.jsonl"
        saved = records(ledger, self.config_hash)
        specs = [None]+specifications()
        # Counterbalance condition order by image hash, avoiding a systematic
        # warm-up/thermal advantage; initialization RNG remains separate.
        import random
        random.Random(fingerprint+":condition-order-v1").shuffle(specs)
        for spec in specs:
            name = spec.name if spec else "dense"
            key = (sample["sample_id"], name)
            if key in saved:
                continue
            output, compact, compression_ms = None, None, None
            if spec:
                torch.cuda.synchronize(bundle["device"])
                t = time.perf_counter()
                output = cluster(state["features"], height=state["height"], width=state["width"], spec=spec)
                compact = compact_anchor(state, output)
                torch.cuda.synchronize(bundle["device"])
                compression_ms = 1000*(time.perf_counter()-t)
            result = timed_evaluate(bundle, state, compact, max_new=16 if "chartqa" in sample["task_name"] else 32,
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
                "status": "completed", "checkpoint": CHECKPOINT, "revision": REVISION, "uses_mlp": False,
                "prediction": prediction, "primary_task_score": primary_task_score(sample["task_name"], metrics),
                "strict_score": strict, "native_task_metrics": metrics, "repetition_penalty": 1.,
                "geometry_spec": asdict(spec) if spec else None, "diagnostic_compression_ms": compression_ms,
                "comparison_to_cosine": compare(reference, output) if output else None,
                "geometry": output["geometry"] if output else None,
                "compression_trace": {k:output[k] for k in ["anchors", "indices", "members", "centroids", "cluster_sizes", "partition_hash", "position_policy"]} if output else None}
            append_json(ledger, row); saved[key] = row
        if len(saved) != len(specs):
            raise ValueError("Incomplete or duplicated geometry conditions")
        a, b = saved[(sample["sample_id"], "cosine_farthest")], saved[(sample["sample_id"], "unit_l2_farthest")]
        if b["comparison_to_cosine"]["partition_identical"] and a["generated_token_ids"] != b["generated_token_ids"]:
            raise ValueError("Identical compressed states produced different greedy output")
        if self.stage == "smoke" and not saved[(sample["sample_id"], "dense")].get("native_generation_identity_passed"):
            raise ValueError("Native generation equivalence not verified")
        return {"sample_id": sample["sample_id"], "task": sample["task_name"], "stage": self.stage,
            "image_fingerprint": fingerprint, "config_hash": self.config_hash, "implementation_hash": self.code_hash,
            "condition_path": str(ledger), "condition_count": len(saved), "elapsed_s": time.perf_counter()-started,
            "cosine_control_bitidentical": True,
            "native_generation_identity_passed": saved[(sample["sample_id"], "dense")].get("native_generation_identity_passed")}
