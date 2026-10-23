"""Second metric study using the exact previous 500-image sample manifest."""
import hashlib
import json
from pathlib import Path
import random
import time

from mllm_comp.analysis.common import token_metric_ablation as geometry
from mllm_comp.analysis.common.anchor_distance_ablation import compare
from mllm_comp.analysis.common.compression_training_check import compact_anchor, implementation_hash as shared_hash
from mllm_comp.analysis.common.condition_ledger import append_json, records
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.learned_hybrid_runtime import CHECKPOINT, REVISION, prepare_state
from mllm_comp.analysis.common.sample import prepare_teacher_forced_analysis_forward_state
from mllm_comp.analysis.common.visual_token_generation import primary_task_score
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.analysis.probes.anchor_distance_ablation import timed_evaluate, implementation_hash as legacy_hash
from mllm_comp.analysis.probes.visual_evidence_interaction import _Session as EvidenceSession, question_positions


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hash():
    paths = [Path(__file__), Path(geometry.__file__),
        Path(__file__).parents[3]/"scripts/analysis/qwen25_vl/run_token_metric_ablation.py"]
    h = hashlib.sha256((shared_hash()+legacy_hash()).encode())
    for p in paths:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def native_response_gate(bundle, state, question_indices, own):
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
    prefix = state["prefix"]
    attn = bundle["llm"].layers[0].self_attn
    captured, handles = {}, []
    def pre(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        if hidden.shape[1] == prefix:
            captured["rope"] = kwargs["position_embeddings"]
    def projection(name):
        def hook(module, args, output):
            if output.shape[1] == prefix:
                captured[name] = output.detach().clone()
        return hook
    try:
        handles.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
        for name in ["q", "k", "v"]:
            handles.append(getattr(attn, name+"_proj").register_forward_hook(projection(name)))
        with torch.inference_mode():
            bundle["llm"](inputs_embeds=state["hidden"][:, :prefix], position_ids=state["positions"][:, :, :prefix],
                attention_mask=torch.ones((1, prefix), dtype=torch.long, device=bundle["device"]), use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    q = captured["q"].view(1, prefix, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = captured["k"].view(1, prefix, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = captured["v"].view(1, prefix, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    q, k = apply_multimodal_rotary_pos_emb(q, k, *captured["rope"], attn.rope_scaling["mrope_section"])
    qi = torch.tensor(question_indices, device=q.device)
    vi = torch.tensor(state["visual"], device=q.device)
    native = geometry.response_embedding(q[0].index_select(1, qi), k[0].index_select(1, vi), v[0].index_select(1, vi))
    for key, value in zip(["embedding", "probabilities", "scores", "values"], native):
        if not torch.allclose(value, own[key], rtol=1e-5, atol=1e-5):
            raise ValueError(f"Native RMSNorm/QKV/M-RoPE response mismatch: {key}")
    return True


class TokenMetricAblation:
    name = "token_metric_ablation"
    requires_teacher_forcing = True


class Session(EvidenceSession):
    def __post_init__(self):
        from safetensors.torch import load_file
        if self.runtime.revision != REVISION or Path(self.runtime.model_path_or_id).resolve() != Path(CHECKPOINT).resolve():
            raise ValueError("Metric experiment requires the frozen original 3B checkpoint")
        self.root, self.stage = Path(self.config["artifact_root"]), self.config["stage"]
        self.protocol = json.loads((self.root/"protocol.json").read_text())
        self.source = Path(self.protocol["source_geometry_run"])
        self.code_hash = implementation_hash()
        if self.protocol["implementation_hash"] != self.code_hash:
            raise ValueError("Frozen metric implementation changed")
        self.calibration = file_sha(self.root/"calibration.safetensors")
        self.diagonal = load_file(str(self.root/"calibration.safetensors"), device=str(self.runtime.device))["diagonal"]
        self.config_hash = digest(json.dumps(self.config, sort_keys=True)+self.code_hash+self.calibration+
            file_sha(self.root/"cohort.json"))
        self.completed = []
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _process(self, sample, entry, folder):
        import torch
        started = time.perf_counter()
        if media_fingerprint(sample["media"]) != entry["image_fingerprint"]:
            raise ValueError("Same-500 image fingerprint mismatch")
        handles = self.backbone.model_adapter.structural_handles(self.runtime.model)
        bundle = {"facade": self.backbone, "runtime": self.runtime, "model": self.runtime.model,
            "processor": self.runtime.processor, "llm": handles["llm"], "vision": handles["vision_encoder"], "device": self.runtime.device}
        teacher, _ = prepare_teacher_forced_analysis_forward_state(backbone=self.backbone, runtime=self.runtime,
            sample=sample, probe_name=self.plugin.name)
        state = prepare_state(bundle, sample["processor_outputs"], teacher.native_model_inputs)
        source_file = self.source/"samples"/self.stage/str(sample["sample_id"]).replace(":", "_")/"conditions.jsonl"
        source_rows = {r["variant"]:r for r in map(json.loads, source_file.read_text().splitlines())}
        old = {name: source_rows[name] for name in ["dense", "cosine_farthest"]}
        reference = geometry.select_and_merge(state["features"], height=state["height"], width=state["width"], method="cosine_farthest")
        if reference["members"] != old["cosine_farthest"]["compression_trace"]["members"]:
            raise ValueError("Current cosine partition differs from the previous experiment")
        legacy_config = {"artifact_root": str(self.source), "cohort_path": str(self.source/"cohort.json"), "stage": self.stage}
        legacy_config_hash = digest(json.dumps(legacy_config, sort_keys=True)+legacy_hash()+file_sha(self.source/"cohort.json"))
        ledger = folder/"conditions.jsonl"
        saved = records(ledger, self.config_hash)
        max_new = 16 if "chartqa" in sample["task_name"] else 32
        for name, row in old.items():
            key = (sample["sample_id"], name)
            if row["config_hash"] != legacy_config_hash or row["image_fingerprint"] != entry["image_fingerprint"] or row["revision"] != REVISION:
                raise ValueError("Legacy baseline provenance mismatch")
            if row["gold_answer_in_decode_context"] or row["source_visual_tokens"] != len(state["features"]):
                raise ValueError("Legacy baseline leakage or source token mismatch")
            if key not in saved:
                copied = {**row, "config_hash": self.config_hash, "reuse_provenance": {"source": str(source_file), "sha256": file_sha(source_file)},
                    "timing_comparable_to_new_run": False}
                if self.stage == "smoke":
                    compact = compact_anchor(state, reference) if name != "dense" else None
                    current = timed_evaluate(bundle, state, compact, max_new=max_new, native_check=name == "dense")
                    if current["generated_token_ids"] != row["generated_token_ids"] or abs(current["answer_nll"]-row["answer_nll"]) > 1e-5:
                        raise ValueError("Fresh smoke differs from previous dense/cosine baseline")
                    copied["fresh_baseline_identity_passed"] = True
                append_json(ledger, copied); saved[key] = copied
        question = str(sample["raw_doc"]["question"])
        question = question.capitalize() if "textvqa" in sample["task_name"] else question
        suffix = state["visual"][-1]+1
        indices = [i+suffix for i in question_positions(bundle["processor"].tokenizer,
            state["prompt_inputs"]["input_ids"][:, suffix:], question)]
        torch.cuda.synchronize(bundle["device"]); begin = time.perf_counter()
        responses = geometry.extract_responses(bundle, state, indices)
        torch.cuda.synchronize(bundle["device"]); response_ms = 1000*(time.perf_counter()-begin)
        if self.stage == "smoke":
            native_response_gate(bundle, state, indices, responses)
        methods = list(geometry.NEW_METHODS)
        random.Random(entry["image_fingerprint"]+":metric-order-v2").shuffle(methods)
        for name in methods:
            key = (sample["sample_id"], name)
            if key in saved:
                continue
            output = geometry.select_and_merge(state["features"], height=state["height"], width=state["width"], method=name,
                diagonal=self.diagonal, responses=responses)
            compact = compact_anchor(state, output)
            result = timed_evaluate(bundle, state, compact, max_new=max_new, native_check=False)
            if result["retained_visual_tokens"] != 128 or result["first_appended_token_position"] != int(compact["positions"][:, :, :compact["prefix"]].max())+1:
                raise ValueError("Fixed budget / M-RoPE cached decode gate failed")
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
            response_method = name.startswith("query_response")
            row = {**result, "variant": name, "sample_id": sample["sample_id"], "task": sample["task_name"], "stage": self.stage,
                "status": "completed", "config_hash": self.config_hash, "image_fingerprint": entry["image_fingerprint"],
                "checkpoint": CHECKPOINT, "revision": REVISION, "prediction": prediction, "uses_mlp": False,
                "primary_task_score": primary_task_score(sample["task_name"], metrics), "native_task_metrics": metrics,
                "strict_score": strict, "repetition_penalty": 1., "timing_comparable_to_new_run": True,
                "selection_ms": output["timing"]["selection_ms"],
                "aggregation_and_statistics_ms": output["timing"]["aggregation_and_statistics_ms"],
                "response_extraction_ms": response_ms if response_method else 0.,
                "fresh_cosine_clustering_timing": reference["timing"],
                "calibration_sha256": self.calibration if name == "diag_mahalanobis" else None,
                "comparison_to_cosine": compare(reference, output), "geometry": output["geometry"],
                "response_definition": {"query_indices": indices, "layer": 0, "native_mrope": True, "gold_answer_used": False,
                    "qk_mean_squared": float(responses["scores"].square().mean()), "value_mean_squared": float(responses["values"].square().mean()),
                    "normalization": "softmax over flattened question-position x head responses, not softmax over visual keys"} if response_method else None,
                "compression_trace": {k:output[k] for k in ["anchors", "indices", "members", "centroids", "cluster_sizes", "partition_hash", "position_policy"]}}
            append_json(ledger, row); saved[key] = row
        if {key[1] for key in saved} != set(geometry.METHODS) or len(saved) != len(geometry.METHODS):
            raise ValueError("Incomplete metric condition set")
        return {"sample_id": sample["sample_id"], "task": sample["task_name"], "stage": self.stage,
            "image_fingerprint": entry["image_fingerprint"], "config_hash": self.config_hash, "implementation_hash": self.code_hash,
            "condition_path": str(ledger), "condition_count": len(saved), "elapsed_s": time.perf_counter()-started,
            "fresh_baseline_and_native_response_gates": self.stage == "smoke"}
