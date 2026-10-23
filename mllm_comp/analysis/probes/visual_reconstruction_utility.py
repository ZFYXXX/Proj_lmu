"""Test whether reconstruction error predicts answer-relevant missing evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.sample import prepare_teacher_forced_analysis_forward_state
from mllm_comp.analysis.common.spectral_residual import spectral_plan, summary_plan, tile_plan
from mllm_comp.analysis.common.visual_token_generation import greedy_decode_embeddings, primary_task_score
from mllm_comp.analysis.common.visual_token_phenomena import answer_predictor_positions, compact_answer_logit_metrics, media_fingerprint
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.analysis.probes.visual_evidence_interaction import _Session as EvidenceSession, cached_decode, question_positions
from mllm_comp.analysis.probes.visual_token_phenomena import _compact_layer_state, _run_decoder_suffix
from mllm_comp.backbones.qwen25_vl.implementation import _move_value
from mllm_comp.core.serialization import atomic_write_json, to_jsonable


def implementation_hash():
    root = Path(__file__).parents[2]
    files = [Path(__file__), root / "analysis/common/spectral_residual.py",
             root / "analysis/common/cache_parity.py",
             root / "analysis/probes/visual_evidence_interaction.py",
             root / "analysis/probes/visual_token_phenomena.py",
             root / "analysis/common/evidence_interaction.py", root / "analysis/common/evidence_cohort.py",
             root / "analysis/common/visual_token_generation.py", root / "analysis/common/visual_token_phenomena.py",
             root / "analysis/common/sample.py", root / "integrations/lmms_eval/adapter.py",
             root / "compressors/image/fixed_visual_merge.py", root / "compressors/image/prumerge.py",
             root / "analysis/common/visual_merge_attribution.py", root / "backbones/qwen25_vl/provider_bridge.py"]
    checksum = hashlib.sha256()
    for path in files:
        checksum.update(str(path.relative_to(root)).encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


class VisualReconstructionUtility:
    name = "visual_reconstruction_utility"
    version = "1.0.0"
    requires_teacher_forcing = True

    def validate_config(self, config):
        if config.get("stage") not in {"smoke", "development", "confirmation"}:
            raise ValueError("Explicit reconstruction experiment stage required.")
        if config.get("budget", 128) != 128 or config.get("base_tokens", 96) != 96:
            raise ValueError("Frozen budget split is 96+32=128.")
        if config.get("save_raw", False):
            raise ValueError("Raw hidden state saving is disabled.")
        for key in ["cohort_path", "artifact_root"]:
            if not config.get(key):
                raise ValueError(f"Missing {key}.")

    def open(self, *, backbone, runtime, config, phase, output_dir):
        self.validate_config(config)
        return Session(self, backbone, runtime, config, phase, output_dir)


class Session(EvidenceSession):
    def __post_init__(self):
        if self.backbone.capabilities().name != "qwen25_vl" or self.runtime.revision != "1b5a7674d40a0ac9acef541a46fed12d710bd36b":
            raise ValueError("Frozen Qwen2.5-VL checkpoint/revision required.")
        self.root = Path(self.config["artifact_root"])
        self.stage = self.config["stage"]
        self.code_hash = implementation_hash()
        self.cohort_hash = hashlib.sha256(Path(self.config["cohort_path"]).read_bytes()).hexdigest()
        self.config_hash = digest(json.dumps(self.config, sort_keys=True) + self.cohort_hash + self.code_hash)
        if self.stage == "confirmation":
            freeze = json.loads((self.root / "frozen.json").read_text())
            if freeze["implementation_hash"] != self.code_hash or freeze["cohort_hash"] != self.cohort_hash:
                raise ValueError("Frozen reconstruction experiment changed.")
        self.completed = []
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _process(self, sample, entry, folder):
        import torch
        from transformers import GenerationConfig
        from mllm_comp.compressors.image.fixed_visual_merge import fixed_visual_merge, FixedVisualMergeSpec
        from mllm_comp.compressors.image.prumerge import prumerge_image_tokens, PruMergeSpec
        from mllm_comp.backbones.qwen25_vl.provider_bridge import _merged_vision_metrics, _apply_rotary_pos_emb_vision_4573

        started = time.perf_counter()
        fingerprint = media_fingerprint(sample["media"])
        if fingerprint != entry["image_fingerprint"] or not callable(sample.get("score_prediction")):
            raise ValueError("Frozen image fingerprint or native task scorer mismatch.")
        model, processor = self.runtime.model, self.runtime.processor
        handles = self.backbone.model_adapter.structural_handles(model)
        llm, vision = handles["llm"], handles["vision_encoder"]
        if llm.config._attn_implementation != "eager" or self.runtime.dtype != torch.bfloat16:
            raise ValueError("BF16/eager runtime required.")
        embedding, head = model.get_input_embeddings(), model.lm_head
        inputs = {k: _move_value(v, device=self.runtime.device, dtype=self.runtime.dtype)
                  for k, v in sample["processor_outputs"].items()}
        grid = inputs["image_grid_thw"]
        if len(grid) != 1 or int(grid[0, 0]) != 1 or inputs["input_ids"].shape[0] != 1 or not bool((inputs["attention_mask"] == 1).all()):
            raise ValueError("Unpadded single-image batch size one required.")
        height, width = int(grid[0, 1]) // vision.spatial_merge_size, int(grid[0, 2]) // vision.spatial_merge_size
        visual = torch.nonzero(inputs["input_ids"][0] == model.config.image_token_id).flatten().tolist()
        if len(visual) != height * width or len(visual) < 128:
            raise ValueError("Visual grid/placeholder count mismatch or insufficient source tokens.")
        captured = {}

        def capture_input(module, args):
            captured["hidden"] = args[0].detach()

        def capture_vision(module, args, kwargs):
            captured["vision_hidden"] = (args[0] if args else kwargs["hidden_states"]).detach()
            captured["vision_positions"] = kwargs["position_embeddings"]

        with HookSession() as hooks:
            hooks.register_forward_pre_hook(llm.layers[0], capture_input)
            hooks.register_forward_pre_hook(vision.blocks[-1].attn, capture_vision, with_kwargs=True)
            dense = model(**inputs, output_attentions=True, output_hidden_states=False, use_cache=False, return_dict=True)
        prompt_hidden = captured.pop("hidden")
        question = str(sample["raw_doc"]["question"])
        question_text = question.capitalize() if "textvqa" in sample["task_name"] else question
        query_positions = question_positions(processor.tokenizer, inputs["input_ids"], question_text)
        attention = dense.attentions[0][0][:, query_positions, :][:, :, visual].float().mean((0, 1))
        del dense
        features = prompt_hidden[0, visual]
        attention_block = vision.blocks[-1].attn
        raw = attention_block.qkv(captured["vision_hidden"]).reshape(-1, 3, attention_block.num_heads, attention_block.head_dim)
        q, k = _apply_rotary_pos_emb_vision_4573(raw[:, 0], raw[:, 1], *captured["vision_positions"])
        window, _ = vision.get_window_index(grid)
        vision_attention, vision_keys = _merged_vision_metrics(q, k, patch_counts=[int(raw.shape[0])], feature_counts=[len(visual)],
            window_index=window, spatial_merge_unit=vision.spatial_merge_size ** 2, scaling=float(attention_block.scaling))
        del raw, q, k
        captured.clear()
        # All plans are constructed from the gold-free prompt before teacher forcing.
        plan = summary_plan(features, height, width, attention, fingerprint)
        spectral = spectral_plan(features, height, width, fingerprint)
        tiles = tile_plan(height, width)
        atomic_write_json(folder / "selection_plan.json", {
            "base_indices": plan["base_indices"], "base_members": plan["members"],
            "selections": plan["selections"], "plan_hash": plan["plan_hash"], "tiles": tiles,
            "local_residual_scores": plan["residual_scores"].tolist(), "pca_scores": plan["pca_scores"].tolist(),
            "prompt_attention_scores": attention.tolist(), "feature_norm_scores": features.float().norm(dim=-1).tolist(),
            "selection_uses_answer": False, "summary_position_policy": "native_representative",
        }, ensure_ascii=False)
        teacher_state, objective = prepare_teacher_forced_analysis_forward_state(
            backbone=self.backbone, runtime=self.runtime, sample=sample, probe_name=self.plugin.name)
        teacher_inputs = {k: _move_value(v, device=self.runtime.device, dtype=self.runtime.dtype)
                          for k, v in teacher_state.native_model_inputs.items()}
        prefix = int(objective["prefix_token_counts"][0])
        if prefix != prompt_hidden.shape[1] or not torch.equal(teacher_inputs["input_ids"][:, :prefix], inputs["input_ids"]):
            raise ValueError("Teacher prompt is not identical to the gold-free generation prompt.")
        with HookSession() as hooks:
            hooks.register_forward_pre_hook(llm.layers[0], capture_input)
            teacher = model(**teacher_inputs)
        teacher_hidden = captured.pop("hidden")
        if not torch.equal(teacher_hidden[:, :prefix], prompt_hidden):
            raise ValueError("Teacher answer contaminated prompt input states.")
        predictors, targets = answer_predictor_positions(teacher_inputs["labels"])
        positions, _ = model.model.get_rope_index(teacher_inputs["input_ids"], grid, attention_mask=teacher_inputs["attention_mask"])
        if positions.shape[0] == 4:
            positions = positions[1:]
        source_positions = positions[:, 0]
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        max_new = 16 if "chartqa" in sample["task_name"] else 32
        common = {"sample_id": sample["sample_id"], "task": sample["task_name"], "stage": self.stage,
                  "image_fingerprint": fingerprint, "split": entry["split"], "grid": [height, width],
                  "source_visual_tokens": len(visual), "checkpoint": str(self.runtime.model_path_or_id),
                  "revision": self.runtime.revision, "question": question, "target_text": sample["target_text"],
                  "config_hash": self.config_hash, "selection_uses_answer": False}
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
                    if row["config_hash"] != self.config_hash or row["request_id"] in recorded:
                        raise ValueError("Mismatched/duplicate resume ledger condition.")
                    recorded[row["request_id"]] = row

        def evaluate(name, indices, *, values=None, centroid_members=None, extra=None):
            if name in recorded:
                return recorded[name]
            before = time.perf_counter()
            indices = list(indices)
            if indices != sorted(set(indices)):
                raise ValueError("Retained indices must be unique and sorted in source order.")
            compact = _compact_layer_state(hidden=teacher_hidden, position_ids=source_positions,
                visual_positions=visual, members=tuple((i,) for i in indices), operator="retain", token_scores=None, budget=len(indices))
            hidden, ids = compact["hidden"], compact["position_ids"]
            mapping = compact["source_to_compact"]
            gather = torch.tensor(sorted(mapping, key=mapping.get), device=hidden.device)
            tokens = teacher_inputs["input_ids"].index_select(1, gather)
            labels = teacher_inputs["labels"].index_select(1, gather)
            mask = teacher_inputs["attention_mask"].index_select(1, gather)
            if tokens.shape != labels.shape or labels.shape != mask.shape or hidden.shape[1] != tokens.shape[1] or not bool((mask == 1).all()) or int((tokens == model.config.image_token_id).sum()) != len(indices):
                raise ValueError("Compressed placeholder/mask/label alignment failed.")
            cp, ct = answer_predictor_positions(labels)
            if ct != targets or cp != [mapping[p] for p in predictors]:
                raise ValueError("Compaction changed answer labels or predictor positions.")
            if values is not None:
                if values.shape != (len(indices), features.shape[1]):
                    raise ValueError("Replacement value shape mismatch.")
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("Nonfinite intervention feature.")
                hidden = hidden.clone()
                for slot, index in enumerate(indices):
                    mapped = mapping[visual[index]]
                    hidden[0, mapped] = values[slot].to(hidden.dtype)
                    if centroid_members is not None:
                        original = [visual[i] for i in centroid_members[slot]]
                        ids[:, 0, mapped] = source_positions[:, original].float().mean(1).round().long()
            for original, mapped in mapping.items():
                if original not in visual and not torch.equal(ids[:, 0, mapped], source_positions[:, original]):
                    raise ValueError("Text M-RoPE positions changed.")
            logits = _run_decoder_suffix(llm=llm, lm_head=head, hidden=hidden, position_ids=ids, start_layer=0)
            metrics = compact_answer_logit_metrics(teacher.logits, logits, source_predictor_positions=predictors,
                                                  compact_predictor_positions=cp, targets=targets)
            if any(isinstance(v, float) and not math.isfinite(v) for v in metrics.values()):
                raise ValueError("Nonfinite answer metric.")
            del logits
            compressed_prefix = prefix - len(visual) + len(indices)
            ph, pp = hidden[:, :compressed_prefix], ids[:, :, :compressed_prefix]
            generated = cached_decode(llm=llm, head=head, embedding=embedding, hidden=ph, positions=pp, eos_ids=eos, max_new=max_new)
            if generated["first_appended_token_position"] != int(pp.max()) + 1:
                raise ValueError("Compressed decode offset mismatch.")
            if self.stage == "smoke" and name in {"dense", "hybrid/high_residual"}:
                plain = greedy_decode_embeddings(prompt_hidden=ph, prompt_position_ids=pp, token_embedding=embedding,
                    forward_logits=lambda h, p: _run_decoder_suffix(llm=llm, lm_head=head, hidden=h, position_ids=p, start_layer=0, last_token_only=True),
                    max_new_tokens=max_new, eos_token_ids=eos)
                if plain["generated_token_ids"] != generated["generated_token_ids"]:
                    from mllm_comp.analysis.common.cache_parity import verify_bf16_boundary
                    reference = verify_bf16_boundary(llm=llm, head=head, embedding=embedding, hidden=ph,
                        positions=pp, eos_ids=eos, max_new=max_new)
                    atomic_write_json(folder / (name.replace("/", "_") + "_cache_parity.json"),
                        {"condition": name, "cached": generated, "cache_free": plain, "reference": reference}, ensure_ascii=False)
                    if not reference["passed"]:
                        raise ValueError(f"Cache lifecycle failed FP32 or BF16 same-prefix parity: {name}.")
                    generated["cache_parity_fp32_reference_passed"] = True
                    generated["cache_free_identity_passed"] = False
                else:
                    generated["cache_free_identity_passed"] = True
            if self.stage == "smoke" and name == "dense":
                native = model.generate(**inputs, generation_config=GenerationConfig(max_new_tokens=max_new, do_sample=False,
                    use_cache=True, repetition_penalty=1., eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id))[0, prefix:].tolist()
                if native != generated["generated_token_ids"] or abs(metrics["answer_nll_delta"]) > .01 or metrics["teacher_to_candidate_answer_kl"] > .001:
                    raise ValueError("Native generation / dense logit identity failed.")
                generated["native_generation_identity_passed"] = True
            text = processor.tokenizer.decode(generated["generated_token_ids"], skip_special_tokens=True).strip()
            stops = sample.get("generation_stops", [])
            stops = [stops] if isinstance(stops, str) else stops
            for stop in stops:
                if stop:
                    text = text.split(stop, 1)[0].strip()
            native_metrics = sample["score_prediction"](text)
            strict = None
            if "chartqa" in sample["task_name"]:
                answers = sample["raw_doc"]["answer"]
                answers = answers if isinstance(answers, list) else [answers]
                strict = float(any(text.strip().casefold() == str(a).strip().casefold() for a in answers))
            row = {**common, "request_id": name, "status": "completed", "retained_indices": indices,
                   "retained_visual_tokens": len(indices), "prediction": text, "native_task_metrics": native_metrics,
                   "primary_task_score": primary_task_score(sample["task_name"], native_metrics), "strict_answer_match": strict,
                   **metrics, **generated, "elapsed_s": time.perf_counter() - before,
                   "placeholder_mask_labels_alignment_passed": True, "text_positions_preserved": True,
                   "visual_position_policy": "centroid" if centroid_members is not None else "native_representative",
                   **(extra or {})}
            encoded = json.dumps(to_jsonable(row), ensure_ascii=False, allow_nan=False)
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            recorded[name] = json.loads(encoded)
            return recorded[name]

        all_indices = list(range(len(visual)))
        evaluate("dense", all_indices)
        refs = fixed_visual_merge(features, grid_height=height, grid_width=width, spec=FixedVisualMergeSpec(target_count=128))
        evaluate("reference/feature_mean_native128", refs.indices.tolist(), values=refs.tokens)
        evaluate("reference/feature_representative128", refs.indices.tolist())
        evaluate("reference/feature_mean_centroid128", refs.indices.tolist(), values=refs.tokens, centroid_members=refs.metadata["cluster_members"])
        pru = prumerge_image_tokens(features, vision_attention[0], vision_keys[0],
            spec=PruMergeSpec(variant="plus", adaptive=False, target_count=128, cluster_size=32))
        evaluate("reference/prumerge_plus128", pru.indices.tolist(), values=pru.tokens, extra={"compression_trace": dict(pru.metadata)})
        evaluate("summary96", plan["base_indices"], values=plan["base_values"])
        for name, selected in plan["selections"].items():
            lookup = {i: plan["base_values"][j] for j, i in enumerate(plan["base_indices"])}
            lookup.update({i: features[i] for i in selected})
            indices = sorted(lookup)
            if len(indices) != 128:
                raise ValueError("Hybrid failed strict K128 budget.")
            values = torch.stack([lookup[i] for i in indices])
            evaluate(f"hybrid/{name}", indices, values=values,
                     extra={"base_tokens": 96, "original_detail_tokens": 32, "detail_indices": sorted(selected), "plan_hash": plan["plan_hash"]})
        for state in spectral:
            realized = (features.float() - state["values"].to(features.dtype).float()).square().sum()
            intended = state["trace"]["perturbation_energy"]
            if not math.isclose(float(realized), intended, rel_tol=.03, abs_tol=1e-5):
                raise ValueError("BF16 realized frequency dose differs by more than 3% from matched FP32 dose.")
            trace = {**state["trace"], "bf16_realized_perturbation_energy": float(realized),
                     "bf16_energy_relative_error": float(realized) / intended - 1}
            evaluate(state["name"], all_indices, values=state["values"], extra={"frequency_trace": trace, "length_preserved": True})
        for tile_id, members in enumerate(tiles):
            values = features.clone()
            values[members] = features[members].float().mean(0).to(features.dtype)
            evaluate(f"tile/{tile_id}", all_indices, values=values,
                     extra={"tile_id": tile_id, "tile_members": members, "length_preserved": True,
                            "intervention": "within_tile_mean_replacement", "tile_residual_score": float(plan["residual_scores"][members].mean()),
                            "tile_pca_score": float(plan["pca_scores"][members].mean()), "tile_attention_score": float(attention[members].mean()),
                            "tile_feature_norm": float(features[members].float().norm(dim=-1).mean())})
        expected = 6 + len(plan["selections"]) + len(spectral) + len(tiles)
        if len(recorded) != expected:
            raise ValueError(f"Incomplete or extraneous condition ledger: {len(recorded)} != {expected}.")
        torch.cuda.synchronize(features.device) if features.is_cuda else None
        return {**common, "implementation_hash": self.code_hash, "condition_count": expected,
                "condition_path": str(ledger), "plan_hash": plan["plan_hash"],
                "native_generation_identity_passed": recorded["dense"].get("native_generation_identity_passed"),
                "cache_free_identity_passed": recorded["hybrid/high_residual"].get("cache_free_identity_passed"),
                "cache_parity_fp32_reference_passed": recorded["hybrid/high_residual"].get("cache_parity_fp32_reference_passed", False),
                "elapsed_s": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(features.device) if features.is_cuda else 0}
