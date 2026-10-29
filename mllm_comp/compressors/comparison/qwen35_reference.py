"""Qwen-specific external baselines with native Qwen3.5 layout/cache handling.

This separate binding keeps the trained-head implementation hash unchanged.
Selection cores follow their released Qwen paths; resolution, fixed budget and
the Qwen3.5 runtime remain explicit controlled adaptations.
"""
from contextlib import nullcontext
from contextvars import ContextVar
import types

import torch

from mllm_comp.analysis.common.learned_topk_experiment import digest_json
from mllm_comp.analysis.common.qwen35_transfer import compact_prompt
from mllm_comp.compressors.image.bifocal import (
    BifocalSpec, bifocal_extract_query, bifocal_image_tokens, bifocal_word_embeddings,
)
from mllm_comp.compressors.image.mmtok import MMTokSpec, mmtok_image_tokens
from .eadp import normalize_eadp_config, select_eadp_indices
from .prunesid import select_prunesid_indices


METHODS = ("eadp", "prunesid", "mmtok", "bifocal_stage1")


def user_instruction_text(model, tokenizer, inputs):
    """Reconstruct only the last complete user text, using this model's tokenizer."""
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("External Qwen selectors require batch size one")
    values = ids[0]
    mask = inputs.get("attention_mask")
    if mask is not None:
        if mask.shape != ids.shape:
            raise ValueError("Question mask does not match input IDs")
        values = values[mask[0].bool()]
    labels = inputs.get("labels")
    if labels is not None:
        if mask is not None and not bool(mask.all()):
            raise ValueError("Teacher question extraction requires an unpadded prompt")
        boundary = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if len(boundary):
            values = values[:int(boundary[0])]
    values = values.tolist()
    prefix = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    starts = [i + len(prefix) for i in range(len(values) - len(prefix) + 1)
              if values[i:i + len(prefix)] == prefix]
    if not starts:
        raise ValueError("Missing complete Qwen user message")
    values = values[starts[-1]:]
    if end not in values:
        raise ValueError("Unterminated Qwen user message")
    values = values[:values.index(end)]
    sentinels = {getattr(model.config, name, None) for name in
                 ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")}
    values = [i for i in values if i not in sentinels]
    text = tokenizer.decode(values, skip_special_tokens=True).strip()
    if not text:
        raise ValueError("Empty user instruction after removing vision sentinels")
    return text


@torch.no_grad()
def reference_question_embeddings(model, tokenizer, inputs, method):
    """Each release has its own text recipe; a shared raw span is not equivalent."""
    if method not in METHODS:
        raise ValueError("Unknown external selector")
    if method == "prunesid":
        return None, {"reads_question": False, "question_recipe": "image_only"}
    text = user_instruction_text(model, tokenizer, inputs)
    if method == "bifocal_stage1":
        query = bifocal_extract_query(text)
        embeddings = bifocal_word_embeddings(tokenizer, model.get_input_embeddings(), model.device, query, mode=0)
        recipe = "bifocal_query_without_task_suffix_lowercase_words_mean_subwords"
        if embeddings is None or not len(embeddings):
            raise ValueError("Bifocal has no query word embeddings")
        encoded_text = query
    else:
        encoded_text = f"Question: {text}" if method == "mmtok" else text
        kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        if method == "mmtok":
            encoded = tokenizer(encoded_text.split(), is_split_into_words=True, **kwargs)["input_ids"]
            recipe = "mmtok_qwen_Question_prefix_split_and_retokenize"
            if encoded.shape[0] != 1:
                raise ValueError("MMTok pretokenized question unexpectedly formed multiple rows")
            if encoded.shape[1] > 1 and int(encoded[0, 0]) == tokenizer.bos_token_id:
                encoded = encoded[:, 1:]
        else:
            encoded = tokenizer(encoded_text, add_special_tokens=True, **kwargs)["input_ids"]
            recipe = "eadp_qwen_stripped_user_text_retokenized"
        if encoded.ndim != 2 or encoded.shape[0] != 1 or encoded.shape[1] == 0:
            raise ValueError("Invalid external question tokenization")
        embeddings = model.get_input_embeddings()(encoded.to(model.device))[0]
    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite external question embeddings")
    return embeddings.detach(), {
        "reads_question": True, "question_recipe": recipe,
        "question_text_sha256": digest_json(encoded_text), "question_vectors": len(embeddings),
        "question_source": "last_complete_user_message_no_answer",
    }


@torch.no_grad()
def reference_selection(features, *, method, question, grid, raw_features=None, attention=None, target_count=128):
    """Fixed K128 evaluation adapter around the released selection cores."""
    h, w = grid
    if h * w != len(features):
        raise ValueError("Post-merger rectangular grid does not match visual population")
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
        raise ValueError("Invalid post-merger token budget")
    k = min(target_count, len(features))
    if method == "eadp":
        selections, stats = select_eadp_indices(features[None], question, patches_per_side=w, grid_height=h,
            cfg=normalize_eadp_config({"tokens_per_crop": k, "alpha": .5, "beta": 2.}))
        indices = selections[0]
        output = {"indices": indices, "tokens": features[indices]}
    elif method == "prunesid":
        selections, stats = select_prunesid_indices(features[None], cfg={"tokens_per_crop": k})
        indices = selections[0]
        output = {"indices": indices, "tokens": features[indices], "official_order": True}
    elif method == "mmtok":
        if raw_features is None or len(raw_features) != len(features):
            raise ValueError("MMTok requires aligned native pre-merger feature groups")
        result = mmtok_image_tokens(features, raw_features, question,
            spec=MMTokSpec(target_count=k, text_temperature=.01, vision_temperature=.2, alpha=.5))
        output = {"indices": result.indices, "tokens": result.tokens}
        stats = dict(result.metadata)
    elif method == "bifocal_stage1":
        if attention is None or len(attention) != len(features):
            raise ValueError("Bifocal requires native non-CLS received-attention statistics")
        result = bifocal_image_tokens(features, {"mass": attention}, grid_height=h, grid_width=w,
            word_embeddings=question, spec=BifocalSpec(target_count=k))
        output = {"indices": result.indices, "tokens": result.tokens}
        stats = dict(result.metadata)
    else:
        raise ValueError("Unknown external selector")
    stats.update(trained=False, controlled_adaptation=True, official_paper_reproduction=False,
                 adapter_version="qwen_native_reference_v2", fixed_budget=k)
    return output, stats


class Qwen35ReferenceProjector:
    def __init__(self, model, config, *, runtime=None, tokenizer=None):
        if set(config) - {"method", "target_count"} or config.get("method") not in METHODS:
            raise ValueError("Unknown external Qwen3.5 configuration")
        target = config.get("target_count", 128)
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            raise ValueError("Invalid post-merger token budget")
        if model.config.model_type != "qwen3_5":
            raise ValueError("Reference binding requires native Qwen3.5")
        self.model, self.method = model, config["method"]
        self.target = target
        self.tokenizer = tokenizer or getattr(getattr(runtime, "processor", None), "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("Native tokenizer is required")
        self.record = ContextVar(f"qwen35_reference_{id(self)}", default=None)
        self.installed = False

    def install(self):
        if self.installed:
            return self
        self.original = self.model.generate
        self.had_generate = "generate" in self.model.__dict__
        owner = self

        def generate(_model, *args, **kwargs):
            if args or kwargs.get("pixel_values") is None:
                return owner.original(*args, **kwargs)
            if kwargs.get("num_beams", 1) != 1 or kwargs.get("do_sample", False) or kwargs.get("labels") is not None:
                raise ValueError("Reference evaluation requires answer-free greedy generation")
            original_ids = kwargs["input_ids"]
            grid = kwargs["image_grid_thw"]
            if grid.shape != (1, 3) or int(grid[0, 0]) != 1:
                raise ValueError("Reference comparison supports exactly one still image")
            previous_delta = owner.model.model.rope_deltas
            owner.record.set(None)
            hooks = []
            try:
                from mllm_comp.backbones.qwen3_5.vision_attention import (
                    capture_qwen3_5_premerger_features, capture_qwen3_5_final_vision_attention,
                    pool_premerger_features_to_merger_tokens, pool_patch_statistics_to_merger_tokens,
                )
                visual = owner.model.model.visual
                merge = owner.model.config.vision_config.spatial_merge_size
                context = (capture_qwen3_5_premerger_features(visual) if owner.method == "mmtok" else
                           capture_qwen3_5_final_vision_attention(visual) if owner.method == "bifocal_stage1" else nullcontext(None))
                with torch.no_grad(), context as captured:
                    features, = owner.model.get_image_features(kwargs["pixel_values"], grid, return_dict=True).pooler_output
                raw, attention = None, None
                if owner.method == "mmtok":
                    if len(captured) != 1:
                        raise ValueError("Missing native pre-merger states")
                    raw, = pool_premerger_features_to_merger_tokens(captured[0], grid, spatial_merge_size=merge)
                if owner.method == "bifocal_stage1":
                    if len(captured) != 1:
                        raise ValueError("Missing native final vision attention")
                    attention, _ = pool_patch_statistics_to_merger_tokens(captured[0],
                        merged_token_count=len(features), spatial_merge_size=merge)
                q, audit = reference_question_embeddings(owner.model, owner.tokenizer, kwargs, owner.method)
                output, stats = reference_selection(features, method=owner.method, question=q,
                    grid=tuple(int(v) // merge for v in grid[0, 1:]), raw_features=raw, attention=attention,
                    target_count=owner.target)
                prepared, record, _ = compact_prompt(owner.model, kwargs, features, output)
                layers = owner.model.model.language_model.layers
                record.update(stats, **audit, method=owner.method, new_base_training=False,
                    observed_visual_tokens_per_decoder_layer=[None] * len(layers),
                    vision_feature_space="native_post_merger_llm_space",
                    vision_attention_block=len(visual.blocks)-1 if owner.method == "bifocal_stage1" else None)
                owner.record.set(record)
                prefill = True

                def verify_input(_module, _args, call):
                    nonlocal prefill
                    if prefill:
                        prefill = False
                        if call["inputs_embeds"].shape[1] != record["output_prompt_tokens"]:
                            raise ValueError("Prefill sequence was not physically compacted")
                    elif record["decode_position_verified"] is None:
                        seen = call["position_ids"][-3:, :, 0]
                        if not bool((seen == record["expected_first_decode_position"]).all()):
                            raise ValueError("Native hybrid-cache decode M-RoPE mismatch")
                        record.update(decode_position_verified=True, first_decode_position=seen.flatten().tolist())

                hooks.append(owner.model.model.language_model.register_forward_pre_hook(verify_input, with_kwargs=True))
                for i, layer in enumerate(layers):
                    def observe(_layer, args, call, index=i):
                        hidden = call.get("hidden_states", args[0] if args else None)
                        if hidden is not None and hidden.shape[1] == record["output_prompt_tokens"]:
                            record["observed_visual_tokens_per_decoder_layer"][index] = record["result_count"]
                    hooks.append(layer.register_forward_pre_hook(observe, with_kwargs=True))
                raw_output = owner.original(**prepared)
                sequence = getattr(raw_output, "sequences", raw_output)
                restored = torch.cat((original_ids, sequence[:, prepared["input_ids"].shape[1]:]), dim=-1)
                if record["observed_visual_tokens_per_decoder_layer"] != [record["result_count"]] * len(layers):
                    raise ValueError("Incomplete hybrid-decoder length verification")
                if hasattr(raw_output, "sequences"):
                    raw_output.sequences = restored
                    return raw_output
                return restored
            finally:
                for hook in reversed(hooks):
                    hook.remove()
                owner.model.model.rope_deltas = previous_delta

        self.model.generate = types.MethodType(generate, self.model)
        self.installed = True
        return self

    def uninstall(self):
        if self.installed:
            if self.had_generate:
                self.model.generate = self.original
            else:
                delattr(self.model, "generate")
            self.installed = False

    def clear_diagnostic_records(self):
        self.record.set(None)

    def runtime_metadata(self):
        record = self.record.get()
        return {"backbone": "qwen3_5", "method": self.method, "installed": self.installed,
                "selection": record, "layout_delta_records": [record] if record else [],
                "budget_policy": {"fixed_tokens": self.target}, "hybrid_attention_native_cache": True,
                "adapter_version": "qwen_native_reference_v2"}


def create_executor(*, model, stage, runtime=None, **kwargs):
    return Qwen35ReferenceProjector(model, stage.config, runtime=runtime)


def register_contracts(registry):
    from mllm_comp.compressors.contracts import AlgorithmDescriptor, ModalityVariantDescriptor, BackbonePlacementBinding, BindingTarget
    from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation, Modality, ExecutionPhase
    from mllm_comp.core.support import SupportUnit, SupportUnitId
    name = "qwen35_reference_projector"
    registry.register_algorithm(AlgorithmDescriptor(name=name, version="v2.0.0", method=CompressionMethod.TOKEN_COMPRESSION,
        kernel=reference_selection, supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
        operations=frozenset({TokenOperation.DROP, TokenOperation.MERGE}),
        metadata={"claim": "Released Qwen text/core recipes; native resolution, fixed K128, Qwen3.5 hybrid cache"}))
    registry.register_variant(ModalityVariantDescriptor(name=f"{name}.image", algorithm=name,
        target_modalities=frozenset({Modality.IMAGE}), conditioning_modalities=frozenset({Modality.TEXT})))
    binding = f"{name}.image.qwen3_5"
    registry.register_binding(BackbonePlacementBinding(name=binding, backbone="qwen3_5", variant=f"{name}.image",
        targets=(BindingTarget(name="reference_projector", placement=Placement.PROJECTOR_LEVEL,
            target_site="post_merger_pre_llm_scatter", target_modalities=frozenset({Modality.IMAGE}),
            required_handles=frozenset({"vision_projector", "llm"})),),
        supported_phases=frozenset({ExecutionPhase.GENERATE}), executor_factory=create_executor,
        legalizer="qwen35.native_sparse_mrope_hybrid_cache", exclusive_runtime_slot="image_comparison",
        conflicts=frozenset(b.name for b in registry.bindings() if b.backbone == "qwen3_5")))
    registry.extend_support([SupportUnit(identity=SupportUnitId(algorithm=name, variant=f"{name}.image",
        backbone="qwen3_5", target_modality=Modality.IMAGE, phase=ExecutionPhase.GENERATE), binding=binding)])
