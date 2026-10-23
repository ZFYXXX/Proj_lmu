"""Fixed-budget, answer-blind region-replacement factorial diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
import traceback
from typing import Any, Mapping

from mllm_comp.analysis.bindings import resolve_analysis_binding
from mllm_comp.analysis.common.config import positive_integer, validate_common_probe_config
from mllm_comp.analysis.common.evidence_interaction import (
    SEEDS, digest, farthest_indices, interactions, materialize_plan, plan_regions,
)
from mllm_comp.analysis.common.sample import prepare_teacher_forced_analysis_forward_state
from mllm_comp.analysis.common.visual_token_generation import greedy_decode_embeddings, primary_task_score
from mllm_comp.analysis.common.visual_token_phenomena import answer_predictor_positions, compact_answer_logit_metrics, media_fingerprint
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.analysis.probes.visual_token_phenomena import _compact_layer_state, _run_decoder_suffix
from mllm_comp.backbones.qwen25_vl.implementation import _move_value
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord
from mllm_comp.core.serialization import atomic_write_json, to_jsonable


def implementation_hash() -> str:
    base = Path(__file__).parents[1]
    paths = [Path(__file__), base / "common/evidence_interaction.py", base / "common/evidence_cohort.py",
             base / "common/visual_token_generation.py", base / "probes/visual_token_phenomena.py",
             Path(__file__).parents[2] / "integrations/lmms_eval/adapter.py"]
    value = hashlib.sha256()
    for path in paths:
        value.update(str(path.name).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def question_positions(tokenizer: Any, prompt_ids: Any, question: str) -> list[int]:
    """Locate only question tokens; fail rather than silently score system text."""
    ids = prompt_ids[0].tolist()
    question = question.strip()
    if not question:
        raise ValueError("Empty question.")
    text = tokenizer.decode(ids,clean_up_tokenization_spaces=False)
    if text.count(question) != 1:
        raise ValueError("Question cannot be located unambiguously in the gold-free prompt.")
    start = text.index(question); end = start + len(question)
    # BPE can merge the final '?' with the following newline, so matching
    # independently tokenized question IDs is not a valid span locator.
    if getattr(tokenizer,"is_fast",False):
        encoded = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        if encoded["input_ids"] != ids:
            raise ValueError("Prompt decode/re-encode changed token IDs.")
        offsets = encoded["offset_mapping"]
    else:
        boundaries = [0] + [len(tokenizer.decode(ids[:i],clean_up_tokenization_spaces=False)) for i in range(1,len(ids)+1)]
        offsets = list(zip(boundaries[:-1],boundaries[1:]))
    selected = [i for i,(left,right) in enumerate(offsets) if left < end and right > start]
    if not selected:
        raise ValueError("Question span contains no prompt tokens.")
    return selected


def cached_decode(*, llm, head, embedding, hidden, positions, eos_ids, max_new):
    import torch
    if positions.shape != (3, 1, hidden.shape[1]) or hidden.shape[0] != 1:
        raise ValueError("Misaligned compressed prompt / M-RoPE positions.")
    ids, past, state = [], None, hidden
    first_position = int(positions.max()) + 1
    length = hidden.shape[1]
    with torch.inference_mode():
        for step in range(max_new):
            current_positions = positions if step == 0 else torch.full((3,1,1), first_position + step - 1, dtype=positions.dtype, device=state.device)
            mask = torch.ones((1,length + step), dtype=torch.long, device=state.device)
            output = llm(inputs_embeds=state, position_ids=current_positions,
                         attention_mask=mask, past_key_values=past, use_cache=True,
                         output_attentions=False, output_hidden_states=False, return_dict=True)
            logits = head(output.last_hidden_state[:, -1:])[0,-1].float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite decode logits.")
            token = int(logits.argmax())
            ids.append(token)
            if token in eos_ids or step + 1 == max_new:
                break
            past = output.past_key_values
            if past is None or past.get_seq_length() != length + step:
                raise ValueError("Unexpected KV cache sequence length.")
            state = embedding(torch.tensor([[token]], device=state.device))
    return {"generated_token_ids":ids, "stopped_on_eos":ids[-1] in eos_ids,
            "first_appended_token_position":first_position, "use_cache":True,
            "max_new_tokens":max_new, "gold_answer_in_decode_context":False}


class VisualEvidenceInteraction:
    name = "visual_evidence_interaction"
    version = "1.0.0"
    requires_teacher_forcing = True

    def validate_config(self, config):
        validate_common_probe_config(config, context=self.name)
        if positive_integer(config,"budget",128,context=self.name) != 128:
            raise ValueError("The preregistered main budget is 128.")
        if config.get("random_seeds",list(SEEDS)) != list(SEEDS):
            raise ValueError("Preregistered seeds must be [11,23,47].")
        if config.get("stage") not in {"smoke","development","confirmation","question_review","budget_review"}:
            raise ValueError("An explicit preregistered stage is required.")
        for name in ["artifact_root","cohort_path"]:
            if not config.get(name):
                raise ValueError(f"Missing {name}.")
        if config.get("save_raw",False):
            raise ValueError("This probe records statistics and indices, not full hidden states.")

    def open(self, *, backbone, runtime, config, phase, output_dir):
        self.validate_config(config)
        return _Session(self,backbone,runtime,config,phase,output_dir)


@dataclass
class _Session:
    plugin: Any
    backbone: Any
    runtime: Any
    config: dict
    phase: str
    output_dir: str

    def __post_init__(self):
        if self.backbone.capabilities().name != "qwen25_vl":
            raise BackendIntegrationError("Evidence interaction requires Qwen2.5-VL.")
        if self.runtime.revision != "1b5a7674d40a0ac9acef541a46fed12d710bd36b":
            raise BackendIntegrationError("Unexpected checkpoint revision.")
        resolve_analysis_binding(self.backbone,self.runtime.model)
        self.root = Path(self.config["artifact_root"])
        self.stage = self.config["stage"]
        self.code_hash = implementation_hash()
        self.cohort_hash = hashlib.sha256(Path(self.config["cohort_path"]).read_bytes()).hexdigest()
        self.config_hash = digest(json.dumps(self.config,sort_keys=True) + self.cohort_hash + self.code_hash)
        if self.stage in {"confirmation","question_review","budget_review"}:
            freeze = json.loads((self.root / "frozen.json").read_text())
            if freeze["implementation_hash"] != self.code_hash or freeze["cohort_hash"] != self.cohort_hash:
                raise BackendIntegrationError("Frozen implementation/cohort changed; confirmation must not continue.")
        self.completed = []
        Path(self.output_dir).mkdir(parents=True,exist_ok=True)

    def process(self, sample: Mapping[str,Any]):
        entry = sample.get("metadata",{}).get("evidence_cohort",{})
        if not entry or entry["sample_id"] != sample["sample_id"] or entry["stage"] != self.stage:
            raise ValueError("Sample is not in the explicit frozen cohort/stage.")
        folder = self.root / "samples" / self.stage / str(sample["sample_id"]).replace(":","_")
        folder.mkdir(parents=True,exist_ok=True)
        try:
            summary_path = folder / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary["config_hash"] != self.config_hash:
                    raise ValueError("Cannot resume results from a different implementation/configuration.")
            else:
                with __import__("torch").inference_mode():
                    summary = self._process(sample,entry,folder)
                atomic_write_json(summary_path,summary,ensure_ascii=False)
            self.completed.append(summary)
            print(f"EVIDENCE_COMPLETE {self.stage} {sample['sample_id']} states={summary['condition_count']} elapsed_s={summary['elapsed_s']:.1f}",flush=True)
        except Exception as exc:
            atomic_write_json(folder / "failure.json",{"sample_id":sample["sample_id"],"stage":self.stage,
                              "config_hash":self.config_hash,"error":str(exc),"traceback":traceback.format_exc()},ensure_ascii=False)
            raise  # Never let the analysis lifecycle silently skip a failed image.

    def _process(self,sample,entry,folder):
        import torch
        from transformers import GenerationConfig
        from mllm_comp.backbones.qwen25_vl.provider_bridge import _merged_vision_metrics, _apply_rotary_pos_emb_vision_4573
        from mllm_comp.compressors.image.fixed_visual_merge import fixed_visual_merge, FixedVisualMergeSpec
        from mllm_comp.compressors.image.prumerge import prumerge_image_tokens, PruMergeSpec

        started = time.perf_counter()
        fingerprint = media_fingerprint(sample["media"])
        if fingerprint != entry["image_fingerprint"]:
            raise ValueError("Frozen image fingerprint mismatch.")
        if not callable(sample.get("score_prediction")):
            raise ValueError("Native task scorer required.")
        model, processor = self.runtime.model, self.runtime.processor
        handles = self.backbone.model_adapter.structural_handles(model)
        llm, vision = handles["llm"], handles["vision_encoder"]
        if llm.config._attn_implementation != "eager":
            raise ValueError("Eager attention required.")
        embedding, head = model.get_input_embeddings(), model.lm_head
        prompt_inputs = {k:_move_value(v,device=self.runtime.device,dtype=self.runtime.dtype) for k,v in sample["processor_outputs"].items()}
        if prompt_inputs["input_ids"].shape[0] != 1 or not bool((prompt_inputs["attention_mask"] == 1).all()):
            raise ValueError("Only unpadded batch-size-one image inputs are supported.")
        grid = prompt_inputs["image_grid_thw"]
        if len(grid) != 1 or int(grid[0,0]) != 1 or prompt_inputs.get("pixel_values_videos") is not None:
            raise ValueError("Exactly one still image required.")
        height,width = int(grid[0,1])//vision.spatial_merge_size,int(grid[0,2])//vision.spatial_merge_size
        visual = torch.nonzero(prompt_inputs["input_ids"][0] == model.config.image_token_id).flatten().tolist()
        if len(visual) != height*width or len(visual) < 160:
            raise ValueError("Grid/placeholder mismatch or insufficient tokens for K128 factorial.")
        captured = {}
        def text_pre(module,args): captured["hidden"] = args[0].detach()
        def vision_pre(module,args,kwargs):
            captured["vision_hidden"] = (args[0] if args else kwargs["hidden_states"]).detach()
            captured["vision_positions"] = kwargs["position_embeddings"]
        with HookSession() as hooks:
            hooks.register_forward_pre_hook(llm.layers[0],text_pre)
            hooks.register_forward_pre_hook(vision.blocks[-1].attn,vision_pre,with_kwargs=True)
            dense_prompt = model(**prompt_inputs,output_attentions=True,output_hidden_states=False,use_cache=False,return_dict=True)
        prompt_hidden = captured["hidden"]
        question = str(sample["raw_doc"]["question"])
        # Upstream TextVQA's doc_to_text applies str.capitalize(). Preserve
        # that native prompt instead of changing the benchmark to fit us.
        prompt_question = question.capitalize() if "textvqa" in sample["task_name"] else question
        question_indices = question_positions(processor.tokenizer,prompt_inputs["input_ids"],prompt_question)
        attn = dense_prompt.attentions[0][0]
        scores = attn[:,question_indices,:][:,:,visual].float().mean(dim=(0,1))
        del dense_prompt,attn
        features = prompt_hidden[0,visual]
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Nonfinite native visual state.")
        window_index,_ = vision.get_window_index(grid)
        attention_block = vision.blocks[-1].attn
        raw_qkv = attention_block.qkv(captured["vision_hidden"]).reshape(-1,3,attention_block.num_heads,attention_block.head_dim)
        query,key = _apply_rotary_pos_emb_vision_4573(raw_qkv[:,0],raw_qkv[:,1],*captured["vision_positions"])
        visual_attentions,visual_keys = _merged_vision_metrics(query,key,patch_counts=[int(raw_qkv.shape[0])],feature_counts=[len(visual)],
                  window_index=window_index,spatial_merge_unit=vision.spatial_merge_size**2,scaling=float(attention_block.scaling))
        del raw_qkv,query,key
        captured.clear()
        teacher_state,objective = prepare_teacher_forced_analysis_forward_state(backbone=self.backbone,runtime=self.runtime,
                                        sample=sample,probe_name=self.plugin.name)
        teacher_inputs = {k:_move_value(v,device=self.runtime.device,dtype=self.runtime.dtype) for k,v in teacher_state.native_model_inputs.items()}
        prefix = int(objective["prefix_token_counts"][0])
        if prefix != prompt_hidden.shape[1] or not torch.equal(teacher_inputs["input_ids"][:,:prefix],prompt_inputs["input_ids"]):
            raise ValueError("Teacher input prefix differs from the gold-free prompt.")
        with HookSession() as hooks:
            hooks.register_forward_pre_hook(llm.layers[0],text_pre)
            teacher = model(**teacher_inputs)
        teacher_hidden = captured.pop("hidden")
        if not torch.equal(teacher_hidden[:,:prefix],prompt_hidden):
            raise ValueError("Appending the teacher answer changed the LLM input prompt state.")
        predictors,targets = answer_predictor_positions(teacher_inputs["labels"])
        positions,_ = model.model.get_rope_index(teacher_inputs["input_ids"],grid,attention_mask=teacher_inputs["attention_mask"])
        if positions.shape[0] == 4: positions = positions[1:]
        source_positions = positions[:,0]
        prompt_positions = source_positions[:,:prefix].unsqueeze(1)
        max_new = 16 if "chartqa" in sample["task_name"] else 32
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids,int) else list(eos_ids)
        ledger = folder / "conditions.jsonl"
        recorded = {}
        if ledger.exists():
            with ledger.open("rb+") as handle:
                while True:
                    offset = handle.tell(); line = handle.readline()
                    if not line: break
                    if not line.endswith(b"\n"):
                        handle.truncate(offset); break
                    row = json.loads(line)
                    if row["config_hash"] != self.config_hash or row["request_id"] in recorded:
                        raise ValueError("Mismatched or duplicate condition in resume ledger.")
                    recorded[row["request_id"]] = row
        common = {"sample_id":sample["sample_id"],"primary_sample_id":entry["primary_sample_id"],
                  "task":sample["task_name"],"split":entry["split"],"stage":self.stage,"image_fingerprint":fingerprint,
                  "checkpoint":str(self.runtime.model_path_or_id),"revision":self.runtime.revision,
                  "config_hash":self.config_hash,"source_visual_tokens":len(visual),"grid":[height,width],
            "target_text":sample["target_text"],"question":question,
            "question_token_positions":question_indices,"selection_uses_answer":False}

        def evaluate(request_id,retained,*,budget=None,operator="retain",values=None,centroids=None,extra=None):
            if request_id in recorded:
                return recorded[request_id]
            before = time.perf_counter()
            compact = _compact_layer_state(hidden=teacher_hidden,position_ids=source_positions,visual_positions=visual,
                         members=tuple((i,) for i in retained),operator="retain",token_scores=None,budget=len(retained))
            hidden,ids = compact["hidden"],compact["position_ids"]
            if budget is not None and len(retained) != budget:
                raise ValueError("Actual retained visual tokens differ from budget.")
            if len(retained) != len(set(retained)):
                raise ValueError("Duplicate retained visual token.")
            source_indices = sorted(compact["source_to_compact"],key=compact["source_to_compact"].get)
            gather = torch.tensor(source_indices,device=hidden.device,dtype=torch.long)
            compact_tokens = teacher_inputs["input_ids"].index_select(1,gather)
            compact_labels = teacher_inputs["labels"].index_select(1,gather)
            compact_mask = teacher_inputs["attention_mask"].index_select(1,gather)
            if (compact_tokens.shape != compact_labels.shape or compact_labels.shape != compact_mask.shape
                    or compact_tokens.shape[1] != hidden.shape[1]
                    or int((compact_tokens == model.config.image_token_id).sum()) != len(retained)
                    or not bool((compact_mask == 1).all())):
                raise ValueError("Placeholder, mask, labels and compressed embeddings are misaligned.")
            candidate_predictors,candidate_targets = answer_predictor_positions(compact_labels)
            if candidate_targets != targets or candidate_predictors != [compact["source_to_compact"][p] for p in predictors]:
                raise ValueError("Answer labels/predictors changed during compaction.")
            if values is not None:
                hidden = hidden.clone()
                for j,index in enumerate(retained):
                    p = compact["source_to_compact"][visual[index]]
                    hidden[0,p] = values[j]
                    if centroids is not None: ids[:,0,p] = centroids[j]
            # All text/answer positions must remain identical after compaction.
            for original,mapped in compact["source_to_compact"].items():
                if original not in visual and not torch.equal(ids[:,0,mapped],source_positions[:,original]):
                    raise ValueError("Text position IDs changed during visual compaction.")
            candidate = _run_decoder_suffix(llm=llm,lm_head=head,hidden=hidden,position_ids=ids,start_layer=0)
            metrics = compact_answer_logit_metrics(teacher.logits,candidate,source_predictor_positions=predictors,
                        compact_predictor_positions=[compact["source_to_compact"][p] for p in predictors],targets=targets)
            del candidate
            compressed_prefix = prefix - len(visual) + len(retained)
            ph,pp = hidden[:,:compressed_prefix],ids[:,:,:compressed_prefix]
            decoded = cached_decode(llm=llm,head=head,embedding=embedding,hidden=ph,positions=pp,eos_ids=eos_ids,max_new=max_new)
            if decoded["first_appended_token_position"] != int(pp.max())+1:
                raise ValueError("Decode M-RoPE position offset mismatch.")
            text = processor.tokenizer.decode(decoded["generated_token_ids"],skip_special_tokens=True).strip()
            stops = sample.get("generation_stops",[])
            stops = [stops] if isinstance(stops,str) else stops
            for stop in stops:
                if stop: text = text.split(stop,1)[0].strip()
            task_metrics = sample["score_prediction"](text)
            if self.stage == "smoke" and (request_id == "reference/baseline" or request_id == "128/11/00"):
                plain = greedy_decode_embeddings(prompt_hidden=ph,prompt_position_ids=pp,token_embedding=embedding,
                         forward_logits=lambda h,p:_run_decoder_suffix(llm=llm,lm_head=head,hidden=h,position_ids=p,start_layer=0,last_token_only=True),
                         max_new_tokens=max_new,eos_token_ids=eos_ids)
                if plain["generated_token_ids"] != decoded["generated_token_ids"]:
                    raise ValueError("Cached/cache-free greedy decode mismatch.")
                decoded["cache_free_identity_passed"] = True
            if request_id == "reference/baseline" and self.stage == "smoke":
                generation = GenerationConfig(max_new_tokens=max_new,do_sample=False,use_cache=True,repetition_penalty=1.,
                            eos_token_id=eos_ids,pad_token_id=processor.tokenizer.pad_token_id)
                native = model.generate(**prompt_inputs,generation_config=generation)[0,prefix:].tolist()
                if native != decoded["generated_token_ids"]:
                    raise ValueError(f"Native/diagnostic dense generation mismatch: {native} != {decoded['generated_token_ids']}")
                decoded["native_generation_identity_passed"] = True
                if abs(metrics["answer_nll_delta"]) > .01 or metrics["teacher_to_candidate_answer_kl"] > .001:
                    raise ValueError("Dense identity teacher-logit check failed.")
            strict = None
            if "chartqa" in sample["task_name"]:
                references = sample["raw_doc"]["answer"]
                references = references if isinstance(references,list) else [references]
                strict = float(any(text.strip().casefold() == str(answer).strip().casefold() for answer in references))
            row = {**common,"status":"completed","request_id":request_id,"retained_indices":list(retained),"retained_visual_tokens":len(retained),
                   "operator":operator,"budget":budget,"prediction":text,"native_task_metrics":task_metrics,
                   "primary_task_score":primary_task_score(sample["task_name"],task_metrics),"strict_answer_match":strict,
                   **metrics,**decoded,"elapsed_s":time.perf_counter()-before,
                   "placeholder_mask_labels_alignment_passed":True,"text_positions_preserved":True,**(extra or {})}
            def finite(v):
                if isinstance(v,float) and not __import__("math").isfinite(v): raise ValueError("Nonfinite condition statistic.")
                if isinstance(v,dict):
                    for x in v.values(): finite(x)
                if isinstance(v,(list,tuple)):
                    for x in v: finite(x)
            finite(row)
            encoded = json.dumps(to_jsonable(row),ensure_ascii=False,allow_nan=False)
            with ledger.open("a",encoding="utf-8") as output:
                output.write(encoded+"\n"); output.flush(); __import__("os").fsync(output.fileno())
            recorded[request_id] = json.loads(encoded)
            return recorded[request_id]

        evaluate("reference/baseline",list(range(len(visual))))
        if self.stage not in {"question_review","budget_review"}:
            merge = fixed_visual_merge(features,grid_height=height,grid_width=width,spec=FixedVisualMergeSpec(target_count=128))
            reps = merge.indices.tolist()
            centroids = [source_positions[:,[visual[i] for i in cluster]].float().mean(1).round().long() for cluster in merge.metadata["cluster_members"]]
            evaluate("reference/feature_anchor_mean_k128",reps,budget=128,operator="cluster_mean",values=merge.tokens,centroids=centroids,
                     extra={"cluster_members":merge.metadata["cluster_members"],"partition_hash":merge.metadata["partition_hash"]})
            evaluate("reference/feature_anchor_representative_k128",reps,budget=128,
                     extra={"cluster_members":merge.metadata["cluster_members"],"partition_hash":merge.metadata["partition_hash"]})
            pru = prumerge_image_tokens(features,visual_attentions[0],visual_keys[0],spec=PruMergeSpec(variant="plus",adaptive=False,target_count=128,cluster_size=32))
            pru_trace = {**dict(pru.metadata),"global_residual_indices":[i for i in range(len(visual)) if i not in set(pru.metadata["center_indices"])],"actual_output_count":len(pru.indices)}
            evaluate("reference/prumerge_plus_k128",pru.indices.tolist(),budget=128,operator="prumerge_plus",values=pru.tokens,extra={"compression_trace":pru_trace})
            for seed in SEEDS:
                keep = sorted(random.Random(f"{seed}:{fingerprint}:random-prune").sample(range(len(visual)),128))
                evaluate(f"reference/random_k128/{seed}",keep,budget=128,extra={"seed":seed})

        saved = self.root / "plans" / f"{digest(entry['primary_sample_id'])}.json"
        budgets = [64,256] if self.stage == "budget_review" else [128]
        if self.stage in {"question_review","budget_review"}:
            source_plan = json.loads(saved.read_text())
            if source_plan["implementation_hash"] != self.code_hash or source_plan["image_fingerprint"] != fingerprint:
                raise ValueError("Review must reuse the frozen primary image plans.")
            region_plans = source_plan["region_plans"]
        else:
            region_plans = [plan_regions(features=features,scores=scores,height=height,width=width,sample_id=entry["primary_sample_id"],seed=seed) for seed in SEEDS]
            payload = {"implementation_hash":self.code_hash,"image_fingerprint":fingerprint,"region_plans":region_plans}
            if saved.exists() and json.loads(saved.read_text()) != payload:
                raise ValueError("Region planning changed between repeated runs.")
            saved.parent.mkdir(parents=True,exist_ok=True)
            atomic_write_json(saved,payload,ensure_ascii=False)
        all_interactions,na = [],[]
        for budget in budgets:
            if len(visual) < budget+32:
                na.append({"budget":budget,"reason":"source_tokens_below_budget_plus_32","source_tokens":len(visual)})
                continue
            for region in region_plans:
                plan = materialize_plan(region,features=features,budget=budget)
                path = folder / f"plan_{budget}_{region['seed']}.json"
                atomic_write_json(path,plan,ensure_ascii=False)
                plan_hash = digest(json.dumps(plan,sort_keys=True))
                rows = [evaluate(f"{budget}/{region['seed']}/{condition['condition_id']}",condition["retained_indices"],budget=budget,
                                 extra={"seed":region["seed"],"condition_id":condition["condition_id"],"changed_slots":condition["changed_slots"],"grouping_hash":plan_hash})
                        for condition in plan["states"]]
                all_interactions.extend({**row,"budget":budget,"seed":region["seed"]} for row in interactions(rows))
        atomic_write_json(folder / "interactions.json",all_interactions,ensure_ascii=False)
        self._region_map(sample,region_plans,folder)
        if prompt_hidden.is_cuda: torch.cuda.synchronize(prompt_hidden.device)
        return {**common,"config_hash":self.config_hash,"implementation_hash":self.code_hash,
                "condition_count":len(recorded),"interactions_count":len(all_interactions),"not_applicable":na,
                "elapsed_s":time.perf_counter()-started,"native_generation_identity_passed":recorded["reference/baseline"].get("native_generation_identity_passed"),
                "condition_path":str(ledger),"interactions_path":str(folder / "interactions.json"),
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(prompt_hidden.device) if prompt_hidden.is_cuda else 0}

    @staticmethod
    def _region_map(sample,plans,folder):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        plan = plans[0]
        image = sample["media"][0]
        fig,ax = plt.subplots(figsize=(8,6))
        ax.imshow(image,extent=(0,1,1,0)); ax.axis("off")
        for slot,indices in enumerate(plan["candidate_regions"]):
            r,c = divmod(indices[0],plan["width"])
            x,y = c/plan["width"],r/plan["height"]
            ax.add_patch(Rectangle((x,y),2/plan["width"],2/plan["height"],fill=False,color="red" if slot<4 else "cyan"))
            ax.text(x,y,str(slot),fontsize=8,color="red" if slot<4 else "blue")
        ax.set_title(str(sample["raw_doc"]["question"])[:120],fontsize=9,wrap=True)
        fig.savefig(folder/"regions.png",dpi=120,bbox_inches="tight"); plt.close(fig)

    def finish(self):
        if not self.completed:
            raise BackendIntegrationError("No completed evidence samples in this shard.")
        path = Path(self.output_dir)/"completed.json"
        atomic_write_json(path,{"samples":self.completed,"implementation_hash":self.code_hash},ensure_ascii=False)
        return AnalysisRecord(name=self.plugin.name,phase=self.phase,method=self.plugin.name,
                              summary={"completed_images":len(self.completed),"stage":self.stage},
                              artifact_paths={"completed":str(path)},sample_count=len(self.completed))

    def close(self):
        pass
