"""Isolated Qwen3.5 projector transfer; never load a Qwen2.5 head as Qwen3.5."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct

import torch

from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.compression_training_check import AnchorAdapterSpec, create_anchor_adapter
from mllm_comp.analysis.common.learned_topk_experiment import digest_json
from mllm_comp.compressors.image.learned_visual_topk import LearnedVisualTopK, LearnedVisualTopKSpec
from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec
from mllm_comp.compressors.image.word_visual_topk import WordVisualTopK, WordVisualTopKSpec, clean_ids, template_patterns
from mllm_comp.compressors.image.protected_visual_topk import ProtectedTopKSpec, ProtectedWordSelector

KINDS = ("mlp", "visual", "question", "word")
METHOD_KIND = {"old_mlp":"mlp", "visual_selector":"visual", "question_selector":"question",
               "protected_selector":"word"}
EXTERNAL = ("eadp", "prunesid", "mmtok", "bifocal_stage1")
CHECKPOINT = "/home/WuHao/checkpoints/ckpts-vlm/Qwen3.5-9B"


def identify(path=CHECKPOINT):
    folder = Path(path).resolve()
    cfg = json.loads((folder/"config.json").read_text())
    text = cfg.get("text_config", {})
    if cfg.get("model_type") != "qwen3_5" or text.get("hidden_size") != 4096 or text.get("num_hidden_layers") != 32:
        raise ValueError("Expected complete Qwen3.5-9B, not Qwen3-VL, Qwen3.8 or Qwen2.5")
    index = json.loads((folder/"model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 4:
        raise ValueError("Expected all four Qwen3.5-9B weight shards")
    for name in shards:
        with (folder/name).open("rb") as stream:
            size, = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(size))
        last = max(v["data_offsets"][1] for k,v in header.items() if k != "__metadata__")
        if (folder/name).stat().st_size != 8+size+last:
            raise ValueError("Incomplete Qwen3.5 weight shard")
    names = shards + ["config.json", "model.safetensors.index.json", "preprocessor_config.json",
                     "video_preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                     "vocab.json", "merges.txt", "chat_template.jinja"]
    files = {}
    for name in names:
        source = folder/name
        before = source.stat()
        checksum = file_sha(source)
        after = source.stat()
        if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
            raise ValueError("Checkpoint changed while fingerprinting")
        files[name] = {"sha256":checksum, "bytes":after.st_size, "mtime_ns":after.st_mtime_ns}
    digest = digest_json({k:{a:v[a] for a in ("sha256","bytes")} for k,v in files.items()})
    return {"checkpoint":str(folder), "model_type":"qwen3_5", "hidden_size":4096,
            "num_hidden_layers":32, "identity_sha256":digest, "revision":f"local-sha256:{digest}",
            "upstream_revision":None, "files":files, "layer_types":text["layer_types"]}


def implementation_hash():
    root = Path(__file__).resolve().parents[2]
    names = ["analysis/common/qwen35_transfer.py", "compressors/comparison/qwen35_topk.py",
             "compressors/image/learned_visual_topk.py", "compressors/image/context_visual_topk.py",
             "compressors/image/word_visual_topk.py", "compressors/image/protected_visual_topk.py",
             "compressors/image/fixed_visual_merge.py", "analysis/common/compression_training_check.py"]
    return digest_json({name:file_sha(root/name) for name in names})


def create_head(kind, dimension, *, visual=None, patterns=()):
    if kind not in KINDS:
        raise ValueError("Unknown transferred head")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        if kind == "mlp":
            return create_anchor_adapter(AnchorAdapterSpec(dimension))
        if kind == "visual":
            return LearnedVisualTopK(LearnedVisualTopKSpec(dimension))
        if visual is None:
            raise ValueError("Conditioned heads must warm-start the completed same-base visual head")
        if kind == "word":
            return WordVisualTopK(WordVisualTopKSpec(dimension, template_token_ids=patterns), visual=visual)
        return ContextVisualTopK(ContextVisualTopKSpec(dimension), visual=visual)


def save_head(module, folder, *, kind, identity, provenance):
    from safetensors.torch import save_file
    from mllm_comp.core.serialization import atomic_write_json
    folder = Path(folder)
    if folder.exists():
        raise ValueError("Refuse head checkpoint overwrite")
    folder.mkdir(parents=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in module.state_dict().items()}, str(folder/"head.safetensors"))
    atomic_write_json(folder/"head_config.json", {"format":"qwen35_projector_transfer_v1", "kind":kind,
        "spec":asdict(module.spec), "base_identity_sha256":identity["identity_sha256"],
        "implementation_hash":implementation_hash(), "weights_sha256":file_sha(folder/"head.safetensors"),
        "provenance":provenance})


def load_head(folder, *, identity, device):
    from safetensors.torch import load_file
    folder = Path(folder)
    cfg = json.loads((folder/"head_config.json").read_text())
    if (cfg.get("format") != "qwen35_projector_transfer_v1" or
            cfg["base_identity_sha256"] != identity["identity_sha256"] or
            cfg["spec"]["hidden_size"] != identity["hidden_size"] or
            cfg["implementation_hash"] != implementation_hash() or
            cfg["weights_sha256"] != file_sha(folder/"head.safetensors")):
        raise ValueError("Wrong base, source, format, dimension or corrupted Qwen3.5 head")
    visual = LearnedVisualTopK(LearnedVisualTopKSpec(identity["hidden_size"]))
    module = create_head(cfg["kind"], identity["hidden_size"], visual=visual,
                         patterns=cfg["spec"].get("template_token_ids", ()))
    module.load_state_dict(load_file(str(folder/"head.safetensors")), strict=True)
    return module.to(device)


def question_embeddings(model, tokenizer, inputs, *, wordwise=False):
    """Parse answer-free ChatML using this tokenizer, never Qwen2 token constants."""
    ids = inputs["input_ids"][0].tolist()
    labels = inputs.get("labels")
    if labels is not None:
        where = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if len(where): ids = ids[:int(where[0])]
    prefix = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    starts = [i+len(prefix) for i in range(len(ids)-len(prefix)+1) if ids[i:i+len(prefix)] == prefix]
    if not starts:
        raise ValueError("No serialized Qwen3.5 user question")
    values = ids[starts[-1]:]
    if end not in values:
        raise ValueError("No complete user span before the answer")
    values = values[:values.index(end)]
    sentinels = {model.config.image_token_id, model.config.vision_start_token_id, model.config.vision_end_token_id}
    values = [i for i in values if i not in sentinels]
    if wordwise: values = clean_ids(values, template_patterns(tokenizer))
    if not values: raise ValueError("Empty Qwen3.5 user question")
    with torch.no_grad():
        q = model.get_input_embeddings()(torch.tensor(values,device=model.device,dtype=torch.long)).detach()
    return q, {"question_tokens":len(values), "question_ids_sha256":digest_json(values),
               "question_source":"qwen35_tokenizer_last_complete_user_span_no_answer"}


def select_tokens(features, *, method, module, question=None, grid=None, target=128):
    from mllm_comp.compressors.image.fixed_visual_merge import FixedVisualMergeSpec, fixed_visual_merge
    n, k = len(features), min(target,len(features))
    if n <= target:
        ids = torch.arange(n,device=features.device)
        return {"tokens":features, "indices":ids}, {"identity":True, "trained":module is not None}
    if method == "old_mlp":
        height,width = grid
        base = fixed_visual_merge(features,grid_height=height,grid_width=width,
            spec=FixedVisualMergeSpec(partition="feature_anchor",target_count=k))
        with torch.autocast(features.device.type,dtype=torch.bfloat16,enabled=features.device.type=="cuda"):
            out = module({"tokens":base.tokens,"indices":base.indices,
                          "members":base.metadata["cluster_members"]})
        return out, {"trained":True, "cluster_members":base.metadata["cluster_members"],
                     "partition_hash":base.metadata["partition_hash"]}
    if method == "protected_selector":
        out = ProtectedWordSelector(module,ProtectedTopKSpec())(features,question=question)
        return out, out["word_diagnostics"]
    out = module(features,**({} if method=="visual_selector" else {"question":question}))
    return out, {"trained":True}


def compact_prompt(model, source, features, output):
    """One image, unpadded batch1; physical cache indices != rotary coordinates."""
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    ids = source["input_ids"]
    types = source.get("mm_token_type_ids")
    grid = source.get("image_grid_thw")
    if ids.ndim != 2 or ids.shape[0] != 1 or grid is None or len(grid)!=1 or int(grid[0,0])!=1 or types is None:
        raise ValueError("Qwen3.5 compression requires one still image, types and batch1")
    if source.get("pixel_values_videos") is not None or source.get("past_key_values") is not None:
        raise ValueError("No videos or externally continued caches in projector transfer")
    mask = source.get("attention_mask",torch.ones_like(ids))
    if mask.shape!=ids.shape or not bool((mask==1).all()): raise ValueError("Unpadded input required")
    visual = (ids[0]==model.config.image_token_id).nonzero(as_tuple=True)[0]
    local = output["indices"]
    if not torch.is_tensor(local): local=torch.tensor(local,device=ids.device)
    if (len(visual)!=len(features) or local.dtype!=torch.long or len(local.unique())!=len(local)
            or (not output.get("official_order",False) and not torch.equal(local,local.sort().values))
            or (local<0).any() or (local>=len(features)).any()
            or output["tokens"].shape!=(len(local),features.shape[1]) or not torch.isfinite(output["tokens"]).all()):
        raise ValueError("Bad visual population, representatives or compressed features")
    positions,delta = model.model.get_rope_index(ids,mm_token_type_ids=types,
        image_grid_thw=grid,attention_mask=mask)
    keep = ids[0]!=model.config.image_token_id
    keep[visual[local]] = True
    gather = keep.nonzero(as_tuple=True)[0]
    if output.get("official_order",False):
        if not torch.equal(visual,torch.arange(int(visual[0]),int(visual[-1])+1,device=visual.device)):
            raise ValueError("Official ordered selection requires one contiguous visual span")
        gather=torch.cat((gather[gather<visual[0]],visual[local],gather[gather>visual[-1]]))
    rotary = positions[:,:,gather].clone()
    destinations = (ids[:,gather][0]==model.config.image_token_id).nonzero(as_tuple=True)[0]
    members = output.get("members")
    if members is not None:
        if sorted(i for group in members for i in group)!=list(range(len(features))):
            raise ValueError("Anchor clusters lost or duplicated sources")
        for j,group in enumerate(members):
            rotary[:,0,destinations[j]] = positions[:,0,visual[list(group)]].float().mean(-1).round().to(rotary.dtype)
    physical = torch.arange(len(gather),device=ids.device).view(1,1,-1)
    prepared = {k:v for k,v in source.items() if k not in
        {"pixel_values","image_grid_thw","pixel_values_videos","video_grid_thw","labels","position_ids","inputs_embeds"}}
    prepared.update(input_ids=ids[:,gather],attention_mask=mask[:,gather],mm_token_type_ids=types[:,gather],
        position_ids=torch.cat((physical,rotary),dim=0),
        mm_encoder_outputs={"image":BaseModelOutputWithPooling(pooler_output=(output["tokens"],))})
    removed = ids.shape[1]-len(gather)
    model.model.rope_deltas = delta+removed
    trace = {"source_count":len(features),"result_count":len(local),"selected_indices":local.tolist(),
        "input_prompt_tokens":ids.shape[1],"output_prompt_tokens":len(gather),
        "expected_first_decode_position":int(rotary.max())+1,"decode_position_verified":None,
        "original_visual_mrope_preserved":members is None,"text_mrope_preserved":True,
        "position_policy":"rounded_cluster_centroid" if members is not None else "original_sparse_mrope",
        "mm_encoder_outputs_used":True,"physical_sequence_compacted":True,"enable_thinking":False}
    return prepared, trace, gather


def load_bundle(identity, device):
    from mllm_comp.backbones.qwen3_5 import create_qwen3_5_backbone
    validate_identity(identity)
    facade = create_qwen3_5_backbone()
    processor = facade.processor_adapter.load_processor(identity["checkpoint"],local_files_only=True,
        use_fast=False,min_pixels=200704,max_pixels=1605632)
    model = facade.model_adapter.load_model(identity["checkpoint"],train=False,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="eager").to(device).eval().requires_grad_(False)
    return {"model":model,"llm":model.model.language_model,"processor":processor,"facade":facade,
            "device":torch.device(device)}


def prepare_training_example(bundle, example):
    from io import BytesIO
    from PIL import Image
    from mllm_comp.backbones.qwen3_5 import Qwen3_5Sample
    from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
    from mllm_comp.analysis.common.learned_hybrid_runtime import move_inputs
    raw=example["image"]
    with Image.open(BytesIO(raw["bytes"]) if raw.get("bytes") else raw["path"]) as im:
        image=im.convert("RGB")
    if media_fingerprint([image])!=example["image_fingerprint"]: raise ValueError("Training image changed")
    sample=Qwen3_5Sample(example["sample_id"],[
        {"role":"system","content":[{"type":"text","text":"You are a helpful assistant."}]},
        {"role":"user","content":[{"type":"image","image":image,"min_pixels":200704,"max_pixels":1605632},
                                  {"type":"text","text":example["question"]}]}],enable_thinking=False)
    adapter=bundle["facade"].processor_adapter
    prompt=adapter.prepare_inputs(bundle["processor"],sample,phase="train")
    teacher=adapter.prepare_teacher_forced_inputs(bundle["processor"],sample,target_text=example["target"],prompt_batch=prompt,phase="train")
    prompt=move_inputs(prompt.model_inputs,bundle["device"])
    teacher=move_inputs(teacher.model_inputs,bundle["device"])
    prefix=prompt["input_ids"].shape[1]
    if not torch.equal(prompt["input_ids"],teacher["input_ids"][:,:prefix]): raise ValueError("Answer-free prefix changed")
    model=bundle["model"]
    with torch.no_grad():
        groups=model.get_image_features(prompt["pixel_values"],prompt["image_grid_thw"],return_dict=True).pooler_output
        features,=groups
        visual=(teacher["input_ids"][0]==model.config.image_token_id).nonzero(as_tuple=True)[0]
        if len(visual)!=len(features) or len(features)<128: raise ValueError("Training token budget/grid not applicable")
        hidden=model.get_input_embeddings()(teacher["input_ids"]).clone()
        hidden[0,visual]=features
        positions,_=model.model.get_rope_index(teacher["input_ids"],mm_token_type_ids=teacher["mm_token_type_ids"],
            image_grid_thw=teacher["image_grid_thw"],attention_mask=teacher["attention_mask"])
        physical=torch.arange(hidden.shape[1],device=hidden.device).view(1,1,-1)
        positions=torch.cat((physical,positions),dim=0)
    return {"prompt":prompt,"teacher":teacher,"features":features.detach(),"hidden":hidden.detach(),
            "positions":positions,"visual":visual,"grid":tuple(int(v)//2 for v in prompt["image_grid_thw"][0,1:])}


def answer_logits(bundle, hidden, positions, labels):
    from mllm_comp.analysis.common.visual_token_phenomena import answer_predictor_positions
    predictors,targets=answer_predictor_positions(labels)
    out=bundle["llm"](inputs_embeds=hidden,position_ids=positions,
        attention_mask=torch.ones(hidden.shape[:2],device=hidden.device,dtype=torch.long),use_cache=False,return_dict=True)
    logits=bundle["model"].lm_head(out.last_hidden_state[:,predictors])[0].float()
    return logits,torch.tensor(targets,device=hidden.device)
