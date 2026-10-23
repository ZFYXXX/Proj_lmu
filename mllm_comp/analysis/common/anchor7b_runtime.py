"""Pinned Qwen2.5-VL-7B runtime; preserves every historical 3B checkpoint hash."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct

from mllm_comp.analysis.common.compression_training_check import (
    AnchorAdapterSpec, create_anchor_adapter, anchor_base, compact_anchor, implementation_hash as shared_hash,
)
from mllm_comp.analysis.common.learned_hybrid_runtime import prepare_state, answer_logits

CHECKPOINT = "/home/WuHao/checkpoints/ckpts-vlm/Qwen2.5-VL-7B-Instruct"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identify_checkpoint(path=CHECKPOINT):
    """Content-address local weights rather than invent an upstream revision."""
    folder = Path(path).resolve()
    config = json.loads((folder / "config.json").read_text())
    text = config.get("text_config", config)
    if config.get("model_type") != "qwen2_5_vl" or text.get("hidden_size") != 3584 or text.get("num_hidden_layers") != 28:
        raise ValueError("Expected native Qwen2.5-VL-7B, not 3B or a text-only checkpoint")
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if len(names) != 5:
        raise ValueError("Expected all five 7B safetensors shards")
    manifest = {}
    for name in names:
        source = folder / name
        with source.open("rb") as stream:
            header_size, = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(header_size))
        last_byte = max(v["data_offsets"][1] for k,v in header.items() if k != "__metadata__")
        if 8 + header_size + last_byte != source.stat().st_size:
            raise ValueError(f"Truncated/invalid weight shard: {source}")
    names += ["config.json", "model.safetensors.index.json", "generation_config.json", "preprocessor_config.json",
              "tokenizer_config.json", "tokenizer.json", "merges.txt", "vocab.json", "chat_template.json"]
    for name in names:
        source = folder / name
        before = source.stat()
        checksum = file_sha(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Checkpoint changed while fingerprinting")
        manifest[name] = {"sha256": checksum, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns}
    content = {name: {"sha256": v["sha256"], "bytes": v["bytes"]} for name,v in manifest.items()}
    content_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return {"checkpoint": str(folder), "model_type": "qwen2_5_vl", "hidden_size": 3584, "num_hidden_layers": 28,
            "identity_sha256": content_hash, "revision": f"local-sha256:{content_hash}",
            "upstream_revision": None, "revision_note": "Local ModelScope download; upstream commit not asserted. Full file hashes pin this run.",
            "files": manifest}


def validate_identity(identity, *, full_hash=False):
    folder = Path(identity["checkpoint"])
    for name, entry in identity["files"].items():
        source = folder / name
        stat = source.stat()
        if stat.st_size != entry["bytes"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"Pinned 7B checkpoint file changed: {source}")
        if full_hash and file_sha(source) != entry["sha256"]:
            raise ValueError(f"Pinned 7B checksum mismatch: {source}")


def implementation_hash():
    root = Path(__file__).parents[2]
    paths = [Path(__file__), root / "compressors/image/fixed_visual_merge.py",
             root / "analysis/common/visual_merge_attribution.py", root / "analysis/common/condition_ledger.py"]
    h = hashlib.sha256(shared_hash().encode())
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()


def load_runtime(identity, device):
    import torch
    from mllm_comp.backbones.base import BackboneRuntime
    from mllm_comp.backbones.qwen25_vl import create_qwen25_vl_backbone
    validate_identity(identity)
    path = identity["checkpoint"]
    facade = create_qwen25_vl_backbone()
    processor = facade.processor_adapter.load_processor(path, local_files_only=True, use_fast=False,
        min_pixels=200704, max_pixels=1605632)
    model = facade.model_adapter.load_model(path, train=False, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    model.requires_grad_(False)
    handles = facade.model_adapter.structural_handles(model)
    llm = handles["llm"]
    if llm.config.hidden_size != 3584 or float(getattr(llm.config, "attention_dropout", 0)) != 0:
        raise ValueError("Unexpected 7B hidden size or stochastic decoder")
    runtime = BackboneRuntime(model=model, processor=processor, model_path_or_id=path, device=torch.device(device),
        dtype=torch.bfloat16, revision=identity["revision"], metadata={"checkpoint_identity": identity["identity_sha256"]})
    return {"facade": facade, "runtime": runtime, "model": model, "processor": processor, "llm": llm,
            "vision": handles["vision_encoder"], "device": torch.device(device)}


def save_adapter(module, folder, identity):
    from safetensors.torch import save_file
    from mllm_comp.core.serialization import atomic_write_json
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if module.spec.hidden_size != identity["hidden_size"]:
        raise ValueError("Adapter/base hidden-size mismatch")
    save_file({k:v.detach().cpu().contiguous() for k,v in module.state_dict().items()}, str(folder / "projector.safetensors"))
    atomic_write_json(folder / "projector_config.json", {"format": "qwen25_vl_anchor7b_v1", "spec": asdict(module.spec),
        "implementation_hash": implementation_hash(), "base_identity_sha256": identity["identity_sha256"]})


def load_adapter(folder, identity, device):
    from safetensors.torch import load_file
    folder = Path(folder)
    config = json.loads((folder / "projector_config.json").read_text())
    if config.get("format") != "qwen25_vl_anchor7b_v1" or config["implementation_hash"] != implementation_hash():
        raise ValueError("Not a compatible, implementation-pinned 7B adapter")
    if config["base_identity_sha256"] != identity["identity_sha256"] or config["spec"]["hidden_size"] != identity["hidden_size"]:
        raise ValueError("Adapter belongs to another base checkpoint/dimension")
    module = create_anchor_adapter(AnchorAdapterSpec(**config["spec"]))
    module.load_state_dict(load_file(str(folder / "projector.safetensors")), strict=True)
    return module.to(device)


def global_batch_indices(step, *, world_size, accumulation, rank):
    if world_size < 1 or accumulation < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed batch configuration")
    start = step * world_size * accumulation
    return [start + micro * world_size + rank for micro in range(accumulation)]


def explicit_generate(bundle, state, *, max_new):
    """Same explicit greedy protocol as compressed decoding, no inherited 1.05 penalty."""
    from transformers import GenerationConfig
    model, processor = bundle["model"], bundle["processor"]
    config = GenerationConfig(max_new_tokens=max_new, do_sample=False, use_cache=True, repetition_penalty=1.,
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id)
    return model.generate(**state["prompt_inputs"], generation_config=config)[0, state["prefix"]:].tolist()
