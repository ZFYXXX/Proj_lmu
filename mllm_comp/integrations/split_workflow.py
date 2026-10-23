"""Small, explicit HF SFT / lmms-eval workflow for modality split.

This is not a claim of compatibility with arbitrary third-party trainers.
The wrapper is installed before optimizer creation and saved with native weights.
"""
from __future__ import annotations

import json
from pathlib import Path

IMAGE_TASKS = (
    "gqa", "vqav2_val", "ok_vqa_val2014", "textvqa_val", "docvqa_val", "ocrbench",
    "scienceqa_img", "ai2d", "mmstar", "refcoco_bbox_rec_testA", "refcoco_bbox_rec_testB",
    "refcoco+_bbox_rec_testA", "refcoco+_bbox_rec_testB", "refcocog_bbox_rec_test",
)


def reusable_training_checkpoint(directory, config):
    directory = Path(directory)
    evidence_path = directory / "sft_evidence.json"
    if not evidence_path.is_file():
        return False
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("effective_config") != config:
        raise ValueError("Training configuration changed; use a new output directory instead of reusing old weights.")
    return evidence.get("status") == "trained_pending_reload" and all(
        (directory / name).is_file() for name in ("modality_split.pt", "reload_probe.pt"))


def assistant_labels(input_ids, prompt_ids, attention_mask):
    import torch
    length = prompt_ids.shape[1]
    if length >= input_ids.shape[1]:
        raise ValueError("No assistant target tokens.")
    if not torch.equal(input_ids[:, :length], prompt_ids):
        raise ValueError("Full conversation does not preserve the prompt prefix.")
    labels = input_ids.clone()
    labels[:, :length] = -100
    labels[~attention_mask.bool()] = -100
    if not (labels[:, 1:] != -100).any():
        raise ValueError("No shifted assistant target tokens.")
    return labels


def load_model_processor(backbone, checkpoint, *, dtype="bfloat16", device="cuda:0", max_pixels=200704):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
    cls = {"qwen25_vl": Qwen2_5_VLForConditionalGeneration,
           "qwen3_vl": Qwen3VLForConditionalGeneration}[backbone]
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True,
                    min_pixels=3136, max_pixels=max_pixels)
    model = cls.from_pretrained(checkpoint, local_files_only=True,
                    torch_dtype=getattr(torch, dtype), attn_implementation="sdpa",
                    device_map=device)
    return model, processor


def sft_batch(processor, record, device):
    from PIL import Image
    image = Image.open(record["image"]).convert("RGB")
    prompt = [{"role": "user", "content": [{"type": "image", "image": image},
              {"type": "text", "text": record["question"]}]}]
    full = prompt + [{"role": "assistant", "content": [{"type": "text", "text": record["answer"]}]}]
    p = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    f = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    prompt_inputs = processor(text=[p], images=[image], return_tensors="pt")
    inputs = processor(text=[f], images=[image], return_tensors="pt")
    inputs["labels"] = assistant_labels(inputs.input_ids, prompt_inputs.input_ids, inputs.attention_mask)
    return {key: value.to(device) for key, value in inputs.items()}


def eval_plugin_spec(backbone, checkpoint, output, *, limit=2, split_spec=None, max_pixels=200704):
    checkpoint = str(Path(checkpoint).resolve())
    output = str(Path(output).resolve())
    backbone_config = {"torch_dtype": "bfloat16", "attn_implementation": "sdpa",
         "model_load_kwargs": {"local_files_only": True},
         "processor_load_kwargs": {"local_files_only": True, "min_pixels": 3136, "max_pixels": max_pixels}}
    if split_spec is None:
        if not (Path(checkpoint) / "modality_split.json").is_file():
            raise FileNotFoundError("Trained checkpoint lacks modality_split.json")
        backbone_config["modality_split_checkpoint"] = checkpoint
    else:
        backbone_config["modality_split"] = split_spec
    cfg = {"experiment": {"name": "modality_split_smoke", "seed": 42},
       "backbone": {"name": backbone, "model_path": checkpoint, "config": backbone_config},
       "pipeline": {"stages": []}, "analysis": {}, "train": {},
       "eval": {"backend": "lmms_eval", "compression_mode": "compressed", "tasks": list(IMAGE_TASKS),
                "runtime_batch_size": 1, "sample_limit": limit},
       "data": {}, "output": {"run_dir": output, "manifest_path": output + "/manifest.json"}}
    return {"schema_version": 2, "effective_config": cfg, "backbone_name": backbone,
        "checkpoint_path": checkpoint, "compression_mode": "compressed", "device": "cuda:0",
        "expected_world_size": 1, "request_mode": "generate", "generation": {},
        "compression_manifest_path": output + "/split_manifest.json",
        "diagnostic_trace_path": output + "/requests.jsonl"}
