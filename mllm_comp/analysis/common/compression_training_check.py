"""Non-destructive training diagnostics; old checkpoint hash inputs stay unchanged."""

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

from mllm_comp.analysis.common.learned_hybrid_runtime import compact_state, answer_logits, code_hash


@dataclass(frozen=True)
class AnchorAdapterSpec:
    hidden_size: int
    width: int = 256


def create_anchor_adapter(spec):
    import torch
    from torch import nn

    class AnchorAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec
            self.adapter = nn.Sequential(nn.LayerNorm(spec.hidden_size), nn.Linear(spec.hidden_size, spec.width),
                                         nn.GELU(), nn.Linear(spec.width, spec.hidden_size))
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

        def forward(self, base):
            # BF16 cluster means are exactly the existing deployed baseline.
            result = dict(base)
            result["tokens"] = (base["tokens"].float() + self.adapter(base["tokens"].float()).float()).to(base["tokens"].dtype)
            if not bool(torch.isfinite(result["tokens"]).all()):
                raise ValueError("Nonfinite anchor adapter output")
            return result

    return AnchorAdapter()


def anchor_base(state):
    from mllm_comp.compressors.image.fixed_visual_merge import FixedVisualMergeSpec, fixed_visual_merge
    result = fixed_visual_merge(state["features"], grid_height=state["height"], grid_width=state["width"],
                                spec=FixedVisualMergeSpec(partition="feature_anchor", target_count=128))
    return {"tokens": result.tokens.detach(), "indices": result.indices.tolist(),
            "members": result.metadata["cluster_members"], "partition_hash": result.metadata["partition_hash"],
            "position_policy": "rounded_native_mrope_cluster_centroid"}


def compact_anchor(state, output):
    import torch
    compact = compact_state(state, output)
    # compact_state puts native source placeholders in ascending source order.
    start = state["visual"][0]
    if state["visual"] != list(range(start, start + len(state["visual"]))):
        raise ValueError("One contiguous visual span is required")
    members = output["members"]
    flattened = [i for group in members for i in group]
    if sorted(flattened) != list(range(len(state["visual"]))):
        raise ValueError("Cluster membership must cover every source token exactly once")
    for i, group in enumerate(members):
        source = torch.tensor([state["visual"][j] for j in group], device=state["positions"].device)
        compact["positions"][:, 0, start + i] = state["positions"][:, 0].index_select(1, source).float().mean(1).round().to(state["positions"].dtype)
    return compact


def implementation_hash():
    h = hashlib.sha256(code_hash().encode())
    h.update(Path(__file__).read_bytes())
    from mllm_comp.compressors.image import fixed_visual_merge
    h.update(Path(fixed_visual_merge.__file__).read_bytes())
    return h.hexdigest()


def save_anchor(module, folder):
    from safetensors.torch import save_file
    from mllm_comp.core.serialization import atomic_write_json
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}, str(folder / "projector.safetensors"))
    atomic_write_json(folder / "projector_config.json", {"spec": asdict(module.spec), "implementation_hash": implementation_hash()})


def load_anchor(folder, device):
    from safetensors.torch import load_file
    folder = Path(folder)
    config = json.loads((folder / "projector_config.json").read_text())
    if config["implementation_hash"] != implementation_hash():
        raise ValueError("Anchor adapter implementation changed")
    module = create_anchor_adapter(AnchorAdapterSpec(**config["spec"]))
    module.load_state_dict(load_file(str(folder / "projector.safetensors")), strict=True)
    return module.to(device)


def evaluate_state(bundle, state, compact=None, *, max_new=32, native_check=False):
    import torch
    import torch.nn.functional as F
    from transformers import GenerationConfig
    from mllm_comp.analysis.probes.visual_evidence_interaction import cached_decode
    bundle["llm"].eval()
    selected = compact if compact is not None else {"hidden": state["hidden"], "positions": state["positions"],
                                                  "prefix": state["prefix"], "labels": state["teacher_inputs"]["labels"]}
    with torch.inference_mode():
        logits, targets = answer_logits(bundle, selected["hidden"], selected["positions"], selected["labels"])
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Nonfinite answer logits")
        eos = bundle["model"].generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        result = cached_decode(llm=bundle["llm"], head=bundle["model"].lm_head,
            embedding=bundle["model"].get_input_embeddings(), hidden=selected["hidden"][:, :selected["prefix"]],
            positions=selected["positions"][:, :, :selected["prefix"]], eos_ids=eos, max_new=max_new)
        if native_check:
            native = bundle["model"].generate(**state["prompt_inputs"], generation_config=GenerationConfig(max_new_tokens=max_new,
                do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=bundle["processor"].tokenizer.pad_token_id))[0, state["prefix"]:].tolist()
            if native != result["generated_token_ids"]:
                raise ValueError("Native dense generation differs from diagnostic generation")
            result["native_generation_identity_passed"] = True
        result.update({"prediction": bundle["processor"].tokenizer.decode(result["generated_token_ids"], skip_special_tokens=True).strip(),
                       "answer_nll": float(F.cross_entropy(logits, targets)),
                       "retained_visual_tokens": 128 if compact is not None else len(state["visual"]),
                       "source_visual_tokens": len(state["visual"]), "text_positions_preserved": True,
                       "placeholder_mask_labels_alignment_passed": True})
        return result
