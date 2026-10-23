"""Generate native CLI configs, not an independent evaluation/data loop."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.eval.lmms_eval import run_projector_ratio_platform as reference
from mllm_comp.core.serialization import atomic_write_json, atomic_write_text

SOURCES = {
    "eadp": {"url": "https://github.com/SJTU-DeepVisionLab/EADP", "commit": "e1a08801461c181871b9ff1cb0803b8e9966f083",
        "adaptation": "Official Qwen selector; beta=2; retain native rectangular resolution, not fixed 1008x1008"},
    "prunesid": {"url": "https://github.com/ZhengyaoFang/PruneSID", "commit": "4ef17d96b390027f31243bc7e1d27bdee77792a5",
        "adaptation": "Qwen2 selector on Qwen2.5; local seeded PCA; exact K; original source order and MRoPE, not official ratio-to-576 budget/grouped order"},
    "mmtok": {"url": "https://github.com/Ironieser/MMTok", "commit": "8e2686836ed5bbe05d064f1d1c20e50c73c70b94",
        "adaptation": "Official Qwen tau_t=.01, Question-prefix user text, mean pre-merger features; fixed K rather than original .2 ratio"},
}


def make_config(method, phase, output):
    count = 2048 if phase == "identity" else 128
    cfg = reference.config(output, method, "smoke" if phase != "full" else "full", target_count=count)
    stage = cfg["pipeline"]["stages"][0]
    stage["config"] = ({"target_count": count, "alpha": .5, "text_temperature": .01, "vision_temperature": .2}
        if method == "mmtok" else {"tokens_per_crop": count,
            **({"alpha": .5, "beta": 2.} if method == "eadp" else {"pca_seed": 0})})
    cfg["experiment"]["name"] = f"{phase}_{method}_k{count}"
    path = output / phase / method
    cfg["output"] = {"run_dir": str(path), "manifest_path": str(path / "manifest.json")}
    return cfg


def main():
    import yaml
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs/eval/lmms_eval/projector_trainfree_20261023")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/eval/qwen25_vl/projector_trainfree_20261023")
    args = parser.parse_args()
    for path in (args.config_dir, args.output):
        if not path.resolve().is_relative_to(Path("/home/ZhouFengyu").resolve()):
            raise ValueError("Generated artifacts must remain inside the owner's directory")
    args.config_dir.mkdir(parents=True, exist_ok=True)
    configs = {}
    for method in SOURCES:
        for phase in ("identity", "smoke", "full"):
            path = args.config_dir / f"{phase}_{method}.yaml"
            if path.exists():
                raise ValueError(f"Refusing to overwrite existing experiment config: {path}")
            content = yaml.safe_dump(make_config(method, phase, args.output), allow_unicode=True, sort_keys=False)
            atomic_write_text(path, content)
            configs[str(path)] = hashlib.sha256(content.encode()).hexdigest()
    atomic_write_json(args.config_dir / "protocol.json", {
        "status": "prepared_not_launched", "profile": "projector", "training": False,
        "sources": SOURCES, "configs": configs, "native_resolution": True,
        "budget": "K_i=min(128,N_i); no duplicated tokens or per-layer average budget",
        "identity_budget": "2048: processor upper bound, therefore no compression",
        "tasks": reference.TASKS, "entrypoint": "python -m mllm_comp.cli.main eval --config <yaml>",
        "gate": "Run identity then smoke before full; CPU gates do not establish paper quality"})
    print(json.dumps({"prepared": len(configs), "config_dir": str(args.config_dir)}, indent=2))


if __name__ == "__main__":
    main()
