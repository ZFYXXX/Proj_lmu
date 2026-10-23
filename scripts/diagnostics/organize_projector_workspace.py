"""Owner-scoped, manifest-backed archival; dry-run unless --apply is supplied.

Never removes data, model weights, experiment outputs, environments or source
repositories. Symlinks are never followed. Restore only to unoccupied paths.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OWNER = Path("/home/ZhouFengyu")
WORKSPACE = OWNER / "fastv-repro"
ARCHIVE = WORKSPACE / "archive/projector_cleanup_20261023"
sys.path.insert(0, str(ROOT))


def guarded(path, *, must_exist=True):
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(WORKSPACE):
        raise ValueError(f"Out-of-scope path: {path}")
    if not path.resolve().is_relative_to(WORKSPACE.resolve()):
        raise ValueError(f"Out-of-scope symlink target: {path}")
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(WORKSPACE)):
        raise ValueError(f"Refusing symlink: {path}")
    if must_exist and path.stat().st_uid != os.getuid():
        raise ValueError(f"Not owned by this user: {path}")
    if must_exist and path.is_dir():
        for item in path.rglob("*"):
            if item.is_symlink() or item.stat().st_uid != os.getuid():
                raise ValueError(f"Mixed ownership or symlink under archive candidate: {item}")
    return path


def candidates():
    import yaml
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    full = builtin_multimodal_compression_contracts(profile="full")
    active = builtin_multimodal_compression_contracts(profile="projector")
    active_bindings = {b.name for b in active.bindings()}
    known = {a.name for a in full.algorithms()}
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text())
        except (ValueError, yaml.YAMLError):
            continue
        if not isinstance(cfg, dict):
            continue
        stages = (cfg.get("pipeline") or {}).get("stages", [])
        inactive = [stage for stage in stages if isinstance(stage, dict)
            and stage.get("enabled", True) and stage.get("algorithm") in known
            and (stage.get("placement") in {"llm_level", "encoder_level"}
                or stage.get("binding") and stage["binding"] not in active_bindings)]
        if inactive:
            yield path, "config outside active projector-only profile"
    # Only old, clearly named modification snapshots / patch transports, not
    # full results or general-purpose source checkouts.
    for path in sorted(WORKSPACE.iterdir()):
        if path.name.startswith("qwen38-") and "-prechange-" in path.name:
            yield path, "historical pre-change source backup"
        elif path.is_file() and path.name.endswith(".tar.gz") and (
            "patch" in path.name or "runner" in path.name or "full-sequence" in path.name):
            yield path, "historical patch transport archive"


def write_manifest(path, data):
    # Operational ledger, not source editing; replace atomically after each move.
    from mllm_comp.core.serialization import atomic_write_json
    atomic_write_json(path, data)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    if args.restore:
        ledger = guarded(args.restore)
        data = json.loads(ledger.read_text())
        for row in reversed(data["moves"]):
            source, destination = Path(row["source"]), Path(row["destination"])
            if destination.exists():
                guarded(destination)
                guarded(source, must_exist=False)
                if source.exists():
                    raise ValueError(f"Restore would overwrite: {source}")
                if args.apply:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(source))
        print("Restored" if args.apply else "Restore dry-run passed")
        return
    rows = [{"source": str(guarded(path)),
        "destination": str(ARCHIVE / "items" / path.relative_to(WORKSPACE)),
        "reason": reason, "status": "planned"} for path, reason in candidates()]
    print(json.dumps({"count": len(rows), "moves": rows}, indent=2))
    if not args.apply:
        return
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    manifest = ARCHIVE / "moves.json"
    if manifest.exists():
        raise ValueError("Existing archive manifest; refusing to overwrite it")
    data = {"created_utc": datetime.now(timezone.utc).isoformat(), "owner_scope": str(OWNER), "moves": rows}
    write_manifest(manifest, data)
    for row in rows:
        destination = guarded(Path(row["destination"]), must_exist=False)
        if destination.exists():
            raise ValueError(f"Archive destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(row["source"], str(destination))
        row["status"] = "archived"
        write_manifest(manifest, data)


if __name__ == "__main__":
    main()
