"""Recoverable source-only cleanup of legacy audio/video experiment launchers.

Core package, tests, licenses, outputs and frozen worktrees are never targets.
Retain candidates referenced by the remaining code/tests/configs to a fixed point.
All moved files have an explicit SHA256 and a same-owner archive destination.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PREFIXES=("scripts/analysis/audio_","scripts/analysis/silent_redundancy_mmsu/",
    "scripts/train/qwen2_5_omni/","scripts/eval/lmms_eval/lmu/")


def candidate(name):
    return any(name.startswith(p) for p in PREFIXES) or (
        name.startswith("scripts/eval/lmms_eval/qwen25_omni_") and name.endswith(".sh"))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(root=ROOT):
    names=subprocess.check_output(["git","ls-files"],cwd=root,text=True).splitlines()
    proposed={x for x in names if candidate(x)}
    counts=Counter(Path(x).name for x in names)
    texts={x:(root/x).read_text(errors="replace") for x in names
        if Path(x).parts[0] in {"mllm_comp","scripts","tests","configs","examples"}}
    protected={}
    while True:
        changed=False
        remaining=[x for x in texts if x not in proposed]
        for name in sorted(proposed):
            module=str(Path(name).with_suffix("")).replace("/",".")
            tokens=[name,module]
            if counts[Path(name).name]==1:tokens.append(Path(name).name)
            incoming=next((x for x in remaining if any(t in texts[x] for t in tokens)),None)
            if incoming:
                proposed.remove(name);protected[name]=incoming;changed=True
        if not changed:break
    rows=[]
    for name in sorted(proposed):
        p=root/name
        if not p.is_file() or p.is_symlink() or p.stat().st_uid!=os.getuid():raise ValueError(f"Unsafe cleanup target: {name}")
        rows.append({"path":name,"sha256":digest(p),"bytes":p.stat().st_size})
    return {"root":str(root),"previous_files":len(names),"files":rows,"move_count":len(rows),
        "source_bytes":sum(x["bytes"] for x in rows),"protected_dependencies":protected,
        "scope":"legacy optional audio launchers/configs only; no core/tests/outputs/checkpoints/worktrees"}


def apply(saved,archive):
    root=Path(saved["root"]).resolve();archive=archive.resolve()
    owner_root=Path("/home/ZhouFengyu/fastv-repro/archive")
    if root!=ROOT.resolve():raise ValueError("Cleanup root must be this repository")
    if not archive.is_relative_to(owner_root) or archive==owner_root or archive.exists():raise ValueError("Use a fresh, owner-local archive leaf")
    # Validate ALL targets before the first move.
    for row in saved["files"]:
        p=root/row["path"]
        if not candidate(row["path"]) or p.is_symlink() or p.stat().st_uid!=os.getuid() or digest(p)!=row["sha256"]:
            raise ValueError(f"Plan changed or unsafe target: {row['path']}")
    archive.mkdir(parents=True)
    journal={**saved,"archive":str(archive),"status":"moving","moved":[]}
    def save():
        from mllm_comp.core.serialization import atomic_write_json
        atomic_write_json(archive/"manifest.json",journal)
    save()
    try:
        for row in saved["files"]:
            source=root/row["path"];destination=archive/row["path"]
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(destination))
            if digest(destination)!=row["sha256"]:raise ValueError("Archived checksum mismatch")
            journal["moved"].append(row["path"]);save()
        journal["status"]="complete";save()
    except BaseException:
        journal["status"]="failed";save();raise
    return {"status":"complete","moved":len(journal["moved"]),"archive":str(archive)}


if __name__=="__main__":
    import sys
    sys.path.insert(0,str(ROOT))
    parser=argparse.ArgumentParser();parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--apply",action="store_true");parser.add_argument("--archive",type=Path)
    args=parser.parse_args()
    if args.apply:print(json.dumps(apply(json.loads(args.plan.read_text()),args.archive)))
    else:
        from mllm_comp.core.serialization import atomic_write_json
        result=plan();atomic_write_json(args.plan,result)
        print(json.dumps({k:v for k,v in result.items() if k not in {"files","protected_dependencies"}}))
        print(json.dumps({"dependency_protected_count":len(result["protected_dependencies"])}))
