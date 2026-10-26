"""Audit/export an explicit source-only snapshot; never read model or data roots.

The export is JSON on stdout for the connected GitHub Git-data API. Credentials
are neither requested nor read. Repository writes happen separately via git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess

ROOT=Path(__file__).resolve().parents[2]
DIRECTORIES={"mllm_comp","configs","scripts","tests","docs","examples","patches","requirements","constraints"}
ROOT_FILES={".gitignore",".gitmodules","LICENSE","README.md","pyproject.toml","third_party/README.md"}
SECRET=re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{32,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
FORBIDDEN={".env",".envrc","credentials.json","secrets.json","id_rsa","id_ed25519"}
SUFFIXES={".safetensors",".pt",".pth",".bin",".parquet",".arrow",".pem",".key",".zip",".gz",".pyc"}


def allowed(name):
    p=Path(name)
    return (name in ROOT_FILES or p.parts[0] in DIRECTORIES) and not (
        p.name in FORBIDDEN or p.suffix in SUFFIXES or "__pycache__" in p.parts)


def candidates():
    raw=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard","-z"],cwd=ROOT)
    return sorted(set(x.decode() for x in raw.split(b"\0") if x))


def audit():
    rows=[];errors=[]
    for name in candidates():
        if not allowed(name):errors.append({"file":name,"reason":"outside_source_allowlist"});continue
        p=ROOT/name
        if not p.exists() or p.is_symlink():errors.append({"file":name,"reason":"missing_or_symlink"});continue
        if not stat.S_ISREG(p.stat().st_mode):errors.append({"file":name,"reason":"not_regular"});continue
        raw=p.read_bytes()
        try:text=raw.decode("utf-8")
        except UnicodeDecodeError:errors.append({"file":name,"reason":"non_utf8"});continue
        if SECRET.search(text):errors.append({"file":name,"reason":"possible_credential"});continue
        if len(raw)>2*1024**2:errors.append({"file":name,"reason":"unexpected_large_source"});continue
        rows.append({"path":name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
    return {"root":str(ROOT),"files":rows,"file_count":len(rows),"total_bytes":sum(r["bytes"] for r in rows),"errors":errors}


def export(commit,batch,byte_limit):
    listing=subprocess.check_output(["git","ls-tree","-r","-z",commit],cwd=ROOT)
    objects=[]
    for item in listing.split(b"\0"):
        if item:
            info,name=item.split(b"\t",1);mode,kind,sha=info.decode().split()
            if kind!="blob" or not allowed(name.decode()):raise ValueError(f"Unexpected committed path: {name.decode()}")
            objects.append((mode,sha,name.decode()))
    # One Git process, rather than one process per file per export batch.
    payload=subprocess.check_output(["git","cat-file","--batch"],cwd=ROOT,
        input=("\n".join(sha for _,sha,_ in objects)+"\n").encode())
    cursor=0
    chunks=[[]];size=0
    for mode,sha,name in objects:
        end=payload.index(b"\n",cursor)
        actual,kind,length=payload[cursor:end].decode().split()
        if actual!=sha or kind!="blob":raise ValueError("Git export object mismatch")
        cursor=end+1;length=int(length)
        raw=payload[cursor:cursor+length];cursor+=length+1
        content=raw.decode("utf-8")
        if SECRET.search(content):raise ValueError(f"Credential-pattern match: {name}")
        entry={"path":name,"mode":mode,"type":"blob","content":content}
        count=len(json.dumps(entry,ensure_ascii=True))
        if chunks[-1] and size+count>byte_limit:chunks.append([]);size=0
        chunks[-1].append(entry);size+=count
    if batch is None:return {"batches":len(chunks),"files":sum(map(len,chunks)),"commit":commit}
    return {"batch":batch,"batches":len(chunks),"entries":chunks[batch]}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("audit","export"))
    parser.add_argument("--commit",default="HEAD");parser.add_argument("--batch",type=int)
    parser.add_argument("--byte-limit",type=int,default=65000);args=parser.parse_args()
    result=audit() if args.command=="audit" else export(args.commit,args.batch,args.byte_limit)
    print(json.dumps(result,ensure_ascii=True))
    if result.get("errors"):raise SystemExit(1)
