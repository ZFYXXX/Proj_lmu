"""Single catalog for portable method aliases, options and checkpoint roles."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "configs/workflow/methods.json"


def load_tasks():
    return json.loads((ROOT / "configs/workflow/tasks.json").read_text())["tasks"]


def load_catalog(path=None):
    payload = json.loads(Path(path or DEFAULT_CATALOG).read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("methods"), dict):
        raise ValueError("Method catalog needs schema_version=1 and methods")
    methods = payload["methods"]
    if "dense" not in methods:
        raise ValueError("Method catalog must include dense")
    for name, spec in methods.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not isinstance(spec, dict):
            raise ValueError(f"Invalid method catalog entry: {name}")
        if name != "dense" and not spec.get("algorithm"):
            raise ValueError(f"Method {name} needs a registered algorithm")
        if not isinstance(spec.get("options", {}), dict):
            raise ValueError(f"Method {name} options must be an object")
    return methods


def method_options(name, *, method_checkpoint=None, scorer_checkpoint=None, identity_path=None, catalog_path=None):
    spec = load_catalog(catalog_path)[name]
    values = {"method_checkpoint": method_checkpoint, "scorer_checkpoint": scorer_checkpoint,
              "identity_path": identity_path}

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            key = value[2:-1]
            if not values.get(key):
                raise ValueError(f"Method {name} requires {key}")
            return str(values[key])
        return value

    return spec, expand(copy.deepcopy(spec.get("options", {})))
