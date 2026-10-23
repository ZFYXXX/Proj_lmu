"""Resumable condition ledgers, importable by installed analysis entrypoints."""
import json
import os
from mllm_comp.core.serialization import to_jsonable


def append_json(path, row):
    encoded = json.dumps(to_jsonable(row), ensure_ascii=False, allow_nan=False)
    with path.open("a") as stream:
        stream.write(encoded+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def records(path, config_hash):
    result = {}
    if not path.exists():
        return result
    with path.open("rb+") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                break
            row = json.loads(line)
            key = (row["sample_id"], row["variant"])
            if row["config_hash"] != config_hash or key in result:
                raise ValueError("Mixed configuration or duplicate ledger condition")
            result[key] = row
    return result
