"""Score exported HallusionBench records with the upstream external judge.

This sends questions, reference answers and predictions to the configured API.
It is never invoked by the GPU evaluation queue.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("judge_input", type=Path)
    parser.add_argument("--confirm-external-judge", action="store_true")
    parser.add_argument("--allow-limited", action="store_true")
    args = parser.parse_args()
    if not args.confirm_external_judge:
        parser.error("Scoring sends data to an external API; pass --confirm-external-judge explicitly")
    api_type = os.environ.get("API_TYPE", "openai")
    credential = "AZURE_API_KEY" if api_type == "azure" else "OPENAI_API_KEY"
    if api_type not in {"openai", "azure"} or not os.environ.get(credential):
        parser.error("Configure the upstream API_TYPE and corresponding API key")
    source = args.judge_input.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["limited"] and not args.allow_limited:
        parser.error("Limited smoke input requires --allow-limited and is not a benchmark result")
    import requests
    from lmms_eval.tasks.hallusion_bench import evaluate_hb
    from lmms_eval.tasks.hallusion_bench import utils as judge_utils

    class JudgeFailure(BaseException):
        pass

    original_post = requests.post
    def checked_post(*values, **kwargs):
        try:
            response = original_post(*values, **kwargs)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty judge response")
            return response
        except Exception as exc:
            # Upstream silently converts network failures into an unclear label.
            # Stop instead, preserving prior per-request caches for inspection.
            raise JudgeFailure(f"External judge request failed: {type(exc).__name__}") from exc

    destination = source.parent / "hallusion_official_scores"
    destination.mkdir(exist_ok=True)
    for method in manifest["methods"]:
        input_path = source / f"{method}.jsonl"
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
        out = destination / method
        out.mkdir(exist_ok=True)
        identity_path = out / "input_identity.json"
        identity = {"sha256": digest, "api_type": api_type,
                    "endpoint_sha256": hashlib.sha256(judge_utils.API_URL.encode()).hexdigest(),
                    "judge_model": "gpt-4 (upstream default)" if api_type == "openai" else "configured Azure deployment"}
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError(f"{method}: judge cache belongs to different inputs or configuration")
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
        records = [json.loads(line)["record"] for line in input_path.read_text().splitlines() if line.strip()]
        if not records:
            raise ValueError(f"{method}: empty judge input")
        official_args = SimpleNamespace(output_path=str(out))
        try:
            requests.post = checked_post
            scores = {name: float(getattr(evaluate_hb, f"hb_aggregation_result_{name}")(records, official_args))
                      for name in ("aAcc", "qAcc", "fAcc")}
        except JudgeFailure as exc:
            raise SystemExit(str(exc))
        finally:
            requests.post = original_post
        (out / "scores.json").write_text(json.dumps({"status": "scored", "scores": scores,
            "samples": len(records), "limited": manifest["limited"], **identity}, indent=2) + "\n")
        print(f"Scored {method}: {scores}")


if __name__ == "__main__":
    main()
