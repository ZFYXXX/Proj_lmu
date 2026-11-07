"""Keep one platform model per worker across native lmms-eval CLI calls.

Only model construction is cached. The platform adapter, native task loading,
generation, metrics, prediction logging and result collection remain in use.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch


def model_key(spec):
    config = spec["effective_config"]
    identity = {name: spec.get(name) for name in (
        "checkpoint_path", "backbone_name", "compression_mode", "runtime_mode",
        "request_mode", "generation", "system_prompt", "use_audio_in_video")}
    identity.update(backbone=config["backbone"], pipeline=config["pipeline"])
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class ResidentEvaluator:
    def __init__(self, state_path):
        self.state_path = Path(state_path)
        self.model = None
        self.key = None
        self.loads = 0
        self.reuses = 0
        self.load_seconds = 0.

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None
            self.key = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_model(self, spec, spec_path, seeds):
        key = model_key(spec)
        if key != self.key:
            self.close()
            import numpy as np
            import torch
            from mllm_comp.analysis.common.learned_topk_experiment import shared_policy
            policy = shared_policy()
            if policy is not None and torch.cuda.is_available():
                # Apply before constructing the model, not just before the
                # first generation, so loading also obeys the allocator cap.
                torch.cuda.set_per_process_memory_fraction(policy['allocator_fraction'], 0)
            from mllm_comp.integrations.lmms_eval.plugin import MLLMCompLMM
            for seed, setter in zip(seeds, (random.seed, np.random.seed, torch.manual_seed)):
                if seed is not None:
                    setter(seed)
            start = time.monotonic()
            self.model = MLLMCompLMM(pretrained=spec["checkpoint_path"],
                mllm_comp_spec=str(spec_path), batch_size=1, device="cuda:0")
            self.load_seconds += time.monotonic() - start
            self.key = key
            self.loads += 1
            print(f"MODEL_LOAD {key[:12]} loads={self.loads}", flush=True)
        else:
            from mllm_comp.integrations.lmms_eval.model_bridge import LMMSEvalRequestBridge
            previous = self.model._platform_bridge
            # A fresh request bridge isolates task_dict, manifests and traces;
            # the installed compressor and base model remain resident.
            self.model._platform_bridge = LMMSEvalRequestBridge(
                facade=previous.facade, runtime=previous.runtime,
                system_prompt=spec.get("system_prompt"),
                use_audio_in_video=bool(spec.get("use_audio_in_video")),
                generation_overrides=spec.get("generation"),
                diagnostic_trace_path=spec.get("diagnostic_trace_path"),
                compression_manifest_path=spec.get("compression_manifest_path"),
                request_mode=spec.get("request_mode", "generate"))
            self.model.task_dict = {}
            self.model._install_grounding_scorer()
            self.reuses += 1
            print(f"MODEL_REUSE {key[:12]} reuses={self.reuses}", flush=True)
        from scripts.eval.lmms_eval.qwen25_portable_queue import dump
        dump(self.state_path, {"model_loads": self.loads, "model_reuses": self.reuses,
            "model_load_seconds": self.load_seconds, "model_key": key,
            "pid": os.getpid(), "updated_at": time.time()})
        return self.model

    def run_native(self, plan):
        from lmms_eval import __main__ as cli, evaluator
        from mllm_comp.integrations.native_cli import NativeCLIResult
        os.environ.update(plan.environment)
        for name in plan.unset_environment:
            os.environ.pop(name, None)
        with patch.object(sys, "argv", ["lmms_eval", *plan.lmms_eval_argv[3:]]):
            _, args = cli.parse_eval_args()
        owner = self
        native_get_model = evaluator.models.get_model

        class ResidentFactory:
            @classmethod
            def create_from_arg_string(cls, model_args, additional_config):
                return owner.get_model(plan.plugin_spec, plan.plugin_spec_path, args.seed[:3])

        def get_model(name, *positional, **kwargs):
            return ResidentFactory if name == args.model else native_get_model(name, *positional, **kwargs)

        # Keep the native model ID string in result metadata. Passing the model
        # object into simple_evaluate would serialize its compressor ContextVars.
        with patch.object(evaluator.models, "get_model", get_model):
            result = cli.cli_evaluate_single(args)
        if not result or result[0] is None:
            raise RuntimeError("Native lmms-eval returned no result")
        return NativeCLIResult(returncode=0)

    def evaluate(self, attempt, manifest, log_path):
        from mllm_comp.config.loader import load_config
        from mllm_comp.integrations.lmms_eval.adapter import LMMSEvalAdapter
        from scripts.eval.lmms_eval.p6000_dataset_smoke import finish_evaluation, save
        manifest["status"] = "running"
        save(attempt / "manifest.json", manifest)
        os.environ["MLLM_EVAL_OUTPUT_ROOT"] = str(attempt.resolve())
        os.environ["MLLM_TOPK_PERFORMANCE"] = str(attempt / "request_performance.jsonl")
        # Preserve the same platform launch plan and native CLI with an in-process
        # command runner, so chunk boundaries no longer reload model weights.
        original_stderr = sys.stderr
        with log_path.open("x") as stream, redirect_stdout(stream), redirect_stderr(stream):
            from loguru import logger
            logger.remove()
            logger.add(stream, level="INFO")
            try:
                config = load_config(attempt / "eval.yaml")
                LMMSEvalAdapter(command_runner=self.run_native).evaluate(
                    config=config, checkpoint_path=manifest["model"], device="cuda:0",
                    compressed=manifest["method"] != "dense")
            finally:
                # Native evaluator may replace all loguru handlers itself.
                logger.remove()
                logger.add(original_stderr, level="INFO")
        if not finish_evaluation(attempt, manifest, 0):
            raise RuntimeError(f"Native evaluation is incomplete: {attempt}")
