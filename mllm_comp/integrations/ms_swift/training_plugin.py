"""ms-swift external plugin for real dynamic-compression training forwards."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys
from threading import Lock
from types import MethodType, SimpleNamespace
from typing import Any

from mllm_comp.backbones.base import BackboneRuntime, ProcessedBatch
from mllm_comp.backbones.qwen25_omni.v2_compression import (
    resolve_qwen25_omni_stage_execution,
)
from mllm_comp.backbones.qwen25_omni.implementation import (
    _apply_debug_compression_reports,
    _capture_training_forward_timing,
    _prepare_runtime_compression,
    _runtime_compression_report,
)
from mllm_comp.backbones.qwen25_omni.token_debug import (
    Qwen25OmniTokenDebugCollector,
    prepare_qwen25_omni_semantic_labels,
)
from mllm_comp.backbones.qwen25_omni.training_compression import (
    build_qwen25_omni_training_state,
    model_inputs_from_forward_call,
)
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.serialization import to_jsonable
from mllm_comp.core.vocabulary import CompressionMode, ExecutionPhase, RuntimeMode
from mllm_comp.compressors.contracts import CompressionContractRegistry
from mllm_comp.compressors.extensions import compression_contracts
from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher
from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.registry.backbones import get_backbone


_SPEC_ENV = "MLLM_COMP_TRAIN_COMPRESSION_SPEC"
_EVIDENCE_DIR_ENV = "MLLM_COMP_TRAIN_COMPRESSION_EVIDENCE_DIR"
_VSCODE_DEBUG_SUBPROCESSES_ENV = "MLLM_COMP_VSCODE_DEBUG_SUBPROCESSES"
_PATCH_MARKER = "_mllm_comp_training_loader_patch"
_TEMPLATE_PATCH_MARKER = "_mllm_comp_training_template_patch"
_IMAGE_TEMPLATE_PATCH_MARKER = "_mllm_comp_image_training_template_patch"
_OUTPUT_PATCH_MARKER = "_mllm_comp_training_output_patch"
_OUTPUT_RECORDER = "_mllm_comp_training_output_recorder"
_PENDING_FORWARD = "_mllm_comp_pending_training_forward"
_LABEL_CAPTURE = "_mllm_comp_training_label_capture"


def _enable_vscode_debugging_for_current_thread() -> None:
    """Install debugpy tracing on callbacks entered from native worker threads."""
    if os.environ.get(_VSCODE_DEBUG_SUBPROCESSES_ENV) != "1":
        return
    try:
        import debugpy
    except ImportError:
        return
    if debugpy.is_client_connected():
        debugpy.debug_this_thread()


def _load_spec() -> tuple[dict[str, Any], str]:
    encoded = os.environ.get(_SPEC_ENV)
    if not encoded:
        raise BackendIntegrationError(
            f"ms-swift imported the compression plugin without {_SPEC_ENV}."
        )
    try:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise BackendIntegrationError(
            "The ms-swift training compression specification is malformed."
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2, 3, 4, 5}:
        raise BackendIntegrationError(
            "The ms-swift training compression specification has an unsupported schema."
        )
    return payload, hashlib.sha256(raw).hexdigest()


class _EvidenceRecorder:
    def __init__(
        self,
        *,
        spec_hash: str,
        stage: CompressionStageConfig,
        runtime_mode: RuntimeMode,
    ) -> None:
        output_dir = os.environ.get(_EVIDENCE_DIR_ENV)
        if not output_dir:
            raise BackendIntegrationError(
                f"Compressed training requires {_EVIDENCE_DIR_ENV}."
            )
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        if runtime_mode is RuntimeMode.DEBUG:
            self._filename = (
                "debug_diagnostic.jsonl"
                if rank == "0"
                else f"debug_diagnostic_rank_{rank}.jsonl"
            )
        else:
            self._filename = f"compression_training_rank_{rank}.jsonl"
        self.path = Path(output_dir) / self._filename
        self.spec_hash = spec_hash
        self.stage = stage
        self._lock = Lock()
        self._initialized = False
        self._forward_index = 0
        self._default_forward_evidence_recorded: set[str] = set()

    def set_output_dir(self, output_dir: str | os.PathLike[str]) -> None:
        """Point evidence at ms-swift's resolved, versioned output directory."""
        with self._lock:
            if self._initialized:
                raise InvariantError(
                    "Cannot relocate compression training evidence after recording began."
                )
            self.path = Path(output_dir) / self._filename

    def _initialize_locked(self) -> None:
        if self._initialized:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._initialized = True

    def record(
        self, event: str, *, stage: CompressionStageConfig | None = None, **values: Any,
    ) -> None:
        stage = stage or self.stage
        payload = {
            "schema_version": 1,
            "event": event,
            "spec_hash": self.spec_hash,
            "stage_name": stage.name,
            "plugin": stage.plugin,
            "execution_phase": stage.execution_phase.value,
            **values,
        }
        line = json.dumps(
            to_jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
        )
        with self._lock:
            self._initialize_locked()
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    def next_forward_index(self) -> int:
        with self._lock:
            result = self._forward_index
            self._forward_index += 1
            return result

    def claim_default_forward_evidence(self, stage_name: str = "") -> bool:
        with self._lock:
            if stage_name in self._default_forward_evidence_recorded:
                return False
            self._default_forward_evidence_recorded.add(stage_name)
            return True


def _loss_tensor(output: Any) -> Any | None:
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, dict):
        loss = output.get("loss")
    return loss


def _dynamic_records(model: Any) -> tuple[Any, ...]:
    return tuple(
        record for record in model._mllm_comp_training_session.active_records
        if callable(getattr(record.activation, "prepare_state", None))
    )


def _llm_training_controller(model: Any) -> Any | None:
    return next((
        record.activation for record in _dynamic_records(model)
        if record.activation.execution.executor_family == "llm_token"
    ), None)


def _device_from_state(state: Any) -> Any:
    return getattr(state.input_ids, "device", None)


def _processed_batch(state: Any) -> ProcessedBatch:
    return ProcessedBatch(
        model_inputs=dict(state.native_model_inputs),
        sample_ids=list(state.sample_ids or []),
        prompt_lengths=[],
        metadata=dict(state.modality_metadata),
    )


def _requires_semantic_labels(model: Any) -> bool:
    return any(
        "semantic_instruction_mask" in record.activation.execution.required_context
        for record in _dynamic_records(model)
    )


def _start_compressed_forward(
    model: Any,
    original_state: Any,
) -> tuple[Any, dict[str, Any]]:
    runtime = model._mllm_comp_training_runtime
    runtime_mode = model._mllm_comp_training_runtime_mode
    debug_collector = model._mllm_comp_training_debug_collector
    model_training = bool(getattr(model, "training", False))
    processed_batch = None
    if model_training and (
        debug_collector is not None or _requires_semantic_labels(model)
    ):
        processed_batch = _processed_batch(original_state)
    if model_training and _requires_semantic_labels(model):
        prepare_qwen25_omni_semantic_labels(original_state, processed_batch)
    if debug_collector is not None and model_training:
        debug_collector.prepare_sequence_labels(original_state, processed_batch)
    model._mllm_comp_training_session.clear_diagnostic_records()
    active_state, prepared = _prepare_runtime_compression(runtime, original_state)
    stack = ExitStack()
    try:
        timing = stack.enter_context(
            _capture_training_forward_timing(device=_device_from_state(active_state))
            if runtime_mode is RuntimeMode.PROFILE and model_training
            else nullcontext(None)
        )
        if debug_collector is not None and model_training:
            stack.enter_context(debug_collector.request(
                original_state=original_state, active_state=active_state,
                processed_batch=processed_batch,
            ))
        for item in prepared:
            stack.enter_context(item.activation.request(item.plan))
    except BaseException:
        stack.__exit__(*sys.exc_info())
        raise
    return active_state, {
        "stack": stack, "active_state": active_state, "prepared": prepared,
        "model_training": model_training, "timing": timing,
    }


def _record_compressed_forward(model: Any, output: Any, pending: dict[str, Any]) -> None:
    runtime_mode = model._mllm_comp_training_runtime_mode
    debug_collector = model._mllm_comp_training_debug_collector
    recorder = model._mllm_comp_training_recorder
    loss = _loss_tensor(output)
    if not pending["model_training"] or not bool(getattr(loss, "requires_grad", False)):
        return
    prepared = pending["prepared"]
    if debug_collector is not None:
        _apply_debug_compression_reports(debug_collector, prepared)
    _runtime_compression_report(model._mllm_comp_training_runtime, prepared)
    forward_index = recorder.next_forward_index()
    recorded = []
    for record in model._mllm_comp_training_session.active_records:
        activation = record.activation
        family = activation.execution.executor_family
        operation = family == "llm_operation"
        report = (
            {"operation": activation.apply_result.metadata}
            if operation else activation.last_report
        )
        if not operation and int((report or {}).get("original_audio_tokens", 0)) < 1:
            continue
        if runtime_mode is RuntimeMode.DEFAULT and not recorder.claim_default_forward_evidence(record.name):
            continue
        stage = model._mllm_comp_training_stages[record.name]
        recorder.record(
            "forward", stage=stage, forward_index=forward_index,
            runtime_mode=runtime_mode.value, executor_family=family,
            bridge=("swift_loader_pre_optimizer_forward" if operation
                    else "swift_template_pre_forward_hook"),
            loss_requires_grad=True, timing=pending["timing"],
            token_debug=(debug_collector.last_report
                         if debug_collector is not None and not operation else None),
            compression=report,
            compression_state=(
                pending["active_state"].stage_history[record.name].stats
                if pending["active_state"] is not None and not operation else {}
            ),
        )
        recorded.append((stage, family))
    if not recorded:
        return

    def record_backward(gradient: Any) -> Any:
        _enable_vscode_debugging_for_current_thread()
        for stage, family in recorded:
            recorder.record(
                "backward", stage=stage, forward_index=forward_index,
                gradient_present=gradient is not None,
                gradient_shape=(list(gradient.shape) if getattr(gradient, "shape", None) is not None else None),
                gradient_dtype=str(getattr(gradient, "dtype", "unknown")),
                executor_family=family,
            )
        return gradient

    loss.register_hook(record_backward)


@contextmanager
def _capture_compact_labels(model: Any) -> Any:
    """Share this request's final labels with Swift's loss and metrics consumers."""
    if not hasattr(model, "_mllm_comp_training_session") or not _dynamic_records(model):
        yield {}
        return
    capture = getattr(model, _LABEL_CAPTURE, None)
    if capture is not None:
        yield capture
        return
    capture = {}
    setattr(model, _LABEL_CAPTURE, capture)
    try:
        yield capture
    finally:
        delattr(model, _LABEL_CAPTURE)


def _complete_compressed_forward(model: Any, output: Any, pending: dict[str, Any]) -> Any:
    try:
        controller = _llm_training_controller(model)
        if controller is not None and _loss_tensor(output) is None:
            # The LLM controller owns the labels after all upstream stages.
            output = controller.attach_compact_training_loss(output)
        capture = getattr(model, _LABEL_CAPTURE, None)
        if capture is not None:
            capture["labels"] = (
                controller.compact_training_labels() if controller is not None
                else pending["active_state"].labels
            )
    finally:
        pending["stack"].close()
    _record_compressed_forward(model, output, pending)
    return output


def _patch_swift_output_dir(recorder: _EvidenceRecorder) -> None:
    """Follow SftArguments after ms-swift adds its versioned subdirectory."""
    from swift.arguments import SftArguments

    setattr(SftArguments, _OUTPUT_RECORDER, recorder)
    if getattr(SftArguments, _OUTPUT_PATCH_MARKER, False):
        return
    original_add_version = SftArguments._add_version

    @wraps(original_add_version)
    def add_version(args: Any) -> Any:
        result = original_add_version(args)
        active_recorder = getattr(SftArguments, _OUTPUT_RECORDER, None)
        if active_recorder is not None:
            active_recorder.set_output_dir(args.output_dir)
        return result

    SftArguments._add_version = add_version
    setattr(SftArguments, _OUTPUT_PATCH_MARKER, True)


def _compact_position_ids(template: Any, active_inputs: dict[str, Any]) -> Any:
    position_payload = template._get_position_ids(dict(active_inputs))
    position_ids = position_payload.get("position_ids")
    if position_ids is None:
        return None
    # Qwen2.5-Omni's collator removes the leading text-only row and sends the
    # remaining three multimodal rows to the Thinker.
    if getattr(position_ids, "ndim", 0) >= 1 and int(position_ids.shape[0]) == 4:
        position_ids = position_ids[1:]
    return position_ids


def _patch_qwen_template() -> None:
    from swift.template.templates.qwen import Qwen2_5OmniTemplate
    from swift.trainers import Seq2SeqTrainer

    if getattr(Qwen2_5OmniTemplate, _TEMPLATE_PATCH_MARKER, False):
        return
    original_pre_forward_hook = Qwen2_5OmniTemplate.pre_forward_hook
    original_compute_sft_loss = Qwen2_5OmniTemplate.compute_sft_loss
    original_prediction_step = Seq2SeqTrainer.prediction_step

    @wraps(original_pre_forward_hook)
    def pre_forward_hook(
        template: Any,
        swift_model: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        base_model = template.get_base_model(swift_model)
        if not hasattr(base_model, "_mllm_comp_training_session") or not _dynamic_records(base_model):
            return original_pre_forward_hook(template, swift_model, args, kwargs)
        if getattr(base_model, _PENDING_FORWARD, None) is not None:
            raise InvariantError(
                "Nested ms-swift compressed training forwards are unsupported."
            )
        original_state = build_qwen25_omni_training_state(
            base_model,
            kwargs,
            debug_semantics=(
                base_model._mllm_comp_training_runtime_mode is RuntimeMode.DEBUG
                or _requires_semantic_labels(base_model)
            ),
        )
        active_state, pending = _start_compressed_forward(base_model, original_state)
        try:
            active_inputs = dict(active_state.native_model_inputs)
            position_ids = _compact_position_ids(template, active_inputs)
            if position_ids is None:
                active_inputs.pop("position_ids", None)
            else:
                active_inputs["position_ids"] = position_ids
            result = original_pre_forward_hook(
                template,
                swift_model,
                args,
                active_inputs,
            )
            setattr(base_model, _PENDING_FORWARD, pending)
            return result
        except BaseException:
            pending["stack"].__exit__(*sys.exc_info())
            raise

    @wraps(original_compute_sft_loss)
    def compute_sft_loss(
        template: Any,
        swift_model: Any,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
        trainer: Any = None,
    ) -> Any:
        base_model = template.get_base_model(
            trainer.accelerator.unwrap_model(swift_model)
            if trainer is not None else swift_model
        )
        if not hasattr(base_model, "_mllm_comp_training_session") or not _dynamic_records(base_model):
            return original_compute_sft_loss(
                template,
                swift_model,
                inputs,
                num_items_in_batch=num_items_in_batch,
                trainer=trainer,
            )
        controller = _llm_training_controller(base_model)
        with _capture_compact_labels(base_model) as capture:
            try:
                outputs = swift_model(**inputs)
            except BaseException:
                pending = getattr(base_model, _PENDING_FORWARD, None)
                if pending is not None:
                    delattr(base_model, _PENDING_FORWARD)
                    pending["stack"].__exit__(*sys.exc_info())
                raise

            # PEFT/Accelerate may retain a forward captured before our bridge.
            pending = getattr(base_model, _PENDING_FORWARD, None)
            if pending is not None:
                delattr(base_model, _PENDING_FORWARD)
                outputs = _complete_compressed_forward(base_model, outputs, pending)

        if capture.get("labels") is not None:
            inputs["labels"] = capture["labels"]

        if "labels" in inputs:
            labels = inputs["labels"]
            loss = _loss_tensor(outputs)
            if (
                loss is None
                and controller is not None
            ):
                # The base bridge may have completed the request correctly, while
                # a later Swift/Accelerate output wrapper rebuilt the ModelOutput
                # without its loss.  Recompute from the final logits and the last
                # completed compact layout at the exact layer that consumes loss.
                outputs = controller.attach_compact_training_loss(
                    outputs,
                )
                loss = _loss_tensor(outputs)
            if loss is None:
                raise InvariantError(
                    "Qwen2.5-Omni compressed SFT returned no loss after the "
                    "template bridge completed the compression request."
                )
            loss = loss.to(labels.device)
            if num_items_in_batch is not None:
                loss = loss * (
                    (labels[:, 1:] != -100).sum() / num_items_in_batch
                )
            if isinstance(outputs, dict):
                outputs["loss"] = loss
            else:
                outputs.loss = loss
        return outputs

    @wraps(original_prediction_step)
    def prediction_step(
        trainer: Any, model: Any, inputs: Any, prediction_loss_only: bool,
        ignore_keys: Any = None, **gen_kwargs: Any,
    ) -> Any:
        if prediction_loss_only or trainer.args.predict_with_generate:
            return original_prediction_step(
                trainer, model, inputs, prediction_loss_only,
                ignore_keys=ignore_keys, **gen_kwargs,
            )
        base_model = trainer.template.get_base_model(trainer.accelerator.unwrap_model(model))
        with _capture_compact_labels(base_model) as capture:
            result = original_prediction_step(
                trainer, model, inputs, prediction_loss_only,
                ignore_keys=ignore_keys, **gen_kwargs,
            )
        if result[2] is None or capture.get("labels") is None:
            return result
        # Transformers snapshots labels before compute_loss compacts the input.
        return result[0], result[1], capture["labels"].detach()

    Qwen2_5OmniTemplate.pre_forward_hook = pre_forward_hook
    Qwen2_5OmniTemplate.compute_sft_loss = compute_sft_loss
    Seq2SeqTrainer.prediction_step = prediction_step
    setattr(Qwen2_5OmniTemplate, _TEMPLATE_PATCH_MARKER, True)


def _install_model_bridge(
    model: Any,
    *,
    stages: list[CompressionStageConfig],
    runtime_mode: RuntimeMode,
    recorder: _EvidenceRecorder,
    registry: CompressionContractRegistry | None = None,
) -> Any:
    """Install one session after native loading and before tuner/optimizer setup."""
    if hasattr(model, "_mllm_comp_training_session"):
        return model
    native_forward = model.forward
    runtime = BackboneRuntime(
        model=model, processor=None,
        metadata={"runtime_mode": runtime_mode.value, "compression_mode": "compressed", "phase": "train"},
    )
    try:
        session = RuntimeCompressionDispatcher(
            contract_registry=registry or compression_contracts(),
        ).activate(
            config=SimpleNamespace(pipeline=SimpleNamespace(stages=stages)),
            facade=get_backbone("qwen25_omni"), runtime=runtime,
            compression_mode=CompressionMode.COMPRESSED,
            execution_phase=ExecutionPhase.FORWARD,
        )
        if session is None:
            raise InvariantError("Omni compression training did not create a runtime session.")
        model._mllm_comp_training_runtime = runtime
        model._mllm_comp_training_session = session
        model._mllm_comp_training_stages = {stage.name: stage for stage in stages}
        model._mllm_comp_training_runtime_mode = runtime_mode
        model._mllm_comp_training_recorder = recorder
        model._mllm_comp_original_forward = native_forward
        if _llm_training_controller(model) is not None:
            # Native Swift captures thinker.forward while loading; refresh it
            # after the shared session installs the compact-label loss hook.
            from swift.model.utils import use_submodel_func
            use_submodel_func(model, "thinker", func_list=["forward"])
        debug_collector = None
        if runtime_mode is RuntimeMode.DEBUG and _dynamic_records(model):
            debug_collector = Qwen25OmniTokenDebugCollector(model)
            runtime.metadata["_token_debug_collector"] = debug_collector
            debug_collector.install()
        model._mllm_comp_training_debug_collector = debug_collector
        original_forward = model.forward

        @wraps(original_forward)
        def compressed_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            del self
            pending = getattr(model, _PENDING_FORWARD, None)
            if pending is not None:
                delattr(model, _PENDING_FORWARD)
                try:
                    output = original_forward(*args, **kwargs)
                except BaseException:
                    pending["stack"].__exit__(*sys.exc_info())
                    raise
                return _complete_compressed_forward(model, output, pending)
            if not _dynamic_records(model):
                # Prepare-only structural changes preserve native packed inputs.
                with (_capture_training_forward_timing(device=next(model.parameters()).device)
                      if runtime_mode is RuntimeMode.PROFILE else nullcontext(None)) as timing:
                    output = original_forward(*args, **kwargs)
                _record_compressed_forward(model, output, {
                    "active_state": None, "prepared": (), "timing": timing,
                    "model_training": bool(getattr(model, "training", False)),
                })
                return output
            original_state = build_qwen25_omni_training_state(
                model, model_inputs_from_forward_call(original_forward, args, kwargs),
                debug_semantics=(runtime_mode is RuntimeMode.DEBUG or _requires_semantic_labels(model)),
            )
            active_state, pending = _start_compressed_forward(model, original_state)
            try:
                output = original_forward(**active_state.native_model_inputs)
            except BaseException:
                pending["stack"].__exit__(*sys.exc_info())
                raise
            return _complete_compressed_forward(model, output, pending)

        model.forward = MethodType(compressed_forward, model)
        for record in session.active_records:
            activation = record.activation
            values = {}
            if activation.execution.executor_family == "llm_operation":
                values["operation"] = activation.apply_result.metadata
            recorder.record(
                "install", stage=model._mllm_comp_training_stages[record.name],
                runtime_mode=runtime_mode.value, model_class=type(model).__qualname__,
                executor_family=activation.execution.executor_family,
                lifecycle="post_load_pre_optimizer", **values,
            )
    except BaseException:
        try:
            runtime.close()
        finally:
            model.forward = native_forward
            _clear_training_attributes(model)
        raise
    return model


def _clear_training_attributes(model: Any) -> None:
    for name in (
        "_mllm_comp_training_runtime", "_mllm_comp_training_session",
        "_mllm_comp_training_stages", "_mllm_comp_training_runtime_mode",
        "_mllm_comp_training_recorder", "_mllm_comp_training_debug_collector",
        "_mllm_comp_original_forward", _PENDING_FORWARD,
    ):
        if hasattr(model, name):
            delattr(model, name)


def _close_model_bridge(model: Any) -> None:
    runtime = getattr(model, "_mllm_comp_training_runtime", None)
    if runtime is None:
        return
    native_forward = model._mllm_comp_original_forward
    pending = getattr(model, _PENDING_FORWARD, None)
    try:
        if pending is not None:
            pending["stack"].close()
    finally:
        try:
            # Restore the previous wrapper before the compression session removes
            # its own instance patches. Do not reinstall it after session.close.
            model.forward = native_forward
            runtime.close()
        finally:
            _clear_training_attributes(model)


def _patch_swift_training_lifecycle() -> None:
    # SwiftSft.run owns tuner preparation, Trainer.train, native checkpoint saves
    # and the final trainer-state/flash-checkpoint flush. Keep hooks until it exits.
    from swift.pipelines.train.sft import SwiftSft
    if getattr(SwiftSft, "_mllm_comp_training_lifecycle_patch", False):
        return
    original_run = SwiftSft.run

    @wraps(original_run)
    def run(pipeline: Any, *args: Any, **kwargs: Any) -> Any:
        loaded_model = pipeline.model
        base_model = loaded_model.model if "AWQ" in type(loaded_model).__name__ else loaded_model
        try:
            return original_run(pipeline, *args, **kwargs)
        finally:
            _close_model_bridge(base_model)

    SwiftSft.run = run
    SwiftSft._mllm_comp_training_lifecycle_patch = True


def _image_model_device_dtype(model: Any) -> tuple[Any, Any]:
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration) as exc:
        raise BackendIntegrationError(
            "Image compression training could not resolve the loaded model device."
        ) from exc
    return parameter.device, parameter.dtype


def _image_token_id(model: Any, model_inputs: dict[str, Any]) -> int:
    config = getattr(model, "config", None)
    for name in ("image_token_id", "image_token_index"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    input_ids = model_inputs.get("input_ids")
    if input_ids is not None:
        negative = input_ids[input_ids < 0]
        if int(negative.numel()) > 0:
            return int(negative[0].item())
    raise BackendIntegrationError(
        "SGL compression training could not resolve the model's image token ID."
    )


def _image_layout_records(session: Any) -> list[dict[str, Any]]:
    summary = session.runtime_metadata().get("layout_delta_summary", {})
    records = summary.get("records", []) if isinstance(summary, Mapping) else []
    result = [dict(item) for item in records if isinstance(item, Mapping)]
    for activation in session.active_records:
        result.extend(activation.runtime_metadata().get("operation_records", []))
    return result


def _install_image_model_bridge(
    model: Any,
    *,
    backbone_name: str,
    stage: CompressionStageConfig,
    runtime_mode: RuntimeMode,
    recorder: _EvidenceRecorder,
    registry: CompressionContractRegistry | None = None,
    stages: list[CompressionStageConfig] | None = None,
) -> Any:
    device, dtype = _image_model_device_dtype(model)
    executor_family = "image_fixed_mask" if stages else "image_multimodal"
    facade = get_backbone(backbone_name)
    runtime = BackboneRuntime(
        model=model,
        processor=None,
        model_path_or_id="ms-swift-loaded-model",
        device=device,
        dtype=dtype,
        metadata={
            "runtime_mode": runtime_mode.value,
            "compression_mode": CompressionMode.COMPRESSED.value,
        },
    )
    registry = registry or compression_contracts()
    dispatcher = RuntimeCompressionDispatcher(contract_registry=registry)
    session = dispatcher.activate(
        config=SimpleNamespace(pipeline=SimpleNamespace(stages=stages or [stage])),
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase=ExecutionPhase.FORWARD if stages else stage.execution_phase,
    )
    if session is None:  # pragma: no cover - compressed mode and one stage are fixed
        raise InvariantError("Image compression training did not create a runtime session.")
    original_forward = model.forward

    @wraps(original_forward)
    def image_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        del self
        model_inputs = dict(kwargs)
        if stages:
            from mllm_comp.integrations.ms_swift.split_lifecycle import unpacked_swift_inputs
            model_inputs = unpacked_swift_inputs(model_inputs)
        model_inputs["use_cache"] = False
        session.clear_diagnostic_records()
        provider_bridge = getattr(model, "mllm_compression_bridge", None)
        prepare_training_forward = getattr(
            provider_bridge,
            "prepare_training_forward",
            None,
        )
        if callable(prepare_training_forward):
            prepare_training_forward()
        sgl = runtime.metadata.get("_sgl_controller")
        if sgl is not None:
            sgl.prepare_teacher_forced(
                model_inputs=model_inputs,
                image_token_id=_image_token_id(model, model_inputs),
            )
        output = original_forward(*args, **model_inputs)
        loss = _loss_tensor(output)
        requires_grad = bool(getattr(loss, "requires_grad", False))
        records = _image_layout_records(session)
        if stages:
            from mllm_comp.fixed_mask.runtime import session_for
            masks = session_for(model)
            if masks is not None:
                records.append({"fixed_mask": masks.runtime_metadata()})
        if (
            not bool(getattr(model, "training", False))
            or not requires_grad
            or not records
            or (
                runtime_mode is RuntimeMode.DEFAULT
                and not recorder.claim_default_forward_evidence()
            )
        ):
            return output
        forward_index = recorder.next_forward_index()
        recorder.record(
            "forward",
            forward_index=forward_index,
            runtime_mode=runtime_mode.value,
            executor_family=executor_family,
            bridge="swift_loader_pre_optimizer_forward",
            backbone_name=backbone_name,
            loss_requires_grad=True,
            use_cache=False,
            layout_records=records,
        )

        def record_backward(gradient: Any) -> Any:
            _enable_vscode_debugging_for_current_thread()
            recorder.record(
                "backward",
                forward_index=forward_index,
                gradient_present=gradient is not None,
                gradient_shape=(
                    list(gradient.shape)
                    if getattr(gradient, "shape", None) is not None
                    else None
                ),
                gradient_dtype=str(getattr(gradient, "dtype", "unknown")),
                executor_family=executor_family,
            )
            return gradient

        loss.register_hook(record_backward)
        return output

    model.forward = MethodType(image_forward, model)
    model._mllm_comp_training_runtime = runtime
    model._mllm_comp_training_session = session
    model._mllm_comp_training_runtime_mode = runtime_mode
    model._mllm_comp_training_recorder = recorder
    model._mllm_comp_original_forward = original_forward
    recorder.record(
        "install",
        runtime_mode=runtime_mode.value,
        model_class=type(model).__qualname__,
        executor_family=executor_family,
        lifecycle="post_load_pre_optimizer",
        backbone_name=backbone_name,
        binding=stage.binding,
        variant=stage.variant,
    )
    return model


def _patch_image_loader(
    *,
    payload: dict[str, Any],
    spec_hash: str,
    backbone_name: str,
    registry: CompressionContractRegistry,
) -> None:
    if backbone_name == "qwen25_vl":
        from swift.model.models.qwen import Qwen2_5VLLoader as loader_class
    elif backbone_name == "qwen3_vl":
        from swift.model.models.qwen import Qwen3VLLoader as loader_class
    elif backbone_name == "llava_onevision":
        from swift.model.models.llava import LlavaOnevisionQwenLoader as loader_class
    else:
        raise BackendIntegrationError(
            f"Unsupported image compression training backbone {backbone_name!r}."
        )
    _patch_image_template(backbone_name)
    if getattr(loader_class, _PATCH_MARKER, False):
        return
    if payload.get("schema_version") == 4:
        _patch_split_loader(loader_class, payload, spec_hash)
        return
    stages = None
    if payload.get("schema_version") == 5:
        from mllm_comp.fixed_mask.checkpoint import patch_swift_save
        patch_swift_save()
        _patch_swift_training_lifecycle()
        stages = [CompressionStageConfig.from_dict(item) for item in payload["stages"]]
    stage_payload = payload["stages"][0] if stages else payload.get("stage")
    if not isinstance(stage_payload, dict):
        raise BackendIntegrationError(
            "Training compression specification omitted its stage mapping."
        )
    stage = CompressionStageConfig.from_dict(stage_payload)
    runtime_mode = RuntimeMode(payload.get("runtime_mode", RuntimeMode.DEFAULT.value))
    recorder = _EvidenceRecorder(
        spec_hash=spec_hash,
        stage=stage,
        runtime_mode=runtime_mode,
    )
    _patch_swift_output_dir(recorder)
    original_get_model = loader_class.get_model

    @wraps(original_get_model)
    def get_model(self: Any, model_dir: str, *args: Any, **kwargs: Any) -> Any:
        if stages:
            from mllm_comp.fixed_mask.checkpoint import validate_checkpoint
            validate_checkpoint(payload.get("checkpoint") or model_dir, backbone_name, stages)
        model = original_get_model(self, model_dir, *args, **kwargs)
        base_model = model.model if "AWQ" in type(model).__name__ else model
        _install_image_model_bridge(
            base_model,
            backbone_name=backbone_name,
            stage=stage,
            runtime_mode=runtime_mode,
            recorder=recorder,
            registry=registry,
            stages=stages,
        )
        return model

    loader_class.get_model = get_model
    setattr(loader_class, _PATCH_MARKER, True)


def _patch_split_loader(loader_class, payload, spec_hash):
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.modality_split.pipeline import compile_pipeline_split
    from mllm_comp.integrations.ms_swift.split_lifecycle import install_training_split, patch_native_lifecycle
    cfg = experiment_config_from_dict({
        "experiment": {"name": "swift-image-operations"},
        "backbone": {"name": payload["backbone_name"]},
        "pipeline": {"stages": payload["stages"], "extensions": payload.get("extensions", [])}})
    plan = compile_pipeline_split(cfg)
    recorder = _EvidenceRecorder(spec_hash=spec_hash, stage=cfg.pipeline.stages[0],
                                 runtime_mode=RuntimeMode(payload.get("runtime_mode", "default")))
    _patch_swift_output_dir(recorder)
    patch_native_lifecycle()
    original = loader_class.get_model

    @wraps(original)
    def get_model(self, model_dir, *args, **kwargs):
        model = original(self, model_dir, *args, **kwargs)
        return install_training_split(model, plan, recorder=recorder, checkpoint=payload.get("checkpoint"))

    loader_class.get_model = get_model
    setattr(loader_class, _PATCH_MARKER, True)


def _patch_image_template(backbone_name: str) -> None:
    if backbone_name == "qwen25_vl":
        from swift.template.templates.qwen import Qwen2_5VLTemplate as template_class
    elif backbone_name == "qwen3_vl":
        from swift.template.templates.qwen import Qwen3VLTemplate as template_class
    else:
        return
    if getattr(template_class, _IMAGE_TEMPLATE_PATCH_MARKER, False):
        return
    original_post_encode = template_class._post_encode

    @wraps(original_post_encode)
    def post_encode(template: Any, model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        base_model = template.get_base_model(model)
        peft_base = getattr(model, "base_model", None)
        candidates = (
            model,
            base_model,
            getattr(base_model, "model", None),
            peft_base,
            getattr(peft_base, "model", None),
        )
        if not any(
            candidate is not None
            and hasattr(candidate, "_mllm_comp_training_session")
            for candidate in candidates
        ):
            return original_post_encode(template, model, inputs)
        media_names = {
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        }
        return {name: value for name, value in inputs.items() if name in media_names}

    template_class._post_encode = post_encode
    setattr(template_class, _IMAGE_TEMPLATE_PATCH_MARKER, True)


def _patch_qwen_loader() -> None:
    from swift.model.models.qwen import Qwen2_5OmniLoader

    if getattr(Qwen2_5OmniLoader, _PATCH_MARKER, False):
        return
    payload, spec_hash = _load_spec()
    registry = compression_contracts(payload.get("extensions", ()))
    backbone_name = str(payload.get("backbone_name", "qwen25_omni"))
    if backbone_name != "qwen25_omni":
        _patch_image_loader(
            payload=payload,
            spec_hash=spec_hash,
            backbone_name=backbone_name,
            registry=registry,
        )
        return
    stage_payloads = payload.get("stages") if payload["schema_version"] == 3 else [payload.get("stage")]
    if not isinstance(stage_payloads, list) or not stage_payloads or any(
        not isinstance(stage, dict) for stage in stage_payloads
    ):
        raise BackendIntegrationError("Training compression specification omitted its stages.")
    stages = [CompressionStageConfig.from_dict(stage) for stage in stage_payloads]
    executions = [resolve_qwen25_omni_stage_execution(stage, registry=registry) for stage in stages]
    for stage, execution in zip(stages, executions):
        required_phase = (ExecutionPhase.PREPARE if execution.executor_family == "llm_operation"
                          else ExecutionPhase.FORWARD)
        if stage.execution_phase is not required_phase:
            raise InvariantError(
                f"ms-swift compression requires execution_phase={required_phase.value} "
                f"for executor_family={execution.executor_family}."
            )
    runtime_mode = RuntimeMode(payload.get("runtime_mode", RuntimeMode.DEFAULT.value))
    recorder = _EvidenceRecorder(
        spec_hash=spec_hash,
        stage=stages[0],
        runtime_mode=runtime_mode,
    )
    _patch_swift_output_dir(recorder)
    original_get_model = Qwen2_5OmniLoader.get_model
    _patch_swift_training_lifecycle()
    if any(execution.executor_family != "llm_operation" for execution in executions):
        _patch_qwen_template()

    @wraps(original_get_model)
    def get_model(self: Any, model_dir: str, *args: Any, **kwargs: Any) -> Any:
        model = original_get_model(self, model_dir, *args, **kwargs)
        base_model = model.model if "AWQ" in type(model).__name__ else model
        _install_model_bridge(
            base_model,
            stages=stages,
            runtime_mode=runtime_mode,
            recorder=recorder,
            registry=registry,
        )
        return model

    Qwen2_5OmniLoader.get_model = get_model
    setattr(Qwen2_5OmniLoader, _PATCH_MARKER, True)


_patch_qwen_loader()


__all__ = []
