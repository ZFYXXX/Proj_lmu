"""Readable evaluation diagnostics, independent of command launching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.serialization import atomic_write_text, to_jsonable


def _pretty_diagnostic_path(trace_path: Path) -> Path:
    if trace_path.suffix:
        return trace_path.with_suffix(".pretty.json")
    return trace_path.with_name(f"{trace_path.name}.pretty.json")


def _is_numeric_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float))


def _is_shape_key(key: str | None) -> bool:
    if key is None:
        return False
    normalized = key.lower()
    return normalized == "shape" or normalized.endswith("_shape")


def _inline_pretty_diagnostic_value(
    value: Any,
    *,
    key: str | None = None,
    array_items_per_line: int = 10,
    max_width: int = 120,
) -> str | None:
    """Return one-line JSON when a diagnostic value is genuinely compact."""

    if isinstance(value, Mapping):
        if not value:
            return "{}"
        fields: list[str] = []
        for child_key, child_value in value.items():
            normalized_key = str(child_key)
            encoded_value = _inline_pretty_diagnostic_value(
                child_value,
                key=normalized_key,
                array_items_per_line=array_items_per_line,
                max_width=max_width,
            )
            if encoded_value is None:
                return None
            encoded_key = json.dumps(normalized_key, ensure_ascii=False)
            fields.append(f"{encoded_key}: {encoded_value}")
        encoded = f"{{{', '.join(fields)}}}"
        return encoded if len(encoded) <= max_width else None
    if isinstance(value, list):
        if not value:
            return "[]"
        if not all(_is_numeric_json_scalar(item) for item in value):
            return None
        if not _is_shape_key(key) and len(value) > array_items_per_line:
            return None
        return "[" + ", ".join(
            json.dumps(item, ensure_ascii=False, allow_nan=True)
            for item in value
        ) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=True)


def _render_pretty_diagnostic_value(
    value: Any,
    *,
    level: int = 0,
    key: str | None = None,
    array_items_per_line: int = 10,
    max_line_width: int = 160,
) -> list[str]:
    """Render valid JSON with compact mappings, shapes, and numeric arrays."""

    indent = "  " * level
    inline = _inline_pretty_diagnostic_value(
        value,
        key=key,
        array_items_per_line=array_items_per_line,
        max_width=max(1, max_line_width - len(indent)),
    )
    if inline is not None:
        return [f"{indent}{inline}"]
    if isinstance(value, Mapping):
        lines = [f"{indent}{{"]
        items = list(value.items())
        child_indent = "  " * (level + 1)
        for index, (child_key, child_value) in enumerate(items):
            normalized_key = str(child_key)
            child_lines = _render_pretty_diagnostic_value(
                child_value,
                level=level + 1,
                key=normalized_key,
                array_items_per_line=array_items_per_line,
                max_line_width=max_line_width,
            )
            encoded_key = json.dumps(normalized_key, ensure_ascii=False)
            child_lines[0] = (
                f"{child_indent}{encoded_key}: "
                f"{child_lines[0][len(child_indent):]}"
            )
            if index < len(items) - 1:
                child_lines[-1] += ","
            lines.extend(child_lines)
        lines.append(f"{indent}}}")
        return lines
    if isinstance(value, list):
        if all(_is_numeric_json_scalar(item) for item in value):
            encoded = [
                json.dumps(item, ensure_ascii=False, allow_nan=True)
                for item in value
            ]
            if _is_shape_key(key) or len(encoded) <= array_items_per_line:
                return [f"{indent}[{', '.join(encoded)}]"]
            lines = [f"{indent}["]
            row_indent = "  " * (level + 1)
            for start in range(0, len(encoded), array_items_per_line):
                stop = min(start + array_items_per_line, len(encoded))
                suffix = "," if stop < len(encoded) else ""
                lines.append(
                    f"{row_indent}{', '.join(encoded[start:stop])}{suffix}"
                )
            lines.append(f"{indent}]")
            return lines
        lines = [f"{indent}["]
        for index, item in enumerate(value):
            child_lines = _render_pretty_diagnostic_value(
                item,
                level=level + 1,
                array_items_per_line=array_items_per_line,
                max_line_width=max_line_width,
            )
            if index < len(value) - 1:
                child_lines[-1] += ","
            lines.extend(child_lines)
        lines.append(f"{indent}]")
        return lines
    return [f"{indent}{json.dumps(value, ensure_ascii=False, allow_nan=True)}"]


def _sparsify_pretty_token_counts(value: Any, *, key: str | None = None) -> Any:
    """Hide zero-valued token classes in the human-oriented pretty artifact."""

    if isinstance(value, Mapping):
        return {
            str(child_key): _sparsify_pretty_token_counts(
                child_value,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
            if not (
                key == "token_counts"
                and isinstance(child_value, (int, float))
                and not isinstance(child_value, bool)
                and child_value == 0
            )
        }
    if isinstance(value, list):
        return [_sparsify_pretty_token_counts(item) for item in value]
    return value


def _nonzero_token_counts(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): count
        for name, count in value.items()
        if not (
            isinstance(count, (int, float))
            and not isinstance(count, bool)
            and count == 0
        )
    }


def _compact_pretty_phase_side(
    value: Any,
    *,
    stage: str,
    phase: str,
    token_counts_reference: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    if not isinstance(value, Mapping):
        return {}, False
    total = value.get("total_tokens")
    compact: dict[str, Any] = {}
    if total is not None:
        compact["tokens"] = total
    classified = value.get("classified_tokens")
    if classified is not None and classified != total:
        compact["classified_tokens"] = classified
    delta = value.get("classification_delta")
    if delta not in (None, 0):
        compact["classification_delta"] = delta

    counts = _nonzero_token_counts(value.get("token_counts"))
    reference = _nonzero_token_counts(token_counts_reference)
    uses_reference = bool(reference) and counts == reference
    redundant_label = (
        "response" if stage == "llm" and phase == "decode" else "audio"
    )
    redundant_counts = (
        total is not None and counts == {redundant_label: total}
    )
    if counts and not uses_reference and not redundant_counts:
        compact["token_counts"] = counts

    known = {
        "total_tokens",
        "classified_tokens",
        "classification_delta",
        "token_counts",
    }
    for name, child in value.items():
        if name not in known:
            compact[str(name)] = child
    return compact, uses_reference


def _prefix_pretty_phase_side(
    side: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    return {
        (f"{prefix}_tokens" if name == "tokens" else f"{prefix}_{name}"): value
        for name, value in side.items()
    }


def _compact_pretty_phase(
    value: Any,
    *,
    stage: str,
    phase: str,
    token_counts_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    calls = value.get("calls")
    if calls == 0:
        return None
    input_side, input_uses_reference = _compact_pretty_phase_side(
        value.get("input"),
        stage=stage,
        phase=phase,
        token_counts_reference=token_counts_reference,
    )
    output_side, output_uses_reference = _compact_pretty_phase_side(
        value.get("output"),
        stage=stage,
        phase=phase,
        token_counts_reference=token_counts_reference,
    )
    compact: dict[str, Any] = {}
    if calls is not None and (calls != 1 or phase == "decode"):
        compact["calls"] = calls
    if input_side == output_side and input_uses_reference == output_uses_reference:
        compact.update(input_side)
        if input_uses_reference:
            compact["token_counts_ref"] = "post_legalizer_llm_input"
    else:
        compact.update(_prefix_pretty_phase_side(input_side, prefix="input"))
        compact.update(_prefix_pretty_phase_side(output_side, prefix="output"))
        if input_uses_reference:
            compact["input_token_counts_ref"] = "post_legalizer_llm_input"
        if output_uses_reference:
            compact["output_token_counts_ref"] = "post_legalizer_llm_input"

    known = {"calls", "input", "output"}
    for name, child in value.items():
        if name not in known:
            compact[str(name)] = child
    return compact


def _compact_pretty_layer_body(
    value: Mapping[str, Any],
    *,
    token_counts_reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stage = str(value.get("stage", "unknown"))
    phases: dict[str, dict[str, Any]] = {}
    for phase_name in ("encode", "prefill", "decode"):
        compact = _compact_pretty_phase(
            value.get(phase_name),
            stage=stage,
            phase=phase_name,
            token_counts_reference=(
                token_counts_reference
                if stage == "llm" and phase_name == "prefill"
                else None
            ),
        )
        if compact is not None:
            phases[phase_name] = compact
    result: dict[str, Any] = {"stage": stage}
    if stage == "audio_encoder" and tuple(phases) == ("encode",):
        result.update(phases["encode"])
    else:
        result.update(phases)
    return result


def _materialize_pretty_layer_run(
    start: int,
    end: int,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    stage = body.get("stage", "unknown")
    location = {"layer": start} if start == end else {"layer_range": [start, end]}
    return {
        "stage": stage,
        **location,
        **{name: child for name, child in body.items() if name != "stage"},
    }


def _compact_pretty_layers(
    value: Any,
    *,
    token_counts_reference: Mapping[str, Any] | None,
) -> list[Any]:
    if not isinstance(value, list):
        return []
    runs: list[tuple[int, int, dict[str, Any]]] = []
    passthrough: list[Any] = []
    for item in value:
        if not isinstance(item, Mapping):
            passthrough.append(item)
            continue
        layer_index = item.get("layer_index")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            passthrough.append(dict(item))
            continue
        body = _compact_pretty_layer_body(
            item,
            token_counts_reference=token_counts_reference,
        )
        if (
            runs
            and runs[-1][2] == body
            and runs[-1][1] + 1 == layer_index
        ):
            start, _, previous = runs[-1]
            runs[-1] = (start, layer_index, previous)
        else:
            runs.append((layer_index, layer_index, body))
    return [
        _materialize_pretty_layer_run(start, end, body)
        for start, end, body in runs
    ] + passthrough


def _phase_is_unchanged(value: Any) -> bool:
    if not isinstance(value, Mapping) or value.get("calls") == 0:
        return True
    input_side = value.get("input")
    output_side = value.get("output")
    if not isinstance(input_side, Mapping) or input_side != output_side:
        return False
    total = input_side.get("total_tokens")
    return (
        input_side.get("classification_delta") in (None, 0)
        and input_side.get("classified_tokens", total) == total
    )


def _compact_pretty_modules(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    result: list[Any] = []
    for item in value:
        if not isinstance(item, Mapping):
            result.append(item)
            continue
        stage = str(item.get("stage", "unknown"))
        if stage == "audio_post_norm" and all(
            _phase_is_unchanged(item.get(phase_name))
            for phase_name in ("encode", "prefill", "decode")
        ):
            continue
        phases: dict[str, dict[str, Any]] = {}
        for phase_name in ("encode", "prefill", "decode"):
            compact = _compact_pretty_phase(
                item.get(phase_name),
                stage=stage,
                phase=phase_name,
            )
            if compact is not None:
                phases[phase_name] = compact
        compact_item: dict[str, Any] = {"stage": stage}
        if tuple(phases) == ("encode",):
            compact_item.update(phases["encode"])
        else:
            compact_item.update(phases)
        result.append(compact_item)
    return result


def _compact_pretty_token_debug_report(value: Mapping[str, Any]) -> dict[str, Any]:
    snapshots = value.get("stage_snapshots")
    token_counts_reference: Mapping[str, Any] | None = None
    if isinstance(snapshots, list):
        token_counts_reference = next(
            (
                item.get("token_counts")
                for item in snapshots
                if isinstance(item, Mapping)
                and item.get("stage") == "post_legalizer_llm_input"
                and isinstance(item.get("token_counts"), Mapping)
            ),
            None,
        )
    result: dict[str, Any] = {}
    for name, child in value.items():
        if name == "layers":
            result[name] = _compact_pretty_layers(
                child,
                token_counts_reference=token_counts_reference,
            )
        elif name == "modules":
            result[name] = _compact_pretty_modules(child)
        else:
            result[str(name)] = _compact_pretty_debug_reports(child)
    return result


def _compact_pretty_debug_reports(value: Any) -> Any:
    if isinstance(value, Mapping):
        if (
            value.get("mode") == "debug"
            and isinstance(value.get("layers"), list)
            and isinstance(value.get("modules"), list)
        ):
            return _compact_pretty_token_debug_report(value)
        return {
            str(name): _compact_pretty_debug_reports(child)
            for name, child in value.items()
        }
    if isinstance(value, list):
        return [_compact_pretty_debug_reports(item) for item in value]
    return value


def _pretty_diagnostic_json(records: Sequence[Any]) -> str:
    normalized = _compact_pretty_debug_reports(to_jsonable(list(records)))
    normalized = _sparsify_pretty_token_counts(normalized)
    return "\n".join(_render_pretty_diagnostic_value(normalized))


def _materialize_pretty_diagnostics(trace_paths: Sequence[Path]) -> list[Path]:
    """Convert machine-oriented JSONL traces into readable JSON arrays."""

    pretty_paths: list[Path] = []
    for trace_path in trace_paths:
        records: list[Any] = []
        try:
            with trace_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise BackendIntegrationError(
                            "Cannot parse lmms-eval diagnostic trace "
                            f"{trace_path} line {line_number}: {exc}"
                        ) from exc
        except OSError as exc:
            raise BackendIntegrationError(
                f"Cannot read lmms-eval diagnostic trace {trace_path}: {exc}"
            ) from exc
        pretty_path = _pretty_diagnostic_path(trace_path)
        atomic_write_text(pretty_path, _pretty_diagnostic_json(records))
        pretty_paths.append(pretty_path)
    return pretty_paths
