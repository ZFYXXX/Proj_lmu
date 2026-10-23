"""Minimal native report inputs for publishing and phase-boundary checks."""


def decoder_evidence(sequence_length: int, *, decode_steps: int = 3) -> dict[str, object]:
    record = {
        "prefill": {"query_length": sequence_length},
        "invariants": {
            "prefill_query_positive": True,
            "attention_mask_aligned_or_native_none": True,
            "position_ids_aligned_or_native_none": True,
            "cache_position_aligned_or_native_none": True,
        },
    }
    if decode_steps:
        record["decode_query_lengths"] = [1] * decode_steps
    return record


def generation_evidence(
    sequence_length: int, *, decode_steps: int = 0, generated_tokens: int | None = None
) -> dict[str, object]:
    record = {
        "logits_finite": True,
        "decoder": decoder_evidence(sequence_length, decode_steps=decode_steps),
    }
    if generated_tokens is not None:
        record["generated_token_count"] = generated_tokens
    return record


def dycoke_training_metadata() -> dict[str, object]:
    return {
        "execution_phase": "forward",
        "decode_view_invocations": 0,
        "decode_kv_views": (),
        "live_cache_controllers": 0,
        "decode_trigger_layer": 2,
        "sparse_view_first_layer": 3,
        "single_token_decode_only": True,
        "physical_cache_retained": True,
        "training_free_compressor": True,
        "training_gradients_preserved": True,
    }
