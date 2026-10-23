"""Gold-free greedy generation for controlled, cache-free deletion diagnostics."""

from typing import Any, Callable, Sequence


def greedy_decode_embeddings(
    *, prompt_hidden: Any, prompt_position_ids: Any,
    token_embedding: Callable, forward_logits: Callable,
    max_new_tokens: int, eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    import torch

    if prompt_hidden.ndim != 3 or prompt_hidden.shape[0] != 1 or prompt_position_ids.shape != (3, 1, prompt_hidden.shape[1]):
        raise ValueError("Generation requires one prompt and aligned three-axis M-RoPE IDs.")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")
    state = prompt_hidden
    positions = prompt_position_ids
    first_new_position = int(positions.max().item()) + 1
    generated = []
    eos = set(int(value) for value in eos_token_ids)
    with torch.inference_mode():
        for step in range(max_new_tokens):
            logits = forward_logits(state, positions)[0, -1].float()
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError("Nonfinite free-generation logits.")
            token = int(logits.argmax().item())
            generated.append(token)
            if token in eos or step + 1 == max_new_tokens:
                break
            new_embedding = token_embedding(torch.tensor([[token]], device=state.device))
            state = torch.cat((state, new_embedding.to(dtype=state.dtype)), dim=1)
            new_position = torch.full((3, 1, 1), first_new_position + step, device=positions.device, dtype=positions.dtype)
            positions = torch.cat((positions, new_position), dim=2)
    return {
        "generated_token_ids": generated, "stopped_on_eos": generated[-1] in eos,
        "first_appended_token_position": first_new_position,
        "max_new_tokens": max_new_tokens, "use_cache": False,
        "do_sample": False, "repetition_penalty": 1.0,
        "gold_answer_in_decode_context": False,
    }


def primary_task_score(task: str, metrics: dict[str, Any]) -> float:
    if "chartqa" in task:
        value = metrics["relaxed_overall"]
    elif "pope" in task:
        value = metrics["pope_accuracy"]["score"]
    else:
        value = metrics["exact_match"]
    score = float(value)
    if not 0 <= score <= 1:
        raise ValueError(f"Unexpected per-sample task score: {score}.")
    return score
