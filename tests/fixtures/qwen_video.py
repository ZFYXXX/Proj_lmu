"""Deterministic inputs for the Qwen video bridge's isolated splice tests."""


def source_call(*, labels=None, pixels=None):
    import torch

    if pixels is None:
        pixels = torch.arange(128, dtype=torch.float32).reshape(32, 4)
    values = {
        "input_ids": torch.tensor([[1, *([8] * 8), 2]]),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
        "pixel_values_videos": pixels,
        "video_grid_thw": torch.tensor([[2, 4, 4]], dtype=torch.long),
        "second_per_grid_ts": torch.tensor([0.5]),
    }
    if labels is not None:
        values["labels"] = labels
    return values
