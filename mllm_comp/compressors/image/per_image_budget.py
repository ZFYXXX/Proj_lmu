"""Map native Qwen image grids to visual spans without mixing images."""


def image_spans_and_budgets(visual_mask, grids, *, merge_size, target_tokens=128):
    """Return ``(start, stop, budget)`` per image; text-only yields an empty list.

    Positions are in the original prompt, and spans exclude surrounding vision
    markers. Each grid is independently capped; no tokens are redistributed.
    """
    if merge_size < 1 or target_tokens < 1:
        raise ValueError("merge_size and target_tokens must be positive")
    mask = [bool(value) for value in visual_mask]
    spans = []
    for index, is_visual in enumerate(mask):
        if is_visual:
            if spans and spans[-1][1] == index:
                spans[-1][1] = index + 1
            else:
                spans.append([index, index + 1])
    rows = [] if grids is None else [list(map(int, row)) for row in grids]
    if len(spans) != len(rows):
        raise ValueError("image placeholder spans do not match image grids")
    result = []
    for (start, stop), grid in zip(spans, rows):
        if len(grid) != 3:
            raise ValueError("expected a THW grid per image")
        t, height, width = grid
        if t != 1 or height < 1 or width < 1 or height % merge_size or width % merge_size:
            raise ValueError("expected an image grid aligned to the spatial merger")
        count = (height // merge_size) * (width // merge_size)
        if stop - start != count:
            raise ValueError("image placeholder span does not match the post-merger grid")
        result.append((start, stop, min(target_tokens, count)))
    return result
