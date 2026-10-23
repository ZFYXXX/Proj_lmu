"""Pure metrics used by the visual-token phenomenon discovery probe."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence


IGNORE_INDEX = -100


def answer_predictor_positions(labels: Any, *, ignore_index: int = IGNORE_INDEX) -> tuple[list[int], list[int]]:
    """Return causal predictor positions and their answer-token labels.

    Causal language-model logits at position ``p - 1`` predict the supervised
    label stored at position ``p``.  Keeping this mapping explicit avoids the
    common off-by-one error in answer-level intervention probes.
    """

    row = labels[0] if getattr(labels, "ndim", 0) == 2 else labels
    positions: list[int] = []
    targets: list[int] = []
    for label_position, raw_target in enumerate(row.tolist()):
        target = int(raw_target)
        if target == ignore_index:
            continue
        if label_position == 0:
            raise ValueError("A supervised token at position zero has no causal predictor.")
        positions.append(label_position - 1)
        targets.append(target)
    if not positions:
        raise ValueError("Teacher-forced sample contains no supervised answer tokens.")
    return positions, targets


def answer_logit_metrics(
    baseline_logits: Any,
    candidate_logits: Any,
    *,
    predictor_positions: Sequence[int],
    targets: Sequence[int],
) -> dict[str, Any]:
    """Compare candidate answer logits with a fixed uncompressed teacher."""

    import torch
    import torch.nn.functional as functional

    if len(predictor_positions) != len(targets) or not predictor_positions:
        raise ValueError("Answer predictor positions and targets must be non-empty and aligned.")
    device = baseline_logits.device
    positions = torch.tensor(predictor_positions, device=device, dtype=torch.long)
    target_ids = torch.tensor(targets, device=device, dtype=torch.long)
    teacher = baseline_logits[0].index_select(0, positions).float()
    student = candidate_logits[0].index_select(0, positions.to(candidate_logits.device)).float()
    target_ids_student = target_ids.to(student.device)
    teacher_log_probs = functional.log_softmax(teacher, dim=-1)
    student_log_probs = functional.log_softmax(student, dim=-1)
    teacher_nll = functional.nll_loss(teacher_log_probs, target_ids, reduction="mean")
    student_nll = functional.nll_loss(student_log_probs, target_ids_student, reduction="mean")
    teacher_probs = teacher_log_probs.exp()
    kl = (teacher_probs * (teacher_log_probs - student_log_probs.to(device))).sum(dim=-1).mean()
    teacher_top1 = teacher.argmax(dim=-1)
    student_top1 = student.argmax(dim=-1).to(device)
    target_exact_teacher = bool(torch.equal(teacher_top1, target_ids))
    target_exact_student = bool(torch.equal(student_top1, target_ids))
    agreement = float((teacher_top1 == student_top1).float().mean().item())
    return {
        "teacher_answer_nll": float(teacher_nll.item()),
        "candidate_answer_nll": float(student_nll.item()),
        "answer_nll_delta": float((student_nll.to(device) - teacher_nll).item()),
        "answer_nll_relative_delta": float(
            ((student_nll.to(device) - teacher_nll) / teacher_nll.clamp_min(1e-12)).item()
        ),
        "teacher_to_candidate_answer_kl": float(kl.item()),
        "teacher_top1_agreement": agreement,
        "teacher_answer_exact": target_exact_teacher,
        "candidate_answer_exact": target_exact_student,
        "answer_flip": bool(not torch.equal(teacher_top1, student_top1)),
        "answer_token_count": len(targets),
    }


def compact_answer_logit_metrics(
    baseline_logits: Any,
    compact_logits: Any,
    *,
    source_predictor_positions: Sequence[int],
    compact_predictor_positions: Sequence[int],
    targets: Sequence[int],
) -> dict[str, Any]:
    """Answer metrics when visual compaction changes sequence indices."""

    import torch
    import torch.nn.functional as functional

    if len(source_predictor_positions) != len(compact_predictor_positions):
        raise ValueError("Source and compact predictor positions must align.")
    if len(source_predictor_positions) != len(targets) or not targets:
        raise ValueError("Compact answer positions and targets must be non-empty and aligned.")
    teacher_positions = torch.tensor(source_predictor_positions, device=baseline_logits.device)
    student_positions = torch.tensor(compact_predictor_positions, device=compact_logits.device)
    teacher = baseline_logits[0].index_select(0, teacher_positions).float()
    student = compact_logits[0].index_select(0, student_positions).float()
    target_ids = torch.tensor(targets, device=teacher.device, dtype=torch.long)
    teacher_log_probs = functional.log_softmax(teacher, dim=-1)
    student_log_probs = functional.log_softmax(student, dim=-1)
    teacher_nll = functional.nll_loss(teacher_log_probs, target_ids, reduction="mean")
    student_nll = functional.nll_loss(
        student_log_probs,
        target_ids.to(student.device),
        reduction="mean",
    )
    teacher_probs = teacher_log_probs.exp()
    kl = (
        teacher_probs
        * (teacher_log_probs - student_log_probs.to(teacher.device))
    ).sum(dim=-1).mean()
    teacher_top1 = teacher.argmax(dim=-1)
    student_top1 = student.argmax(dim=-1).to(teacher.device)
    return {
        "teacher_answer_nll": float(teacher_nll.item()),
        "candidate_answer_nll": float(student_nll.item()),
        "answer_nll_delta": float((student_nll.to(teacher.device) - teacher_nll).item()),
        "answer_nll_relative_delta": float(
            ((student_nll.to(teacher.device) - teacher_nll) / teacher_nll.clamp_min(1e-12)).item()
        ),
        "teacher_to_candidate_answer_kl": float(kl.item()),
        "teacher_top1_agreement": float((teacher_top1 == student_top1).float().mean().item()),
        "teacher_answer_exact": bool(torch.equal(teacher_top1, target_ids)),
        "candidate_answer_exact": bool(torch.equal(student_top1, target_ids)),
        "answer_flip": bool(not torch.equal(teacher_top1, student_top1)),
        "answer_token_count": len(targets),
    }


def average_ranks(values: Sequence[float]) -> list[float]:
    """Average-tie ranks in ascending order, implemented without SciPy."""

    ordered = sorted(enumerate(float(value) for value in values), key=lambda item: (item[1], item[0]))
    result = [0.0] * len(ordered)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            result[ordered[index][0]] = rank
        cursor = end
    return result


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_values, right_values)
    )
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left_values))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Spearman rho with average ties; returns ``None`` for constants."""

    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson_correlation(average_ranks(left), average_ranks(right))


def top_fraction_indices(values: Sequence[float], fraction: float) -> tuple[int, ...]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Top fraction must be in (0, 1].")
    count = max(1, int(math.ceil(len(values) * fraction)))
    return tuple(
        index
        for index, _ in sorted(
            enumerate(float(value) for value in values),
            key=lambda item: (-item[1], item[0]),
        )[:count]
    )


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def spatial_coverage(
    selected_clusters: Sequence[int],
    cluster_centroids: Sequence[Sequence[float]],
    *,
    height: int,
    width: int,
    bins: int = 4,
) -> dict[str, float]:
    """Measure coarse spatial coverage and the largest centroid hole."""

    if height < 1 or width < 1 or bins < 1:
        raise ValueError("Spatial coverage requires positive grid dimensions and bins.")
    selected = [cluster_centroids[index] for index in selected_clusters]
    occupied = {
        (
            min(bins - 1, int(float(row) * bins / max(height, 1))),
            min(bins - 1, int(float(col) * bins / max(width, 1))),
        )
        for row, col in selected
    }
    all_points = [tuple(map(float, point)) for point in cluster_centroids]
    if not selected:
        maximum_hole = math.hypot(max(0, height - 1), max(0, width - 1))
    else:
        maximum_hole = max(
            min(math.hypot(row - sr, col - sc) for sr, sc in selected)
            for row, col in all_points
        )
    diagonal = max(1e-12, math.hypot(max(0, height - 1), max(0, width - 1)))
    return {
        "coarse_bin_coverage": len(occupied) / float(bins * bins),
        "maximum_centroid_hole": maximum_hole,
        "maximum_centroid_hole_normalized": maximum_hole / diagonal,
    }


def media_fingerprint(media: Any, raw_doc: Mapping[str, Any] | None = None) -> str:
    """Stable image identity for grouping different questions about one image."""

    digest = hashlib.sha256()
    values = media if isinstance(media, (list, tuple)) else [media]
    for value in values:
        if hasattr(value, "tobytes") and hasattr(value, "size"):
            digest.update(str(getattr(value, "mode", "")).encode())
            digest.update(str(tuple(value.size)).encode())
            digest.update(value.tobytes())
        elif isinstance(value, (bytes, bytearray)):
            digest.update(bytes(value))
        else:
            digest.update(repr(value).encode("utf-8", errors="replace"))
    if not values and raw_doc:
        for key in ("image_id", "image", "img_id", "file_name"):
            if key in raw_doc:
                digest.update(f"{key}:{raw_doc[key]}".encode())
                break
    return digest.hexdigest()


def safe_budget(
    frontier: Sequence[Mapping[str, Any]],
    *,
    nll_relative_limit: float = 0.05,
    kl_limit: float = 0.05,
) -> int | None:
    """Return the smallest budget satisfying the pre-registered safety gate."""

    for row in sorted(frontier, key=lambda item: int(item["budget"])):
        if (
            float(row["answer_nll_relative_delta"]) <= nll_relative_limit
            and float(row["teacher_to_candidate_answer_kl"]) <= kl_limit
            and not bool(row["answer_flip"])
        ):
            return int(row["budget"])
    return None


__all__ = [
    "IGNORE_INDEX",
    "answer_logit_metrics",
    "answer_predictor_positions",
    "average_ranks",
    "compact_answer_logit_metrics",
    "jaccard",
    "media_fingerprint",
    "pearson_correlation",
    "safe_budget",
    "spatial_coverage",
    "spearman_correlation",
    "top_fraction_indices",
]
