"""Analysis sample records."""

from typing import Any, Callable

try:
    from typing import NotRequired, TypedDict
except ImportError:  # Python 3.10
    # ``typing.TypedDict`` on Python 3.10 does not recognize qualifiers from
    # ``typing_extensions``. Keep the factory and qualifier from one provider.
    from typing_extensions import NotRequired, TypedDict


class AnalysisSample(TypedDict):
    sample_id: str | int
    task_name: str
    processor_outputs: dict[str, Any]
    target_text: NotRequired[str]
    target_references: NotRequired[list[str]]
    target_source: NotRequired[str]
    score_prediction: NotRequired[Callable[[str], dict[str, Any]]]
    generation_stops: NotRequired[list[str] | str]
    target_unavailable_reason: NotRequired[str]
    teacher_forced_processor_outputs: NotRequired[dict[str, Any]]
    teacher_forcing_unavailable_reason: NotRequired[str]
    objective: NotRequired[dict[str, Any]]
    raw_doc: NotRequired[Any]
    media: NotRequired[Any]
    metadata: NotRequired[dict[str, Any]]
