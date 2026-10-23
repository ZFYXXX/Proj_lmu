from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones import llamavid_video as llamavid_provider
from mllm_comp.backbones._video_conditioning import user_text_from_conversation
from mllm_comp.backbones.llamavid_video import (
    LLaMAVIDVideoExecutor,
    LLaMAVIDVideoSignalProvider,
)
from mllm_comp.backbones.qwen3_vl.video_compression import (
    Qwen3VLPackedVideoRequest,
    Qwen3VLVideoCompressionResponse,
    Qwen3VLVideoGrid,
)
from mllm_comp.compressors.video import (
    LLaMAVIDVideoKernel,
    VideoTokenCoordinate,
    VideoTokenPayload,
    llamavid_structural_groups,
    llamavid_video,
    replay_llamavid_value_lineage,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError


def _torch():
    return pytest.importorskip("torch", reason="LLaMA-VID tensor tests need PyTorch")


def _coordinates(
    temporal_count: int,
    *,
    side: int,
) -> tuple[VideoTokenCoordinate, ...]:
    return tuple(
        VideoTokenCoordinate(t=temporal, h=row, w=column, frame_index=temporal)
        for temporal in range(temporal_count)
        for row in range(side)
        for column in range(side)
    )


class _TinyQFormer:
    def __init__(self) -> None:
        torch = _torch()
        self.score = torch.nn.Linear(3, 1, bias=False)
        self.context = torch.nn.Linear(3, 4, bias=False)
        self.content = torch.nn.Linear(3, 4, bias=False)

    def compress_video(
        self,
        raw: Any,
        *,
        text_context: str,
        text_tokens: Any,
        num_query: int,
    ) -> tuple[Any, Any]:
        del text_tokens
        assert text_context == "Which action occurs?"
        assert num_query == 32
        weights = self.score(raw).squeeze(-1).softmax(dim=-1)
        context = self.context((weights[:, :, None] * raw).sum(dim=1))
        content = self.content(raw.mean(dim=1))
        return _torch().stack((context, content), dim=1), weights

    def parameters(self):
        yield from self.score.parameters()
        yield from self.context.parameters()
        yield from self.content.parameters()

    def state_dict(self):
        return {
            "score.weight": self.score.weight,
            "context.weight": self.context.weight,
            "content.weight": self.content.weight,
        }

    def load_state_dict(self, state: Any, strict: bool = True):
        assert strict is True
        with _torch().no_grad():
            self.score.weight.copy_(state["score.weight"])
            self.context.weight.copy_(state["context.weight"])
            self.content.weight.copy_(state["content.weight"])


def _payload(
    *,
    temporal_count: int = 2,
    projected_side: int = 3,
    raw_side: int = 4,
    requires_grad: bool = False,
    qformer: Any | None = None,
    dtype: Any | None = None,
) -> VideoTokenPayload:
    torch = _torch()
    dtype = dtype or torch.float32
    projected_count = temporal_count * projected_side * projected_side
    raw_count = temporal_count * raw_side * raw_side
    projected = torch.arange(
        1,
        projected_count * 4 + 1,
        dtype=dtype,
    ).reshape(projected_count, 4)
    raw = torch.arange(
        1,
        raw_count * 3 + 1,
        dtype=dtype,
    ).reshape(raw_count, 3)
    raw = raw.clone().requires_grad_(requires_grad)
    module = qformer or _TinyQFormer()
    return VideoTokenPayload(
        features=projected,
        media_ids=("nextqa-llamavid:video-0",),
        video_offsets=(0, projected_count),
        coordinates=_coordinates(temporal_count, side=projected_side),
        raw_features=raw,
        raw_video_offsets=(0, raw_count),
        raw_coordinates=_coordinates(temporal_count, side=raw_side),
        signals={
            "text_context": "Which action occurs?",
            "bert_qformer": module,
            "projector_dimensions": {
                "raw_hidden_size": 3,
                "output_hidden_size": 4,
                "num_query": 32,
            },
        },
    )


def test_llamavid_control_plane_freezes_recipe_and_causal_anchors() -> None:
    kernel = LLaMAVIDVideoKernel()
    kernel.validate_config({"config": {}})
    kernel.validate_config(
        {
            "config": {
                "num_query": 32,
                "bert_type": "qformer",
                "compress_type": "mean",
            }
        }
    )
    with pytest.raises(ConfigError, match="non-canonical"):
        kernel.validate_config({"config": {"compress_type": "grid:4"}})

    groups, anchors = llamavid_structural_groups(_coordinates(2, side=3))
    assert groups == ((0,), (1,), (9,), (10,))
    assert anchors == (0, 1, 9, 10)
    with pytest.raises(BackendIntegrationError, match="two distinct"):
        llamavid_structural_groups(_coordinates(2, side=1))


def test_llamavid_kernel_is_deterministic_reducing_and_trainable() -> None:
    torch = _torch()
    qformer = _TinyQFormer()
    payload = _payload(requires_grad=True, qformer=qformer)
    first = llamavid_video(payload, qformer_module=qformer)
    second = LLaMAVIDVideoKernel()(payload, qformer_module=qformer)

    assert first.source_count == 18
    assert first.result_count == 4
    assert first.source_groups == (((0,), (1,), (9,), (10,)),)
    assert first.representative_indices == ((0, 1, 9, 10),)
    assert first.metadata["tokens_per_temporal_unit"] == 2
    assert first.metadata["output_order"] == ("context_then_content_per_temporal_unit")
    records = first.metadata["value_records_by_video"][0]
    assert tuple(record["kind"] for record in records) == (
        "context",
        "content",
        "context",
        "content",
    )
    assert all(len(record["source_indices"]) == 9 for record in records)
    assert all(sum(record["weights"]) == pytest.approx(1.0) for record in records)
    torch.testing.assert_close(first.features, second.features)

    first.features.sum().backward()
    assert payload.raw_features.grad is not None
    assert bool((payload.raw_features.grad != 0).any().item())
    assert all(
        parameter.grad is not None and bool((parameter.grad != 0).any().item())
        for parameter in qformer.parameters()
    )


def test_llamavid_fp16_serialized_value_lineage_is_normalized() -> None:
    torch = _torch()
    qformer = _TinyQFormer()
    for module in (qformer.score, qformer.context, qformer.content):
        module.to(dtype=torch.float16)
    payload = _payload(
        temporal_count=1,
        projected_side=14,
        raw_side=27,
        qformer=qformer,
        dtype=torch.float16,
    )

    result = llamavid_video(payload, qformer_module=qformer)

    records = result.metadata["value_records_by_video"][0]
    assert len(records) == 2
    assert all(math.fsum(record["weights"]) == pytest.approx(1.0) for record in records)
    content_counts = [0] * 196
    for raw_h in range(27):
        for raw_w in range(27):
            content_counts[(raw_h * 14 // 27) * 14 + (raw_w * 14 // 27)] += 1
    assert tuple(records[1]["weights"]) == pytest.approx(
        tuple(value / 729 for value in content_counts),
        rel=1e-2,
        abs=1e-6,
    )
    assert bool(torch.isfinite(result.features).all().item())


def test_llamavid_value_lineage_and_qwen3_deepstack_replay() -> None:
    torch = _torch()
    qformer = _TinyQFormer()
    payload = _payload(qformer=qformer)
    result = llamavid_video(payload, qformer_module=qformer)
    branch = payload.features * 5.0
    replayed = replay_llamavid_value_lineage(branch, result)
    assert tuple(replayed.shape) == (4, 4)
    torch.testing.assert_close(
        replayed,
        replay_llamavid_value_lineage(payload.features, result) * 5.0,
    )

    grid = Qwen3VLVideoGrid.from_native(
        (2, 6, 6),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    request = Qwen3VLPackedVideoRequest(
        invocation_id="llamavid-video-1",
        media_id="nextqa-llamavid:video-0",
        payload=payload,
        deepstack_features=(payload.features * 2.0, payload.features * 3.0),
        grid=grid,
        source_placeholder_positions=tuple(range(1, 19)),
        prompt_input_ids=torch.tensor([[1, *([8] * 18), 2]]),
        source_attention_mask=torch.ones(1, 20, dtype=torch.long),
        source_video_grid_thw=torch.tensor([[2, 6, 6]], dtype=torch.long),
        source_second_per_grid_ts=torch.tensor([0.5]),
    )
    executor = LLaMAVIDVideoExecutor(
        backbone="qwen3_vl",
        model=object(),
        qformer_module=qformer,
    )
    response = executor._projector_callback(request)
    assert isinstance(response, Qwen3VLVideoCompressionResponse)
    assert response.compression_result.result_count == 4
    for source, replay in zip(request.deepstack_features, response.deepstack_features):
        torch.testing.assert_close(
            replay,
            replay_llamavid_value_lineage(source, response.compression_result),
        )
    record = executor.runtime_metadata()["algorithm_records"][0]
    assert record["algorithm"] == "llamavid"
    assert record["variant"] == "llamavid.video"
    assert record["source_count"] == 18
    assert record["result_count"] == 4
    assert record["tokens_per_temporal_unit"] == 2
    assert record["text_context_present"] is True
    assert record["deepstack_branch_count"] == 2


@pytest.mark.parametrize("backbone", ("llava_onevision", "qwen25_vl", "qwen3_vl"))
def test_llamavid_executor_registers_qformer_before_bridge_and_uninstalls(
    backbone: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _torch()
    events: list[tuple[str, str]] = []
    qformer = _TinyQFormer()

    class _Vision:
        hidden_size = 3

    if backbone == "llava_onevision":

        class _Model:
            config = SimpleNamespace(hidden_size=4)

            def get_vision_tower(self) -> Any:
                return _Vision()

        model: Any = _Model()
    else:
        block = SimpleNamespace(norm1=SimpleNamespace(normalized_shape=(3,)))
        model = SimpleNamespace(
            model=SimpleNamespace(visual=SimpleNamespace(blocks=(block,))),
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4)),
        )

    class _Provider:
        def bind_qformer(self, module: Any, dimensions: Any) -> None:
            assert module is qformer
            assert dimensions["raw_hidden_size"] == 3
            events.append((backbone, "bind"))

        def install(self) -> None:
            assert getattr(model, "mllm_comp_llamavid_qformer") is qformer
            events.append((backbone, "provider_install"))

        def uninstall(self) -> None:
            events.append((backbone, "provider_uninstall"))

        def runtime_metadata(self) -> dict[str, Any]:
            return {"fake_provider": backbone}

    class _Handle:
        def remove(self) -> None:
            events.append((backbone, "remove"))

    class _Bridge:
        def register_callback(self, site: Any, callback: Any, *, signal_provider: Any):
            assert isinstance(site, str)
            assert callable(callback)
            assert signal_provider is provider
            events.append((backbone, "register"))
            return _Handle()

        def detach(self) -> None:
            events.append((backbone, "detach"))

        def runtime_metadata(self) -> dict[str, Any]:
            return {"fake_bridge": backbone}

    provider = _Provider()
    monkeypatch.setattr(
        llamavid_provider,
        "install_video_bridge",
        lambda _model, selected: (
            _Bridge(),
            selected == backbone,
            f'{selected}:qformer_context_content_pool',
        ),
    )
    executor = LLaMAVIDVideoExecutor(
        backbone=backbone,
        model=model,
        signal_provider=provider,
        qformer_module=qformer,
    )
    executor.install()
    assert executor.runtime_metadata()["qformer_registered_pre_optimizer"] is True
    saved = executor.compression_state_dict()
    executor.load_compression_state_dict(saved)
    executor.uninstall()

    assert not hasattr(model, "mllm_comp_llamavid_qformer")
    assert events == [
        (backbone, "bind"),
        (backbone, "provider_install"),
        (backbone, "register"),
        (backbone, "remove"),
        (backbone, "detach"),
        (backbone, "provider_uninstall"),
    ]


def test_llava_llamavid_provider_attaches_raw_and_bert_tokens() -> None:
    torch = _torch()

    class _Tokenizer:
        def __call__(self, texts: Any, **kwargs: Any) -> Any:
            assert texts == ["Which action occurs?"]
            assert kwargs["max_length"] == 256
            return SimpleNamespace(
                input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
            )

    class _Outer:
        hidden_size = 3

        def forward(self, frames: Any) -> Any:
            return torch.arange(
                1,
                int(frames.shape[0]) * 16 * 3 + 1,
                dtype=frames.dtype,
            ).reshape(int(frames.shape[0]), 16, 3)

    class _Model:
        def __init__(self) -> None:
            self.outer = _Outer()

        def get_vision_tower(self) -> Any:
            return self.outer

    model = _Model()
    native = model.outer.forward
    qformer = _TinyQFormer()
    provider = LLaMAVIDVideoSignalProvider(
        backbone="llava_onevision",
        model=model,
        tokenizer=_Tokenizer(),
    )
    provider.bind_qformer(
        qformer,
        {"raw_hidden_size": 3, "output_hidden_size": 4, "num_query": 32},
    )
    provider.install()
    frames = torch.zeros(2, 3, 8, 8)
    with provider.capture_context(frames=frames) as capture:
        model.outer.forward(frames)
    base = VideoTokenPayload(
        features=torch.ones(18, 4),
        media_ids=("llava-llamavid",),
        video_offsets=(0, 18),
        coordinates=_coordinates(2, side=3),
        signals={"text_context": "Which action occurs?"},
    )
    enriched = provider.enrich_payload(base, capture)
    assert tuple(enriched.raw_features.shape) == (32, 3)
    assert enriched.signals["bert_qformer"] is qformer
    tokens = enriched.signals["llamavid.bert_tokens_by_video"][0]
    assert tuple(tokens["input_ids"].shape) == (1, 3)
    assert enriched.metadata["llamavid_raw_stream"] == (
        "llava_outer_vision_pre_projector"
    )
    provider.uninstall()
    assert getattr(model.outer.forward, "__func__", None) is getattr(
        native, "__func__", None
    )


@pytest.mark.parametrize(
    ("backbone", "source_indices"),
    (
        ("qwen25_vl", (12, 13, 8, 9, 14, 15, 10, 11, 4, 5, 0, 1, 6, 7, 2, 3)),
        ("qwen3_vl", (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)),
    ),
)
def test_qwen_llamavid_provider_restores_native_raster(
    backbone: str,
    source_indices: tuple[int, ...],
) -> None:
    torch = _torch()

    class _Tokenizer:
        def __call__(self, texts: Any, **kwargs: Any) -> Any:
            del texts, kwargs
            return SimpleNamespace(
                input_ids=torch.tensor([[1]], dtype=torch.long),
                attention_mask=torch.ones(1, 1, dtype=torch.long),
            )

    class _Block:
        norm1 = SimpleNamespace(normalized_shape=(4,))

        def forward(self, hidden_states: Any, **kwargs: Any) -> Any:
            del kwargs
            return hidden_states + 1.0

    block = _Block()
    reverse = torch.arange(3, -1, -1)
    visual = SimpleNamespace(
        blocks=(block,),
        spatial_merge_size=2,
        spatial_merge_unit=4,
        get_window_index=lambda _grid: (reverse, torch.tensor([0, 4])),
    )
    model = SimpleNamespace(model=SimpleNamespace(visual=visual))
    provider = LLaMAVIDVideoSignalProvider(
        backbone=backbone,
        model=model,
        tokenizer=_Tokenizer(),
    )
    provider.bind_qformer(
        _TinyQFormer(),
        {"raw_hidden_size": 4, "output_hidden_size": 4, "num_query": 32},
    )
    provider.install()
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    raw = torch.arange(64, dtype=torch.float32).reshape(16, 4).requires_grad_()
    with provider.capture_context(grid_thw=grid) as capture:
        block.forward(raw)
    base = VideoTokenPayload(
        features=torch.ones(4, 4),
        media_ids=("qwen-llamavid",),
        video_offsets=(0, 4),
        coordinates=_coordinates(1, side=2),
        signals={"text_context": "Which action occurs?"},
    )
    enriched = provider.enrich_payload(base, capture)
    assert tuple(enriched.raw_features.shape) == (16, 4)
    torch.testing.assert_close(enriched.raw_features, (raw + 1)[list(source_indices)])
    assert enriched.raw_coordinates == tuple(
        VideoTokenCoordinate(t=0, h=row, w=column, tubelet_index=0)
        for row in range(4)
        for column in range(4)
    )
    enriched.raw_features[2, 1].backward()
    expected_gradient = torch.zeros_like(raw)
    expected_gradient[source_indices[2], 1] = 1.0
    torch.testing.assert_close(raw.grad, expected_gradient)
    provider.uninstall()


def test_llamavid_text_excludes_teacher_forced_answer_and_missing_bert_closes() -> None:
    text = user_text_from_conversation(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "sample.mp4"},
                    {"type": "text", "text": "Which action occurs?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The answer is hidden."}],
            },
        ]
    )
    assert text == "Which action occurs?"

    provider = LLaMAVIDVideoSignalProvider(
        backbone="llava_onevision",
        model=object(),
    )
    with pytest.raises(BackendIntegrationError, match="llamavid_bert_path"):
        provider.prepare_tokenizer()
