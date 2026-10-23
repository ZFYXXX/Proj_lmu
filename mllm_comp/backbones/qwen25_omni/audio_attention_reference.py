"""Carry fixed encoder attention importance to pooled audio token layouts."""

from typing import Any
from time import perf_counter

from mllm_comp.compressors.encoder_level.adjacent_token_merging import qwen_audio_chunk_lengths_after_cnn
from mllm_comp.core.errors import InvariantError

from .audio_compression import _module_execution_device, _torch, sa_map_audio_importance


class AudioAttentionReference:
    def __init__(self, audio_tower: Any, layer_offset: int, *, profile_enabled: bool = False, distinct_importance: bool = False, probe_kind: str = "received_attention") -> None:
        self.tower = audio_tower
        self.profile_enabled = profile_enabled
        self.distinct_importance = distinct_importance
        self.key_vectors = probe_kind == "key_vectors"
        self.layer_index = len(audio_tower.layers) + layer_offset
        if not 0 <= self.layer_index < len(audio_tower.layers):
            raise InvariantError("The audio encoder has no layer for the rule's fixed attention source.")
        self.attention = audio_tower.layers[self.layer_index].self_attn
        self.handles: list[Any] = []
        self.encoder = None
        self.projector = None
        self.clear()

    def start(self, *, enabled: bool) -> None:
        self.clear()
        self.active = enabled

    def clear(self) -> None:
        self.active = False
        self.source = None
        self.chunk_lengths: list[int] = []
        self.audio_chunk_counts: list[int] = []
        self.audio_scores: list[Any] = []
        self.sequence_scores = None
        self.score_seconds = 0.0

    def install(self) -> None:
        self.encoder = getattr(self.tower, "_mllm_encoder_compression", None)
        self.projector = getattr(self.tower, "_mllm_projector_compression", None)
        if self.encoder is not None:
            self.previous_encoder_observer = self.encoder._layout_observer
            self.encoder._layout_observer = self._encoder_layout
        if self.projector is not None:
            self.previous_projector_observer = self.projector._layout_observer
            self.projector._layout_observer = self._projector_layout
        self.handles = [
            self.tower.register_forward_pre_hook(self._encoder_input, with_kwargs=True),
            self.attention.register_forward_pre_hook(self._attention_input, with_kwargs=True),
            self.tower.proj.register_forward_pre_hook(self._projector_input),
        ]

    def uninstall(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.encoder is not None:
            self.encoder._layout_observer = self.previous_encoder_observer
            self.encoder = None
        if self.projector is not None:
            self.projector._layout_observer = self.previous_projector_observer
            self.projector = None
        self.clear()

    def _encoder_input(self, module: Any, args: Any, kwargs: Any) -> None:
        if not self.active:
            return
        lengths = kwargs.get("feature_lens", args[1] if len(args) > 1 else None)
        if lengths is None:
            raise InvariantError("Encoder attention reference requires raw audio feature lengths.")
        chunks = [qwen_audio_chunk_lengths_after_cnn(int(n), self.tower.n_window) for n in lengths.tolist()]
        self.audio_chunk_counts = [len(row) for row in chunks]
        self.chunk_lengths = [n for row in chunks for n in row]

    def _attention_input(self, module: Any, args: Any, kwargs: Any) -> None:
        if not self.active:
            return
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        cu = kwargs.get("cu_seqlens", args[1] if len(args) > 1 else None)
        if hidden is None or cu is None:
            raise InvariantError("Encoder attention reference requires normalized inputs and chunk boundaries.")
        self.source = hidden.detach()
        self.chunk_lengths = [int(n) for n in cu.diff().tolist()]

    def _encoder_layout(self, strategy: Any, plan: Any, chunks: list[int], chunk_counts: list[int]) -> None:
        if self.previous_encoder_observer is not None:
            self.previous_encoder_observer(strategy, plan, chunks, chunk_counts)
        if not self.active:
            return
        if self.source is not None:
            with _torch().no_grad():
                self.source = getattr(strategy, "map_reference", strategy.replay)(self.source, plan).detach()
        self.chunk_lengths = list(chunks)
        self.audio_chunk_counts = list(chunk_counts)

    def _pool_scores(self) -> None:
        if self.audio_scores or self.sequence_scores is not None:
            return
        if self.source is None or not self.audio_chunk_counts:
            raise InvariantError("LLM encoder-attention selection requires an encoder forward in the current request.")
        from .encoder_compression import mean_received_attention_scores

        torch = _torch()
        started = perf_counter() if self.profile_enabled else None
        with torch.no_grad():
            source = self.source.to(_module_execution_device(self.attention.q_proj))
            audio_lengths = []
            chunk_offset = 0
            for count in self.audio_chunk_counts:
                audio_lengths.append(sum(self.chunk_lengths[chunk_offset:chunk_offset + count]))
                chunk_offset += count
            if self.key_vectors:
                keys = self.attention.k_proj(source)
                self.audio_scores = [values[:length // 2 * 2].reshape(-1, 2, keys.shape[-1]).mean(1)
                                     for length, values in zip(audio_lengths, keys.split(audio_lengths))]
            elif self.distinct_importance:
                components = sa_map_audio_importance(self.attention, source, self.chunk_lengths, audio_lengths)
                scores = torch.stack((components["merge"], components["prune"]), dim=-1)
                self.audio_scores = list(scores.split([length // 2 for length in audio_lengths]))
            else:
                scores = mean_received_attention_scores(self.attention, source, self.chunk_lengths)
                self.audio_scores = [values[:length // 2 * 2].reshape(-1, 2).mean(-1)
                                     for length, values in zip(audio_lengths, scores.split(audio_lengths))]
        self.source = None
        if started is not None:
            self.score_seconds += perf_counter() - started

    def _projector_input(self, module: Any, args: Any) -> None:
        if self.active:
            self._pool_scores()

    def _projector_layout(self, strategy: Any, item_index: int, indices: Any, details: Any, scores: Any, keep_count: int) -> None:
        if self.previous_projector_observer is not None:
            self.previous_projector_observer(strategy, item_index, indices, details, scores, keep_count)
        if not self.active:
            return
        self._pool_scores()
        with _torch().no_grad():
            values = self.audio_scores[item_index]
            mapped = strategy.map_reference(
                values if values.ndim > 1 else values.unsqueeze(-1), indices=indices,
                details=details, scores=scores, keep_count=keep_count,
            )
            self.audio_scores[item_index] = (mapped if values.ndim > 1 else mapped.flatten()).detach()

    def scores_for(self, request: Any, positions: Any) -> Any:
        self._pool_scores()
        torch = _torch()
        if self.sequence_scores is None:
            if len(self.audio_scores) != len(request.audio_units):
                raise InvariantError("Encoder attention reference does not match LLM audio units.")
            shape = (*request.original_attention_mask.shape, *self.audio_scores[0].shape[1:])
            rows = torch.zeros(shape, dtype=self.audio_scores[0].dtype if self.key_vectors else torch.float32, device=positions.device)
            for unit, values in zip(request.audio_units, self.audio_scores):
                if values.shape[0] != int(unit["original_count"]):
                    raise InvariantError("Encoder attention reference does not match the LLM audio span length.")
                rows[int(unit["sample_index"]), int(unit["span_start"]):int(unit["span_end"])] = values.to(rows.device)
            self.sequence_scores = rows
            self.audio_scores.clear()
        indices = positions.clamp_min(0)
        if self.sequence_scores.ndim == 3:
            indices = indices.unsqueeze(-1).expand(-1, -1, self.sequence_scores.shape[-1])
        return self.sequence_scores.to(positions.device).gather(1, indices)

    def map_llm_keys(self, request: Any, positions: Any, *, sample: int, slots: Any,
                     kept_absolute: Any, replay: Any, replay_plan: Any) -> None:
        if not self.key_vectors:
            return
        with _torch().no_grad():
            values = self.scores_for(request, positions)[sample, slots]
            mapped = replay(hidden_states=values, replay_plan=replay_plan)
            self.sequence_scores[sample, kept_absolute.to(self.sequence_scores.device)] = mapped.to(self.sequence_scores.device).detach()
