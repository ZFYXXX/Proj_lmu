"""Reference-initialized trainable LLaMA-VID QFormer components.

The local BERT asset supplies embeddings and ordinary BERT layers.  Every
second layer receives the released query-only visual cross-attention block;
those blocks and the query/projector parameters are newly initialized, matching
the reference construction rather than claiming a method-specific checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.compressors.video import _rules as rule_contract
from mllm_comp.compressors.video.llamavid import custom_llamavid_pool


class _QueryCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        bert_hidden_size: int,
        encoder_width: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        initializer_range: float,
    ) -> None:
        super().__init__()
        if bert_hidden_size % num_heads:
            raise BackendIntegrationError(
                "LLaMA-VID BERT hidden size must be divisible by its heads"
            )
        self.num_heads = num_heads
        self.head_size = bert_hidden_size // num_heads
        self.query = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.key = nn.Linear(encoder_width, bert_hidden_size)
        self.value = nn.Linear(encoder_width, bert_hidden_size)
        self.output = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(bert_hidden_size, eps=layer_norm_eps)
        self._reset_parameters(initializer_range)

    def _reset_parameters(self, initializer_range: float) -> None:
        for module in (self.query, self.key, self.value, self.output):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

    def _heads(self, value: Any) -> Any:
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.num_heads, self.head_size).permute(
            0, 2, 1, 3
        )

    def forward(self, queries: Any, encoder: Any) -> Any:
        query = self._heads(self.query(queries))
        key = self._heads(self.key(encoder))
        value = self._heads(self.value(encoder))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_size)
        probabilities = self.dropout(scores.softmax(dim=-1))
        context = torch.matmul(probabilities, value)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(queries.shape[0], queries.shape[1], -1)
        )
        context = self.output_dropout(self.output(context))
        return self.layer_norm(queries + context)


class LLaMAVIDQFormerModule(nn.Module):
    """BERT QFormer plus context/content projectors for one backbone."""

    def __init__(
        self,
        bert_model: Any,
        *,
        raw_hidden_size: int,
        output_hidden_size: int,
        num_query: int = 32,
        cross_attention_frequency: int = 2,
    ) -> None:
        super().__init__()
        config = getattr(bert_model, "config", None)
        embeddings = getattr(bert_model, "embeddings", None)
        encoder = getattr(bert_model, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if config is None or embeddings is None or layers is None:
            raise BackendIntegrationError(
                "LLaMA-VID local BERT model does not expose embeddings/encoder layers"
            )
        bert_hidden = int(getattr(config, "hidden_size", 0))
        num_heads = int(getattr(config, "num_attention_heads", 0))
        if (
            raw_hidden_size < 1
            or output_hidden_size < 1
            or num_query != 32
            or cross_attention_frequency != 2
            or bert_hidden < 1
            or num_heads < 1
        ):
            raise BackendIntegrationError(
                "LLaMA-VID QFormer dimensions/configuration are invalid"
            )
        self.config = config
        self.raw_hidden_size = raw_hidden_size
        self.output_hidden_size = output_hidden_size
        self.num_query = num_query
        self.bert_embeddings = embeddings
        self.bert_layers = layers
        self.cross_attention = nn.ModuleDict(
            {
                str(index): _QueryCrossAttention(
                    bert_hidden_size=bert_hidden,
                    encoder_width=raw_hidden_size,
                    num_heads=num_heads,
                    dropout=float(getattr(config, "attention_probs_dropout_prob", 0.1)),
                    layer_norm_eps=float(getattr(config, "layer_norm_eps", 1e-12)),
                    initializer_range=float(getattr(config, "initializer_range", 0.02)),
                )
                for index in range(0, len(layers), cross_attention_frequency)
            }
        )
        self.query_tokens = nn.Parameter(torch.empty(1, num_query, bert_hidden))
        nn.init.normal_(
            self.query_tokens,
            mean=0.0,
            std=float(getattr(config, "initializer_range", 0.02)),
        )
        self.vlm_att_projector = nn.Linear(bert_hidden, raw_hidden_size)
        self.vlm_att_key_projector = nn.Linear(raw_hidden_size, raw_hidden_size)
        self.vlm_att_val_projector = nn.Linear(raw_hidden_size, output_hidden_size)
        self.content_projector = nn.Sequential(
            nn.Linear(raw_hidden_size, output_hidden_size),
            nn.GELU(),
            nn.Linear(output_hidden_size, output_hidden_size),
        )

    @classmethod
    def from_local_bert(
        cls,
        source: str | Path,
        *,
        vocab_size: int,
        raw_hidden_size: int,
        output_hidden_size: int,
        num_query: int = 32,
    ) -> "LLaMAVIDQFormerModule":
        path = Path(source).expanduser()
        if not path.is_dir():
            raise BackendIntegrationError(
                f"LLaMA-VID BERT directory does not exist: {path}"
            )
        try:
            from transformers import BertConfig, BertModel
        except ImportError as exc:
            raise BackendIntegrationError(
                "LLaMA-VID requires BERT support from the existing Transformers runtime"
            ) from exc
        try:
            config = BertConfig.from_pretrained(str(path), local_files_only=True)
            bert = BertModel.from_pretrained(
                str(path),
                config=config,
                add_pooling_layer=False,
                local_files_only=True,
            )
            bert.resize_token_embeddings(vocab_size)
        except Exception as exc:
            raise BackendIntegrationError(
                "LLaMA-VID could not load the supplied local BERT asset without downloads"
            ) from exc
        return cls(
            bert,
            raw_hidden_size=raw_hidden_size,
            output_hidden_size=output_hidden_size,
            num_query=num_query,
            cross_attention_frequency=2,
        )

    def compress_video(
        self,
        raw_features: Any,
        *,
        text_context: str,
        text_tokens: Mapping[str, Any] | None,
        num_query: int,
        pool_position: str = "before",
        rules: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        if pool_position not in {"before", "after"}:
            raise BackendIntegrationError("LLaMA-VID pool_position must be before or after its output projectors")
        if num_query != self.num_query:
            raise BackendIntegrationError("LLaMA-VID query count changed")
        if (
            not isinstance(raw_features, torch.Tensor)
            or raw_features.ndim != 3
            or int(raw_features.shape[-1]) != self.raw_hidden_size
        ):
            raise BackendIntegrationError(
                "LLaMA-VID raw features must be [temporal, patches, raw_hidden]"
            )
        if not isinstance(text_context, str) or not text_context.strip():
            raise BackendIntegrationError("LLaMA-VID requires non-empty text context")
        if not isinstance(text_tokens, Mapping):
            raise BackendIntegrationError(
                "LLaMA-VID provider did not attach local BERT tokens"
            )
        input_ids = text_tokens.get("input_ids")
        attention_mask = text_tokens.get("attention_mask")
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or int(input_ids.shape[0]) != 1
            or not isinstance(attention_mask, torch.Tensor)
            or tuple(attention_mask.shape) != tuple(input_ids.shape)
        ):
            raise BackendIntegrationError(
                "LLaMA-VID BERT tokens must be one input_ids/attention_mask row"
            )
        temporal = int(raw_features.shape[0])
        input_ids = input_ids.to(device=raw_features.device).expand(temporal, -1)
        attention_mask = attention_mask.to(device=raw_features.device).expand(
            temporal, -1
        )
        query_hidden = self._qformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            raw_features=raw_features,
        )
        if rules and "query_transform" in rules:
            query_hidden = rule_contract.value(rules["query_transform"],
                {"values": query_hidden, "raw_features": raw_features, "module": self,
                 "input_ids": input_ids, "attention_mask": attention_mask,
                 "default_transform": lambda: query_hidden}, reference=query_hidden,
                shape=tuple(query_hidden.shape), name="query_transform")
        text_queries = self.vlm_att_projector(query_hidden)
        keys = self.vlm_att_key_projector(raw_features)
        scores = torch.matmul(text_queries, keys.transpose(-1, -2)) / math.sqrt(
            self.raw_hidden_size
        )
        if rules and "context_scorer" in rules:
            scores = rule_contract.value(rules["context_scorer"],
                {"values": raw_features, "queries": text_queries, "keys": keys,
                 "default_scores": scores}, reference=scores,
                shape=tuple(scores.shape), name="context_scorer")
        per_query_weights = scores.softmax(dim=-1)
        context_weights = per_query_weights.mean(dim=1)
        if pool_position == "before":
            context_raw = torch.matmul(per_query_weights, raw_features).mean(dim=1)
            if rules and "context_aggregator" in rules:
                context_raw = torch.stack(tuple(custom_llamavid_pool(raw_features[t], context_weights[t],
                    rules["context_aggregator"], kind="context", default_value=context_raw[t]) for t in range(temporal)))
            context = self.vlm_att_val_projector(context_raw[:, None, :])
            content_raw = raw_features.mean(dim=1, keepdim=True)
            if rules and "content_aggregator" in rules:
                content_raw = torch.stack(tuple(custom_llamavid_pool(raw_features[t],
                    torch.full_like(context_weights[t], 1.0 / int(raw_features.shape[1])),
                    rules["content_aggregator"], kind="content", default_value=content_raw[t, 0])
                    for t in range(temporal)))[:, None, :]
            content = self.content_projector(content_raw)
        else:
            # QFormer attention keeps its raw-vision source. Only the values
            # pooled by the existing two output paths change spaces.
            context_values = self.vlm_att_val_projector(raw_features)
            context = torch.matmul(per_query_weights, context_values).mean(dim=1, keepdim=True)
            if rules and "context_aggregator" in rules:
                context = torch.stack(tuple(custom_llamavid_pool(context_values[t], context_weights[t],
                    rules["context_aggregator"], kind="context", default_value=context[t, 0])
                    for t in range(temporal)))[:, None, :]
            content_values = self.content_projector(raw_features)
            content = content_values.mean(dim=1, keepdim=True)
            if rules and "content_aggregator" in rules:
                content = torch.stack(tuple(custom_llamavid_pool(content_values[t],
                    torch.full_like(context_weights[t], 1.0 / int(raw_features.shape[1])),
                    rules["content_aggregator"], kind="content", default_value=content[t, 0])
                    for t in range(temporal)))[:, None, :]
        return torch.cat((context, content), dim=1), context_weights

    def _qformer(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        raw_features: Any,
    ) -> Any:
        temporal, text_length = input_ids.shape
        position_ids = (
            torch.arange(
                text_length,
                dtype=torch.long,
                device=input_ids.device,
            )
            .unsqueeze(0)
            .expand(temporal, -1)
        )
        text = self.bert_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
        )
        query = self.query_tokens.expand(temporal, -1, -1)
        query = self.bert_embeddings.LayerNorm(query)
        query = self.bert_embeddings.dropout(query)
        hidden = torch.cat((query, text), dim=1)
        query_mask = torch.ones(
            temporal,
            self.num_query,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask = torch.cat((query_mask, attention_mask), dim=1)
        extended_mask = (1.0 - full_mask[:, None, None, :].to(hidden.dtype)) * (
            torch.finfo(hidden.dtype).min
        )
        for index, layer in enumerate(self.bert_layers):
            attention_output = layer.attention(
                hidden,
                attention_mask=extended_mask,
                output_attentions=False,
            )[0]
            cross = (
                self.cross_attention[str(index)]
                if str(index) in self.cross_attention
                else None
            )
            if cross is not None:
                query_output = cross(
                    attention_output[:, : self.num_query],
                    raw_features,
                )
                attention_output = torch.cat(
                    (query_output, attention_output[:, self.num_query :]),
                    dim=1,
                )
            hidden = layer.output(
                layer.intermediate(attention_output), attention_output
            )
        return hidden[:, : self.num_query]


__all__ = ["LLaMAVIDQFormerModule"]
