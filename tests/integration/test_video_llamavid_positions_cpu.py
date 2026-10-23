"""Real BERT/QFormer pooling on both sides of its own output projectors."""

import copy
import math
from dataclasses import replace

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig

from tests.fixtures import native_image as image_fixture
from tests.integration.test_video_vqtoken_llava_onevision_positions_cpu import _video_model


BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")


def _model(backbone):
    return _video_model() if backbone == "llava_onevision" else image_fixture.qwen_model(backbone)


def _qformer(raw_hidden=16, output_hidden=24, vocab_size=32, dropout=0.0):
    from transformers import BertConfig, BertModel
    from mllm_comp.backbones._llamavid_torch import LLaMAVIDQFormerModule
    config = BertConfig(vocab_size=vocab_size, hidden_size=12, intermediate_size=20,
        num_hidden_layers=2, num_attention_heads=3, hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout, max_position_embeddings=64)
    config._attn_implementation = "eager"
    return LLaMAVIDQFormerModule(BertModel(config, add_pooling_layer=False),
        raw_hidden_size=raw_hidden, output_hidden_size=output_hidden)


def _tokenizer(tmp_path):
    from transformers import BertTokenizer
    path = tmp_path / "vocab.txt"
    path.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwhich\naction\noccurs\n?\n[DEC]\n", encoding="utf-8")
    return BertTokenizer(vocab_file=str(path), truncation_side="left")


def _stage(backbone, position=None, *, phase="forward"):
    return CompressionStageConfig.from_dict({
        "name": "llamavid-own-projectors", "placement": "projector_level", "execution_phase": phase,
        "plugin": "llamavid", "algorithm": "llamavid", "variant": "llamavid.video",
        "binding": f"llamavid.video.{backbone}", "target_modalities": ["video"], "config": {},
        "operation": {} if position is None else {"targets": {"qformer_context_content_pool": {"target_site": position}}},
    })


def _activation(model, stage, tokenizer, qformer=None):
    activation = image_fixture.activation(model, stage)
    activation.signal_provider._tokenizer = tokenizer
    module = _qformer(vocab_size=len(tokenizer)) if qformer is None else qformer
    projector = model.model.mm_projector if activation.backbone == "llava_onevision" else model.model.visual.merger
    reference = next(projector.parameters())
    activation.qformer_module = module.to(device=reference.device, dtype=reference.dtype)
    return activation


def _inputs(torch, backbone):
    if backbone == "llava_onevision":
        ids = torch.tensor([[1, -200, 2, 3]])
        values = dict(images=[torch.randn(2, 3, 54, 54, requires_grad=True)],
            image_sizes=[(54, 54)], modalities=["video"])
    else:
        ids = torch.tensor([[1, *([7, *([11] * 4), 8] * 2), 2]]) if backbone == "qwen3_vl" else torch.tensor([[1, 7, *([11] * 8), 8, 2]])
        values = dict(pixel_values_videos=torch.randn(32, 12, requires_grad=True),
            video_grid_thw=torch.tensor([[2, 4, 4]]),
            **({"second_per_grid_ts": torch.tensor([0.5])} if backbone == "qwen25_vl" else {}))
    labels = torch.full_like(ids, -100)
    labels[0, -1] = ids[0, -1]
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False, **values)


def _request_context(activation):
    metadata = {"conditioning_text": "Which action occurs?"} if activation.backbone == "llava_onevision" else {"video_conditioning_texts": ["Which action occurs?"]}
    return activation._bridge.request_context(metadata)


def _formula(torch, module, raw, query_hidden, position):
    query = module.vlm_att_projector(query_hidden)
    keys = module.vlm_att_key_projector(raw)
    attention = (query @ keys.transpose(-1, -2) / math.sqrt(raw.shape[-1])).softmax(-1)
    if position == "before":
        context = module.vlm_att_val_projector((attention @ raw).mean(1, keepdim=True))
        content = module.content_projector(raw.mean(1, keepdim=True))
    else:
        context = (attention @ module.vlm_att_val_projector(raw)).mean(1, keepdim=True)
        content = module.content_projector(raw).mean(1, keepdim=True)
    return torch.cat((context, content), dim=1), attention.mean(1)


def test_real_bert_pooling_formula_gradient_full_projector_inputs_and_strict_reload():
    torch = pytest.importorskip("torch")
    torch.manual_seed(471)
    module = _qformer(raw_hidden=3, output_hidden=5).train()
    raw = torch.randn(2, 9, 3, requires_grad=True)
    tokens = {"input_ids": torch.tensor([[2, 5, 6, 3]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    calls, queries = {"context": [], "content": []}, []
    handles = [module.vlm_att_val_projector.register_forward_pre_hook(lambda m, a: calls["context"].append(a[0].shape)),
        module.content_projector.register_forward_pre_hook(lambda m, a: calls["content"].append(a[0].shape)),
        module.vlm_att_projector.register_forward_pre_hook(lambda m, a: queries.append(a[0]))]
    baseline = None
    try:
        for position in (None, "before", "after"):
            calls["context"].clear()
            calls["content"].clear()
            queries.clear()
            actual, weights = module.compress_video(raw, text_context="Which action occurs?", text_tokens=tokens,
                num_query=32, **({} if position is None else {"pool_position": position}))
            assert calls == {name: [torch.Size([2, 9 if position == "after" else 1, 3])] for name in calls}
            assert queries[0].shape == (2, 32, 12)
            expected, expected_weights = _formula(torch, module, raw, queries[0], position or "before")
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(weights, expected_weights)
            parameters = (raw, *module.parameters())
            upstream = torch.randn_like(actual)
            torch.testing.assert_close(torch.autograd.grad(actual, parameters, upstream, retain_graph=True),
                torch.autograd.grad(expected, parameters, upstream, retain_graph=True), atol=2e-6, rtol=2e-5)
            if baseline is None:
                baseline = (actual.detach().clone(), weights.detach().clone())
            elif position == "before":
                torch.testing.assert_close((actual, weights), baseline, atol=0, rtol=0)
            else:
                torch.testing.assert_close(weights, baseline[1], atol=0, rtol=0)
                torch.testing.assert_close(actual[:, 0], baseline[0][:, 0], atol=2e-6, rtol=2e-5)
                assert not torch.allclose(actual[:, 1], baseline[0][:, 1], atol=1e-7, rtol=1e-6)
        actual.square().sum().backward()
        assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        clone = _qformer(raw_hidden=3, output_hidden=5).train()
        clone.load_state_dict(copy.deepcopy(module.state_dict()), strict=True)
        torch.testing.assert_close(clone.compress_video(raw, text_context="Which action occurs?", text_tokens=tokens,
            num_query=32, pool_position="after"), (actual, weights), atol=0, rtol=0)
        with pytest.raises(RuntimeError, match="Missing key"):
            clone.load_state_dict({}, strict=True)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("backbone", BACKBONES)
def test_native_learned_pooling_sites_raw_source_default_alias_lineage_and_repeat(backbone, tmp_path):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.video.positions import video_projector_site
    from mllm_comp.compressors.video import replay_llamavid_value_lineage

    torch.manual_seed(472)
    base = _model(backbone).train()
    tokenizer = _tokenizer(tmp_path)
    qformer_base = _qformer(vocab_size=len(tokenizer)).eval()
    baseline = None
    for position in (None, video_projector_site(backbone, "after"), "projector_output"):
        model = copy.deepcopy(base)
        qformer = copy.deepcopy(qformer_base)
        activation = _activation(model, _stage(backbone, position), tokenizer, qformer)
        activation.binding = replace(activation.binding, metadata={**activation.binding.metadata, "verified_phases": ["forward"]})
        requests, results, queries, projector_inputs, decoder_inputs = [], [], [], [], []
        callback = activation._projector_callback

        def capture(request):
            response = callback(request)
            requests.append(request)
            results.append(response)
            return response

        activation._projector_callback = capture
        activation.install()
        assert all(module.training for module in qformer.modules())
        registered = {id(p) for p in model.parameters()}
        assert all(id(p) in registered for p in qformer.parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        assert all(id(p) in {id(v) for group in optimizer.param_groups for v in group["params"]} for p in qformer.parameters())
        assert activation.runtime_metadata()["qformer_registered_pre_optimizer"] is True
        before = position != "projector_output"
        actual_site = "llamavid_pre_output_projectors" if before else "llamavid_post_output_projectors"
        decoder = model.model if backbone == "llava_onevision" else model.model.language_model
        handles = [qformer.vlm_att_projector.register_forward_pre_hook(lambda m, a: queries.append(a[0])),
            qformer.content_projector.register_forward_pre_hook(lambda m, a: projector_inputs.append(a[0])),
            decoder.register_forward_pre_hook(lambda m, a, kw: decoder_inputs.append(kw.get("inputs_embeds")), with_kwargs=True)]
        previous = None
        try:
            for repeat in range(2 if position == "projector_output" else 1):
                requests.clear()
                results.clear()
                queries.clear()
                projector_inputs.clear()
                decoder_inputs.clear()
                model.zero_grad(set_to_none=True)
                torch.manual_seed(473)
                args = _inputs(torch, backbone)
                with _request_context(activation):
                    output = model(**args)
                assert len(requests) == len(results) == 1
                request = requests[0]
                result = results[0].compression_result if backbone == "qwen3_vl" else results[0]
                raw = request.payload.raw_features.reshape(2, -1, 16)
                assert projector_inputs[0].shape == (2, 1 if before else raw.shape[1], 16)
                expected, weights = _formula(torch, qformer, raw, queries[0], "before" if before else "after")
                torch.testing.assert_close(result.features, expected.flatten(0, 1))
                assert result.result_count == 4
                assert result.metadata["tokens_per_temporal_unit"] == 2
                assert request.payload.features.shape[-1] == result.features.shape[-1] == 24
                assert result.metadata["pool_position"] == ("before" if before else "after")
                assert activation._bridge.runtime_metadata()["projector_position"] == "after"
                if backbone == "qwen3_vl":
                    for source, branch in zip(request.deepstack_features, results[0].deepstack_features):
                        torch.testing.assert_close(branch, replay_llamavid_value_lineage(source, result))
                elif backbone == "llava_onevision":
                    entries = [(request.packing.patch_positions[source], result.features[i]) for i, source in enumerate(result.representative_indices[0])]
                    entries += [(i, model.model.image_newline) for i in request.packing.structural_positions]
                    expected_splice = torch.stack([value for _, value in sorted(entries, key=lambda item: item[0])])
                    torch.testing.assert_close(decoder_inputs[0][0, 1:-2], expected_splice)
                output.loss.backward()
                pixels = args["images"][0] if backbone == "llava_onevision" else args["pixel_values_videos"]
                assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in qformer.parameters())
                if backbone == "llava_onevision":
                    assert model.model.image_newline.grad is not None and model.model.image_newline.grad.abs().sum() > 0
                observed = (output.logits.detach().clone(), pixels.grad.clone(), result.features.detach().clone())
                if previous is not None:
                    torch.testing.assert_close(observed, previous, atol=0, rtol=0)
                previous = observed
                assert activation.signal_provider._active.get() is None
                metadata = activation.runtime_metadata()
                assert metadata["target_site"] == metadata["algorithm_records"][0]["target_site"] == actual_site
                assert metadata["projector_layout_deltas"][0]["site"] == actual_site
                assert metadata["real_checkpoint_verified"] is before
                assert metadata["output_projectors_before_pool"] is not before
            if baseline is None:
                baseline = previous
            elif before:
                torch.testing.assert_close(previous, baseline, atol=0, rtol=0)
            else:
                assert not torch.allclose(previous[2][1::2], baseline[2][1::2], atol=1e-7, rtol=1e-6)
            state = copy.deepcopy(activation.compression_state_dict())
            activation.load_compression_state_dict(state)
            assert set(state) == set(qformer.state_dict())
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()
        assert not hasattr(model, "mllm_comp_llamavid_qformer")
        assert not activation.signal_provider.installed


def test_legacy_injected_qformer_default_signature_and_explicit_post_error():
    from mllm_comp.compressors.video import llamavid_video, LLaMAVIDVideoKernel
    from mllm_comp.core.errors import BackendIntegrationError
    from tests.unit.test_llamavid_video_kernel import _TinyQFormer, _payload
    torch = pytest.importorskip("torch")
    custom = _TinyQFormer()
    payload = _payload(qformer=custom)
    torch.testing.assert_close(llamavid_video(payload).features,
        LLaMAVIDVideoKernel()(payload, pool_position="before").features, atol=0, rtol=0)
    with pytest.raises(BackendIntegrationError, match="injected QFormer does not expose this position"):
        llamavid_video(payload, pool_position="after")

    class IgnoredKeywords:
        def compress_video(self, *args, **kwargs):
            kwargs.pop("pool_position", None)
            return custom.compress_video(*args, **kwargs)

    ignored = _payload(qformer=IgnoredKeywords())
    with pytest.raises(BackendIntegrationError, match="injected QFormer does not expose this position"):
        llamavid_video(ignored, pool_position="after")


@pytest.mark.parametrize("backbone", BACKBONES)
def test_native_post_pooling_generate_two_tokens_cache_and_request_cleanup(backbone, tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(474)
    model = _model(backbone).eval()
    tokenizer = _tokenizer(tmp_path)
    activation = _activation(model, _stage(backbone, "projector_output", phase="generate"), tokenizer,
        _qformer(vocab_size=len(tokenizer), dropout=0.2))
    requests = []
    callback = activation._projector_callback

    def capture(request):
        requests.append(request)
        return callback(request)

    activation._projector_callback = capture
    activation.install()
    assert all(not module.training for module in activation.qformer_module.modules())
    decoder = model.model if backbone == "llava_onevision" else model.model.language_model
    lengths = []
    if backbone == "llava_onevision":
        handle = decoder.register_forward_pre_hook(
            lambda module, args, kwargs: lengths.append(kwargs["inputs_embeds"].shape[1])
            if kwargs.get("inputs_embeds") is not None else None, with_kwargs=True)
    else:
        # The Qwen bridge compacts inside language_model.forward, after that
        # module's pre-hooks. Layer 0 observes the actual decoder sequence.
        handle = decoder.layers[0].register_forward_pre_hook(
            lambda module, args, kwargs: lengths.append(
                (args[0] if args else kwargs["hidden_states"]).shape[1]), with_kwargs=True)
    args = _inputs(torch, backbone)
    args.pop("labels")
    args["use_cache"] = True
    if backbone == "llava_onevision":
        args["inputs"] = args.pop("input_ids")
    previous = None
    try:
        for repeat in range(2):
            lengths.clear()
            with torch.no_grad(), _request_context(activation):
                generated = model.generate(**args, min_new_tokens=2, max_new_tokens=2, do_sample=False,
                    eos_token_id=None, pad_token_id=0, return_dict_in_generate=True)
            assert len(requests) == repeat + 1
            assert activation.runtime_metadata()["projector_invocations"] == repeat + 1
            assert activation.runtime_metadata()["signal_provider"]["capture_count"] == repeat + 1
            request = requests[-1]
            if backbone == "llava_onevision":
                prefill_count = 3 + 4 + len(request.packing.structural_positions)
            else:
                prefill_count = args["input_ids"].shape[1] - 8 + 4
            assert lengths[0] == prefill_count
            assert generated.past_key_values.get_seq_length() == prefill_count + 1
            assert all(layer.get_seq_length() == prefill_count + 1 for layer in generated.past_key_values.layers)
            if previous is not None:
                torch.testing.assert_close(generated.sequences, previous, atol=0, rtol=0)
            previous = generated.sequences
            assert activation.signal_provider._active.get() is None
            assert activation.runtime_metadata()["canonical_evidence_applicable"] is False
        assert requests[0].invocation_id != requests[1].invocation_id
    finally:
        handle.remove()
        activation.uninstall()
    assert not activation.signal_provider.installed
    assert not hasattr(model, "mllm_comp_llamavid_qformer")
