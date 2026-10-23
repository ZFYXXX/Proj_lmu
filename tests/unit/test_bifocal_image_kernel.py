from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mllm_comp.compressors.image.bifocal import (  # noqa: E402
    BifocalSpec,
    bifocal_extract_query,
    bifocal_image_tokens,
    bifocal_query_text_from_ids,
)
from mllm_comp.core.layout import OrderingPolicy  # noqa: E402


_REF = "hope"
_REF_ENV = _REF.upper()
_REF_CFG = f"{_REF.capitalize()}Cfg"
_REF_STAGE1 = f"{_REF}_stage1"
_REF_SECTION_START = f"_{_REF_ENV}_N_LAYERS = 28"

_DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[4]
    / f"slurm_{_REF}_vg"
    / "qwen2_5_vl"
    / "modeling_qwen2_5_vl.py"
)
_KNOBS = (
    "s1_frac",
    "block",
    "coarse_frac",
    "k_frac",
    "clsr",
    "no_query",
    "respool",
    "respool_pos",
    "metric",
    "metric_agg",
    "rank_map",
    "rank_mix",
    "nrm_sign",
    "sink_clamp",
    "s1_merge",
    "s1_mu",
)


def _ref_env(name: str) -> str:
    return f"{_REF_ENV}_{name.upper()}"


def _reference_path() -> Path | None:
    raw = os.environ.get("BIFOCAL_REFERENCE_MODELING")
    path = Path(raw) if raw else _DEFAULT_REFERENCE
    return path if path.is_file() else None


def _load_reference() -> dict:
    path = _reference_path()
    if path is None:
        pytest.skip("Bifocal reference modeling_qwen2_5_vl.py not available")
    text = path.read_text(encoding="utf-8")
    start = text.index(_REF_SECTION_START)
    end = text.index("class Qwen2_5_VLMLP")
    namespace: dict = {"torch": torch, "F": F, "math": math, "os": os}
    exec(compile(text[start:end], str(path), "exec"), namespace)
    return namespace


def _to_env(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _reference_cfg(ref: dict, knobs: dict, monkeypatch) -> object:
    for key in (_ref_env("s2_frac"), _ref_env("s2_smooth"), *(_ref_env(k) for k in _KNOBS)):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(_ref_env("s2_frac"), "0")
    monkeypatch.setenv(_ref_env("s2_smooth"), "0")
    for name, value in knobs.items():
        monkeypatch.setenv(_ref_env(name), _to_env(value))
    return ref[_REF_CFG](f"{_REF}_b20", s1_frac=0.25, s2_frac=0.0)


def _inputs(H: int, W: int, D: int = 48, n_word: int = 4, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    N = H * W
    embeds = torch.randn(N, D, generator=gen).to(torch.bfloat16)
    mass = torch.rand(N, generator=gen) * 3.0
    mass[:W] += 4.0
    maps = {
        "mass": mass,
        "nodiag": mass - torch.rand(N, generator=gen) * 0.2,
        "vmass": mass * (0.5 + torch.rand(N, generator=gen)),
    }
    words = torch.randn(n_word, D, generator=gen).to(torch.bfloat16)
    return embeds, maps, words


_CASES = [
    ({}, (14, 18)),
    ({}, (19, 14)),
    ({"s1_frac": 0.20}, (15, 23)),
    ({"s1_frac": 0.50, "block": 3}, (14, 18)),
    ({"k_frac": 0.5}, (14, 18)),
    ({"clsr": "0"}, (14, 18)),
    ({"clsr": "nrm"}, (14, 18)),
    ({"nrm_sign": -1}, (14, 18)),
    ({"nrm_sign": 0}, (14, 18)),
    ({"respool": False}, (14, 18)),
    ({"respool_pos": True}, (14, 18)),
    ({"metric": "uniform"}, (14, 18)),
    ({"metric": "norm", "clsr": "0"}, (14, 18)),
    ({"metric_agg": "vmass"}, (14, 18)),
    ({"rank_map": "vmass"}, (14, 18)),
    ({"rank_map": "norm", "clsr": "nrm"}, (14, 18)),
    ({"rank_mix": 0.5}, (14, 18)),
    ({"sink_clamp": 0.995}, (14, 18)),
    ({"s1_merge": "near", "s1_mu": 0.5}, (14, 18)),
    ({"no_query": True}, (14, 18)),
    ({"coarse_frac": 0.15}, (32, 40)),
]


def _reference_run(ref, cfg, embeds, maps, words, H, W, knobs):
    metric = maps.get(knobs.get("metric_agg", "mass"))
    metric_rank = (
        maps.get(knobs["rank_map"])
        if knobs.get("rank_map") in ("mass", "vmass", "nodiag")
        else None
    )
    return ref[_REF_STAGE1](embeds, metric, H, W, words, cfg, metric_rank=metric_rank)


@pytest.mark.parametrize("knobs,grid", _CASES)
def test_bit_exact_against_original_stage1(knobs, grid, monkeypatch):
    ref = _load_reference()
    H, W = grid
    embeds, maps, words = _inputs(H, W)
    cfg = _reference_cfg(ref, knobs, monkeypatch)
    expected = _reference_run(ref, cfg, embeds, maps, words, H, W, knobs)
    spec = BifocalSpec(**knobs)
    got = bifocal_image_tokens(
        embeds, maps, grid_height=H, grid_width=W, word_embeddings=words, spec=spec
    )
    assert expected is not None
    tok_ref, idx_ref, _ = expected
    assert torch.equal(got.indices, idx_ref)
    assert got.tokens.dtype == tok_ref.dtype
    assert torch.equal(got.tokens, tok_ref)


def test_query_missing_degrades_like_original(monkeypatch):
    ref = _load_reference()
    H, W = 14, 18
    embeds, maps, _ = _inputs(H, W)
    cfg = _reference_cfg(ref, {}, monkeypatch)
    tok_ref, idx_ref, _ = ref[_REF_STAGE1](embeds, maps["mass"], H, W, None, cfg)
    got = bifocal_image_tokens(
        embeds, maps, grid_height=H, grid_width=W, word_embeddings=None, spec=BifocalSpec()
    )
    assert got.metadata["chain"] == "attn"
    assert torch.equal(got.indices, idx_ref)
    assert torch.equal(got.tokens, tok_ref)


def test_budget_matches_original_for_many_grids(monkeypatch):
    ref = _load_reference()
    for s1 in (0.2, 0.25, 0.3125, 0.5, 0.625):
        cfg = _reference_cfg(ref, {"s1_frac": s1}, monkeypatch)
        spec = BifocalSpec(s1_frac=s1)
        for H in range(4, 40, 3):
            for W in range(4, 52, 5):
                N = H * W
                b_ref = cfg.budget(N, H, W)
                b_new = spec.budget(N, H, W)
                if b_ref is None:
                    assert b_new is None
                else:
                    NT, C, n_fine, k, _K = b_ref
                    assert b_new == (NT, C, n_fine, k)


@pytest.mark.parametrize(
    "knobs", [{}, {"respool": False}, {"respool_pos": True}, {"s1_merge": "near"}]
)
def test_layout_delta_is_complete_and_source_ordered(knobs):
    H, W = 14, 18
    embeds, maps, words = _inputs(H, W, seed=3)
    result = bifocal_image_tokens(
        embeds, maps, grid_height=H, grid_width=W, word_embeddings=words,
        spec=BifocalSpec(**knobs),
    )
    delta = result.layout_delta
    NT = int(result.indices.numel())
    assert delta.source_count == H * W
    assert delta.result_count == NT == int(result.tokens.shape[0])
    assert not delta.dropped_indices
    reps = tuple(delta.result_layout["representative_indices"])
    assert reps == tuple(int(v) for v in result.indices.tolist())
    assert list(reps) == sorted(reps)
    assert delta.ordering_policy is OrderingPolicy.EXPLICIT
    for r, src in enumerate(reps):
        assert delta.old_to_new[src] == r
    coarse = set(delta.result_layout["coarse_representatives"])
    for src in reps:
        if src not in coarse and not knobs.get("s1_merge"):
            group = [
                i for i in range(H * W) if delta.old_to_new[i] == delta.old_to_new[src]
            ]
            assert group == [src]


@pytest.mark.parametrize(
    "knobs",
    [{}, {"respool": False}, {"respool_pos": True}, {"s1_merge": "near", "s1_mu": 0.5}],
)
def test_tokens_backpropagate_to_merger_output_without_changing_forward(knobs):
    H, W = 14, 18
    embeds, maps, words = _inputs(H, W, seed=5)
    spec = BifocalSpec(**knobs)
    plain = bifocal_image_tokens(
        embeds, maps, grid_height=H, grid_width=W, word_embeddings=words, spec=spec
    )
    source = embeds.detach().clone().requires_grad_(True)
    traced = bifocal_image_tokens(
        source, maps, grid_height=H, grid_width=W, word_embeddings=words, spec=spec
    )
    assert torch.equal(traced.indices, plain.indices)
    assert torch.equal(traced.tokens.detach(), plain.tokens)
    assert traced.tokens.requires_grad
    traced.tokens.float().sum().backward()
    grad = source.grad
    assert grad is not None and bool(torch.isfinite(grad.float()).all())
    fine = traced.fine_indices
    assert bool((grad[fine] != 0).all())
    if not knobs:
        assert traced.metadata["respool_fallback"] == 0
        assert torch.equal(grad[fine], torch.ones_like(grad[fine]))
    if knobs.get("s1_merge") and traced.metadata["near_merged"]:
        mapping = traced.layout_delta.old_to_new
        kept = set(int(v) for v in traced.indices.tolist())
        fine_set = set(int(v) for v in fine.tolist())
        merged_cells = [
            cell
            for cell in range(H * W)
            if cell not in kept and int(traced.indices[mapping[cell]]) in fine_set
        ]
        assert merged_cells
        assert bool((grad[merged_cells] != 0).any())


def test_no_graph_when_input_does_not_require_grad():
    H, W = 14, 18
    embeds, maps, words = _inputs(H, W, seed=6)
    result = bifocal_image_tokens(
        embeds, maps, grid_height=H, grid_width=W, word_embeddings=words
    )
    assert not result.tokens.requires_grad


def test_small_image_passthrough_is_identity():
    H, W = 2, 2
    embeds, maps, words = _inputs(H, W)
    result = bifocal_image_tokens(
        embeds, maps, grid_height=H, grid_width=W, word_embeddings=words
    )
    assert result.metadata["passthrough"] is True
    assert result.layout_delta.is_identity
    assert torch.equal(result.tokens, embeds)


def test_query_extraction_matches_wrapper():
    assert bifocal_extract_query(
        "Please provide the bounding box coordinate of the region this sentence "
        "describes: man in red."
    ) == "man in red."
    assert bifocal_extract_query(
        "What color is the bus?\nAnswer the question using a single word or phrase."
    ) == "What color is the bus?"

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    ids = [1, 2, 99, 5, 6, 7, 77, 8]
    assert bifocal_query_text_from_ids(ids, _Tok(), 99, 77) == "5 6 7"


def test_spec_rejects_invalid_knobs():
    with pytest.raises(ValueError):
        BifocalSpec(s1_frac=1.2)
    with pytest.raises(ValueError):
        BifocalSpec(rank_map="norm")
    with pytest.raises(ValueError):
        BifocalSpec(s1_merge="near", respool=False)
    assert BifocalSpec(block="4").block == 4


def test_registry_exposes_exact_bifocal_units():
    from mllm_comp.compressors.multimodal_registry import (
        builtin_multimodal_compression_contracts,
    )
    from mllm_comp.core.vocabulary import ExecutionPhase, Placement

    registry = builtin_multimodal_compression_contracts()
    algorithm = registry.algorithm("bifocal")
    assert algorithm.supported_placements == frozenset({Placement.PROJECTOR_LEVEL})
    binding = registry.binding("bifocal.image.qwen25_vl")
    assert binding.variant == "bifocal.image"
    assert binding.position_resolver is None
    assert binding.supported_phases == frozenset(
        {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}
    )
    assert [target.target_site for target in binding.targets] == [
        "post_vision_merger_pre_llm_scatter"
    ]
    assert "visionzip.image.qwen25_vl" in binding.conflicts
    assert "nuwa.image.qwen25_vl" in binding.conflicts
    units = registry.support.find(algorithm="bifocal", backbone="qwen25_vl")
    assert {unit.identity.phase for unit in units} == {
        ExecutionPhase.FORWARD,
        ExecutionPhase.GENERATE,
    }
    assert all(not unit.evidence for unit in units)
    assert not registry.support.find(algorithm="bifocal", backbone="qwen3_vl")


class _FakeTokenizer:
    def __init__(self, base: int = 1000, vision_end: int = 99, im_end: int = 77):
        self.vocab: dict[str, int] = {}
        self.base = base
        self.special = {"<|vision_end|>": vision_end, "<|im_end|>": im_end}

    def __call__(self, text, add_special_tokens=False):
        class _Out:
            pass

        out = _Out()
        out.input_ids = [self.vocab.setdefault(text.strip(), self.base + len(self.vocab))]
        return out

    def decode(self, ids, skip_special_tokens=True):
        inverse = {v: k for k, v in self.vocab.items()}
        return " ".join(inverse[i] for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.special[token]


class _FakeModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embed = torch.nn.Embedding(4096, dim)
        self.config = type("Cfg", (), {"vision_end_token_id": 99})()

    def get_input_embeddings(self):
        return self.embed


def _executor(D: int, tokenizer: _FakeTokenizer, config: dict):
    from mllm_comp.backbones.qwen25_vl.image_compression import (
        QWEN25_VL_IMAGE_BINDING_CONTRACTS,
        Qwen25VLImageCompressionExecutor,
    )

    runtime = type(
        "Runtime", (), {"processor": type("P", (), {"tokenizer": tokenizer})()}
    )()
    return Qwen25VLImageCompressionExecutor(
        model=_FakeModel(D),
        runtime=runtime,
        stage=None,
        contract=QWEN25_VL_IMAGE_BINDING_CONTRACTS["bifocal.image.qwen25_vl"],
        execution_phase="generate",
        config=config,
    )


def test_executor_post_visual_merger_builds_legal_prompt_repair():
    H, W, D = 14, 18, 48
    embeds, maps, _ = _inputs(H, W, D=D)
    tokenizer = _FakeTokenizer()
    words = "what color is the bus".split()
    word_ids = [tokenizer(w).input_ids[0] for w in words]
    prompt = torch.tensor([5, 6, 99, *word_ids, 77, 8])
    executor = _executor(D, tokenizer, {"s1_frac": 0.25})
    N = H * W
    placeholders = tuple(range(3, 3 + N))
    result = executor.post_visual_merger(
        embeds,
        None,
        None,
        placeholder_indices=placeholders,
        source_sequence_length=N + 10,
        image_index=0,
        feature_grid=(H, W),
        vision_maps=maps,
        prompt_input_ids=prompt,
    )
    kernel = result.kernel_result
    assert kernel.metadata["chain"].startswith("salr+crn(5w)")
    plan = result.prompt_repair
    kept = plan.retained_placeholder_indices
    assert len(kept) == int(kernel.tokens.shape[0])
    assert kept == tuple(3 + int(i) for i in kernel.indices.tolist())
    metadata = executor.runtime_metadata()
    record = metadata["layout_delta_summaries"][-1]
    assert record["bifocal_query"] == "what color is the bus"
    assert record["bifocal_chain"] == kernel.metadata["chain"]
    assert metadata["projector_target_sites"] == ["post_vision_merger_pre_llm_scatter"]
    assert metadata["bifocal_spec"]["s1_frac"] == 0.25
    executor.clear_diagnostic_records()
    assert executor.runtime_metadata()["layout_delta_records"] == []


def test_executor_rejects_unknown_config_and_missing_maps():
    from mllm_comp.core.errors import BackendIntegrationError

    with pytest.raises(BackendIntegrationError, match="Invalid Bifocal config"):
        _executor(8, _FakeTokenizer(), {"s1_frac": 0.25, "s2_frac": 0.1})
    executor = _executor(8, _FakeTokenizer(), {})
    embeds, _, _ = _inputs(6, 6, D=8)
    with pytest.raises(BackendIntegrationError, match="column-mass"):
        executor.post_visual_merger(
            embeds, None, None,
            placeholder_indices=tuple(range(36)), source_sequence_length=36,
            image_index=0, feature_grid=(6, 6), vision_maps=None,
            prompt_input_ids=None,
        )


def test_executor_generate_summary_is_published():
    executor = _executor(8, _FakeTokenizer(), {})
    executor.record_projector_generate(
        {"decode_position_verified": True, "prompt_removed": 12}
    )
    executor.record_projector_generate({"decode_position_verified": None})
    summary = executor.runtime_metadata()["bifocal_generate"]
    assert summary["generate_calls"] == 2
    assert summary["decode_position_verified"] == 1
    assert summary["no_decode_step"] == 1
    assert summary["prompt_removed_total"] == 12


def test_eval_yaml_lists_valid_default_knobs():
    import yaml

    from mllm_comp.compressors.image.bifocal import bifocal_spec_from_config

    root = Path(__file__).resolve().parents[2]
    raw = yaml.safe_load(
        (root / "configs/eval/lmms_eval/qwen25_vl_14bench_bifocal.yaml").read_text(
            encoding="utf-8"
        )
    )
    stage = raw["pipeline"]["stages"][0]
    assert stage["algorithm"] == "bifocal"
    assert stage["binding"] == "bifocal.image.qwen25_vl"
    assert bifocal_spec_from_config(stage["config"]) == BifocalSpec()


@pytest.mark.parametrize("phase", ["generate", "forward"])
def test_bifocal_eval_config_resolves_through_v2_definition(phase, monkeypatch):
    from mllm_comp.config.loader import load_config
    from mllm_comp.pipeline.definition import resolve_compression_definition
    from mllm_comp.run_manifest import build_run_manifest

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("QWEN25_VL_MODEL_PATH", "/models/qwen25-vl")
    config = load_config(root / "configs/eval/lmms_eval/qwen25_vl_14bench_bifocal.yaml")
    from mllm_comp.core.vocabulary import ExecutionPhase

    config.pipeline.stages[0].execution_phase = ExecutionPhase(phase)
    definition = resolve_compression_definition(
        config,
        compression_mode=config.eval.compression_mode,
        interface="eval",
        runtime_batch_size=config.eval.runtime_batch_size,
    )
    assert [stage.algorithm for stage in definition.stages] == ["bifocal"]
    record = build_run_manifest(config).compression_stages[0]
    assert record.support_status == "declared"
    assert record.target_sites == ["post_vision_merger_pre_llm_scatter"]


def test_bifocal_train_config_builds_compressed_swift_plan(monkeypatch):
    from mllm_comp.config.loader import load_config
    from mllm_comp.integrations.ms_swift import MSSwiftTrainAdapter
    from mllm_comp.pipeline.definition import resolve_compression_definition

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("QWEN25_VL_MODEL_PATH", "/models/qwen25-vl")
    monkeypatch.setenv("MLLM_COMP_TRAIN_DATASET", "/datasets/train.jsonl")
    monkeypatch.setenv("MLLM_COMP_TRAIN_RUN_DIR", "/runs/bifocal-train")
    config = load_config(root / "configs/train/imagellm/qwen25_vl_bifocal_train_step.yaml")
    definition = resolve_compression_definition(
        config, compression_mode="compressed", interface="train"
    )
    assert [stage.algorithm for stage in definition.stages] == ["bifocal"]
    plan = MSSwiftTrainAdapter().build_launch_plan(config)
    assert plan.compression_training["enabled"] is True
    assert plan.compression_training["plugin"] == "bifocal"
    assert plan.compression_training["executor_family"] == "image_multimodal"
    assert plan.compression_training["execution_phase"] == "forward"


def test_executor_loads_query_tokenizer_when_runtime_has_no_processor(monkeypatch):
    import transformers

    from mllm_comp.backbones.qwen25_vl.image_compression import (
        QWEN25_VL_IMAGE_BINDING_CONTRACTS,
        Qwen25VLImageCompressionExecutor,
    )

    tokenizer = _FakeTokenizer()
    calls = []

    def fake_from_pretrained(source, **kwargs):
        calls.append((source, kwargs))
        return tokenizer

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    model = _FakeModel(8)
    model.config._name_or_path = "/models/qwen25-vl"
    executor = Qwen25VLImageCompressionExecutor(
        model=model,
        runtime=type("Runtime", (), {"processor": None})(),
        stage=None,
        contract=QWEN25_VL_IMAGE_BINDING_CONTRACTS["bifocal.image.qwen25_vl"],
        execution_phase="forward",
        config={},
    )
    word_ids = [tokenizer(w).input_ids[0] for w in ("red", "bus")]
    prompt = torch.tensor([5, 99, *word_ids, 77, 3001, 77])
    text, words = executor._bifocal_query(prompt, device=torch.device("cpu"))
    assert text == "red bus"
    assert words.shape == (2, 8)
    executor._bifocal_query(prompt, device=torch.device("cpu"))
    assert calls == [("/models/qwen25-vl", {"local_files_only": True})]
