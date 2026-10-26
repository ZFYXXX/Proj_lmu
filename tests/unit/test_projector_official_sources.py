"""Optional parity gates against the owner's pinned, unmodified source copies."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
SOURCES = Path(__file__).resolve().parents[4]


def load_file(name, path):
    if not path.is_file():
        pytest.skip(f"Optional official source is not present: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eadp_matches_released_qwen_scores_and_selected_set(dtype):
    from mllm_comp.compressors.comparison.eadp import compute_eadp_importance, normalize_eadp_config, select_eadp_indices
    official = load_file("official_eadp", SOURCES / "research_sources/EADP/Qwen_vl/model/pruner.py")
    torch.manual_seed(314)
    x, text = torch.randn(1, 48, 24), torch.randn(11, 24)
    x, text = x.to(dtype), text.to(dtype)
    cfg = normalize_eadp_config({"tokens_per_crop": 12, "beta": 2.})
    importance, _, _ = compute_eadp_importance(x, text, patches_per_side=8, grid_height=6, cfg=cfg)
    pruner = official.VisualTokenPruner(visual_token_num=12, beta=2., device="cpu")
    expected = pruner.compute_importance(x, text.mean(0, keepdim=True), text[None], 6, 8)
    torch.testing.assert_close(importance, expected, rtol=1e-5, atol=1e-6)
    expected_ids, _ = official._greed_select_impl(expected, official._sim_visual_impl(x), 12)
    ids, _ = select_eadp_indices(x, text, patches_per_side=8, grid_height=6, cfg=cfg)
    assert ids[0].tolist() == expected_ids[0].sort().values.tolist()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mmtok_matches_official_coverage_selector(dtype):
    from mllm_comp.compressors.image.mmtok import MMTokSpec, mmtok_image_tokens
    official = load_file("official_mmtok_selector", SOURCES / "MMTok/mmtok/core/semantic_selector.py")
    torch.manual_seed(315)
    projected, vision, text = torch.randn(64, 24), torch.randn(64, 16), torch.randn(9, 24)
    projected, vision, text = projected.to(dtype), vision.to(dtype), text.to(dtype)
    expected, _ = official.SemanticTokenSelector(target_vision_tokens=16).mm_coverage_selection(
        text, projected, vision, tv_temp=.01, vv_temp=.2)
    actual = mmtok_image_tokens(projected, vision, text, spec=MMTokSpec(target_count=16, text_temperature=.01))
    assert actual.indices.tolist() == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k", [32, 64, 128, 192])
def test_prunesid_matches_released_selection_set(k, dtype):
    from mllm_comp.compressors.comparison.prunesid import select_prunesid_indices
    path = SOURCES / "research_sources/PruneSID/prunesid/prunesid_qwen/modeling_qwen2_vl.py"
    if not path.is_file():
        pytest.skip("Official PruneSID checkout is not present")
    source = ast.parse(path.read_text())
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in {"pca_group", "nms"}]
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Qwen2VisionTransformerPretrainedModel_prunesid")
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    # Extract only the selector, without importing/replacing upstream Transformers.
    begin = next(i for i, node in enumerate(forward.body)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Tuple) for t in node.targets))
    body = forward.body[begin:]
    reference = ast.FunctionDef(name="reference", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="hidden_states"), ast.arg(arg="need_token_num")], kwonlyargs=[], kw_defaults=[], defaults=[]), body=body, decorator_list=[])
    import math
    import numpy as np
    namespace = {"torch": torch, "F": torch.nn.functional, "np": np, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[*functions, reference], type_ignores=[])), str(path), "exec"), namespace)
    torch.manual_seed(316)
    x = torch.randn(1, 240, 48).to(dtype)
    torch.manual_seed(0)
    _, expected = namespace["reference"](x[0], k)
    actual, _ = select_prunesid_indices(x, cfg={"tokens_per_crop": k, "pca_seed": 0})
    assert actual[0].tolist() == expected.tolist()
