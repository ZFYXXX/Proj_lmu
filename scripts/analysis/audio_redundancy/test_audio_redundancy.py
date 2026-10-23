"""Numerical, cross-fitting, native hook and report integration regressions."""
from dataclasses import dataclass
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch

from .algorithms import (change_scores, discrete_measurement, feasible_budget, local_ranks,
                         matched_smoothing, mel_reconstruction, pool_groups, rank_measurement,
                         run_groups, score_groups, subspace, uniform_groups)
from .common import coverage, write_json, write_row
from .plan import build_conditions


def test_independent_experiment_condition_counts():
    expected = {"A": 11, "B": 16, "C": 24, "E": 3, "H": 16}
    for name, count in expected.items():
        conditions, membership = build_conditions([name], stages=["P_in"])
        assert len(conditions) == len(membership[name]) == count


def test_resume_requires_full_phase_artifacts(tmp_path):
    from .launch import phase_complete
    phase = tmp_path / "calibration"
    (phase / "features").mkdir(parents=True)
    with (phase / "samples.rank0.jsonl").open("w", encoding="utf-8") as handle:
        write_row(handle, {"doc_index": 0})
    assert not phase_complete(phase, 1, [], calibration=True)
    np.savez(phase / "features/doc0.npz", hidden_P_in=np.zeros((2, 2)))
    result = phase / "native" / "run_results.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"results": {"mmsufinal": {"accuracy": 1}}}))
    assert phase_complete(phase, 1, [], calibration=True)
    assert json.loads((phase / "coverage.json").read_text())["complete"]


def write_complete_phase(root, phase, conditions):
    directory = root / phase
    directory.mkdir(exist_ok=True)
    with (directory / "samples.rank0.jsonl").open("w") as handle:
        write_row(handle, {"doc_index": 0})
    with (directory / "conditions.rank0.jsonl").open("w") as handle:
        for condition in conditions:
            write_row(handle, {"doc_index": 0, "condition_id": condition["id"]})
    if phase == "calibration":
        (directory / "features").mkdir(exist_ok=True)
        np.savez(directory / "features/doc0.npz", hidden_P_in=np.zeros((2, 2)))
    write_json(directory / "native_results.json", {"results": {"mmsufinal": {"accuracy": 1}}})


@pytest.mark.parametrize("experiments", [["A"], ["B"], ["C"], ["E"], ["H"], ["A", "B", "C", "E", "H"]])
@pytest.mark.parametrize("returncode", [-signal.SIGSEGV, 128 + signal.SIGSEGV])
def test_stage_supervisor_recovers_complete_gpu_passes(tmp_path, monkeypatch, experiments, returncode):
    from .launch import run_stages
    conditions, _ = build_conditions(experiments)
    calibration = bool(set(experiments) & {"B", "E"})
    settings = dict(output_dir=str(tmp_path), experiments=experiments, expected_samples=1,
                    needs_calibration=calibration, conditions=conditions)
    phases = []

    def execute(command, *, env):
        assert "faulthandler" in command and env["PYTHONFAULTHANDLER"] == "1"
        assert command[-1] == "--resume"
        phase = command[command.index("--phase") + 1]
        phases.append(phase)
        if phase in {"calibration", "evaluation"}:
            write_complete_phase(tmp_path, phase, [] if phase == "calibration" else conditions)
            return subprocess.CompletedProcess(command, returncode)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", execute)
    run_stages(settings, ["--experiments", ",".join(experiments)])
    assert phases == (["calibration", "fit"] if calibration else []) + ["evaluation", "report"]
    assert (tmp_path / "evaluation/recovered_launcher_exit.json").is_file()


@pytest.mark.parametrize("failed_phase,returncode,complete", [
    ("calibration", -signal.SIGSEGV, False), ("evaluation", -signal.SIGSEGV, False),
    ("fit", -signal.SIGSEGV, True), ("report", -signal.SIGSEGV, True),
    ("evaluation", 1, True), ("evaluation", -signal.SIGINT, True),
])
def test_stage_supervisor_stops_on_incomplete_or_unrecoverable_exit(
        tmp_path, monkeypatch, failed_phase, returncode, complete):
    from .launch import run_stages
    conditions, _ = build_conditions(["B"])
    settings = dict(output_dir=str(tmp_path), experiments=["B"], expected_samples=1,
                    needs_calibration=True, conditions=conditions)
    phases = []

    def execute(command, *, env):
        phase = command[command.index("--phase") + 1]
        phases.append(phase)
        if complete and phase in {"calibration", "evaluation"}:
            write_complete_phase(tmp_path, phase, [] if phase == "calibration" else conditions)
        return subprocess.CompletedProcess(command, returncode if phase == failed_phase else 0)

    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(RuntimeError, match=f"{failed_phase} process failed"):
        run_stages(settings, ["--experiments", "B"])
    assert phases[-1] == failed_phase
    assert not list(tmp_path.rglob("recovered_launcher_exit.json"))


def test_compaction_accepts_frozen_runner_and_restores_facade():
    from types import SimpleNamespace
    from .worker import mutable_compact_audio_request

    @dataclass(frozen=True)
    class Runner:
        generate: object

    @dataclass(frozen=True)
    class Facade:
        runner: object

    original_generate = lambda: "original"
    facade = Facade(Runner(original_generate))
    bridge = SimpleNamespace(facade=facade)
    controller = SimpleNamespace(keep_indices=[0])

    @contextmanager
    def compact(active_bridge, _):
        runner = active_bridge.facade.runner
        saved = runner.generate
        runner.generate = lambda: "compact"
        try:
            yield
        finally:
            runner.generate = saved

    with pytest.raises(RuntimeError, match="interrupted"):
        with mutable_compact_audio_request(bridge, controller, compact):
            assert bridge.facade.runner.generate() == "compact"
            raise RuntimeError("interrupted")
    assert bridge.facade is facade
    assert bridge.facade.runner.generate() == "original"


def test_portable_export_keeps_paired_numbers_without_bulk_artifacts(tmp_path):
    import csv
    import gzip
    from .export import export_portable

    report = tmp_path / "report"
    evaluation = tmp_path / "evaluation"
    report.mkdir()
    evaluation.mkdir()
    write_json(tmp_path / "analysis.json", dict(expected_samples=2, experiments=["C"],
        conditions=[dict(id="uniform_r0.5")], example_doc_ids=[0, 1],
        document_ids=["0", "1"], audio_paths=["0.wav", "1.wav"], folds=None))
    write_json(report / "summary.json", dict(experiments=["C"], questions=2,
        conditions={"uniform_r0.5": dict(accuracy=.5)}))
    (report / "README.zh.md").write_text("result\n")
    (report / "C_performance.png").write_bytes(b"plot")
    (report / "C_segments_doc0.png").write_bytes(b"example 0")
    (report / "C_segments_doc1.png").write_bytes(b"example 1")
    (evaluation / "example.doc0.npz").write_bytes(b"large detail")
    with (evaluation / "samples.rank0.jsonl").open("w") as baseline, \
         (evaluation / "conditions.rank0.jsonl").open("w") as changed:
        for index in range(2):
            write_row(baseline, dict(doc_index=index, id=str(index), audio_path=f"{index}.wav",
                accuracy=1., prediction="A"))
            write_row(changed, dict(doc_index=index, condition_id="uniform_r0.5",
                accuracy=float(index == 0), baseline_accuracy=1., prediction="B",
                intervention=dict(retention=.5, mse=.01)))
    portable = export_portable(tmp_path)
    with gzip.open(portable / "condition_scores.csv.gz", "rt") as handle:
        parsed = list(csv.DictReader(handle))
    assert len(parsed) == 2 and parsed[0]["retention"] == "0.5"
    assert (portable / "figures/C_segments_doc0.png").is_file()
    assert not (portable / "figures/C_segments_doc1.png").exists()
    assert not list(portable.rglob("*.npz"))
    assert json.loads((portable / "manifest.json").read_text())["condition_rows"] == 2


def test_rank_sample_capacity_mean_and_reconstruction():
    t = torch.arange(12, dtype=torch.float32)[:, None]
    hidden = t @ torch.tensor([[1., 2., 3., 4.]]) + torch.tensor([8., -4., 2., 0.])
    stats, _ = rank_measurement(hidden)
    assert stats["rank_capacity"] == 4
    assert stats["effective_rank"] == pytest.approx(1, abs=1e-4)
    reconstructed, _ = subspace(hidden, 1, "pca")
    assert torch.allclose(reconstructed, hidden, atol=2e-5)
    constant, _ = rank_measurement(torch.ones(7, 18))
    assert constant["effective_rank"] is None
    short, _ = rank_measurement(torch.randn(5, 100))
    assert short["rank_capacity"] == 4
    assert short["normalized_effective_rank"] <= 1.00001


def test_local_rank_never_joins_hard_chunks():
    records = local_ranks(torch.randn(12, 8), np.arange(12) * 40., np.repeat([0, 1], 6), [500, 1000])
    assert len(records) == 4
    assert all(r["tokens"] == 6 and r["rank_capacity"] == 5 for r in records)


@pytest.mark.parametrize("ratio", [.1, .25, .5, .75, 1.])
def test_all_segment_controls_match_budget_and_partition(ratio):
    chunks = np.array([0] * 13 + [-8] + [1] * 9)
    requested, budget = feasible_budget(chunks, ratio)
    scores = np.random.default_rng(3).random(len(chunks))
    for groups in (uniform_groups(chunks, budget), score_groups(chunks, budget, scores)):
        assert len(groups) == budget
        assert np.array_equal(np.concatenate(groups), np.arange(len(chunks)))
        assert all(len(set(chunks[g])) == 1 for g in groups)
    assert budget >= requested


def test_dynamic_allocation_selects_change_and_broadcast_preserves_length():
    hidden = torch.tensor([[1., 0.]] * 5 + [[0., 1.]] * 5)
    groups = score_groups(np.zeros(10, int), 2, change_scores(hidden, "cosine"))
    assert [g.tolist() for g in groups] == [list(range(5)), list(range(5, 10))]
    compressed, error = pool_groups(hidden, groups)
    broadcast, _ = pool_groups(hidden, groups, broadcast=True)
    assert compressed.shape == (2, 2) and error == 0
    assert torch.equal(broadcast, hidden)


def toy_unit_model():
    return dict(mean=np.zeros(2, np.float32), components=np.eye(2, dtype=np.float32),
                centers=np.eye(2, dtype=np.float32), prior=np.array([.5, .5]),
                transition=np.array([[.999, .001], [.2, .8]]))


def test_entropy_surprisal_and_reset_are_distinct():
    hidden = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])
    result = discrete_measurement(hidden, np.array([0, 0, 1, 1]), toy_unit_model(), .16)
    assert result["entropy"][1] < .02
    assert result["surprisal"][1] > 9
    assert result["entropy"][2] == pytest.approx(1.)
    assert result["surprisal"][2] == pytest.approx(1.)
    groups = run_groups(np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1]))
    assert len(groups) == 2


def test_mel_rank_preserves_mean_and_matched_control():
    torch.manual_seed(4)
    mel = torch.randn(30, 12)
    low, info = mel_reconstruction(mel, 2)
    assert torch.allclose(low.mean(0), mel.mean(0), atol=1e-6)
    full, _ = mel_reconstruction(mel, 128)
    assert torch.allclose(full, mel, atol=3e-6)
    for axis in (0, 1):
        smooth, details = matched_smoothing(mel, info["mse"], axis)
        assert smooth.shape == mel.shape
        assert details["matched"]
        assert details["mse"] == pytest.approx(info["mse"], rel=.01)
    _, impossible = matched_smoothing(torch.arange(10.)[:, None].expand(10, 3), 100., 1)
    assert not impossible["matched"]


def test_crossfit_groups_duplicate_audio_and_ignores_held_out_features(tmp_path):
    from .calibration import assign_folds, fit_models
    assert assign_folds(["a", "b", "a"])[0] == assign_folds(["a", "b", "a"])[2]
    roots = [tmp_path / "first", tmp_path / "changed"]
    rng = np.random.default_rng(4)
    blocks = [rng.normal(size=(16, 5)).astype(np.float16) for _ in range(4)]
    for root in roots:
        cache = root / "calibration" / "features"
        cache.mkdir(parents=True)
        for index, block in enumerate(blocks):
            if root.name == "changed" and index % 2 == 0:
                block = np.ones_like(block) * 30
            np.savez(cache / f"doc{index}.npz", hidden_P_in=block,
                     time_P_in=np.arange(16) * 40, chunks_P_in=np.zeros(16))
        if root.name == "changed":
            (root / "models").mkdir()  # B/E may leave this empty after an interrupted fit.
        fit_models(dict(output_dir=str(root), folds=[0, 1, 0, 1], stages=["P_in"],
            seed=42, fit_max_tokens=32, codebook_sizes=[2], codebook_pca_dim=3,
            fit_threads=1, markov_alpha=.1), resume_incomplete=root.name == "changed")
    with np.load(roots[0] / "models/fold0_P_in_K2.npz") as a, np.load(roots[1] / "models/fold0_P_in_K2.npz") as b:
        for field in a.files:
            assert np.array_equal(a[field], b[field]), field


@pytest.mark.parametrize("fit_threads", [1, 4])
def test_codebook_fit_resumes_without_scipy_decomposition(tmp_path, monkeypatch, fit_threads):
    from scipy import linalg
    from .calibration import ModelBank, fit_models
    from .launch import models_complete

    def reject_decomposition(*args, **kwargs):
        raise AssertionError("B/E codebook fitting reached a SciPy decomposition")

    for name in ("lu", "qr", "svd"):
        monkeypatch.setattr(linalg, name, reject_decomposition)
    cache = tmp_path / "calibration" / "features"
    cache.mkdir(parents=True)
    rng = np.random.default_rng(19)
    for index in range(4):
        np.savez(cache / f"doc{index}.npz",
                 hidden_P_in=rng.normal(size=(32, 24)).astype(np.float16),
                 time_P_in=np.arange(32) * 40, chunks_P_in=np.zeros(32))
    output = tmp_path / "models"
    output.mkdir()
    # A failed fit can leave an incomplete model file and no completion metadata.
    (output / "fold0_P_in_K4.npz").write_bytes(b"interrupted fit")
    settings = dict(output_dir=str(tmp_path), folds=[0, 1, 0, 1], stages=["P_in"],
                    seed=42, fit_max_tokens=64, codebook_sizes=[4], codebook_pca_dim=6,
                    fit_threads=fit_threads, markov_alpha=.1)
    assert not models_complete(output, settings)
    fit_models(settings, resume_incomplete=True)
    assert models_complete(output, settings)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["pca_backend"] == "torch.pca_lowrank"
    assert metadata["pca_compute_dtype"] == "float64"
    assert metadata["pca_threads"] == 1
    assert metadata["fit_threads"] == fit_threads
    environment = json.loads((output / "fit_environment.json").read_text())
    assert all(environment[key] for key in ("numpy", "scipy", "sklearn", "torch", "torch_build"))
    assert isinstance(environment["threadpools"], list)
    bank = ModelBank(output)
    for fold in (0, 1):
        model = bank.get(fold, "P_in", 4)
        assert model["components"].shape == (6, 24)
        assert model["centers"].shape == (4, 6)
        assert all(np.isfinite(values).all() for values in model.values())
        np.testing.assert_allclose(model["components"] @ model["components"].T,
                                   np.eye(6), atol=2e-6)
        np.testing.assert_allclose(model["transition"].sum(axis=1), 1.)
        np.testing.assert_allclose(model["prior"].sum(), 1.)


@pytest.mark.parametrize("shape", [(96, 48), (32, 96)])
def test_torch_pca_matches_reference_and_preserves_state(shape):
    from .calibration import fit_pca
    rng = np.random.default_rng(8)
    sample = (rng.normal(size=(shape[0], 8)) @ rng.normal(size=(8, shape[1]))
              + .005 * rng.normal(size=shape)).astype(np.float32)
    original = sample.copy()
    random_state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    mean, components = fit_pca(sample, 4, seed=42)
    assert torch.equal(torch.get_rng_state(), random_state)
    assert torch.get_num_threads() == threads
    assert np.array_equal(sample, original)
    mean_again, components_again = fit_pca(sample, 4, seed=42)
    assert np.array_equal(mean, mean_again) and np.array_equal(components, components_again)
    centered = sample.astype(np.float64) - sample.astype(np.float64).mean(0)
    _, _, reference = np.linalg.svd(centered, full_matrices=False)
    np.testing.assert_allclose(mean, sample.astype(np.float64).mean(0), atol=1e-6)
    np.testing.assert_allclose(components.T @ components, reference[:4].T @ reference[:4], atol=2e-5)


def test_torch_pca_restores_state_after_failure(monkeypatch):
    from .calibration import fit_pca
    random_state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()

    def fail(sample, **kwargs):
        assert sample.device.type == "cpu" and sample.dtype == torch.float64
        assert torch.get_num_threads() == 1
        torch.rand(4)
        raise RuntimeError("PCA failed")

    monkeypatch.setattr(torch, "pca_lowrank", fail)
    with pytest.raises(RuntimeError, match="PCA failed"):
        fit_pca(np.eye(8, dtype=np.float32), 4, seed=42)
    assert torch.equal(torch.get_rng_state(), random_state)
    assert torch.get_num_threads() == threads


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_torch_pca_rejects_nonfinite_features(invalid):
    from .calibration import fit_pca
    sample = np.eye(8, dtype=np.float32)
    sample[0, 0] = invalid
    with pytest.raises(ValueError, match="non-finite"):
        fit_pca(sample, 4, seed=42)


@pytest.fixture
def native_capture():
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
    from .capture import RepresentationCapture
    from .controller import RedundancyController
    torch.manual_seed(5)
    config = Qwen2_5OmniAudioEncoderConfig(num_mel_bins=128, encoder_layers=2, encoder_attention_heads=2,
        encoder_ffn_dim=32, d_model=16, output_dim=12, max_source_positions=32, n_window=8,
        attention_dropout=0., dropout=0.)
    config._attn_implementation = "eager"
    tower = Qwen2_5OmniAudioEncoder(config).eval()
    features = torch.randn(128, 31)
    arguments = dict(input_features=features, feature_lens=torch.tensor([31]), aftercnn_lens=torch.tensor([16]))
    with torch.no_grad():
        original = tower(**arguments).last_hidden_state
    capture = RepresentationCapture(tower, sample_rate=16000, hop_length=160, layers=[0, 1])
    controller = RedundancyController(tower, capture)
    capture.install()
    controller.install()
    controller.reset()
    capture.begin()
    with torch.no_grad():
        hooked = tower(**arguments).last_hidden_state
    layout, _ = capture.finish()
    assert torch.equal(hooked, original)
    assert layout["encoder_tokens"] == 16 and layout["projector_tokens"] == 8
    yield tower, capture, controller, arguments, original
    controller.close()
    capture.close()
    with torch.no_grad():
        restored = tower(**arguments).last_hidden_state
    assert torch.equal(original, restored)


@pytest.mark.parametrize("stage", ["E_l0", "E_L", "P_in", "P_out"])
def test_native_encoder_identity_and_subspace_hooks(native_capture, stage):
    tower, capture, controller, arguments, original = native_capture
    for method in ("identity", "pca", "random"):
        controller.prepare(dict(id="test", family="subspace", stage=stage, method=method, k=2), {})
        with torch.no_grad():
            output = tower(**arguments).last_hidden_state
        details = controller.finish()
        assert output.shape == original.shape
        if method == "identity":
            assert torch.equal(output, original)
        else:
            assert details["retained_k"] == 2


def test_native_mel_hooks_and_all_compression_families(native_capture):
    tower, capture, controller, arguments, original = native_capture
    hidden = capture.hidden["P_in"]
    chunks = capture.layouts["P_in"][1]
    unit = dict(groups=run_groups(np.zeros(len(hidden)), chunks),
                entropy=np.arange(len(hidden), dtype=float), surprisal=np.arange(len(hidden), dtype=float)[::-1])
    discrete = {("P_in", 2): unit}
    conditions, _ = build_conditions(["B", "C", "E", "H"], ratios=[.25], codebook_sizes=[2], mel_ranks=[2])
    for condition in conditions:
        controller.prepare(condition, discrete, example=True)
        with torch.no_grad():
            result = tower(**arguments).last_hidden_state
        details = controller.finish()
        assert len(result) == details["output_tokens"]
        if condition["family"] == "mel":
            assert result.shape == original.shape
            if condition["method"] == "identity":
                assert torch.equal(result, original)
        if controller.keep_indices is not None:
            assert len(np.unique(controller.keep_indices)) == len(result)
            assert np.all(np.diff(controller.keep_indices) > 0)


def test_shared_plan_and_worker_dispatch(tmp_path):
    from .launch import worker_plan
    conditions, membership = build_conditions(["A", "B", "C", "E", "H"])
    ids = [c["id"] for c in conditions]
    assert len(ids) == len(set(ids))
    assert set(membership["B"]) & set(membership["C"]) == {f"uniform_r{r:g}" for r in (.75, .5, .25, .1)}

    @dataclass
    class Plan:
        argv: list
        environment: dict
    native = Plan(["python", "-m", "accelerate.commands.launch", "--num_processes", "4", "-m", "lmms_eval"], {})
    changed = worker_plan(native, tmp_path / "analysis.json", "evaluation")
    assert changed.argv[2] == "accelerate.commands.launch"
    assert changed.argv[-1] == "scripts.analysis.audio_redundancy.worker"
    assert changed.environment["AUDIO_REDUNDANCY_PHASE"] == "evaluation"


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("experiment", ["B", "C", "E"])
def test_compaction_contract_restores_methods(native_capture, monkeypatch, fail, experiment):
    from types import SimpleNamespace
    from scripts.analysis.audio_temporal.controller import compact_audio_request
    from mllm_comp.backbones.qwen25_omni import audio_compression, create_qwen25_omni_backbone, Qwen25OmniRunner
    from .worker import mutable_compact_audio_request
    tower, capture, controller, arguments, original = native_capture
    conditions, _ = build_conditions([experiment], ratios=[.5], codebook_sizes=[2])
    chunks = capture.layouts["P_in"][1]
    unit = dict(groups=run_groups(np.zeros(len(original)), chunks),
                entropy=np.arange(len(original), dtype=float), surprisal=np.arange(len(original), dtype=float)[::-1])
    recorded = {}

    def legalize(state, *, token_actions, policy, stage_name):
        recorded["action"] = token_actions[0]
        assert policy.position_policy == "compact"
        return state

    monkeypatch.setattr(audio_compression, "legalize_qwen25_omni_audio_tokens", legalize)

    def rope(*args, **kwargs):
        return kwargs["audio_seqlens"]

    def generate(self, runtime, state, *, processed_batch, **kwargs):
        return state

    monkeypatch.setattr(Qwen25OmniRunner, "generate", generate)
    thinker = SimpleNamespace(get_rope_index=rope)
    facade = create_qwen25_omni_backbone()
    runner = facade.runner
    original_generate = runner.generate
    bridge = SimpleNamespace(runtime=SimpleNamespace(model=SimpleNamespace(thinker=thinker)),
                             facade=facade)
    original_lengths = tower._get_feat_extract_output_lengths
    state = SimpleNamespace(modality_token_spans=[SimpleNamespace(modality="audio", sample_index=0,
                                                                  start=3, end=3 + len(original))])
    for condition in conditions:
        controller.prepare(condition, {("P_in", 2): unit})
        recorded.clear()
        try:
            with mutable_compact_audio_request(bridge, controller, compact_audio_request):
                _, output_length = tower._get_feat_extract_output_lengths(torch.tensor([31]))
                assert int(output_length[0]) == len(controller.changed)
                assert bridge.facade.runner.generate(None, state, processed_batch=None) is state
                if controller.keep_indices is not None:
                    assert int(thinker.get_rope_index(audio_seqlens=torch.tensor([31]))[0]) == 4 * len(controller.changed)
                    assert recorded["action"].target_indices == controller.keep_indices.tolist()
                else:
                    assert not recorded and bridge.facade is facade
                if fail:
                    raise RuntimeError("interrupted condition")
                with torch.no_grad():
                    assert len(tower(**arguments).last_hidden_state) == len(controller.changed)
                assert controller.finish()["output_tokens"] == len(controller.changed)
        except RuntimeError as error:
            assert fail and str(error) == "interrupted condition"
        assert bridge.facade is facade
        assert runner.generate == original_generate and thinker.get_rope_index is rope
        assert tower._get_feat_extract_output_lengths == original_lengths


@pytest.mark.parametrize("experiment,needs_calibration,condition_count", [
    ("all", True, 66), ("B", True, 16), ("C", False, 24), ("E", True, 3),
])
def test_dry_plan_has_full_dataset_and_no_writes(tmp_path, experiment, needs_calibration, condition_count):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", dict(thinker_config=dict(audio_config=dict(encoder_layers=32))))
    source = Path(__file__).resolve().parents[5] / "AudioMCQevaluator/MMSU.jsonl"
    if not source.exists():
        source = tmp_path / "samples.jsonl"
        source.write_text("\n".join(json.dumps(dict(id=str(i), audio_path=f"{i}.wav")) for i in range(5000)))
    target = tmp_path / "run"
    result = subprocess.run([sys.executable, "-m", "scripts.analysis.audio_redundancy.launch",
        "--checkpoint", str(checkpoint), "--mmsu-jsonl", str(source), "--output-dir", str(target),
        "--experiments", experiment, "--plan-only"],
        capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["full_mmsufinal"] and plan["num_processes"] == 4
    assert set(plan["experiment_conditions"]) == ({"A", "B", "C", "E", "H"} if experiment == "all" else {experiment})
    assert plan["needs_calibration"] is needs_calibration
    assert plan["unique_conditions"] == condition_count
    assert not target.exists()


def test_coverage_rejects_wrong_id_set_and_duplicates(tmp_path):
    with (tmp_path / "samples.rank0.jsonl").open("w") as f:
        for i in [1, 2]:
            write_row(f, dict(doc_index=i))
    assert not coverage(tmp_path, 2, [])["complete"]
    with (tmp_path / "samples.rank0.jsonl").open("w") as f:
        for i in [0, 1, 1]:
            write_row(f, dict(doc_index=i))
    assert not coverage(tmp_path, 2, [])["complete"]


def test_end_to_end_report_all_five_experiments(tmp_path, native_capture):
    from .measurements import observe
    from .report import report
    tower, capture, controller, arguments, original = native_capture
    conditions, membership = build_conditions(["A", "B", "C", "E", "H"], ranks=[2], ratios=[.5], codebook_sizes=[2], mel_ranks=[2])
    settings = dict(output_dir=str(tmp_path), experiments=["A", "B", "C", "E", "H"],
        expected_samples=2, full_mmsufinal=False, stages=["E_l0", "E_L", "P_in", "P_out"],
        intervention_stages=["P_in"], codebook_sizes=[2], windows_ms=[100., 200.], conditions=conditions,
        experiment_conditions=membership, bootstrap_repeats=20, tolerance_pp=1.)
    write_json(tmp_path / "analysis.json", settings)
    output = tmp_path / "evaluation"
    output.mkdir()

    class Bank:
        def get(self, fold, stage, k):
            d = capture.hidden[stage].shape[1]
            components = np.eye(d, dtype=np.float32)[:2]
            return {**toy_unit_model(), "mean": np.zeros(d, np.float32), "components": components}

    measured, discrete, bases, arrays = observe(capture, settings, Bank(), 0, example=True)
    np.savez_compressed(output / "example.doc0.npz", **arrays)
    with (output / "samples.rank0.jsonl").open("w") as samples, (output / "conditions.rank0.jsonl").open("w") as modified:
        for index in range(2):
            reference = dict(doc_index=index, id=str(index), audio_path=f"{index}.wav", prediction="A", accuracy=1.,
                             question_type="perception", sub_category="Linguistics", task_name="toy", **measured)
            if index == 0:
                reference["example_file"] = "example.doc0.npz"
            write_row(samples, reference)
            for condition in conditions:
                controller.prepare(condition, discrete, example=True)
                with torch.no_grad():
                    tower(**arguments)
                details = controller.finish()
                if index == 0 and controller.arrays:
                    np.savez_compressed(output / f"example.doc0.{condition['id']}.npz", **controller.arrays)
                write_row(modified, dict(doc_index=index, id=str(index), audio_path=f"{index}.wav", prediction="A",
                    accuracy=1., baseline_accuracy=1., condition_id=condition["id"], condition=condition,
                    intervention=details, analysis_preparation_seconds=.1, diagnostic_generation_seconds=.2))
    report(tmp_path)
    summary = json.loads((tmp_path / "report/summary.json").read_text())
    assert summary["identity_checks_passed"]
    assert all(row["matched_budget"] for row in summary["matched_comparisons"])
    for experiment in "ABCEH":
        assert (tmp_path / f"report/{experiment}_performance.png").exists()
    assert (tmp_path / "report/H_mel_doc0.pdf").exists()
    assert (tmp_path / "report/paired_task_results.csv").exists()
