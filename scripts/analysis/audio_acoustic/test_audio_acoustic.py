"""CPU numerical and native-Transformers contract tests; no checkpoint download."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .algorithms import temporal_spectrum, time_lowpass
from .launch import coverage, worker_plan
from .plan import EXPERIMENTS, build_conditions
from .report import interaction_damage, paired_interval
from .signal import acoustic_targets, audio_intervention, filter_wave, mel_intervention, reduce_mel_power, spectrum


def test_shared_plan_reuses_only_identical_operations():
    from .plan import build_suite_conditions
    shared, experiments = build_suite_conditions()
    assert len(shared) == 56
    assert (len(shared) + 1) * 5000 == 285000
    assert sum(len(rows) + 1 for rows in experiments.values()) * 5000 == 425000
    by_id = {row["id"]: row for row in shared}
    for experiment, members in experiments.items():
        assert len(members) == len(build_conditions(experiment))
        for row in members:
            source = by_id[row["source_condition_id"]]
            assert {k: v for k, v in row.items() if k not in {"id", "source_condition_id"}} == {
                k: v for k, v in source.items() if k != "id"}
    assert experiments["representation_spectrum"][0]["source_condition_id"] == "time_lowpass1"
    assert experiments["energy_importance"][1]["source_condition_id"] == experiments["band_importance"][1]["source_condition_id"]
    # Physical frequency controls remain distinct; every condition preserves token count.
    assert {row["id"] for row in shared} >= {
        "band0_raw", "band0_rms", "band0_matched_energy"}
    assert {row["family"] for row in shared} == {"wave", "mel", "time_lowpass"}
    assert all("compact" not in row and "keep_ratio" not in row for row in shared)
    assert all(experiments[name] == [] for name in ("pitch", "envelope", "rhythm"))


def test_acoustic_profile_measures_input_and_keeps_unavailable_pitch():
    from .signal import acoustic_profile
    wave = np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)
    profile = acoustic_profile(wave, acoustic_targets(wave))
    scaled = acoustic_profile(wave * 0.5, acoustic_targets(wave * 0.5))
    assert profile["median_f0_hz"] == pytest.approx(200, abs=3)
    assert profile["duration_s"] == 1
    assert profile["valid_f0_frames"] >= 5
    assert scaled["mean_log_rms"] - profile["mean_log_rms"] == pytest.approx(np.log(0.5))
    assert scaled["log_rms_std"] == pytest.approx(profile["log_rms_std"])
    silence = np.zeros(16000)
    empty = acoustic_profile(silence, acoustic_targets(silence))
    assert empty["median_f0_hz"] is None and empty["f0_iqr_cents"] is None
    assert empty["voiced_fraction"] == 0 and empty["proxy_pause_fraction"] == 1
    json.dumps(empty, allow_nan=False)


def test_attribute_terciles_ignore_repeated_questions_keep_ties_and_missing():
    from .attributes import input_terciles
    grouped = input_terciles([1] * 100 + [2, 3, 4, None], ["a"] * 100 + ["b", "c", "d", "e"])
    assert grouped["cutoffs"] == [2, 3]
    assert grouped["cutoff_audio_count"] == 4
    assert len(set(grouped["labels"][:100])) == 1
    assert grouped["labels"][-1] == "unavailable"
    constant = input_terciles([0, 0, None], ["a", "b", "c"])
    assert constant["groups"] == ["constant", "unavailable"]
    assert input_terciles([None], ["a"])["groups"] == ["unavailable"]


def write_shared_fixture(directory):
    from .plan import build_suite_conditions
    from .signal import acoustic_profile
    from scripts.analysis.audio_temporal.metrics import jsonable
    shared, experiments = build_suite_conditions()
    settings = dict(experiment="all", conditions=shared, experiment_conditions=experiments,
                    expected_samples=6, full_benchmark=False, expected_generation_calls=57 * 6,
                    independent_generation_calls=85 * 6, generation_calls_saved=28 * 6)
    (directory / "analysis.json").write_text(json.dumps(settings))
    native = []
    for i, j in enumerate([0, 0, 1, 2, 3, 4]):
        wave = (np.sin(2 * np.pi * (100 + j * 50) * np.arange(16000) / 16000) / (j + 1)) if j != 3 else np.zeros(16000)
        profile = acoustic_profile(wave, acoustic_targets(wave))
        native.append(dict(doc_index=i, accuracy=float(i < 3), prediction="A" if i < 3 else "B", audio_path=f"clip{j}.wav",
            question_type="perception", sub_category="fixture", task_name=f"task{j % 2}", dataset_name="fixture",
            projector_tokens=10 + j, acoustic_proxy_summary=profile, spectrum=spectrum(wave),
            stages=[dict(name="P_in", tokens=10 + j, temporal_spectrum=None)]))
        if i == 0:
            targets = acoustic_targets(wave)
            np.savez(directory / "example.rank0.doc0.npz", target_time_ms=targets["time_ms"], target_log_rms=targets["log_rms"])
            native[-1]["example_file"] = "example.rank0.doc0.npz"
    (directory / "samples.rank0.jsonl").write_text('\n'.join(json.dumps(jsonable(row)) for row in native))
    with (directory / "conditions.rank0.jsonl").open("w") as handle:
        for condition in shared:
            for i, row in enumerate(native):
                changed = dict(row, baseline_accuracy=row["accuracy"],
                    condition_id=condition["id"], condition=condition,
                    stages=[dict(row["stages"][0], drift=dict(relative_l2=0.0, mean_cosine=1.0))])
                handle.write(json.dumps(jsonable(changed)) + '\n')
    return settings


def test_shared_report_reuses_frequency_conditions_and_attributes_use_only_baseline(tmp_path):
    from .report import build_summary
    write_shared_fixture(tmp_path)
    band = build_summary(tmp_path, experiment="band_importance", repeats=10)
    energy = build_summary(tmp_path, experiment="energy_importance", repeats=10)
    for name, result in band["conditions"].items():
        assert result["accuracy"] == energy["conditions"][name]["accuracy"]
        assert result["damage"] == energy["conditions"][name]["damage"]
    pitch = build_summary(tmp_path, experiment="pitch", repeats=10)
    assert pitch["conditions"] == {}
    attr = pitch["attribute_analysis"]["attributes"]["median_f0_hz"]
    assert "unavailable" in attr["groups"]
    groups = attr["baseline_groups"]
    assert sum(row["n"] for row in groups.values()) == 6
    assert sum(row["accuracy"] * row["n"] for row in groups.values() if row["n"]) / 6 == 0.5
    assert groups["unavailable"]["accuracy"] == 0
    assert groups["unavailable"]["audio_clusters"] == 1
    assert set(groups["unavailable"]["controls"]) == {"duration_s", "native_tokens", "task_name", "sub_category"}
    assert all("probes" not in row for row in pitch["baseline_stages"].values())
    # The statistical attribute groups cannot depend on the QA scores.
    rows = [json.loads(line) for line in (tmp_path / "samples.rank0.jsonl").read_text().splitlines()]
    from .attributes import prepare_attributes
    before = prepare_attributes("pitch", rows)
    for row in rows:
        row["accuracy"] = 0
    assert before == prepare_attributes("pitch", rows)


def test_all_report_writes_ten_views_without_inference(tmp_path, monkeypatch):
    import csv
    import sys
    from . import report
    write_shared_fixture(tmp_path)
    rendered = []
    make_figures = report.make_figures
    monkeypatch.setattr(report, "make_figures", lambda result, target: rendered.append(result["experiment"]))
    monkeypatch.setattr(sys, "argv", ["report", "--run-dir", str(tmp_path), "--bootstrap", "10"])
    report.main()
    assert rendered == list(EXPERIMENTS)
    index = json.loads((tmp_path / "report/summary.json").read_text())
    assert index["generation_calls_saved"] == 168
    assert len(index["experiments"]) == 10 and "merge" not in index["experiments"]
    for experiment in ("pitch", "envelope", "rhythm"):
        target = tmp_path / "report" / experiment
        summary = json.loads((target / "summary.json").read_text())
        assert summary["coverage"]["complete"]
        assert not list(target.glob("probe_*"))
        for name in ("attribute_baseline", "attribute_assignments", "attribute_controls"):
            with (target / f"{name}.csv").open() as source:
                rows = list(csv.reader(source))
            assert len(rows) > 1 and all(len(row) == len(rows[0]) for row in rows)
    target = tmp_path / "report/pitch"
    make_figures(json.loads((target / "summary.json").read_text()), target)
    assert (target / "attribute_median_f0_hz.png").is_file()
    assert (target / "attribute_f0_iqr_cents.pdf").is_file()
    assert not list(target.glob("*merge*"))
    assert not (target / "attribute_damage.csv").exists()
    envelope = json.loads((tmp_path / "report/envelope/summary.json").read_text())
    assert envelope["envelope_examples"][0]["doc_index"] == 0


@pytest.mark.parametrize("experiment", ["pitch", "envelope", "rhythm", "all"])
def test_plan_accepts_acoustic_experiments_without_probe(tmp_path, monkeypatch, capsys, experiment):
    import sys
    from .launch import main
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(dict(thinker_config=dict(audio_config=dict(encoder_layers=2)))))
    output = tmp_path / "never_created"
    monkeypatch.setattr(sys, "argv", ["launch", "--experiment", experiment, "--checkpoint", str(checkpoint),
                        "--output-dir", str(output), "--plan-only"])
    main()
    plan = json.loads(capsys.readouterr().out)
    assert "probe_model" not in plan
    assert plan["expected_generation_calls"] == (285000 if experiment == "all" else 5000)
    if experiment == "all":
        assert plan["generation_calls_saved"] == 140000
    assert not output.exists()


def test_two_gpu_full_runner_has_no_probe_preflight_and_runs_one_launch(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    calls = tmp_path / "calls.jsonl"
    python = tmp_path / "record_python"
    python.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
with open(os.environ["CALL_RECORD"], "a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
if os.environ.get("FAIL_LAUNCH") and "scripts.analysis.audio_acoustic.launch" in sys.argv:
    sys.exit(7)
''')
    python.chmod(0o755)
    script = Path(__file__).with_name("run_all_mmsufinal_2gpu.sh")
    env = dict(os.environ, PYTHON_BIN=str(python), CHECKPOINT_PATH="/unused/checkpoint", CUDA_VISIBLE_DEVICES="2,3",
               NUM_PROCESSES="99", PROBE_MODEL="/nonexistent/ignored", CALL_RECORD=str(calls), RUN_GROUP=str(tmp_path / "out"))
    subprocess.run(["bash", str(script), "--plan-only"], env=env, check=True, capture_output=True)
    assert not (tmp_path / "out.log").exists() and not (tmp_path / "out").exists()
    calls.unlink()
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(recorded) == 2
    assert recorded[0][recorded[0].index("--experiment") + 1] == "all"
    assert recorded[0][-2:] == ["--num-processes", "2"]
    assert "scripts.analysis.audio_acoustic.report" in recorded[1]
    calls.unlink()
    env.update(FAIL_LAUNCH="1", RUN_GROUP=str(tmp_path / "failed"))
    failed = subprocess.run(["bash", str(script)], env=env, capture_output=True)
    assert failed.returncode != 0 and len(calls.read_text().splitlines()) == 1


def tones():
    time = np.arange(32000) / 16000
    return (np.sin(2 * np.pi * 500 * time) + 0.3 * np.sin(2 * np.pi * 5500 * time)).astype(np.float32)


def test_filter_keeps_duration_and_separates_physical_frequencies():
    wave = tones()
    base = spectrum(wave)
    assert base["band_fraction"][0] > 0.85
    assert sum(base["band_fraction"]) == pytest.approx(1)
    identity, _ = filter_wave(wave, dict(operation="identity"))
    np.testing.assert_allclose(identity, wave, atol=1e-7)
    low, details = filter_wave(wave, dict(operation="lowpass", cutoff_hz=2000, rms_match=False))
    matched, matched_details = filter_wave(wave, dict(operation="lowpass", cutoff_hz=2000, rms_match=True))
    assert len(wave) == len(low) == len(matched)
    assert spectrum(low)["band_fraction"][5] < 1e-4
    assert details["removed_power_fraction"] == pytest.approx(0.09 / 1.09, abs=0.002)
    assert np.mean(matched ** 2) == pytest.approx(np.mean(wave ** 2), rel=1e-6)
    assert matched_details["rms_gain"] > 1


def test_equal_removed_energy_is_matched_in_linear_power():
    rng = np.random.default_rng(4)
    wave = (rng.normal(size=16000) * 0.1 + tones()[:16000]).astype(np.float32)
    fractions = []
    for index in range(8):
        _, details = filter_wave(wave, dict(operation="matched_energy", bands=[index], band_width_hz=1000))
        fractions.append(details["removed_power_fraction"])
        assert fractions[-1] == pytest.approx(details["requested_removed_power_fraction"], abs=1e-10)
    assert np.ptp(fractions) < 1e-10


def test_silence_has_no_fake_pitch_or_rolloff():
    targets = acoustic_targets(np.zeros(1000))
    assert "spectral_flux" not in targets
    assert np.isnan(targets["log_f0"]).all()
    assert not targets["voiced"].any()
    assert spectrum(np.zeros(1000))["rolloff_hz"]["0.95"] is None
    assert np.isfinite(filter_wave(np.zeros(1000), dict(operation="lowpass", cutoff_hz=1000, rms_match=True))[0]).all()
    wave = np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)
    targets = acoustic_targets(wave)
    assert np.median(np.exp(targets["log_f0"][20:-20])) == pytest.approx(200, abs=3)


def test_mel_identity_matches_native_transformers_and_conserves_linear_power():
    from transformers import WhisperFeatureExtractor
    extractor = WhisperFeatureExtractor(feature_size=128)
    wave = tones()[None, :]
    original_method = extractor._torch_extract_fbank_features
    native = original_method(wave)
    with mel_intervention(extractor, dict(family="mel", groups=128, method="mean")):
        np.testing.assert_array_equal(extractor._torch_extract_fbank_features(wave), native)
    assert extractor._torch_extract_fbank_features == original_method
    power = torch.rand(2, 128, 17)
    coarse = reduce_mel_power(power, 17)
    torch.testing.assert_close(coarse.sum(-2), power.sum(-2))
    assert coarse.shape == power.shape
    with pytest.raises(RuntimeError):
        with mel_intervention(extractor, dict(family="mel", groups=32, method="mean")):
            raise RuntimeError("test restoration")
    assert extractor._torch_extract_fbank_features == original_method


def test_native_audio_loader_restores_frozen_adapter_on_failure():
    from mllm_comp.backbones.qwen25_omni.implementation import Qwen25OmniProcessorAdapter
    wave = tones()
    loader = lambda *args, **kwargs: ([wave], None, None)
    adapter = Qwen25OmniProcessorAdapter(media_loader=loader)
    capture = {}
    with pytest.raises(RuntimeError):
        with audio_intervention(adapter, dict(family="wave", operation="lowpass", cutoff_hz=1000), capture):
            output = adapter.media_loader([])
            assert not np.array_equal(output[0][0], wave)
            np.testing.assert_array_equal(capture["original_wave"], wave)
            raise RuntimeError("intentional")
    assert adapter.media_loader is loader


def test_temporal_spectrum_and_filter_respect_chunks_and_axis():
    chunks = np.array([0] * 64 + [1] * 64)
    constant = torch.cat([torch.ones(64, 3), 7 * torch.ones(64, 3)])
    power = temporal_spectrum(constant, chunks)
    assert power["demean"] is None
    torch.testing.assert_close(time_lowpass(constant, chunks, 0.25), constant)
    time = torch.arange(128).float()
    values = torch.sin(2 * torch.pi * 0.375 * time)[:, None].repeat(1, 3)
    spectrum_result = temporal_spectrum(values, np.zeros(128))
    assert np.argmax(spectrum_result["demean"]) in (11, 12)
    assert time_lowpass(values, np.zeros(128), 0.25).square().mean() < 0.05 * values.square().mean()


@pytest.fixture
def tower():
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
    torch.manual_seed(3)
    config = Qwen2_5OmniAudioEncoderConfig(num_mel_bins=128, encoder_layers=2,
        encoder_attention_heads=4, encoder_ffn_dim=32, d_model=16, max_source_positions=64,
        n_window=15, output_dim=12)
    config._attn_implementation = "eager"
    return Qwen2_5OmniAudioEncoder(config).eval()


@pytest.mark.parametrize("length", [28, 91, 100])
def test_real_encoder_hooks_preserve_native_layout_for_time_smoothing(tower, length):
    from .observer import AcousticObserver
    features, lengths = torch.randn(128, length), torch.tensor([length])
    cnn, _ = tower._get_feat_extract_output_lengths(lengths)
    with torch.inference_mode():
        native = tower(features, feature_lens=lengths, aftercnn_lens=cnn).last_hidden_state
    observer = AcousticObserver(tower, sample_rate=16000, hop_length=160, layers=[0, 1])
    observer.install()
    try:
        with torch.inference_mode():
            observer.begin(example=True)
            baseline = tower(features, feature_lens=lengths, aftercnn_lens=cnn).last_hidden_state
            measured, arrays = observer.finish()
            torch.testing.assert_close(baseline, native, rtol=0, atol=0)
            assert {row["name"] for row in measured["stages"]} == {"C", "E_l0", "E_L", "P_in", "P_out"}
            assert all("lag_similarity" not in row for row in measured["stages"])
            assert "region_proxy" not in measured and "projector_provenance" not in measured
            assert not any("flux" in key or "region" in key for key in arrays)
            np.testing.assert_array_equal(arrays["mel_time_ms"], np.arange(length) * 10.0)
            observer.begin(condition=dict(family="time_lowpass", nyquist_ratio=1))
            identity = tower(features, feature_lens=lengths, aftercnn_lens=cnn).last_hidden_state
            observer.finish()
            torch.testing.assert_close(identity, native, rtol=0, atol=0)
            for ratio in (0.75, 0.5, 0.25):
                observer.begin(condition=dict(family="time_lowpass", nyquist_ratio=ratio))
                smoothed = tower(features, feature_lens=lengths, aftercnn_lens=cnn).last_hidden_state
                measured, _ = observer.finish()
                assert smoothed.shape == native.shape
                assert measured["projector_tokens"] == len(native)
                assert measured["stages"][-1]["temporal_spectrum"] is not None
                assert all(row["drift"]["alignment"] == "native_grid" for row in measured["stages"])
    finally:
        observer.close()
    with torch.inference_mode():
        restored = tower(features, feature_lens=lengths, aftercnn_lens=cnn).last_hidden_state
    torch.testing.assert_close(restored, native, rtol=0, atol=0)


def test_deletion_interaction_sign_and_audio_cluster_bootstrap():
    # A and B are redundant backups: either alone suffices, removing both hurts.
    np.testing.assert_array_equal(interaction_damage(np.ones(4), np.ones(4), np.ones(4), np.zeros(4)), np.ones(4))
    interval = paired_interval([1, 1, 0], ["a", "a", "b"], repeats=400)
    assert interval["mean"] == pytest.approx(2 / 3)
    assert interval["audio_clusters"] == 2
    assert interval["ci95"] == [0, 1]


def test_all_remaining_interventions_preserve_native_token_counts(tower):
    from transformers import WhisperFeatureExtractor
    from mllm_comp.backbones.qwen25_omni.implementation import Qwen25OmniProcessorAdapter
    from .observer import AcousticObserver
    from .plan import build_suite_conditions

    wave = tones()[:4800]
    extractor = WhisperFeatureExtractor(feature_size=128)
    adapter = Qwen25OmniProcessorAdapter(media_loader=lambda *args, **kwargs: ([wave], None, None))
    observer = AcousticObserver(tower, sample_rate=16000, hop_length=160, layers=[0, 1])
    observer.install()
    native = None
    try:
        with torch.inference_mode():
            for condition in [None] + build_suite_conditions()[0]:
                observer.begin(condition=condition)
                with audio_intervention(adapter, condition, {}), mel_intervention(extractor, condition):
                    changed_wave = adapter.media_loader([])[0][0]
                    assert len(changed_wave) == len(wave)
                    features = torch.from_numpy(extractor._torch_extract_fbank_features(changed_wave[None, :])[0])
                    lengths = torch.tensor([features.shape[-1]])
                    cnn, _ = tower._get_feat_extract_output_lengths(lengths)
                    output = tower(features, feature_lens=lengths, aftercnn_lens=cnn).last_hidden_state
                measurements, _ = observer.finish()
                if condition is None:
                    native = measurements
                assert measurements["mel_length"] == native["mel_length"]
                assert measurements["encoder_tokens"] == native["encoder_tokens"]
                assert measurements["projector_tokens"] == native["projector_tokens"] == len(output)
                assert [row["tokens"] for row in measurements["stages"]] == [row["tokens"] for row in native["stages"]]
    finally:
        observer.close()


def test_coverage_rejects_condition_ids_with_correct_count_wrong_documents(tmp_path):
    (tmp_path / "samples.rank0.jsonl").write_text('\n'.join(json.dumps(dict(doc_index=i)) for i in range(2)))
    condition = dict(id="test")
    file = tmp_path / "conditions.rank0.jsonl"
    file.write_text('\n'.join(json.dumps(dict(doc_index=i, condition_id="test")) for i in [1, 2]))
    assert not coverage(tmp_path, 2, [condition])["complete"]
    file.write_text('\n'.join(json.dumps(dict(doc_index=i, condition_id="test")) for i in [0, 1]))
    assert coverage(tmp_path, 2, [condition])["complete"]


def test_experiments_are_independent_and_accelerate_worker_is_wrapped():
    assert len(EXPERIMENTS) == 10
    for removed in ("locality", "boundary", "merge"):
        assert removed not in EXPERIMENTS
        with pytest.raises(ValueError, match="Unknown experiment"):
            build_conditions(removed)
    assert build_conditions("observe") == []
    for experiment in EXPERIMENTS:
        conditions = build_conditions(experiment)
        assert len(conditions) == len({row["id"] for row in conditions})
        assert all("protect_boundary" not in row for row in conditions)
    for experiment in ("pitch", "envelope", "rhythm"):
        assert build_conditions(experiment) == []
    assert len(build_conditions("frequency_energy")) == 11

    @dataclass
    class Plan:
        argv: list
        environment: dict

    original = Plan(["accelerate", "launch", "-m", "lmms_eval", "--tasks", "mmsufinal"], {})
    wrapped = worker_plan(original, "/tmp/analysis.json")
    assert wrapped.argv[3] == "scripts.analysis.audio_acoustic.worker"
    assert original.argv[3] == "lmms_eval"


def test_plan_only_prints_cost_without_creating_output(tmp_path, monkeypatch, capsys):
    import sys
    from .launch import main
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(dict(thinker_config=dict(audio_config=dict(encoder_layers=2)))))
    output = tmp_path / "never_created"
    monkeypatch.setattr(sys, "argv", ["launch", "--experiment", "frequency_energy", "--checkpoint", str(checkpoint),
        "--output-dir", str(output), "--plan-only", "--cutoffs-hz", "4000"])
    main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["expected_generation_calls"] == 20000
    assert plan["expected_samples"] == 5000
    assert "lags_ms" not in plan
    assert not output.exists()


@pytest.mark.parametrize("arguments", [
    ["--experiment", "locality"],
    ["--experiment", "boundary"],
    ["--experiment", "merge"],
    ["--experiment", "observe", "--keep-ratios", "0.5"],
    ["--experiment", "observe", "--merge-methods", "mean"],
    ["--experiment", "observe", "--lags-ms", "80"],
])
def test_launch_rejects_removed_experiments_and_option(tmp_path, monkeypatch, capsys, arguments):
    import sys
    from .launch import main
    output = tmp_path / "never_created"
    monkeypatch.setattr(sys, "argv", ["launch", "--output-dir", str(output), *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("experiment", ["band_importance", "representation_spectrum"])
def test_full_report_pipeline_and_identity_control_failure(tmp_path, tower, experiment):
    from .observer import AcousticObserver
    from .report import build_summary, make_figures
    from scripts.analysis.audio_temporal.metrics import jsonable
    observer = AcousticObserver(tower, sample_rate=16000, hop_length=160, layers=[0, 1])
    observer.install()
    lengths = torch.tensor([100])
    cnn, _ = tower._get_feat_extract_output_lengths(lengths)
    try:
        with torch.inference_mode():
            observer.begin()
            tower(torch.randn(128, 100), feature_lens=lengths, aftercnn_lens=cnn)
            measurements, _ = observer.finish()
    finally:
        observer.close()
    conditions = build_conditions(experiment)
    (tmp_path / "analysis.json").write_text(json.dumps(dict(experiment=experiment,
        conditions=conditions, expected_samples=2, full_benchmark=False)))
    baseline = [dict(measurements, doc_index=i, accuracy=1.0, prediction="A", audio_path=f"clip{i}.wav",
                     question_type="perception", sub_category="synthetic", task_name="fixture", dataset_name="fixture",
                     spectrum=spectrum(tones())) for i in range(2)]
    changed = []
    for condition in conditions:
        for record in baseline:
            stages = [dict(stage, drift=dict(relative_l2=0.0, mean_cosine=1.0)) for stage in record["stages"]]
            changed.append(dict(record, stages=stages, condition_id=condition["id"], baseline_accuracy=1.0))
    (tmp_path / "samples.rank0.jsonl").write_text('\n'.join(json.dumps(jsonable(row)) for row in baseline))
    path = tmp_path / "conditions.rank0.jsonl"
    path.write_text('\n'.join(json.dumps(jsonable(row)) for row in changed))
    result = build_summary(tmp_path, repeats=10)
    assert result["identity_controls_passed"]
    assert all(row["within_tolerance"] is None for row in result["conditions"].values())
    assert all("lags" not in stage for stage in result["baseline_stages"].values())
    figures = tmp_path / "report"
    figures.mkdir()
    make_figures(result, figures)
    expected_figure = "band_energy_importance.png" if experiment == "band_importance" else "paired_damage.png"
    assert (figures / expected_figure).exists()
    assert (figures / "condition_layer_drift.pdf").exists()
    assert (figures / "representation_temporal_spectrum.png").exists()
    assert not list(figures.glob("temporal_locality.*"))
    changed[0]["prediction"] = "B"
    path.write_text('\n'.join(json.dumps(jsonable(row)) for row in changed))
    assert not build_summary(tmp_path, repeats=10)["identity_controls_passed"]


def test_real_frozen_runner_and_thinker_preserve_audio_positions():
    from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
    from mllm_comp.core.state import ForwardState, TokenSpan
    from .observer import AcousticObserver
    from mllm_comp.backbones.base import BackboneRuntime, ProcessedBatch
    from mllm_comp.backbones.qwen25_omni.implementation import Qwen25OmniRunner
    config = Qwen2_5OmniThinkerConfig(
        audio_config=dict(d_model=8, encoder_layers=2, encoder_attention_heads=2,
            encoder_ffn_dim=16, num_mel_bins=4, max_source_positions=8, n_window=4, output_dim=16),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, out_hidden_size=16,
            num_heads=2, patch_size=2, spatial_merge_size=1, fullatt_block_indexes=[0]),
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2,
            rope_scaling=dict(rope_type="default", mrope_section=[2, 1, 1]),
            pad_token_id=0, bos_token_id=1, eos_token_id=2),
        audio_token_index=12, audio_start_token_id=10, audio_end_token_id=11,
        image_token_index=13, video_token_index=14, vision_start_token_id=15)
    config._attn_implementation = "eager"
    thinker = Qwen2_5OmniThinkerForConditionalGeneration(config).eval()
    features = torch.randn(1, 4, 41)
    ids = torch.tensor([[1, 10] + [12] * 10 + [11, 2]])
    state = ForwardState(input_ids=ids, attention_mask=torch.ones_like(ids), sample_ids=["fixture"],
        audio_values=features, audio_attention_mask=torch.ones(1, 41, dtype=torch.long),
        audio_feature_lengths=torch.tensor([41]),
        native_model_inputs=dict(input_ids=ids, attention_mask=torch.ones_like(ids),
            input_features=features, feature_attention_mask=torch.ones(1, 41, dtype=torch.long)),
        modality_token_spans=[TokenSpan(sample_index=0, start=2, end=12, modality="audio")],
        modality_metadata=dict(token_ids={"audio": 12}, pad_token_id=0, padding_side="right",
                               expected_token_counts_by_sample=[{"audio": 10}]))
    prompt_lengths = []

    def generate(*, return_audio, speaker, use_audio_in_video, **inputs):
        assert return_audio is False and use_audio_in_video is False
        prompt_lengths.append(inputs["input_ids"].shape[1])
        return thinker.generate(**inputs, max_new_tokens=2, do_sample=False,
                                eos_token_id=2, pad_token_id=0)

    runner = Qwen25OmniRunner()
    runtime = BackboneRuntime(model=SimpleNamespace(thinker=thinker, generate=generate),
        processor=SimpleNamespace(batch_decode=lambda ids, **kwargs: ["fixture"] * len(ids)),
        model_path_or_id="tiny-random-fixture", device=torch.device("cpu"), dtype=torch.float32,
        metadata={"runtime_mode": "default"})
    batch = ProcessedBatch(model_inputs=state.native_model_inputs, sample_ids=state.sample_ids,
                           prompt_lengths=[ids.shape[1]], metadata=state.modality_metadata)
    original_generate = runner.generate
    observer = AcousticObserver(thinker.audio_tower, sample_rate=16000, hop_length=160,
                                layers=[0, 1])
    observer.install()
    try:
        with torch.inference_mode():
            observer.begin()
            baseline = runner.generate(runtime, state, processed_batch=batch)
            native, _ = observer.finish()
            for condition in build_conditions("representation_spectrum"):
                observer.begin(condition=condition)
                result = runner.generate(runtime, state, processed_batch=batch)
                measurement, _ = observer.finish()
                if condition["nyquist_ratio"] == 1:
                    assert torch.equal(result.text.token_ids, baseline.text.token_ids)
                assert measurement["projector_tokens"] == native["projector_tokens"] == 10
                assert result.metadata["compression_mode"] == "none"
                assert result.metadata["input_sequence_length"] == ids.shape[1]
                assert result.text.token_ids.numel() > 0
                assert runner.generate == original_generate
            assert prompt_lengths == [ids.shape[1]] * 5
    finally:
        observer.close()
