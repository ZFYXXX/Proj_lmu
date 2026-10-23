"""Source provenance must distinguish code without reading experiment outputs."""

from pathlib import Path
import json
import stat
import subprocess

import pytest

from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp import run_manifest


def test_manifest_preserves_reloadable_effective_config_after_overrides() -> None:
    config = experiment_config_from_dict({
        "experiment": {"name": "effective-config"},
        "backbone": {"name": "qwen25_omni", "model_path": "${MODEL}"},
        "eval": {"tasks": ["old-task"], "generation": {"max_new_tokens": 16}},
        "output": {"run_dir": "old-run", "manifest_path": "old-run/manifest.json"},
    }, path="source.yaml")
    config.backbone.model_path = "actual-model"
    config.eval.tasks = ["actual-task"]
    config.eval.generation["max_new_tokens"] = 32
    config.output.run_dir = "actual-run"
    config.output.manifest_path = "actual-run/manifest.json"
    config.metadata["internal_note"] = "not part of the experiment"

    manifest = run_manifest.build_run_manifest(config)
    snapshot = json.loads(json.dumps(manifest.effective_config))
    restored = experiment_config_from_dict(snapshot)
    assert restored.backbone.model_path == "actual-model"
    assert restored.eval.tasks == ["actual-task"]
    assert restored.eval.generation["max_new_tokens"] == 32
    assert restored.eval.runtime_batch_size == 1
    assert restored.output == config.output
    assert manifest.config_path == "source.yaml"
    assert not {"raw", "metadata", "path"} & snapshot.keys()

    config.eval.generation["max_new_tokens"] = 64
    assert manifest.effective_config["eval"]["generation"]["max_new_tokens"] == 32


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    # Git creates read-only objects on Windows; allow the workspace fixture to
    # clean up only this test repository after assertions have finished.
    for path in (root / ".git" / "objects").rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD | stat.S_IWRITE)


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch):
    (root / "mllm_comp").mkdir()
    source = root / "mllm_comp" / "run_manifest.py"
    source.write_text("value = 1\n", encoding="utf-8")
    (root / "README.md").write_text("documentation\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "baseline")
    monkeypatch.setattr(run_manifest, "__file__", str(source))
    config = experiment_config_from_dict({
        "experiment": {"name": "source-fixture"},
        "backbone": {"name": "qwen25_omni", "model_id": "fixture"},
    })
    # The configuration need not reside in the executing source checkout.
    config.path = str(root.parent / "external.yaml")
    return config, source


def test_source_identity_ignores_artifacts_without_opening_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    config, source = _fixture(root, monkeypatch)
    initial = run_manifest._git_source_identity(config)
    assert initial[0] and initial[1] is False and initial[2]
    for directory in (".goal-runs", ".goal-logs", "outputs", "tests", "docs"):
        path = root / directory
        path.mkdir()
        (path / "run.py").write_text("not executed platform source", encoding="utf-8")
    (root / "README.md").write_text("edited docs", encoding="utf-8")
    original_open = Path.open
    opened = []

    def guarded_open(path, *args, **kwargs):
        assert path == source, f"Read unrelated artifact: {path}"
        opened.append(path)
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", guarded_open)
        assert run_manifest._git_source_identity(config) == initial
    assert opened == [source]
    source.write_text("value = 2\n", encoding="utf-8")
    edited = run_manifest._git_source_identity(config)
    assert edited[0] == initial[0] and edited[1] is True
    assert edited[2] != initial[2]


def test_untracked_added_deleted_and_staged_sources_affect_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    config, source = _fixture(root, monkeypatch)
    initial = run_manifest._git_source_identity(config)
    extra = source.with_name("new_kernel.py")
    extra.write_text("new_kernel = True\n", encoding="utf-8")
    untracked = run_manifest._git_source_identity(config)
    assert untracked[1] is True and untracked[2] != initial[2]
    _git(root, "add", str(extra))
    assert run_manifest._git_source_identity(config) == untracked
    source.unlink()
    deleted = run_manifest._git_source_identity(config)
    assert deleted[1] is True and deleted[2] != untracked[2]


def test_documentation_commit_does_not_change_runtime_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    config, _ = _fixture(root, monkeypatch)
    initial = run_manifest._git_source_identity(config)
    (root / "README.md").write_text("new docs\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "docs")
    updated = run_manifest._git_source_identity(config)
    assert updated[0] != initial[0]
    assert updated[1:] == initial[1:]


def test_source_deletion_identity_is_independent_of_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    config, source = _fixture(root, monkeypatch)
    initial = run_manifest._git_source_identity(config)
    source.unlink()
    deleted = run_manifest._git_source_identity(config)
    assert deleted[1] is True and deleted[2] != initial[2]
    _git(root, "add", "-u")
    assert run_manifest._git_source_identity(config) == deleted
    _git(root, "reset", "--quiet")
    assert run_manifest._git_source_identity(config) == deleted
    _git(root, "add", "-u")
    _git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "delete source")
    committed = run_manifest._git_source_identity(config)
    assert committed[0] != deleted[0]
    assert committed[1] is False and committed[2] == deleted[2]


@pytest.mark.parametrize("phase", ["analyze", "eval", "train"])
def test_command_manifests_publish_scoped_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    config, _ = _fixture(tmp_path, monkeypatch)
    kwargs = {"analysis_records": []} if phase == "analyze" else {}
    if phase == "train":
        from mllm_comp.core.manifest import BackendRecord
        kwargs["train_backend"] = BackendRecord(name="ms_swift", version="fixture")
    manifest = run_manifest.build_run_manifest(config, **kwargs)
    assert manifest.source_scope == "runtime_sources_v1"
    assert manifest.source_tree_hash and manifest.source_commit
    assert manifest.source_dirty is False


def test_installed_package_does_not_identify_config_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _fixture(tmp_path, monkeypatch)
    config.path = str(tmp_path / "config.yaml")
    monkeypatch.setattr(run_manifest, "__file__", str(tmp_path / "installed" / "mllm_comp" / "run_manifest.py"))
    assert run_manifest._git_source_identity(config) == (None, None, None)


@pytest.mark.parametrize("command", ["eval", "analyze_default", "analyze_explicit"])
def test_vendored_runtime_changes_are_scoped_to_executed_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    root = tmp_path
    config, _ = _fixture(root, monkeypatch)
    from mllm_comp.core.manifest import BackendRecord

    if command == "eval":
        config.eval.backend = "lmms_eval"
    elif command == "analyze_explicit":
        config.analysis.backend = "lmms_eval"
    kwargs = {} if command == "eval" else {"analysis_records": []}
    kwargs["eval_backend"] = BackendRecord(name="lmms_eval", version="fixture")

    def identity():
        manifest = run_manifest.build_run_manifest(config, **kwargs)
        return manifest.source_commit, manifest.source_dirty, manifest.source_tree_hash
    vendor = root / "third_party" / "frameworks" / "lmms-eval"
    package = vendor / "lmms_eval"
    package.mkdir(parents=True)
    source = package / "model.py"
    source.write_text("model = 1\n", encoding="utf-8")
    _git(vendor, "init")
    _git(vendor, "add", ".")
    _git(vendor, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "model")
    initial = identity()
    (vendor / "outputs").mkdir()
    (vendor / "outputs" / "samples.json").write_text("[]", encoding="utf-8")
    assert identity() == initial
    source.write_text("model = 2\n", encoding="utf-8")
    changed = identity()
    assert changed[1] is True and changed[2] != initial[2]
