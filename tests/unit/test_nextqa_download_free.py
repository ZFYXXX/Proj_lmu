from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


_TASK_DIR = (
    Path(__file__).resolve().parents[2]
    / "third_party/frameworks/lmms-eval/lmms_eval/tasks/nextqa"
)
_UTILS_PATH = _TASK_DIR / "utils.py"


def _package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = []  # type: ignore[attr-defined]
    return package


def _load_utils(monkeypatch: pytest.MonkeyPatch, suffix: str) -> ModuleType:
    video_loader = ModuleType("lmms_eval.tasks._task_utils.video_loader")
    video_loader.get_cache_dir = lambda _config, subdir: f"/cache/{subdir}"
    video_loader.get_video = lambda cache, video: f"{cache}/{video}.mp4"
    modules = {
        "lmms_eval": _package("lmms_eval"),
        "lmms_eval.tasks": _package("lmms_eval.tasks"),
        "lmms_eval.tasks._task_utils": _package("lmms_eval.tasks._task_utils"),
        "lmms_eval.tasks._task_utils.video_loader": video_loader,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    module_name = f"_nextqa_download_free_{suffix}"
    spec = importlib.util.spec_from_file_location(module_name, _UTILS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_oe_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_resources: bool,
) -> SimpleNamespace:
    calls = SimpleNamespace(find=[], download=[])

    class _Data:
        @staticmethod
        def find(path: str) -> object:
            calls.find.append(path)
            if missing_resources:
                raise LookupError(path)
            return object()

    class _Synset:
        @staticmethod
        def wup_similarity(_other: object) -> float:
            return 0.75

    nltk = _package("nltk")
    nltk.data = _Data()
    nltk.download = lambda *args, **kwargs: calls.download.append((args, kwargs))
    corpus = ModuleType("nltk.corpus")
    corpus.wordnet = SimpleNamespace(synsets=lambda _word: [_Synset()])
    tokenize = ModuleType("nltk.tokenize")
    tokenize.word_tokenize = lambda value: value.split()
    pywsd = _package("pywsd")
    pywsd_utils = ModuleType("pywsd.utils")
    pywsd_utils.lemmatize_sentence = lambda value: value.lower().split()
    for name, module in {
        "nltk": nltk,
        "nltk.corpus": corpus,
        "nltk.tokenize": tokenize,
        "pywsd": pywsd,
        "pywsd.utils": pywsd_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_nextqa_mc_import_and_fixture_never_touch_oe_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = {"nltk", "pandas", "pywsd"}
    attempted: list[str] = []
    native_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in forbidden:
            attempted.append(name)
            raise AssertionError(f"MC import touched OE dependency {name}")
        return native_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = _load_utils(monkeypatch, "mc")
    assert module.__name__ not in sys.modules
    doc = {
        "video": "3064389881",
        "question": "Which option is correct?",
        "a0": "one",
        "a1": "two",
        "a2": "three",
        "a3": "four",
        "a4": "five",
        "answer": 3,
    }

    prompt = module.nextqa_doc_to_text_mc(doc)

    assert attempted == []
    assert module.config["metadata"]["load_package"] is False
    assert module._oe_runtime is None
    assert prompt.splitlines() == [
        "Which option is correct?",
        "A. one",
        "B. two",
        "C. three",
        "D. four",
        "E. five",
    ]
    assert module.nextqa_doc_to_visual(doc) == ["/cache/NExTVideo/3064389881.mp4"]
    assert module.OPTIONS[module.nextqa_doc_to_target(doc)] == "D"
    assert module.nextqa_mc_process_results(doc, ["D"])["exact_match"] is True


def test_nextqa_oe_missing_resources_fail_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_utils(monkeypatch, "missing")
    calls = _install_fake_oe_dependencies(
        monkeypatch,
        missing_resources=True,
    )
    monkeypatch.delitem(sys.modules, "pywsd")
    monkeypatch.delitem(sys.modules, "pywsd.utils")
    native_import = builtins.__import__
    pywsd_imports: list[str] = []

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] == "pywsd":
            pywsd_imports.append(name)
            raise AssertionError(
                "missing NLTK resources should fail before pywsd import"
            )
        return native_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(
        module.NextQAOEDependencyError,
        match="Automatic downloads are disabled",
    ) as exc_info:
        module._load_oe_runtime()

    assert "wordnet" in str(exc_info.value)
    assert len(calls.find) == 6
    assert calls.download == []
    assert pywsd_imports == []
    assert module._oe_runtime is None


def test_nextqa_oe_runtime_loads_existing_resources_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_utils(monkeypatch, "present")
    calls = _install_fake_oe_dependencies(
        monkeypatch,
        missing_resources=False,
    )

    assert module.remove_stop("the cat") == "cat"
    assert module.get_wups("cat", "cat", 0.9) == 1.0
    assert len(calls.find) == 6
    assert calls.download == []

    assert module.remove_stop("the dog") == "dog"
    assert len(calls.find) == 6
