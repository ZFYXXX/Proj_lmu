"""Optional torch-backed dense Qwen modality-split public API."""
__all__ = ["install_modality_split", "load_modality_split", "save_modality_split",
           "native_state_dict", "ModalitySplitSession", "ModalitySplitCache"]


def __getattr__(name):
    # Definition/plan compilation remains usable without importing torch.
    if name in __all__:
        from importlib import import_module
        module = import_module(".cache" if name == "ModalitySplitCache" else ".core", __name__)
        return getattr(module, name)
    raise AttributeError(name)
