"""Deterministic lifecycle for temporary analysis hooks."""

from __future__ import annotations

from typing import Any, Callable

from mllm_comp.core.errors import BackendIntegrationError


class HookSession:
    """Own hook handles and remove them in reverse order, including failures."""

    def __init__(self) -> None:
        self._handles: list[Any] = []
        self._closed = False

    def register_forward_pre_hook(
        self,
        module: Any,
        callback: Callable[..., Any],
        *,
        with_kwargs: bool = False,
    ) -> Any:
        options = {"with_kwargs": True} if with_kwargs else {}
        return self._register(
            module,
            "register_forward_pre_hook",
            callback,
            **options,
        )

    def register_forward_hook(
        self,
        module: Any,
        callback: Callable[..., Any],
    ) -> Any:
        return self._register(module, "register_forward_hook", callback)

    def _register(
        self,
        module: Any,
        method_name: str,
        callback: Callable[..., Any],
        **options: Any,
    ) -> Any:
        if self._closed:
            raise BackendIntegrationError("Cannot register a hook on a closed session.")
        register = getattr(module, method_name, None)
        if not callable(register):
            raise BackendIntegrationError(
                f"Analysis target {type(module).__name__} does not support {method_name}."
            )
        handle = register(callback, **options)
        if not callable(getattr(handle, "remove", None)):
            raise BackendIntegrationError(
                f"{method_name} did not return a removable hook handle."
            )
        self._handles.append(handle)
        return handle

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except Exception as exc:  # cleanup must attempt every owned handle
                errors.append(f"{type(exc).__name__}: {exc}")
        self._handles.clear()
        if errors:
            raise BackendIntegrationError(
                "Analysis hook cleanup failed: " + "; ".join(errors)
            )

    def __enter__(self) -> "HookSession":
        if self._closed:
            raise BackendIntegrationError("Cannot re-enter a closed hook session.")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.close()
        except BackendIntegrationError:
            if exc is None:
                raise
        return False


__all__ = ["HookSession"]
