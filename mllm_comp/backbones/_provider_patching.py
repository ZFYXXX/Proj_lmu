"""Small helpers for reversible instance-level provider patches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
from types import MethodType
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError


def preserve_bound_method_signature(
    wrapper: Callable[..., Any],
    original: Callable[..., Any],
) -> None:
    """Make a ``MethodType`` wrapper inspect like the original bound method.

    ``inspect.signature`` removes the first parameter of a bound function. A
    wrapper therefore needs one synthetic instance parameter in front of the
    already-bound provider signature; otherwise GenerationMixin loses fields
    such as ``position_ids`` when it inspects an instance patch.
    """

    signature = inspect.signature(original)
    instance = inspect.Parameter("_instance", inspect.Parameter.POSITIONAL_ONLY)
    setattr(
        wrapper,
        "__signature__",
        signature.replace(parameters=(instance, *signature.parameters.values())),
    )


@dataclass
class MethodPatch:
    """Own one reversible method replacement without changing its signature."""

    owner: Any
    name: str
    installed: Any
    had_instance_value: bool
    prior_instance_value: Any
    label: str

    @classmethod
    def apply(
        cls,
        owner: Any,
        name: str,
        replacement: Callable[..., Any],
        original: Callable[..., Any],
        *,
        label: str,
    ) -> "MethodPatch":
        preserve_bound_method_signature(replacement, original)
        installed = MethodType(replacement, owner)
        had = name in vars(owner)
        prior = vars(owner).get(name)
        setattr(owner, name, installed)
        return cls(owner, name, installed, had, prior, label)

    def restore(self) -> None:
        current = getattr(self.owner, self.name, None)
        same_function = getattr(current, "__func__", None) is getattr(
            self.installed, "__func__", None
        )
        if current is not self.installed and not same_function:
            raise BackendIntegrationError(
                f"{self.label} patched method {self.name!r} changed while active"
            )
        if self.had_instance_value:
            setattr(self.owner, self.name, self.prior_instance_value)
        else:
            delattr(self.owner, self.name)


__all__ = ["MethodPatch", "preserve_bound_method_signature"]
