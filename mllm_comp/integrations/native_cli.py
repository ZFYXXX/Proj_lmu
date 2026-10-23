"""Shared helpers for launching upstream command-line interfaces.

The platform deliberately inherits the current terminal for stdout and stderr.
It only adds environment variables, forwards termination signals, and returns
the native process status for the thin command wrappers to propagate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
import os
from pathlib import Path
import signal
import subprocess
import threading
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError


@dataclass(frozen=True)
class NativeCLIPlan:
    """One fully resolved upstream command invocation."""

    argv: list[str]
    environment: dict[str, str]
    cwd: str
    expected_output_paths: list[str]
    unset_environment: list[str] = field(default_factory=list, kw_only=True)


@dataclass(frozen=True)
class NativeCLIResult:
    """Completion status returned without capturing terminal output."""

    returncode: int

    @property
    def shell_returncode(self) -> int:
        """Normalize POSIX signal return codes to the conventional 128+signal."""

        return 128 + abs(self.returncode) if self.returncode < 0 else self.returncode


class NativeCLIProcessError(BackendIntegrationError):
    """Raised after an upstream CLI exits unsuccessfully."""

    def __init__(self, backend: str, returncode: int) -> None:
        self.backend = backend
        self.returncode = int(returncode)
        self.shell_returncode = (
            128 + abs(self.returncode) if self.returncode < 0 else self.returncode
        )
        super().__init__(
            f"{backend} native CLI exited with status {self.returncode}."
        )


def distribution_version(*names: str) -> str:
    """Return the installed version for the first matching distribution."""

    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "not-installed"


def installed_versions(*, primary: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Collect actual package versions without importing heavyweight runtimes."""

    versions = {
        label: distribution_version(*distribution_names)
        for label, distribution_names in primary.items()
    }
    versions.setdefault(
        "mllm_comp_platform",
        distribution_version("mllm-comp-platform"),
    )
    versions.setdefault("transformers", distribution_version("transformers"))
    versions.setdefault("torch", distribution_version("torch"))
    return versions


def run_native_cli(
    plan: NativeCLIPlan,
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> NativeCLIResult:
    """Run an upstream CLI with inherited streams and forwarded INT/TERM signals."""

    environment = os.environ.copy()
    for name in plan.unset_environment:
        environment.pop(name, None)
    environment.update(plan.environment)
    process = popen_factory(
        list(plan.argv),
        cwd=plan.cwd,
        env=environment,
    )

    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)

    try:
        return NativeCLIResult(returncode=int(process.wait()))
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        return NativeCLIResult(returncode=int(process.wait()))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def resolved_cwd(value: str | os.PathLike[str] | None = None) -> str:
    """Return a stable absolute working directory for command records."""

    return str(Path(value or Path.cwd()).resolve())
