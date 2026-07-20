"""Locate and import the external MBE physics simulator (``mbe_sim``).

The simulator is a sibling repository, NOT pip-installable (implementation-plan E2). This
module is the ONLY place that knows where it lives: the ``MBE_SIM_PATH``
environment variable points at the repo root (the directory *containing* the
``mbe_sim`` package), defaulting to ``c:\\Users\\Jiaow\\Documents\\github\\MBE sim``.
The path is inserted into ``sys.path`` lazily, on first use — importing this
module (or anything else in ``rig_adapters.mbe``) never touches the sim.

All sim access in ``rig_adapters.mbe`` goes through :func:`load_mbe_sim`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

MBE_SIM_ENV = "MBE_SIM_PATH"
DEFAULT_MBE_SIM_PATH = r"c:\Users\Jiaow\Documents\github\MBE sim"

_SIM: ModuleType | None = None


class MBESimNotFoundError(RuntimeError):
    """The mbe_sim package could not be located (bad/missing MBE_SIM_PATH)."""


def sim_path() -> Path:
    """The configured simulator repo root (may or may not exist)."""
    return Path(os.environ.get(MBE_SIM_ENV, DEFAULT_MBE_SIM_PATH))


def sim_available() -> bool:
    """True if the mbe_sim package is present at the configured path."""
    return (sim_path() / "mbe_sim" / "__init__.py").is_file()


def load_mbe_sim() -> ModuleType:
    """Import (once) and return the ``mbe_sim`` package.

    Raises :class:`MBESimNotFoundError` with an actionable message if the
    simulator repo is not where ``MBE_SIM_PATH`` says it is.
    """
    global _SIM
    if _SIM is not None:
        return _SIM
    root = sim_path()
    if not (root / "mbe_sim" / "__init__.py").is_file():
        raise MBESimNotFoundError(
            f"MBE simulator not found: expected package 'mbe_sim' under {root!s}. "
            f"Set the {MBE_SIM_ENV} environment variable to the 'MBE sim' repo root "
            "(the directory that CONTAINS the mbe_sim package), e.g. "
            rf"set {MBE_SIM_ENV}=c:\path\to\MBE sim"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    import mbe_sim  # noqa: PLC0415 — deliberate lazy import

    _SIM = mbe_sim
    return _SIM
