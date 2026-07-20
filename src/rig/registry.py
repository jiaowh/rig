"""Adapter registry (implementation-plan §3): entry-point discovery ONLY, no static imports.

The hard core/adapter boundary holds transitively only if this module never
statically imports adapters - so adapters SELF-REGISTER via the
``rig.adapters`` packaging entry-point group and are discovered at runtime
with :func:`importlib.metadata.entry_points`. There is no static adapter
import list anywhere in ``rig`` (enforced by import-linter and a test).
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from rig.interfaces import ProcessAdapter, validate_adapter

ENTRY_POINT_GROUP = "rig.adapters"

# In-process registrations for tests (bypass packaging metadata).
_TEST_REGISTRY: dict[str, Callable[[], ProcessAdapter]] = {}


def register_adapter_for_testing(process_id: str, factory: Callable[[], ProcessAdapter]) -> None:
    """Register an adapter factory in-process (tests only - production
    discovery goes through the ``rig.adapters`` entry-point group)."""
    _TEST_REGISTRY[process_id] = factory


def clear_test_registry() -> None:
    """Remove all in-process test registrations."""
    _TEST_REGISTRY.clear()


def list_adapters() -> list[str]:
    """All discoverable adapter process_ids (entry points + test registry)."""
    names = {ep.name for ep in entry_points(group=ENTRY_POINT_GROUP)}
    names.update(_TEST_REGISTRY)
    return sorted(names)


def get_adapter(process_id: str, **kwargs: Any) -> ProcessAdapter:
    """Load, instantiate, and validate the adapter for ``process_id``.

    Entry points must resolve to a zero-/kwargs-arg factory (class or
    function) returning a :class:`~rig.interfaces.ProcessAdapter`. Every
    loaded adapter passes :func:`~rig.interfaces.validate_adapter`
    (includes the D7 physics-vs-verifier independence check).
    """
    factory: Callable[..., ProcessAdapter] | None = _TEST_REGISTRY.get(process_id)
    if factory is None:
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            if ep.name == process_id:
                factory = ep.load()
                break
    if factory is None:
        raise KeyError(
            f"no adapter registered for process_id {process_id!r} (known: {list_adapters()})"
        )
    adapter = factory(**kwargs)
    validate_adapter(adapter)
    return adapter
